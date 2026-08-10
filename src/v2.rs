//! Experimental typed, recording-first API.
//!
//! This module is intentionally additive while the crate validates its next
//! resident-buffer interface. It keeps raw [`wgpu`] interoperability, but
//! carries element type, suballocation range, capacity, and logical extent in
//! Rust values instead of repeating them across primitive-specific methods.

use std::{marker::PhantomData, ops::Range};

use crate::{
    Compactor, Context, CountedSortDispatch, Error, GpuCountPlan, Reducer, Sorter, U32Reduction,
    common::buffers::BufferRange,
};

mod sealed {
    pub trait Sealed {}

    impl Sealed for u32 {}
}

/// Element types supported by typed resident-buffer views.
///
/// The trait is sealed because accepting a Rust type is also a promise that
/// the crate's WGSL kernels understand its exact storage representation.
pub trait GpuElement: sealed::Sealed + bytemuck::Pod {
    #[doc(hidden)]
    const SIZE_BYTES: u64;
}

impl GpuElement for u32 {
    const SIZE_BYTES: u64 = size_of::<Self>() as u64;
}

/// A GPU-resident scalar count written and consumed by compute commands.
#[derive(Clone, Copy)]
pub struct GpuCount<'a> {
    buffer: &'a wgpu::Buffer,
    offset: u64,
}

impl<'a> GpuCount<'a> {
    /// Uses the first `u32` in `buffer` as a logical item count.
    pub fn new(buffer: &'a wgpu::Buffer) -> Result<Self, Error> {
        Self::at(buffer, 0)
    }

    /// Uses the `u32` at `byte_offset` as a logical item count.
    ///
    /// Construction checks buffer bounds. Primitive recording additionally
    /// checks the device's storage-binding offset alignment.
    pub fn at(buffer: &'a wgpu::Buffer, byte_offset: u64) -> Result<Self, Error> {
        BufferRange::new(
            buffer,
            byte_offset,
            size_of::<u32>() as u64,
            "GPU item count",
        )?;
        Ok(Self {
            buffer,
            offset: byte_offset,
        })
    }

    /// Returns the underlying caller-owned buffer.
    pub const fn buffer(self) -> &'a wgpu::Buffer {
        self.buffer
    }

    /// Returns the count's byte offset in its buffer.
    pub const fn byte_offset(self) -> u64 {
        self.offset
    }

    fn range(self) -> BufferRange<'a> {
        BufferRange {
            buffer: self.buffer,
            offset: self.offset,
            size: size_of::<u32>() as u64,
        }
    }
}

/// The number of initialized elements represented by a [`GpuSlice`].
#[derive(Clone, Copy)]
pub enum Extent<'a> {
    /// The CPU knows the exact number of initialized elements.
    Fixed(u32),
    /// A GPU scalar contains the exact length; the slice stores its CPU-known
    /// allocation capacity separately.
    Gpu(GpuCount<'a>),
}

/// A typed, read-only view into a caller-owned GPU buffer.
#[derive(Clone, Copy)]
pub struct GpuSlice<'a, T: GpuElement> {
    buffer: &'a wgpu::Buffer,
    byte_offset: u64,
    capacity: u32,
    extent: Extent<'a>,
    _element: PhantomData<T>,
}

impl<'a, T: GpuElement> GpuSlice<'a, T> {
    /// Creates a fixed-length view from an element-index range.
    pub fn from_range(buffer: &'a wgpu::Buffer, range: Range<u32>) -> Result<Self, Error> {
        let capacity = range
            .end
            .checked_sub(range.start)
            .ok_or(Error::SizeOverflow)?;
        Self::from_parts(buffer, range.start, capacity, Extent::Fixed(capacity))
    }

    /// Creates a capacity-bounded view whose exact length remains on the GPU.
    pub fn counted(
        buffer: &'a wgpu::Buffer,
        range: Range<u32>,
        count: GpuCount<'a>,
    ) -> Result<Self, Error> {
        let capacity = range
            .end
            .checked_sub(range.start)
            .ok_or(Error::SizeOverflow)?;
        Self::from_parts(buffer, range.start, capacity, Extent::Gpu(count))
    }

    fn from_parts(
        buffer: &'a wgpu::Buffer,
        first_element: u32,
        capacity: u32,
        extent: Extent<'a>,
    ) -> Result<Self, Error> {
        let byte_offset = u64::from(first_element)
            .checked_mul(T::SIZE_BYTES)
            .ok_or(Error::SizeOverflow)?;
        let size = u64::from(capacity)
            .checked_mul(T::SIZE_BYTES)
            .ok_or(Error::SizeOverflow)?;
        BufferRange::new(buffer, byte_offset, size, "GPU slice")?;
        Ok(Self {
            buffer,
            byte_offset,
            capacity,
            extent,
            _element: PhantomData,
        })
    }

    /// Returns the physical allocation bound in elements.
    pub const fn capacity(self) -> u32 {
        self.capacity
    }

    /// Returns the fixed or GPU-resident logical extent.
    pub const fn extent(self) -> Extent<'a> {
        self.extent
    }

    /// Returns the underlying caller-owned buffer.
    pub const fn buffer(self) -> &'a wgpu::Buffer {
        self.buffer
    }

    /// Returns the view's byte offset in the underlying buffer.
    pub const fn byte_offset(self) -> u64 {
        self.byte_offset
    }

    fn range(self) -> BufferRange<'a> {
        BufferRange {
            buffer: self.buffer,
            offset: self.byte_offset,
            size: u64::from(self.capacity) * T::SIZE_BYTES,
        }
    }
}

/// A typed writable allocation range in a caller-owned GPU buffer.
///
/// `Mut` describes shader access, not Rust-exclusive ownership: views are
/// copyable because wgpu buffers are handles. A primitive's read and write
/// views must use distinct buffer handles: wgpu treats a writable storage
/// binding as exclusive even when static binding ranges do not overlap.
#[derive(Clone, Copy)]
pub struct GpuSliceMut<'a, T: GpuElement> {
    buffer: &'a wgpu::Buffer,
    byte_offset: u64,
    capacity: u32,
    _element: PhantomData<T>,
}

impl<'a, T: GpuElement> GpuSliceMut<'a, T> {
    /// Creates a writable view from an element-index range.
    pub fn from_range(buffer: &'a wgpu::Buffer, range: Range<u32>) -> Result<Self, Error> {
        let capacity = range
            .end
            .checked_sub(range.start)
            .ok_or(Error::SizeOverflow)?;
        let byte_offset = u64::from(range.start)
            .checked_mul(T::SIZE_BYTES)
            .ok_or(Error::SizeOverflow)?;
        let size = u64::from(capacity)
            .checked_mul(T::SIZE_BYTES)
            .ok_or(Error::SizeOverflow)?;
        BufferRange::new(buffer, byte_offset, size, "writable GPU slice")?;
        Ok(Self {
            buffer,
            byte_offset,
            capacity,
            _element: PhantomData,
        })
    }

    /// Returns the physical allocation bound in elements.
    pub const fn capacity(self) -> u32 {
        self.capacity
    }

    /// Returns the underlying caller-owned buffer.
    pub const fn buffer(self) -> &'a wgpu::Buffer {
        self.buffer
    }

    /// Returns the view's byte offset in the underlying buffer.
    pub const fn byte_offset(self) -> u64 {
        self.byte_offset
    }

    fn range(self) -> BufferRange<'a> {
        BufferRange {
            buffer: self.buffer,
            offset: self.byte_offset,
            size: u64::from(self.capacity) * T::SIZE_BYTES,
        }
    }

    fn initialized(self, capacity: u32, extent: Extent<'a>) -> GpuSlice<'a, T> {
        debug_assert!(capacity <= self.capacity);
        GpuSlice {
            buffer: self.buffer,
            byte_offset: self.byte_offset,
            capacity,
            extent,
            _element: PhantomData,
        }
    }
}

/// Radix-sort policy that does not multiply the number of entry points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SortOptions {
    key_bits: u32,
}

impl SortOptions {
    /// Sorts the full `u32` key width.
    pub const fn full_width() -> Self {
        Self {
            key_bits: u32::BITS,
        }
    }

    /// Declares the number of significant low key bits.
    pub const fn key_bits(mut self, key_bits: u32) -> Self {
        self.key_bits = key_bits;
        self
    }
}

impl Default for SortOptions {
    fn default() -> Self {
        Self::full_width()
    }
}

/// Capacity-dependent workspaces to allocate before command recording.
///
/// Build only the operations a pipeline will use. Fixed and GPU-counted paths
/// can require different scratch, so they are selected independently.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[must_use]
pub struct WorkspaceRequirements {
    capacity: u32,
    compact: bool,
    fixed_sort: bool,
    counted_sort: bool,
    fixed_reduce: bool,
    counted_reduce: bool,
}

impl WorkspaceRequirements {
    /// Starts a workspace request for at most `capacity` `u32` elements.
    pub const fn new(capacity: u32) -> Self {
        Self {
            capacity,
            compact: false,
            fixed_sort: false,
            counted_sort: false,
            fixed_reduce: false,
            counted_reduce: false,
        }
    }

    /// Reserves stream-compaction workspace.
    pub const fn compact(mut self) -> Self {
        self.compact = true;
        self
    }

    /// Reserves fixed-extent radix-sort workspace.
    pub const fn fixed_sort(mut self) -> Self {
        self.fixed_sort = true;
        self
    }

    /// Reserves GPU-counted radix-sort workspace.
    pub const fn counted_sort(mut self) -> Self {
        self.counted_sort = true;
        self
    }

    /// Reserves fixed-extent reduction workspace.
    pub const fn fixed_reduce(mut self) -> Self {
        self.fixed_reduce = true;
        self
    }

    /// Reserves GPU-counted reduction workspace.
    pub const fn counted_reduce(mut self) -> Self {
        self.counted_reduce = true;
        self
    }
}

struct CachedCountPlan {
    count: wgpu::Buffer,
    count_offset: u64,
    capacity: u32,
    plan: GpuCountPlan,
}

/// Reusable primitive pipelines and workspace for recording resident commands.
pub struct Primitives {
    device: wgpu::Device,
    compactor: Compactor,
    sorter: Sorter,
    reducer: Reducer,
    count_plans: Vec<CachedCountPlan>,
    prepared_count_plans: Vec<usize>,
}

impl Primitives {
    /// Creates the experimental recorder over an existing wgpu device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            device: device.clone(),
            compactor: Compactor::new(device, queue),
            sorter: Sorter::new(device, queue),
            reducer: Reducer::new(device, queue),
            count_plans: Vec::new(),
            prepared_count_plans: Vec::new(),
        }
    }

    /// Creates the recorder from the crate's convenience context.
    pub fn from_context(context: &Context) -> Self {
        Self::new(&context.device, &context.queue)
    }

    /// Reserves the requested capacity-dependent GPU workspaces.
    ///
    /// This prevents workspace growth while recording operations up to
    /// `capacity`. Recording still creates lightweight bind groups and uniform
    /// buffers; this method is not an allocation-free-recording guarantee.
    pub fn reserve_workspace(&mut self, requirements: WorkspaceRequirements) -> Result<(), Error> {
        let capacity = requirements.capacity;
        if requirements.compact {
            self.compactor.reserve(capacity)?;
        }
        if requirements.fixed_sort {
            self.sorter.reserve_fixed(capacity)?;
        }
        if requirements.counted_sort {
            self.sorter.reserve_counted(capacity)?;
        }
        if requirements.fixed_reduce {
            self.reducer.reserve_fixed(capacity)?;
        }
        if requirements.counted_reduce {
            self.reducer.reserve_counted(capacity)?;
        }
        Ok(())
    }

    /// Creates and caches the metadata shared by counted sort and reduction.
    ///
    /// Call this before [`Self::record`] when resource creation during command
    /// recording is undesirable.
    pub fn reserve_count(&mut self, count: GpuCount<'_>, capacity: u32) -> Result<(), Error> {
        self.count_plan_index(count, capacity)?;
        self.prepared_count_plans.reserve(self.count_plans.len());
        Ok(())
    }

    /// Drops cached count-specific metadata.
    ///
    /// Use this when an application retires transient count buffers; otherwise
    /// plans remain cached for reuse for the lifetime of `Primitives`.
    pub fn clear_count_cache(&mut self) {
        self.count_plans.clear();
        self.prepared_count_plans.clear();
    }

    /// Borrows the primitive workspace and command encoder for ordered recording.
    pub fn record<'primitives, 'encoder>(
        &'primitives mut self,
        encoder: &'encoder mut wgpu::CommandEncoder,
    ) -> Recorder<'primitives, 'encoder> {
        self.prepared_count_plans.clear();
        Recorder {
            primitives: self,
            encoder,
        }
    }

    fn count_plan_index(&mut self, count: GpuCount<'_>, capacity: u32) -> Result<usize, Error> {
        if let Some(index) = self.count_plans.iter().position(|cached| {
            &cached.count == count.buffer
                && cached.count_offset == count.offset
                && cached.capacity == capacity
        }) {
            return Ok(index);
        }
        let plan = GpuCountPlan::new_with_count_range(
            &self.device,
            count.range(),
            capacity,
            CountedSortDispatch::Indirect,
        )?;
        self.count_plans.push(CachedCountPlan {
            count: count.buffer.clone(),
            count_offset: count.offset,
            capacity,
            plan,
        });
        Ok(self.count_plans.len() - 1)
    }
}

/// Ordered, recording-only access to resident GPU primitives.
///
/// Methods append commands to the borrowed encoder. They never submit, wait,
/// map buffers, or read a GPU-resident extent back to the CPU.
pub struct Recorder<'primitives, 'encoder> {
    primitives: &'primitives mut Primitives,
    encoder: &'encoder mut wgpu::CommandEncoder,
}

impl Recorder<'_, '_> {
    /// Stably compacts a fixed-length input and returns a slice carrying the
    /// GPU-resident selected count.
    pub fn compact<'a>(
        &mut self,
        input: GpuSlice<'a, u32>,
        mask: GpuSlice<'a, u32>,
        output: GpuSliceMut<'a, u32>,
        count: GpuCount<'a>,
    ) -> Result<GpuSlice<'a, u32>, Error> {
        let Extent::Fixed(num_items) = input.extent else {
            return Err(Error::UnsupportedDynamicExtent {
                operation: "stream compaction",
            });
        };
        let Extent::Fixed(mask_items) = mask.extent else {
            return Err(Error::UnsupportedDynamicExtent {
                operation: "stream-compaction mask",
            });
        };
        if num_items != mask_items {
            return Err(Error::CompactionLengthMismatch {
                input: num_items as usize,
                mask: mask_items as usize,
            });
        }
        if output.capacity < num_items {
            return Err(Error::BufferTooSmall {
                name: "compaction output view",
                required: u64::from(num_items) * size_of::<u32>() as u64,
                actual: u64::from(output.capacity) * size_of::<u32>() as u64,
            });
        }
        self.primitives.compactor.record_compact_ranges(
            self.encoder,
            input.range(),
            mask.range(),
            output.range(),
            count.range(),
            num_items,
        )?;
        self.invalidate_count(count);
        Ok(output.initialized(num_items, Extent::Gpu(count)))
    }

    /// Stably sorts a fixed or GPU-counted `u32` slice.
    pub fn sort<'a>(
        &mut self,
        input: GpuSlice<'a, u32>,
        output: GpuSliceMut<'a, u32>,
        options: SortOptions,
    ) -> Result<GpuSlice<'a, u32>, Error> {
        if output.capacity < input.capacity {
            return Err(Error::BufferTooSmall {
                name: "sort output view",
                required: u64::from(input.capacity) * size_of::<u32>() as u64,
                actual: u64::from(output.capacity) * size_of::<u32>() as u64,
            });
        }
        match input.extent {
            Extent::Fixed(num_items) => self.primitives.sorter.record_sort_ranges(
                self.encoder,
                input.range(),
                output.range(),
                num_items,
                options.key_bits,
            )?,
            Extent::Gpu(count) => {
                let plan_index = self.prepare_count(count, input.capacity)?;
                let (sorter, plans) = (&mut self.primitives.sorter, &self.primitives.count_plans);
                sorter.record_sort_ranges_with_count_plan(
                    self.encoder,
                    input.range(),
                    output.range(),
                    &plans[plan_index].plan,
                    options.key_bits,
                )?;
            }
        }
        Ok(output.initialized(input.capacity, input.extent))
    }

    /// Reduces a fixed or GPU-counted `u32` slice into one caller-owned scalar.
    pub fn reduce(
        &mut self,
        input: GpuSlice<'_, u32>,
        output: GpuSliceMut<'_, u32>,
        operation: U32Reduction,
    ) -> Result<(), Error> {
        if output.capacity < 1 {
            return Err(Error::BufferTooSmall {
                name: "reduction output view",
                required: size_of::<u32>() as u64,
                actual: 0,
            });
        }
        match input.extent {
            Extent::Fixed(num_items) => self.primitives.reducer.record_reduce_ranges(
                self.encoder,
                input.range(),
                output.range(),
                num_items,
                operation,
            ),
            Extent::Gpu(count) => {
                let plan_index = self.prepare_count(count, input.capacity)?;
                let (reducer, plans) = (&mut self.primitives.reducer, &self.primitives.count_plans);
                reducer.record_reduce_ranges_with_count_plan(
                    self.encoder,
                    input.range(),
                    output.range(),
                    &plans[plan_index].plan,
                    operation,
                )
            }
        }
    }

    fn prepare_count(&mut self, count: GpuCount<'_>, capacity: u32) -> Result<usize, Error> {
        let index = self.primitives.count_plan_index(count, capacity)?;
        if !self.primitives.prepared_count_plans.contains(&index) {
            self.primitives.count_plans[index]
                .plan
                .record_prepare(self.encoder);
            self.primitives.prepared_count_plans.push(index);
        }
        Ok(index)
    }

    fn invalidate_count(&mut self, count: GpuCount<'_>) {
        let (prepared, plans) = (
            &mut self.primitives.prepared_count_plans,
            &self.primitives.count_plans,
        );
        prepared.retain(|&index| {
            let cached = &plans[index];
            &cached.count != count.buffer || cached.count_offset != count.offset
        });
    }
}

#[cfg(test)]
mod tests {
    use super::SortOptions;

    #[test]
    fn sort_options_default_to_full_width() {
        assert_eq!(SortOptions::default().key_bits, u32::BITS);
        assert_eq!(SortOptions::default().key_bits(16).key_bits, 16);
    }
}
