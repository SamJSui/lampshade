//! Typed, recording-first GPU primitive composition.
//!
//! This module keeps raw [`wgpu`] interoperability while carrying element
//! type, suballocation range, capacity, and logical extent in Rust values
//! instead of repeating them across primitive-specific methods. It records
//! commands into a caller-owned encoder and never submits or reads back
//! implicitly.

use std::{marker::PhantomData, ops::Range};

use crate::{
    Compactor, Context, CountedSortDispatch, Error, GpuCountPlan, KeyValue, KeyValueCompactor,
    KeyValueField, KeyValueSorter, MaskGenerator, Reducer, RunLengthEncoder, Sorter, U32Predicate,
    U32Reduction, common::buffers::BufferRange, run_length::RunLengthOutputRanges,
};

mod sealed {
    pub trait Sealed {}

    impl Sealed for u32 {}
    impl Sealed for crate::KeyValue {}
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

impl GpuElement for KeyValue {
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

/// Primitive pipelines and capacity-dependent workspaces to prepare before recording.
///
/// Build only the operations a pipeline will use. Fixed and GPU-counted paths
/// can require different scratch, so they are selected independently.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[must_use]
pub struct WorkspaceRequirements {
    capacity: u32,
    predicate: bool,
    compact: bool,
    compact_key_values: bool,
    fixed_sort: bool,
    counted_sort: bool,
    fixed_key_value_sort: bool,
    counted_key_value_sort: bool,
    fixed_reduce: bool,
    counted_reduce: bool,
    run_length_encode: bool,
}

impl WorkspaceRequirements {
    /// Starts a workspace request for at most `capacity` elements or records.
    pub const fn new(capacity: u32) -> Self {
        Self {
            capacity,
            predicate: false,
            compact: false,
            compact_key_values: false,
            fixed_sort: false,
            counted_sort: false,
            fixed_key_value_sort: false,
            counted_key_value_sort: false,
            fixed_reduce: false,
            counted_reduce: false,
            run_length_encode: false,
        }
    }

    /// Prepares predicate-mask pipelines.
    pub const fn predicate(mut self) -> Self {
        self.predicate = true;
        self
    }

    /// Reserves stream-compaction workspace.
    pub const fn compact(mut self) -> Self {
        self.compact = true;
        self
    }

    /// Reserves key-value stream-compaction workspace.
    pub const fn compact_key_values(mut self) -> Self {
        self.compact_key_values = true;
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

    /// Reserves fixed-extent key-value radix-sort workspace.
    pub const fn fixed_key_value_sort(mut self) -> Self {
        self.fixed_key_value_sort = true;
        self
    }

    /// Reserves GPU-counted key-value radix-sort workspace.
    pub const fn counted_key_value_sort(mut self) -> Self {
        self.counted_key_value_sort = true;
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

    /// Reserves run-length-encoding workspace.
    pub const fn run_length_encode(mut self) -> Self {
        self.run_length_encode = true;
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
    queue: wgpu::Queue,
    adapter_info: Option<wgpu::AdapterInfo>,
    generator: Option<MaskGenerator>,
    compactor: Option<Compactor>,
    key_value_compactor: Option<KeyValueCompactor>,
    sorter: Option<Sorter>,
    key_value_sorter: Option<KeyValueSorter>,
    reducer: Option<Reducer>,
    run_length_encoder: Option<RunLengthEncoder>,
    count_plans: Vec<CachedCountPlan>,
    prepared_count_plans: Vec<usize>,
}

impl Primitives {
    /// Creates the primitive workspace over an existing wgpu device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
            adapter_info: None,
            generator: None,
            compactor: None,
            key_value_compactor: None,
            sorter: None,
            key_value_sorter: None,
            reducer: None,
            run_length_encoder: None,
            count_plans: Vec::new(),
            prepared_count_plans: Vec::new(),
        }
    }

    /// Creates the recorder over an existing device and queue with adapter
    /// metadata available for hardware-specific primitive selection.
    ///
    /// Prefer this constructor when the application owns its wgpu context and
    /// already has the [`wgpu::AdapterInfo`] returned by [`wgpu::Adapter::get_info`].
    /// [`Self::new`] remains the portable fallback when adapter metadata is not
    /// available.
    pub fn new_for_adapter(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter_info: &wgpu::AdapterInfo,
    ) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
            adapter_info: Some(adapter_info.clone()),
            generator: None,
            compactor: None,
            key_value_compactor: None,
            sorter: None,
            key_value_sorter: None,
            reducer: None,
            run_length_encoder: None,
            count_plans: Vec::new(),
            prepared_count_plans: Vec::new(),
        }
    }

    /// Creates the recorder from the crate's convenience context.
    pub fn from_context(context: &Context) -> Self {
        Self::new_for_adapter(&context.device, &context.queue, &context.adapter_info)
    }

    /// Prepares the requested pipelines and capacity-dependent GPU workspaces.
    ///
    /// This prevents workspace growth while recording operations up to
    /// `capacity`. Recording still creates lightweight bind groups and uniform
    /// buffers; this method is not an allocation-free-recording guarantee.
    pub fn reserve_workspace(&mut self, requirements: WorkspaceRequirements) -> Result<(), Error> {
        let capacity = requirements.capacity;
        if requirements.predicate {
            self.generator();
        }
        if requirements.compact {
            self.compactor().reserve(capacity)?;
        }
        if requirements.compact_key_values {
            self.key_value_compactor().reserve(capacity)?;
        }
        if requirements.fixed_sort {
            self.sorter().reserve_fixed(capacity)?;
        }
        if requirements.counted_sort {
            self.sorter().reserve_counted(capacity)?;
        }
        if requirements.fixed_key_value_sort {
            self.key_value_sorter().reserve_fixed(capacity)?;
        }
        if requirements.counted_key_value_sort {
            self.key_value_sorter().reserve_counted(capacity)?;
        }
        if requirements.fixed_reduce {
            self.reducer().reserve_fixed(capacity)?;
        }
        if requirements.counted_reduce {
            self.reducer().reserve_counted(capacity)?;
        }
        if requirements.run_length_encode {
            self.run_length_encoder().reserve(capacity)?;
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

    fn generator(&mut self) -> &MaskGenerator {
        self.generator
            .get_or_insert_with(|| MaskGenerator::new(&self.device, &self.queue))
    }

    fn compactor(&mut self) -> &mut Compactor {
        if self.compactor.is_none() {
            self.compactor = Some(match &self.adapter_info {
                Some(adapter_info) => Compactor::from_context(&Context {
                    adapter_info: adapter_info.clone(),
                    device: self.device.clone(),
                    queue: self.queue.clone(),
                }),
                None => Compactor::new(&self.device, &self.queue),
            });
        }
        self.compactor.as_mut().expect("compactor is initialized")
    }

    fn sorter(&mut self) -> &mut Sorter {
        if self.sorter.is_none() {
            self.sorter = Some(match &self.adapter_info {
                Some(adapter_info) => {
                    Sorter::new_for_adapter(&self.device, &self.queue, adapter_info)
                }
                None => Sorter::new(&self.device, &self.queue),
            });
        }
        self.sorter.as_mut().expect("sorter is initialized")
    }

    fn reducer(&mut self) -> &mut Reducer {
        self.reducer
            .get_or_insert_with(|| Reducer::new(&self.device, &self.queue))
    }

    fn run_length_encoder(&mut self) -> &mut RunLengthEncoder {
        if self.run_length_encoder.is_none() {
            self.run_length_encoder = Some(match &self.adapter_info {
                Some(adapter_info) => RunLengthEncoder::from_context(&Context {
                    adapter_info: adapter_info.clone(),
                    device: self.device.clone(),
                    queue: self.queue.clone(),
                }),
                None => RunLengthEncoder::new(&self.device, &self.queue),
            });
        }
        self.run_length_encoder
            .as_mut()
            .expect("run-length encoder is initialized")
    }

    fn key_value_compactor(&mut self) -> &mut KeyValueCompactor {
        if self.key_value_compactor.is_none() {
            self.key_value_compactor = Some(match &self.adapter_info {
                Some(adapter_info) => {
                    let context = Context {
                        adapter_info: adapter_info.clone(),
                        device: self.device.clone(),
                        queue: self.queue.clone(),
                    };
                    KeyValueCompactor::from_context(&context)
                }
                None => KeyValueCompactor::new(&self.device, &self.queue),
            });
        }
        self.key_value_compactor
            .as_mut()
            .expect("key-value compactor is initialized")
    }

    fn key_value_sorter(&mut self) -> &mut KeyValueSorter {
        if self.key_value_sorter.is_none() {
            self.key_value_sorter = Some(match &self.adapter_info {
                Some(adapter_info) => {
                    KeyValueSorter::new_for_adapter(&self.device, &self.queue, adapter_info)
                }
                None => KeyValueSorter::new(&self.device, &self.queue),
            });
        }
        self.key_value_sorter
            .as_mut()
            .expect("key-value sorter is initialized")
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

/// GPU-counted unique values and their adjacent run lengths.
#[derive(Clone, Copy)]
pub struct RunLengthOutput<'a> {
    /// One value for each adjacent input run.
    pub unique_values: GpuSlice<'a, u32>,
    /// The item count for each adjacent input run.
    pub run_lengths: GpuSlice<'a, u32>,
}

impl Recorder<'_, '_> {
    /// Tests a fixed-length `u32` slice and returns a same-length `0`/`1` mask.
    pub fn mask<'a>(
        &mut self,
        input: GpuSlice<'a, u32>,
        output: GpuSliceMut<'a, u32>,
        predicate: U32Predicate,
    ) -> Result<GpuSlice<'a, u32>, Error> {
        let Extent::Fixed(num_items) = input.extent else {
            return Err(Error::UnsupportedDynamicExtent {
                operation: "predicate mask",
            });
        };
        if output.capacity < num_items {
            return Err(Error::BufferTooSmall {
                name: "predicate mask output view",
                required: u64::from(num_items) * size_of::<u32>() as u64,
                actual: u64::from(output.capacity) * size_of::<u32>() as u64,
            });
        }
        self.primitives.generator().record_mask_ranges(
            self.encoder,
            input.range(),
            output.range(),
            num_items,
            predicate,
        )?;
        Ok(output.initialized(num_items, Extent::Fixed(num_items)))
    }

    /// Tests one field of fixed-length key-value records and returns a
    /// same-length `0`/`1` mask.
    pub fn mask_key_values<'a>(
        &mut self,
        input: GpuSlice<'a, KeyValue>,
        output: GpuSliceMut<'a, u32>,
        field: KeyValueField,
        predicate: U32Predicate,
    ) -> Result<GpuSlice<'a, u32>, Error> {
        let Extent::Fixed(num_items) = input.extent else {
            return Err(Error::UnsupportedDynamicExtent {
                operation: "key-value predicate mask",
            });
        };
        if output.capacity < num_items {
            return Err(Error::BufferTooSmall {
                name: "predicate mask output view",
                required: u64::from(num_items) * size_of::<u32>() as u64,
                actual: u64::from(output.capacity) * size_of::<u32>() as u64,
            });
        }
        self.primitives.generator().record_key_value_mask_ranges(
            self.encoder,
            input.range(),
            output.range(),
            num_items,
            field,
            predicate,
        )?;
        Ok(output.initialized(num_items, Extent::Fixed(num_items)))
    }

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
        self.primitives.compactor().record_compact_ranges(
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

    /// Stably compacts fixed-length key-value records and carries their
    /// GPU-resident selected count into later operations.
    pub fn compact_key_values<'a>(
        &mut self,
        input: GpuSlice<'a, KeyValue>,
        mask: GpuSlice<'a, u32>,
        output: GpuSliceMut<'a, KeyValue>,
        count: GpuCount<'a>,
    ) -> Result<GpuSlice<'a, KeyValue>, Error> {
        let Extent::Fixed(num_items) = input.extent else {
            return Err(Error::UnsupportedDynamicExtent {
                operation: "key-value stream compaction",
            });
        };
        let Extent::Fixed(mask_items) = mask.extent else {
            return Err(Error::UnsupportedDynamicExtent {
                operation: "key-value stream-compaction mask",
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
                name: "key-value compaction output view",
                required: u64::from(num_items) * size_of::<KeyValue>() as u64,
                actual: u64::from(output.capacity) * size_of::<KeyValue>() as u64,
            });
        }
        self.primitives
            .key_value_compactor()
            .record_compact_ranges(
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

    /// Encodes adjacent equal values from a fixed or GPU-counted input.
    ///
    /// Both returned slices share `run_count` as their GPU-resident extent.
    pub fn run_length_encode<'a>(
        &mut self,
        input: GpuSlice<'a, u32>,
        unique_values: GpuSliceMut<'a, u32>,
        run_lengths: GpuSliceMut<'a, u32>,
        run_count: GpuCount<'a>,
    ) -> Result<RunLengthOutput<'a>, Error> {
        for (output, name) in [
            (unique_values, "run-length unique-values output view"),
            (run_lengths, "run-length lengths output view"),
        ] {
            if output.capacity < input.capacity {
                return Err(Error::BufferTooSmall {
                    name,
                    required: u64::from(input.capacity) * size_of::<u32>() as u64,
                    actual: u64::from(output.capacity) * size_of::<u32>() as u64,
                });
            }
        }
        match input.extent {
            Extent::Fixed(num_items) => {
                self.primitives.run_length_encoder().record_encode_ranges(
                    self.encoder,
                    input.range(),
                    RunLengthOutputRanges {
                        unique_values: unique_values.range(),
                        run_lengths: run_lengths.range(),
                        run_count: run_count.range(),
                    },
                    num_items,
                )?;
            }
            Extent::Gpu(input_count) => {
                self.primitives
                    .run_length_encoder()
                    .record_encode_counted_ranges(
                        self.encoder,
                        input.range(),
                        input_count.range(),
                        RunLengthOutputRanges {
                            unique_values: unique_values.range(),
                            run_lengths: run_lengths.range(),
                            run_count: run_count.range(),
                        },
                        input.capacity,
                    )?;
            }
        }
        self.invalidate_count(run_count);
        let extent = Extent::Gpu(run_count);
        Ok(RunLengthOutput {
            unique_values: unique_values.initialized(input.capacity, extent),
            run_lengths: run_lengths.initialized(input.capacity, extent),
        })
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
            Extent::Fixed(num_items) => self.primitives.sorter().record_sort_ranges(
                self.encoder,
                input.range(),
                output.range(),
                num_items,
                options.key_bits,
            )?,
            Extent::Gpu(count) => {
                let plan_index = self.prepare_count(count, input.capacity)?;
                self.primitives.sorter();
                let (sorter, plans) = (&mut self.primitives.sorter, &self.primitives.count_plans);
                sorter
                    .as_mut()
                    .expect("sorter is initialized")
                    .record_sort_ranges_with_count_plan(
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

    /// Stably sorts fixed or GPU-counted key-value records by their `u32` key.
    pub fn sort_by_key<'a>(
        &mut self,
        input: GpuSlice<'a, KeyValue>,
        output: GpuSliceMut<'a, KeyValue>,
        options: SortOptions,
    ) -> Result<GpuSlice<'a, KeyValue>, Error> {
        if output.capacity < input.capacity {
            return Err(Error::BufferTooSmall {
                name: "key-value sort output view",
                required: u64::from(input.capacity) * size_of::<KeyValue>() as u64,
                actual: u64::from(output.capacity) * size_of::<KeyValue>() as u64,
            });
        }
        match input.extent {
            Extent::Fixed(num_items) => {
                self.primitives.key_value_sorter().record_sort_ranges(
                    self.encoder,
                    input.range(),
                    output.range(),
                    num_items,
                    options.key_bits,
                )?;
            }
            Extent::Gpu(count) => {
                let plan_index = self.prepare_count(count, input.capacity)?;
                self.primitives.key_value_sorter();
                let (sorter, plans) = (
                    &mut self.primitives.key_value_sorter,
                    &self.primitives.count_plans,
                );
                sorter
                    .as_mut()
                    .expect("key-value sorter is initialized")
                    .record_sort_ranges_with_count_plan(
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
            Extent::Fixed(num_items) => self.primitives.reducer().record_reduce_ranges(
                self.encoder,
                input.range(),
                output.range(),
                num_items,
                operation,
            ),
            Extent::Gpu(count) => {
                let plan_index = self.prepare_count(count, input.capacity)?;
                self.primitives.reducer();
                let (reducer, plans) = (&mut self.primitives.reducer, &self.primitives.count_plans);
                reducer
                    .as_mut()
                    .expect("reducer is initialized")
                    .record_reduce_ranges_with_count_plan(
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
