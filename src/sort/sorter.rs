use crate::context::Context;
use crate::{Error, GpuCountPlan, GpuProfile, common::buffers::BufferRange};

use super::core::{RadixSorter, validate_key_for_bits};
use super::counted::CountedSorter;
use super::pipeline::SortItemKind;

/// Performs an unsigned 32-bit LSD radix sort on a wgpu device.
///
/// GPU-buffer entry points require distinct input and output buffers.
pub struct Sorter {
    core: RadixSorter,
    counted: Option<CountedSorter>,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Sorter {
    /// Creates a sorter that submits work through an existing wgpu device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            core: RadixSorter::new(device, queue, SortItemKind::Key),
            counted: None,
            device: device.clone(),
            queue: queue.clone(),
        }
    }

    /// Creates a sorter from the crate's optional convenience context.
    pub fn from_context(ctx: &Context) -> Self {
        Self::new(&ctx.device, &ctx.queue)
    }

    /// Uploads values, sorts them on the GPU, and downloads the sorted result.
    pub async fn sort(&mut self, input: &[u32]) -> Result<Vec<u32>, Error> {
        self.core.sort_slice(input).await
    }

    /// Uploads and sorts values known to fit within `key_bits` significant bits.
    ///
    /// Fewer bits reduce the number of passes. Every input value is checked
    /// before upload. `key_bits` must be at most 32; zero is valid only when
    /// every input value is zero.
    pub async fn sort_with_key_bits(
        &mut self,
        input: &[u32],
        key_bits: u32,
    ) -> Result<Vec<u32>, Error> {
        for &key in input {
            validate_key_for_bits(key, key_bits)?;
        }
        self.core.sort_slice_with_key_bits(input, key_bits).await
    }

    /// Sorts caller-owned GPU buffers and submits the work immediately.
    pub fn sort_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        self.core.sort_gpu_to_gpu(input, output, num_items)
    }

    /// Sorts GPU buffers using only the declared number of significant key bits.
    ///
    /// The bound is trusted: this method does not read the input back to validate
    /// it. If any key needs more than `key_bits`, the output may be only partially
    /// sorted. `key_bits` must be at most 32; zero means every key is zero.
    pub fn sort_gpu_to_gpu_with_key_bits(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        self.core
            .sort_gpu_to_gpu_with_key_bits(input, output, num_items, key_bits)
    }

    /// Profiles a GPU-buffer radix sort using GPU timestamps.
    pub async fn profile_sort_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<GpuProfile, Error> {
        self.core
            .profile_sort_gpu_to_gpu(input, output, num_items)
            .await
    }

    /// Profiles a GPU-buffer sort using a trusted significant-key-bit bound.
    pub async fn profile_sort_gpu_to_gpu_with_key_bits(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        key_bits: u32,
    ) -> Result<GpuProfile, Error> {
        self.core
            .profile_sort_gpu_to_gpu_with_key_bits(input, output, num_items, key_bits)
            .await
    }

    /// Records a GPU radix sort without submitting or waiting for the work.
    pub fn record_sort(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        self.core.record_sort(encoder, input, output, num_items)
    }

    /// Records a GPU-buffer sort using a trusted significant-key-bit bound.
    pub fn record_sort_with_key_bits(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        self.core
            .record_sort_with_key_bits(encoder, input, output, num_items, key_bits)
    }

    pub(crate) fn record_sort_ranges(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        num_items: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        self.core
            .record_sort_ranges(encoder, input, output, num_items, key_bits)
    }

    pub(crate) fn reserve_fixed(&mut self, capacity: u32) -> Result<(), Error> {
        self.core.reserve(capacity)
    }

    pub(crate) fn reserve_counted(&mut self, capacity: u32) -> Result<(), Error> {
        self.counted().reserve(capacity)
    }

    /// Sorts the prefix selected by a GPU-resident item count and submits it.
    ///
    /// `capacity` is the maximum number of readable input and writable output
    /// values. The GPU count is clamped to that capacity before indirect
    /// dispatch arguments are produced. All three buffers require `STORAGE` and
    /// must be distinct. Only the first `min(count, capacity)` output values are
    /// valid; the remaining output capacity is unspecified.
    pub fn sort_counted_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
    ) -> Result<(), Error> {
        self.counted()
            .sort_gpu_to_gpu(input, output, count, capacity, u32::BITS)
    }

    /// Sorts a GPU-counted prefix using a trusted significant-key-bit bound.
    ///
    /// As with [`Self::sort_counted_gpu_to_gpu`], output beyond the clamped
    /// count is unspecified.
    pub fn sort_counted_gpu_to_gpu_with_key_bits(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        self.counted()
            .sort_gpu_to_gpu(input, output, count, capacity, key_bits)
    }

    /// Records a capacity-bounded radix sort whose actual length remains on the GPU.
    pub fn record_sort_counted(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
    ) -> Result<(), Error> {
        self.counted()
            .record_sort(encoder, input, output, count, capacity, u32::BITS)
    }

    /// Records a GPU-counted sort using a trusted significant-key-bit bound.
    pub fn record_sort_counted_with_key_bits(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        self.counted()
            .record_sort(encoder, input, output, count, capacity, key_bits)
    }

    /// Records a GPU-counted sort using metadata shared by several primitives.
    ///
    /// Record [`GpuCountPlan::record_prepare`] after the count producer and
    /// before this method in the same encoder. The plan capacity is the buffer
    /// bound; output beyond the clamped count is unspecified.
    pub fn record_sort_with_count_plan(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        plan: &GpuCountPlan,
    ) -> Result<(), Error> {
        self.counted()
            .record_sort_with_plan(encoder, input, output, plan, u32::BITS)
    }

    /// Records a shared-plan GPU-counted sort with a trusted key-width bound.
    pub fn record_sort_with_count_plan_and_key_bits(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        plan: &GpuCountPlan,
        key_bits: u32,
    ) -> Result<(), Error> {
        self.counted()
            .record_sort_with_plan(encoder, input, output, plan, key_bits)
    }

    pub(crate) fn record_sort_ranges_with_count_plan(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        plan: &GpuCountPlan,
        key_bits: u32,
    ) -> Result<(), Error> {
        self.counted()
            .record_sort_ranges_with_plan(encoder, input, output, plan, key_bits)
    }

    /// Profiles a capacity-bounded sort whose actual length is GPU-resident.
    pub async fn profile_sort_counted_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
    ) -> Result<GpuProfile, Error> {
        self.counted()
            .profile_sort(input, output, count, capacity, u32::BITS)
            .await
    }

    /// Profiles a GPU-counted sort using a trusted key-width bound.
    pub async fn profile_sort_counted_gpu_to_gpu_with_key_bits(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
        key_bits: u32,
    ) -> Result<GpuProfile, Error> {
        self.counted()
            .profile_sort(input, output, count, capacity, key_bits)
            .await
    }

    fn counted(&mut self) -> &mut CountedSorter {
        if self.counted.is_none() {
            self.counted = Some(CountedSorter::new(&self.device, &self.queue));
        }
        self.counted
            .as_mut()
            .expect("counted sorter is initialized")
    }
}
