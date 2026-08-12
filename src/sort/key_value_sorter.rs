use crate::context::Context;
use crate::{Error, GpuCountPlan, GpuProfile, common::buffers::BufferRange};

use super::core::{RadixSorter, validate_key_for_bits};
use super::counted::CountedSorter;
use super::pipeline::SortItemKind;

/// A `u32` key and its associated `u32` value.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KeyValue {
    pub key: u32,
    pub value: u32,
}

impl KeyValue {
    pub const fn new(key: u32, value: u32) -> Self {
        Self { key, value }
    }
}

/// Performs a stable LSD radix sort of `KeyValue` items by key on a wgpu device.
///
/// GPU-buffer entry points require distinct input and output buffers.
pub struct KeyValueSorter {
    core: RadixSorter,
    counted: Option<CountedSorter>,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl KeyValueSorter {
    /// Creates a sorter that submits work through an existing wgpu device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            core: RadixSorter::new(device, queue, SortItemKind::KeyValue),
            counted: None,
            device: device.clone(),
            queue: queue.clone(),
        }
    }

    /// Creates a sorter specialized for the supplied adapter when a measured
    /// fast path is available.
    ///
    /// NVIDIA Vulkan adapters with enabled, fixed 32-wide subgroups use the
    /// 8-bit radix kernel. Discrete NVIDIA Vulkan devices without compatible
    /// subgroups use the 4-bit kernel, and all remaining adapters use the
    /// portable 2-bit kernel.
    pub fn new_for_adapter(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter_info: &wgpu::AdapterInfo,
    ) -> Self {
        Self {
            core: RadixSorter::new_for_adapter(device, queue, SortItemKind::KeyValue, adapter_info),
            counted: None,
            device: device.clone(),
            queue: queue.clone(),
        }
    }

    /// Creates a sorter from the crate's optional convenience context.
    pub fn from_context(ctx: &Context) -> Self {
        Self::new_for_adapter(&ctx.device, &ctx.queue, &ctx.adapter_info)
    }

    /// Uploads items, stably sorts them by key, and downloads the result.
    pub async fn sort(&mut self, input: &[KeyValue]) -> Result<Vec<KeyValue>, Error> {
        self.core.sort_slice(input).await
    }

    /// Uploads and stably sorts items whose keys fit within `key_bits` bits.
    ///
    /// Fewer bits reduce the number of passes on every radix path. Every key is
    /// checked before upload. `key_bits` must be at most 32; zero is valid only
    /// when every key is zero.
    pub async fn sort_with_key_bits(
        &mut self,
        input: &[KeyValue],
        key_bits: u32,
    ) -> Result<Vec<KeyValue>, Error> {
        for item in input {
            validate_key_for_bits(item.key, key_bits)?;
        }
        self.core.sort_slice_with_key_bits(input, key_bits).await
    }

    /// Stably sorts caller-owned GPU buffers and submits the work immediately.
    pub fn sort_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        self.core.sort_gpu_to_gpu(input, output, num_items)
    }

    /// Stably sorts GPU buffers using a trusted significant-key-bit bound.
    ///
    /// This method does not read the input back to validate the bound. If any key
    /// needs more than `key_bits`, the output may be only partially sorted.
    /// `key_bits` must be at most 32; zero means every key is zero.
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

    /// Profiles a stable GPU-buffer key-value radix sort using GPU timestamps.
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

    /// Profiles a stable GPU-buffer sort using a trusted key-bit bound.
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

    /// Records a stable GPU key-value radix sort without submitting or waiting.
    pub fn record_sort(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        self.core.record_sort(encoder, input, output, num_items)
    }

    /// Records a stable GPU-buffer sort using a trusted key-bit bound.
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
        if let Some(sorter) = self.core.eight_bit_mut() {
            sorter.reserve_counted(capacity)
        } else {
            self.counted().reserve(capacity)
        }
    }

    /// Sorts a capacity-bounded prefix of key-value records selected by a
    /// GPU-resident count and submits the work immediately.
    ///
    /// The count is clamped to `capacity`. Input, output, and count buffers
    /// require `STORAGE` and must use distinct handles. Output records beyond
    /// the clamped count are unspecified.
    pub fn sort_counted_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
    ) -> Result<(), Error> {
        self.sort_counted_gpu_to_gpu_with_key_bits(input, output, count, capacity, u32::BITS)
    }

    /// Sorts a GPU-counted key-value prefix using a trusted key-width bound.
    pub fn sort_counted_gpu_to_gpu_with_key_bits(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        if let Some(sorter) = self.core.eight_bit_mut() {
            sorter.sort_counted_gpu_to_gpu(input, output, count, 0, capacity, key_bits)
        } else {
            self.counted()
                .sort_gpu_to_gpu(input, output, count, capacity, key_bits)
        }
    }

    /// Records a capacity-bounded key-value sort whose exact length remains on
    /// the GPU.
    pub fn record_sort_counted(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
    ) -> Result<(), Error> {
        self.record_sort_counted_with_key_bits(encoder, input, output, count, capacity, u32::BITS)
    }

    /// Records a GPU-counted key-value sort with a trusted key-width bound.
    pub fn record_sort_counted_with_key_bits(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        if let Some(sorter) = self.core.eight_bit_mut() {
            sorter.record_sort_counted(encoder, input, output, count, 0, capacity, key_bits)
        } else {
            self.counted()
                .record_sort(encoder, input, output, count, capacity, key_bits)
        }
    }

    /// Records a GPU-counted key-value sort using shared prepared count
    /// metadata.
    ///
    /// Record [`GpuCountPlan::record_prepare`] after the count producer and
    /// before this method in the same encoder.
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

    /// Records a shared-plan key-value sort with a trusted key-width bound.
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

    /// Profiles a capacity-bounded key-value sort whose exact length remains
    /// GPU-resident.
    pub async fn profile_sort_counted_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
    ) -> Result<GpuProfile, Error> {
        self.profile_sort_counted_gpu_to_gpu_with_key_bits(
            input,
            output,
            count,
            capacity,
            u32::BITS,
        )
        .await
    }

    /// Profiles a GPU-counted key-value sort using a trusted key-width bound.
    pub async fn profile_sort_counted_gpu_to_gpu_with_key_bits(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
        key_bits: u32,
    ) -> Result<GpuProfile, Error> {
        if let Some(sorter) = self.core.eight_bit_mut() {
            sorter
                .profile_sort_counted(input, output, count, 0, capacity, key_bits)
                .await
        } else {
            self.counted()
                .profile_sort(input, output, count, capacity, key_bits)
                .await
        }
    }

    fn counted(&mut self) -> &mut CountedSorter {
        if self.counted.is_none() {
            self.counted = Some(CountedSorter::new(
                &self.device,
                &self.queue,
                SortItemKind::KeyValue,
            ));
        }
        self.counted
            .as_mut()
            .expect("counted key-value sorter is initialized")
    }
}
