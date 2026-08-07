use crate::context::Context;
use crate::{Error, GpuProfile};

use super::core::{RadixSorter, validate_key_for_bits};
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
}

impl KeyValueSorter {
    /// Creates a sorter that submits work through an existing wgpu device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            core: RadixSorter::new(device, queue, SortItemKind::KeyValue),
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
}
