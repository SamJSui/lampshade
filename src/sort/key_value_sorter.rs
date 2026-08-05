use crate::Error;
use crate::context::Context;

use super::core::RadixSorter;
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

    /// Creates a sorter from the crate's optional convenience context.
    pub fn from_context(ctx: &Context) -> Self {
        Self::new(&ctx.device, &ctx.queue)
    }

    /// Uploads items, stably sorts them by key, and downloads the result.
    pub async fn sort(&mut self, input: &[KeyValue]) -> Result<Vec<KeyValue>, Error> {
        self.core.sort_slice(input).await
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
}
