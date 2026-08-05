use crate::context::Context;
use crate::{Error, GpuProfile};

use super::core::RadixSorter;
use super::pipeline::SortItemKind;

/// Performs an unsigned 32-bit LSD radix sort on a wgpu device.
pub struct Sorter {
    core: RadixSorter,
}

impl Sorter {
    /// Creates a sorter that submits work through an existing wgpu device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            core: RadixSorter::new(device, queue, SortItemKind::Key),
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

    /// Sorts caller-owned GPU buffers and submits the work immediately.
    pub fn sort_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        self.core.sort_gpu_to_gpu(input, output, num_items)
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
}
