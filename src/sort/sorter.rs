use crate::context::Context;
use crate::{Error, GpuProfile};

use super::core::{RadixSorter, validate_key_for_bits};
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

    /// Uploads and sorts values known to fit within `key_bits` significant bits.
    ///
    /// Fewer bits reduce the number of passes on portable and wide radix paths.
    /// Every input value is checked before upload. `key_bits` must be at most 32;
    /// zero is valid only when every input value is zero.
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
}
