use crate::{
    Error, KeyValue, common::buffers::BufferRange, context::Context, profiling::GpuProfile,
};

use super::{core::CompactCore, pipeline::CompactItemKind};

/// Stably packs selected [`KeyValue`] records into a contiguous output buffer.
///
/// Both fields move together, and selected records retain their original order.
/// Masks contain one `u32` per record and must contain only `0` (discard) or
/// `1` (keep). GPU-buffer entry points trust the mask contents.
pub struct KeyValueCompactor {
    core: CompactCore,
}

impl KeyValueCompactor {
    /// Creates a key-value compactor for an existing device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            core: CompactCore::new(device, queue, CompactItemKind::KeyValue),
        }
    }

    /// Creates a key-value compactor from the crate's convenience context.
    pub fn from_context(context: &Context) -> Self {
        Self {
            core: CompactCore::from_context(context, CompactItemKind::KeyValue),
        }
    }

    /// Uploads records and a mask, compacts them on the GPU, and downloads the result.
    pub async fn compact(
        &mut self,
        input: &[KeyValue],
        mask: &[u32],
    ) -> Result<Vec<KeyValue>, Error> {
        self.core.compact_slice(input, mask).await
    }

    /// Compacts caller-owned GPU buffers and submits the work immediately.
    ///
    /// `input` and `output` require `STORAGE`; `mask` requires `STORAGE |
    /// COPY_SRC`; and the four-byte `output_count` requires `STORAGE |
    /// COPY_DST`. `output` must have capacity for `num_items` eight-byte records.
    pub fn compact_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        output: &wgpu::Buffer,
        output_count: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        self.core
            .compact_gpu_to_gpu(input, mask, output, output_count, num_items)
    }

    /// Records stable key-value compaction without submitting or waiting.
    pub fn record_compact(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        output: &wgpu::Buffer,
        output_count: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        self.core
            .record_compact(encoder, input, mask, output, output_count, num_items)
    }

    pub(crate) fn record_compact_ranges(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        mask: BufferRange<'_>,
        output: BufferRange<'_>,
        output_count: BufferRange<'_>,
        num_items: u32,
    ) -> Result<(), Error> {
        self.core
            .record_compact_ranges(encoder, input, mask, output, output_count, num_items)
    }

    pub(crate) fn reserve(&mut self, capacity: u32) -> Result<(), Error> {
        self.core.reserve(capacity)
    }

    /// Profiles GPU-buffer key-value compaction using hardware timestamp queries.
    pub async fn profile_compact_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        output: &wgpu::Buffer,
        output_count: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<GpuProfile, Error> {
        self.core
            .profile_compact_gpu_to_gpu(input, mask, output, output_count, num_items)
            .await
    }
}
