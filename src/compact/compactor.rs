use crate::{Error, common::buffers::BufferRange, context::Context, profiling::GpuProfile};

use super::{core::CompactCore, pipeline::CompactItemKind};

/// Stably packs selected `u32` values into a contiguous output buffer.
///
/// Masks contain one `u32` per input item and must contain only `0` (discard)
/// or `1` (keep). GPU-buffer entry points trust this contract to avoid a
/// validation pass or readback.
pub struct Compactor {
    core: CompactCore,
}

impl Compactor {
    /// Creates a compactor that submits work through an existing device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            core: CompactCore::new(device, queue, CompactItemKind::Value),
        }
    }

    /// Creates a compactor from the crate's optional convenience context.
    pub fn from_context(context: &Context) -> Self {
        Self {
            core: CompactCore::from_context(context, CompactItemKind::Value),
        }
    }

    /// Uploads values and a mask, compacts them on the GPU, and downloads the result.
    pub async fn compact(&mut self, input: &[u32], mask: &[u32]) -> Result<Vec<u32>, Error> {
        self.core.compact_slice(input, mask).await
    }

    /// Compacts caller-owned GPU buffers and submits the work immediately.
    ///
    /// `input` and `output` require `STORAGE`; `mask` requires `STORAGE |
    /// COPY_SRC`; and the four-byte `output_count` requires `STORAGE |
    /// COPY_DST`. `output` must have capacity for `num_items` values. The input,
    /// mask, output, and count buffers remain GPU-resident.
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

    /// Records stable stream compaction without submitting or waiting.
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

    /// Profiles GPU-buffer compaction using hardware timestamp queries.
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
