use crate::{
    Error, common,
    context::Context,
    profiling::{GpuProfile, TimestampRecorder},
    scan::Scanner,
};

use super::pipeline::{CompactDispatch, CompactPipeline};

const ITEM_SIZE_BYTES: u64 = size_of::<u32>() as u64;
const COUNT_SIZE_BYTES: u64 = size_of::<u32>() as u64;

/// Stably packs selected `u32` values into a contiguous output buffer.
///
/// Masks contain one `u32` per input item and must contain only `0` (discard)
/// or `1` (keep). GPU-buffer entry points trust this contract to avoid a
/// validation pass or readback.
pub struct Compactor {
    pipeline: CompactPipeline,
    scanner: Scanner,
    device: wgpu::Device,
    queue: wgpu::Queue,
    offsets: Option<wgpu::Buffer>,
    offsets_capacity_bytes: u64,
}

impl Compactor {
    /// Creates a compactor that submits work through an existing device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            pipeline: CompactPipeline::new(device),
            scanner: Scanner::new(device, queue),
            device: device.clone(),
            queue: queue.clone(),
            offsets: None,
            offsets_capacity_bytes: 0,
        }
    }

    /// Creates a compactor from the crate's optional convenience context.
    pub fn from_context(context: &Context) -> Self {
        Self::new(&context.device, &context.queue)
    }

    /// Uploads values and a mask, compacts them on the GPU, and downloads the result.
    pub async fn compact(&mut self, input: &[u32], mask: &[u32]) -> Result<Vec<u32>, Error> {
        if input.len() != mask.len() {
            return Err(Error::CompactionLengthMismatch {
                input: input.len(),
                mask: mask.len(),
            });
        }
        for (index, &value) in mask.iter().enumerate() {
            if value > 1 {
                return Err(Error::InvalidCompactionFlag { index, value });
            }
        }
        if input.is_empty() {
            return Ok(Vec::new());
        }

        let num_items = common::math::checked_u32(input.len() as u64)?;
        let selected_count = mask.iter().map(|&value| value as usize).sum();
        let input_buffer = common::buffers::create_storage_buffer(&self.device, input);
        let mask_buffer = common::buffers::create_storage_buffer(&self.device, mask);
        let output_buffer =
            common::buffers::create_empty_storage_buffer(&self.device, input_buffer.size());
        let count_buffer =
            common::buffers::create_empty_storage_buffer(&self.device, COUNT_SIZE_BYTES);

        self.compact_gpu_to_gpu(
            &input_buffer,
            &mask_buffer,
            &output_buffer,
            &count_buffer,
            num_items,
        )?;

        if selected_count == 0 {
            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })?;
            return Ok(Vec::new());
        }
        common::buffers::download_buffer(&self.device, &self.queue, &output_buffer, selected_count)
            .await
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.record_compact(&mut encoder, input, mask, output, output_count, num_items)?;
        self.queue.submit(Some(encoder.finish()));
        Ok(())
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
        self.record_commands(encoder, input, mask, output, output_count, num_items, None)
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
        if num_items == 0 {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Profiled Empty Stream Compaction"),
                });
            self.record_commands(
                &mut encoder,
                input,
                mask,
                output,
                output_count,
                num_items,
                None,
            )?;
            let submission = self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })?;
            return Ok(GpuProfile::empty());
        }

        let span_count = self
            .scanner
            .compute_pass_count(num_items)
            .checked_add(1)
            .ok_or(Error::SizeOverflow)?;
        let mut profiler = TimestampRecorder::new(&self.device, &self.queue, span_count)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Profiled Stream Compaction"),
            });
        self.record_commands(
            &mut encoder,
            input,
            mask,
            output,
            output_count,
            num_items,
            Some(&mut profiler),
        )?;
        profiler.resolve(&mut encoder);
        let submission = self.queue.submit(Some(encoder.finish()));
        profiler.read(&self.device, submission).await
    }

    #[allow(clippy::too_many_arguments)]
    fn record_commands(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        output: &wgpu::Buffer,
        output_count: &wgpu::Buffer,
        num_items: u32,
        mut profiler: Option<&mut TimestampRecorder>,
    ) -> Result<(), Error> {
        validate_distinct_buffers(input, mask, output, output_count)?;
        common::buffers::validate_buffer(
            output_count,
            "compaction output count",
            COUNT_SIZE_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        if num_items == 0 {
            encoder.clear_buffer(output_count, 0, Some(COUNT_SIZE_BYTES));
            return Ok(());
        }

        let size_bytes = common::math::checked_byte_size(u64::from(num_items), ITEM_SIZE_BYTES)?;
        common::buffers::validate_buffer(
            input,
            "compaction input",
            size_bytes,
            wgpu::BufferUsages::STORAGE,
        )?;
        common::buffers::validate_buffer(
            mask,
            "compaction mask",
            size_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        common::buffers::validate_buffer(
            output,
            "compaction output",
            size_bytes,
            wgpu::BufferUsages::STORAGE,
        )?;
        self.ensure_offsets(size_bytes)?;
        let offsets = self
            .offsets
            .as_ref()
            .expect("compaction offsets exist for non-empty inputs");

        if let Some(profiler) = profiler.as_deref_mut() {
            self.scanner.record_profiled_exclusive_scan(
                encoder,
                mask,
                offsets,
                num_items,
                "compact.scan",
                profiler,
            )?;
        } else {
            self.scanner
                .record_exclusive_scan(encoder, mask, offsets, num_items)?;
        }
        self.pipeline.dispatch(
            &self.device,
            encoder,
            CompactDispatch {
                input,
                mask,
                offsets,
                output,
                output_count,
                num_items,
            },
            profiler,
        );
        Ok(())
    }

    fn ensure_offsets(&mut self, size_bytes: u64) -> Result<(), Error> {
        let limit = self.device.limits().max_buffer_size;
        if size_bytes > limit {
            return Err(Error::BufferLimitExceeded {
                requested: size_bytes,
                limit,
            });
        }
        if self.offsets.is_none() || size_bytes > self.offsets_capacity_bytes {
            self.offsets = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Stream Compaction Offsets"),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.offsets_capacity_bytes = size_bytes;
        }
        Ok(())
    }
}

fn validate_distinct_buffers(
    input: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    output: &wgpu::Buffer,
    output_count: &wgpu::Buffer,
) -> Result<(), Error> {
    for (first, first_name, second, second_name) in [
        (input, "compaction input", output, "compaction output"),
        (mask, "compaction mask", output, "compaction output"),
        (
            input,
            "compaction input",
            output_count,
            "compaction output count",
        ),
        (
            mask,
            "compaction mask",
            output_count,
            "compaction output count",
        ),
        (
            output,
            "compaction output",
            output_count,
            "compaction output count",
        ),
    ] {
        if first == second {
            return Err(Error::BufferAlias {
                first: first_name,
                second: second_name,
            });
        }
    }
    Ok(())
}
