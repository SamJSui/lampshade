use crate::{
    Error, common,
    common::buffers::BufferRange,
    common::{runtime::CommandSession, runtime::ProfileSession, workspace::ReusableBuffer},
    context::Context,
    profiling::{GpuProfile, TimestampRecorder},
    scan::Scanner,
};

use super::pipeline::{CompactDispatch, CompactItemKind, CompactPipeline};

const MASK_ITEM_SIZE_BYTES: u64 = size_of::<u32>() as u64;
const COUNT_SIZE_BYTES: u64 = size_of::<u32>() as u64;

pub(crate) struct CompactCore {
    pipeline: CompactPipeline,
    scanner: Scanner,
    device: wgpu::Device,
    queue: wgpu::Queue,
    item_size_bytes: u64,
    offsets: ReusableBuffer,
}

impl CompactCore {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item_kind: CompactItemKind,
    ) -> Self {
        Self {
            pipeline: CompactPipeline::new(device, item_kind),
            scanner: Scanner::new(device, queue),
            device: device.clone(),
            queue: queue.clone(),
            item_size_bytes: item_kind.size_bytes(),
            offsets: ReusableBuffer::default(),
        }
    }

    pub(crate) fn from_context(context: &Context, item_kind: CompactItemKind) -> Self {
        Self {
            pipeline: CompactPipeline::new(&context.device, item_kind),
            scanner: Scanner::from_context(context),
            device: context.device.clone(),
            queue: context.queue.clone(),
            item_size_bytes: item_kind.size_bytes(),
            offsets: ReusableBuffer::default(),
        }
    }

    pub(crate) async fn compact_slice<T: bytemuck::Pod>(
        &mut self,
        input: &[T],
        mask: &[u32],
    ) -> Result<Vec<T>, Error> {
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

        debug_assert_eq!(size_of::<T>() as u64, self.item_size_bytes);
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

    pub(crate) fn compact_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        output: &wgpu::Buffer,
        output_count: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        let mut commands = CommandSession::new(&self.device, None);
        self.record_compact(
            commands.encoder(),
            input,
            mask,
            output,
            output_count,
            num_items,
        )?;
        commands.submit(&self.queue);
        Ok(())
    }

    pub(crate) fn record_compact(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        output: &wgpu::Buffer,
        output_count: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        self.record_compact_ranges(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(mask),
            BufferRange::whole(output),
            BufferRange::whole(output_count),
            num_items,
        )
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
        self.record_commands(encoder, input, mask, output, output_count, num_items, None)
    }

    pub(crate) fn reserve(&mut self, capacity: u32) -> Result<(), Error> {
        if capacity == 0 {
            return Ok(());
        }
        let mask_bytes =
            common::math::checked_byte_size(u64::from(capacity), MASK_ITEM_SIZE_BYTES)?;
        self.ensure_offsets(mask_bytes)?;
        self.scanner.reserve(capacity);
        Ok(())
    }

    pub(crate) async fn profile_compact_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        output: &wgpu::Buffer,
        output_count: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<GpuProfile, Error> {
        let span_count = if num_items == 0 {
            0
        } else {
            self.scanner
                .compute_block_local_pass_count(num_items)
                .checked_add(1)
                .ok_or(Error::SizeOverflow)?
        };
        let label = if num_items == 0 {
            "Profiled Empty Stream Compaction"
        } else {
            "Profiled Stream Compaction"
        };
        let mut profile = ProfileSession::new(&self.device, &self.queue, span_count, label)?;
        let (encoder, profiler) = profile.recording();
        self.record_commands(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(mask),
            BufferRange::whole(output),
            BufferRange::whole(output_count),
            num_items,
            profiler,
        )?;
        profile.finish(&self.device, &self.queue).await
    }

    #[allow(clippy::too_many_arguments)]
    fn record_commands(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        mask: BufferRange<'_>,
        output: BufferRange<'_>,
        output_count: BufferRange<'_>,
        num_items: u32,
        mut profiler: Option<&mut TimestampRecorder>,
    ) -> Result<(), Error> {
        validate_distinct_buffers(input, mask, output, output_count)?;
        output_count.validate(
            "compaction output count",
            COUNT_SIZE_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        output_count.validate_storage_offset(&self.device, "compaction output count")?;
        if num_items == 0 {
            encoder.clear_buffer(
                output_count.buffer,
                output_count.offset,
                Some(COUNT_SIZE_BYTES),
            );
            return Ok(());
        }

        let item_bytes =
            common::math::checked_byte_size(u64::from(num_items), self.item_size_bytes)?;
        let mask_bytes =
            common::math::checked_byte_size(u64::from(num_items), MASK_ITEM_SIZE_BYTES)?;
        self.validate_storage_binding_size(item_bytes)?;
        self.validate_storage_binding_size(mask_bytes)?;
        input.validate("compaction input", item_bytes, wgpu::BufferUsages::STORAGE)?;
        mask.validate(
            "compaction mask",
            mask_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        output.validate("compaction output", item_bytes, wgpu::BufferUsages::STORAGE)?;
        input.validate_storage_offset(&self.device, "compaction input")?;
        mask.validate_storage_offset(&self.device, "compaction mask")?;
        output.validate_storage_offset(&self.device, "compaction output")?;
        let input = BufferRange {
            size: item_bytes,
            ..input
        };
        let mask = BufferRange {
            size: mask_bytes,
            ..mask
        };
        let output = BufferRange {
            size: item_bytes,
            ..output
        };
        let output_count = BufferRange {
            size: COUNT_SIZE_BYTES,
            ..output_count
        };
        self.ensure_offsets(mask_bytes)?;
        let offsets = self
            .offsets
            .get()
            .expect("compaction offsets exist for non-empty inputs");

        let offsets_range = BufferRange {
            buffer: offsets,
            offset: 0,
            size: mask_bytes,
        };
        let scan_items_per_block = self.scanner.record_block_local_exclusive_scan_ranges(
            encoder,
            mask,
            offsets_range,
            num_items,
            "compact.scan",
            profiler.as_deref_mut(),
        )?;
        let block_prefixes = self.scanner.block_prefix_buffer().unwrap_or(offsets);
        self.pipeline.dispatch(
            &self.device,
            encoder,
            CompactDispatch {
                input,
                mask,
                offsets: offsets_range,
                block_prefixes: BufferRange::whole(block_prefixes),
                output,
                output_count,
                num_items,
                scan_items_per_block,
            },
            profiler,
        );
        Ok(())
    }

    fn ensure_offsets(&mut self, size_bytes: u64) -> Result<(), Error> {
        self.validate_storage_binding_size(size_bytes)?;
        self.offsets.ensure(
            &self.device,
            size_bytes,
            "Stream Compaction Offsets",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        Ok(())
    }

    fn validate_storage_binding_size(&self, requested: u64) -> Result<(), Error> {
        let limits = self.device.limits();
        let limit = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        if requested > limit {
            Err(Error::BufferLimitExceeded { requested, limit })
        } else {
            Ok(())
        }
    }
}

fn validate_distinct_buffers(
    input: BufferRange<'_>,
    mask: BufferRange<'_>,
    output: BufferRange<'_>,
    output_count: BufferRange<'_>,
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
        if first.buffer == second.buffer {
            return Err(Error::BufferAlias {
                first: first_name,
                second: second_name,
            });
        }
    }
    Ok(())
}
