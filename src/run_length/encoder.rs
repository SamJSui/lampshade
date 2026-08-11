use crate::{
    Context, Error, common,
    common::workspace::ReusableBuffer,
    common::{buffers::BufferRange, runtime::CommandSession, runtime::ProfileSession},
    profiling::{GpuProfile, TimestampRecorder},
    scan::Scanner,
};

use super::pipeline::RunLengthPipeline;

const ITEM_SIZE_BYTES: u64 = size_of::<u32>() as u64;
const COUNT_SIZE_BYTES: u64 = size_of::<u32>() as u64;

/// Caller-owned GPU buffers written by run-length encoding.
///
/// After a successful operation, `run_count` contains the initialized prefix
/// length for both array outputs. Only `unique_values[..run_count]` and
/// `run_lengths[..run_count]` are defined; any reused buffer tail is
/// unspecified.
#[derive(Clone, Copy)]
pub struct RunLengthOutputBuffers<'a> {
    /// One value for each adjacent run.
    pub unique_values: &'a wgpu::Buffer,
    /// The number of input items in each adjacent run.
    pub run_lengths: &'a wgpu::Buffer,
    /// One `u32` receiving the number of output runs.
    pub run_count: &'a wgpu::Buffer,
}

impl<'a> RunLengthOutputBuffers<'a> {
    /// Groups the three caller-owned output buffers required by RLE.
    pub const fn new(
        unique_values: &'a wgpu::Buffer,
        run_lengths: &'a wgpu::Buffer,
        run_count: &'a wgpu::Buffer,
    ) -> Self {
        Self {
            unique_values,
            run_lengths,
            run_count,
        }
    }

    fn ranges(self) -> RunLengthOutputRanges<'a> {
        RunLengthOutputRanges {
            unique_values: BufferRange::whole(self.unique_values),
            run_lengths: BufferRange::whole(self.run_lengths),
            run_count: BufferRange::whole(self.run_count),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct RunLengthOutputRanges<'a> {
    pub(crate) unique_values: BufferRange<'a>,
    pub(crate) run_lengths: BufferRange<'a>,
    pub(crate) run_count: BufferRange<'a>,
}

#[derive(Clone, Copy)]
enum InputExtent<'a> {
    Fixed(u32),
    Counted {
        count: BufferRange<'a>,
        capacity: u32,
    },
}

impl<'a> InputExtent<'a> {
    const fn capacity(self) -> u32 {
        match self {
            Self::Fixed(items) => items,
            Self::Counted { capacity, .. } => capacity,
        }
    }

    const fn fixed_items(self) -> u32 {
        match self {
            Self::Fixed(items) => items,
            Self::Counted { .. } => 0,
        }
    }

    const fn count(self) -> Option<BufferRange<'a>> {
        match self {
            Self::Fixed(_) => None,
            Self::Counted { count, .. } => Some(count),
        }
    }
}

/// Encodes adjacent equal `u32` values as unique values and run lengths.
///
/// Input does not need to be sorted. Sorting first groups every equal key into
/// one run; unsorted input produces one output entry per consecutive run.
pub struct RunLengthEncoder {
    pipeline: RunLengthPipeline,
    scanner: Scanner,
    device: wgpu::Device,
    queue: wgpu::Queue,
    heads: ReusableBuffer,
    offsets: ReusableBuffer,
}

impl RunLengthEncoder {
    /// Creates an encoder that submits work through an existing device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            pipeline: RunLengthPipeline::new(device),
            scanner: Scanner::new(device, queue),
            device: device.clone(),
            queue: queue.clone(),
            heads: ReusableBuffer::default(),
            offsets: ReusableBuffer::default(),
        }
    }

    /// Creates an encoder from the crate's optional convenience context.
    pub fn from_context(context: &Context) -> Self {
        Self::new(&context.device, &context.queue)
    }

    /// Uploads values, encodes adjacent runs, and downloads values and lengths.
    pub async fn encode(&mut self, input: &[u32]) -> Result<(Vec<u32>, Vec<u32>), Error> {
        if input.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let num_items = common::math::checked_u32(input.len() as u64)?;
        let run_count = 1 + input.windows(2).filter(|pair| pair[0] != pair[1]).count();
        let input_buffer = common::buffers::create_storage_buffer(&self.device, input);
        let output_bytes = common::math::checked_byte_size(input.len() as u64, ITEM_SIZE_BYTES)?;
        let unique_values =
            common::buffers::create_empty_storage_buffer(&self.device, output_bytes);
        let run_lengths = common::buffers::create_empty_storage_buffer(&self.device, output_bytes);
        let output_count =
            common::buffers::create_empty_storage_buffer(&self.device, COUNT_SIZE_BYTES);
        self.encode_gpu_to_gpu(
            &input_buffer,
            RunLengthOutputBuffers::new(&unique_values, &run_lengths, &output_count),
            num_items,
        )?;
        let unique_values =
            common::buffers::download_buffer(&self.device, &self.queue, &unique_values, run_count)
                .await?;
        let run_lengths =
            common::buffers::download_buffer(&self.device, &self.queue, &run_lengths, run_count)
                .await?;
        Ok((unique_values, run_lengths))
    }

    /// Encodes a CPU-known number of items and submits immediately.
    ///
    /// `input`, `unique_values`, and `run_lengths` require `STORAGE` and the
    /// outputs need capacity for `num_items` values. The four-byte `run_count`
    /// requires `STORAGE | COPY_DST`.
    pub fn encode_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        outputs: RunLengthOutputBuffers<'_>,
        num_items: u32,
    ) -> Result<(), Error> {
        let mut commands = CommandSession::new(&self.device, None);
        self.record_encode(commands.encoder(), input, outputs, num_items)?;
        commands.submit(&self.queue);
        Ok(())
    }

    /// Encodes a capacity-bounded prefix selected by a GPU-resident count.
    ///
    /// The input count requires `STORAGE` and is clamped to `capacity`; no CPU
    /// readback occurs. Output requirements match [`Self::encode_gpu_to_gpu`].
    pub fn encode_counted_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        input_count: &wgpu::Buffer,
        outputs: RunLengthOutputBuffers<'_>,
        capacity: u32,
    ) -> Result<(), Error> {
        let mut commands = CommandSession::new(&self.device, None);
        self.record_encode_counted(commands.encoder(), input, input_count, outputs, capacity)?;
        commands.submit(&self.queue);
        Ok(())
    }

    /// Records fixed-extent run-length encoding without submitting or waiting.
    pub fn record_encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        outputs: RunLengthOutputBuffers<'_>,
        num_items: u32,
    ) -> Result<(), Error> {
        self.record_commands(
            encoder,
            BufferRange::whole(input),
            outputs.ranges(),
            InputExtent::Fixed(num_items),
            None,
        )
    }

    /// Records GPU-counted run-length encoding without submitting or waiting.
    ///
    /// Work is capacity-sized; inactive lanes are zeroed before the prefix
    /// scan so only the clamped GPU-resident prefix contributes runs.
    pub fn record_encode_counted(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        input_count: &wgpu::Buffer,
        outputs: RunLengthOutputBuffers<'_>,
        capacity: u32,
    ) -> Result<(), Error> {
        self.record_commands(
            encoder,
            BufferRange::whole(input),
            outputs.ranges(),
            InputExtent::Counted {
                count: BufferRange::whole(input_count),
                capacity,
            },
            None,
        )
    }

    pub(crate) fn record_encode_ranges(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        outputs: RunLengthOutputRanges<'_>,
        num_items: u32,
    ) -> Result<(), Error> {
        self.record_commands(encoder, input, outputs, InputExtent::Fixed(num_items), None)
    }

    pub(crate) fn record_encode_counted_ranges(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        input_count: BufferRange<'_>,
        outputs: RunLengthOutputRanges<'_>,
        capacity: u32,
    ) -> Result<(), Error> {
        self.record_commands(
            encoder,
            input,
            outputs,
            InputExtent::Counted {
                count: input_count,
                capacity,
            },
            None,
        )
    }

    pub(crate) fn reserve(&mut self, capacity: u32) -> Result<(), Error> {
        if capacity > 0 {
            let bytes = common::math::checked_byte_size(u64::from(capacity), ITEM_SIZE_BYTES)?;
            self.ensure_workspace(bytes, capacity)?;
        }
        Ok(())
    }

    /// Profiles fixed-extent GPU-buffer encoding using timestamp queries.
    pub async fn profile_encode_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        outputs: RunLengthOutputBuffers<'_>,
        num_items: u32,
    ) -> Result<GpuProfile, Error> {
        self.profile_commands(input, outputs.ranges(), InputExtent::Fixed(num_items))
            .await
    }

    /// Profiles capacity-bounded GPU-counted encoding using timestamp queries.
    pub async fn profile_encode_counted_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        input_count: &wgpu::Buffer,
        outputs: RunLengthOutputBuffers<'_>,
        capacity: u32,
    ) -> Result<GpuProfile, Error> {
        self.profile_commands(
            input,
            outputs.ranges(),
            InputExtent::Counted {
                count: BufferRange::whole(input_count),
                capacity,
            },
        )
        .await
    }

    async fn profile_commands(
        &mut self,
        input: &wgpu::Buffer,
        outputs: RunLengthOutputRanges<'_>,
        extent: InputExtent<'_>,
    ) -> Result<GpuProfile, Error> {
        let capacity = extent.capacity();
        let span_count = if capacity == 0 {
            0
        } else {
            self.scanner
                .compute_pass_count(capacity)
                .checked_add(3)
                .ok_or(Error::SizeOverflow)?
        };
        let mut profile = ProfileSession::new(
            &self.device,
            &self.queue,
            span_count,
            "Profiled Run-Length Encoding",
        )?;
        let (encoder, profiler) = profile.recording();
        self.record_commands(
            encoder,
            BufferRange::whole(input),
            outputs,
            extent,
            profiler,
        )?;
        profile.finish(&self.device, &self.queue).await
    }

    fn record_commands(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        outputs: RunLengthOutputRanges<'_>,
        extent: InputExtent<'_>,
        mut profiler: Option<&mut TimestampRecorder>,
    ) -> Result<(), Error> {
        let RunLengthOutputRanges {
            unique_values,
            run_lengths,
            run_count,
        } = outputs;
        validate_distinct_buffers(input, extent.count(), unique_values, run_lengths, run_count)?;
        run_count.validate(
            "run-length output count",
            COUNT_SIZE_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        run_count.validate_storage_offset(&self.device, "run-length output count")?;
        if let Some(input_count) = extent.count() {
            input_count.validate(
                "run-length input count",
                COUNT_SIZE_BYTES,
                wgpu::BufferUsages::STORAGE,
            )?;
            input_count.validate_storage_offset(&self.device, "run-length input count")?;
        }
        let capacity = extent.capacity();
        if capacity == 0 {
            encoder.clear_buffer(run_count.buffer, run_count.offset, Some(COUNT_SIZE_BYTES));
            return Ok(());
        }
        let item_bytes = common::math::checked_byte_size(u64::from(capacity), ITEM_SIZE_BYTES)?;
        common::buffers::validate_storage_binding_size(&self.device, item_bytes)?;
        input.validate("run-length input", item_bytes, wgpu::BufferUsages::STORAGE)?;
        unique_values.validate(
            "run-length unique values",
            item_bytes,
            wgpu::BufferUsages::STORAGE,
        )?;
        run_lengths.validate(
            "run-length lengths",
            item_bytes,
            wgpu::BufferUsages::STORAGE,
        )?;
        input.validate_storage_offset(&self.device, "run-length input")?;
        unique_values.validate_storage_offset(&self.device, "run-length unique values")?;
        run_lengths.validate_storage_offset(&self.device, "run-length lengths")?;
        self.ensure_workspace(item_bytes, capacity)?;
        encoder.clear_buffer(run_count.buffer, run_count.offset, Some(COUNT_SIZE_BYTES));

        let input = BufferRange {
            size: item_bytes,
            ..input
        };
        let unique_values = BufferRange {
            size: item_bytes,
            ..unique_values
        };
        let run_lengths = BufferRange {
            size: item_bytes,
            ..run_lengths
        };
        let run_count = BufferRange {
            size: COUNT_SIZE_BYTES,
            ..run_count
        };
        let heads_buffer = self.heads.get().expect("run-length heads are reserved");
        let offsets_buffer = self.offsets.get().expect("run-length offsets are reserved");
        let heads = BufferRange::whole(heads_buffer);
        let offsets = BufferRange::whole(offsets_buffer);
        let input_count = extent.count().map(|range| BufferRange {
            size: COUNT_SIZE_BYTES,
            ..range
        });
        let fixed_items = extent.fixed_items();

        self.pipeline.mark(
            &self.device,
            encoder,
            input,
            heads,
            input_count,
            capacity,
            fixed_items,
            profiler.as_deref_mut(),
        );
        match profiler.as_deref_mut() {
            Some(profiler) => self.scanner.record_profiled_exclusive_scan(
                encoder,
                heads_buffer,
                offsets_buffer,
                capacity,
                "run_length.scan",
                profiler,
            )?,
            None => self.scanner.record_exclusive_scan(
                encoder,
                heads_buffer,
                offsets_buffer,
                capacity,
            )?,
        }
        self.pipeline.scatter(
            &self.device,
            encoder,
            input,
            heads,
            offsets,
            unique_values,
            run_lengths,
            input_count,
            capacity,
            fixed_items,
            profiler.as_deref_mut(),
        );
        self.pipeline.finalize(
            &self.device,
            encoder,
            heads,
            offsets,
            run_lengths,
            input_count,
            run_count,
            capacity,
            fixed_items,
            profiler,
        );
        Ok(())
    }

    fn ensure_workspace(&mut self, item_bytes: u64, capacity: u32) -> Result<(), Error> {
        common::buffers::validate_storage_binding_size(&self.device, item_bytes)?;
        self.heads.ensure(
            &self.device,
            item_bytes,
            "Run-Length Heads",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        self.offsets.ensure(
            &self.device,
            item_bytes,
            "Run-Length Offsets",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        self.scanner.reserve(capacity);
        Ok(())
    }
}

fn validate_distinct_buffers(
    input: BufferRange<'_>,
    input_count: Option<BufferRange<'_>>,
    unique_values: BufferRange<'_>,
    run_lengths: BufferRange<'_>,
    run_count: BufferRange<'_>,
) -> Result<(), Error> {
    let mut pairs = vec![
        (
            input,
            "run-length input",
            unique_values,
            "run-length unique values",
        ),
        (input, "run-length input", run_lengths, "run-length lengths"),
        (
            input,
            "run-length input",
            run_count,
            "run-length output count",
        ),
        (
            unique_values,
            "run-length unique values",
            run_lengths,
            "run-length lengths",
        ),
        (
            unique_values,
            "run-length unique values",
            run_count,
            "run-length output count",
        ),
        (
            run_lengths,
            "run-length lengths",
            run_count,
            "run-length output count",
        ),
    ];
    if let Some(input_count) = input_count {
        pairs.extend([
            (
                input_count,
                "run-length input count",
                unique_values,
                "run-length unique values",
            ),
            (
                input_count,
                "run-length input count",
                run_lengths,
                "run-length lengths",
            ),
            (
                input_count,
                "run-length input count",
                run_count,
                "run-length output count",
            ),
        ]);
    }
    for (first, first_name, second, second_name) in pairs {
        if first.buffer == second.buffer {
            return Err(Error::BufferAlias {
                first: first_name,
                second: second_name,
            });
        }
    }
    Ok(())
}
