use super::pipeline::{ScanDispatch, ScanInputDispatch, ScanPipeline};
use crate::{
    Error, common,
    common::{runtime::CommandSession, runtime::ProfileSession, workspace::ReusableBuffer},
    context::Context,
    profiling::{GpuProfile, TimestampRecorder},
};

#[derive(Clone, Copy)]
enum ScanMode {
    Inclusive,
    Exclusive,
}

struct ScanRecording<'a> {
    input: &'a wgpu::Buffer,
    output: &'a wgpu::Buffer,
    num_items: u32,
    mode: ScanMode,
    profile_prefix: &'a str,
    propagate_output: bool,
}

/// Performs inclusive and exclusive unsigned 32-bit prefix scans on a wgpu device.
pub struct Scanner {
    pipeline: ScanPipeline,
    device: wgpu::Device,
    queue: wgpu::Queue,
    scratch: ReusableBuffer,
}

impl Scanner {
    /// Creates a scanner that submits work through an existing wgpu device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            pipeline: ScanPipeline::new(device),
            device: device.clone(),
            queue: queue.clone(),
            scratch: ReusableBuffer::default(),
        }
    }

    /// Creates a scanner from the crate's optional convenience context.
    pub fn from_context(ctx: &Context) -> Self {
        Self::new(&ctx.device, &ctx.queue)
    }

    /// Uploads values, scans them on the GPU, and downloads the inclusive prefixes.
    pub async fn scan(&mut self, input: &[u32]) -> Result<Vec<u32>, Error> {
        self.scan_slice(input, ScanMode::Inclusive).await
    }

    /// Uploads values, scans them on the GPU, and downloads the exclusive prefixes.
    pub async fn scan_exclusive(&mut self, input: &[u32]) -> Result<Vec<u32>, Error> {
        self.scan_slice(input, ScanMode::Exclusive).await
    }

    async fn scan_slice(&mut self, input: &[u32], mode: ScanMode) -> Result<Vec<u32>, Error> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        let num_items = common::math::checked_u32(input.len() as u64)?;
        let data_buffer = common::buffers::create_storage_buffer(&self.device, input);
        let dst_buffer =
            common::buffers::create_empty_storage_buffer(&self.device, data_buffer.size());

        self.submit_scan(&data_buffer, &dst_buffer, num_items, mode)?;

        common::buffers::download_buffer(&self.device, &self.queue, &dst_buffer, input.len()).await
    }

    /// Scans caller-owned GPU buffers and submits the work immediately.
    pub fn scan_gpu_to_gpu(
        &mut self,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        self.submit_scan(input_buf, output_buf, num_items, ScanMode::Inclusive)
    }

    /// Exclusively scans caller-owned GPU buffers and submits the work immediately.
    pub fn scan_exclusive_gpu_to_gpu(
        &mut self,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        self.submit_scan(input_buf, output_buf, num_items, ScanMode::Exclusive)
    }

    fn submit_scan(
        &mut self,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_items: u32,
        mode: ScanMode,
    ) -> Result<(), Error> {
        let mut commands = CommandSession::new(&self.device, None);
        self.record_scan_with_mode(
            commands.encoder(),
            ScanRecording {
                input: input_buf,
                output: output_buf,
                num_items,
                mode,
                profile_prefix: "scan",
                propagate_output: true,
            },
            None,
        )?;
        commands.submit(&self.queue);
        Ok(())
    }

    /// Profiles an inclusive scan of caller-owned GPU buffers using GPU timestamps.
    pub async fn profile_scan_gpu_to_gpu(
        &mut self,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<GpuProfile, Error> {
        self.profile_scan_with_mode(input_buf, output_buf, num_items, ScanMode::Inclusive)
            .await
    }

    /// Profiles an exclusive scan of caller-owned GPU buffers using GPU timestamps.
    pub async fn profile_exclusive_scan_gpu_to_gpu(
        &mut self,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<GpuProfile, Error> {
        self.profile_scan_with_mode(input_buf, output_buf, num_items, ScanMode::Exclusive)
            .await
    }

    async fn profile_scan_with_mode(
        &mut self,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_items: u32,
        mode: ScanMode,
    ) -> Result<GpuProfile, Error> {
        let span_count = self.pipeline.compute_pass_count(num_items);
        let label = if span_count == 0 {
            "Profiled Trivial Scan"
        } else {
            "Profiled Prefix Scan"
        };
        let mut profile = ProfileSession::new(&self.device, &self.queue, span_count, label)?;
        let (encoder, profiler) = profile.recording();
        self.record_scan_with_mode(
            encoder,
            ScanRecording {
                input: input_buf,
                output: output_buf,
                num_items,
                mode,
                profile_prefix: "scan",
                propagate_output: true,
            },
            profiler,
        )?;
        profile.finish(&self.device, &self.queue).await
    }

    /// Records a GPU prefix scan without submitting or waiting for the work.
    pub fn record_scan(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        self.record_scan_with_mode(
            encoder,
            ScanRecording {
                input: input_buf,
                output: output_buf,
                num_items,
                mode: ScanMode::Inclusive,
                profile_prefix: "scan",
                propagate_output: true,
            },
            None,
        )
    }

    /// Records an exclusive GPU prefix scan without submitting or waiting for the work.
    pub fn record_exclusive_scan(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        self.record_scan_with_mode(
            encoder,
            ScanRecording {
                input: input_buf,
                output: output_buf,
                num_items,
                mode: ScanMode::Exclusive,
                profile_prefix: "scan",
                propagate_output: true,
            },
            None,
        )
    }

    pub(crate) fn record_profiled_scan(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_items: u32,
        profile_prefix: &str,
        profiler: &mut TimestampRecorder,
    ) -> Result<(), Error> {
        self.record_scan_with_mode(
            encoder,
            ScanRecording {
                input: input_buf,
                output: output_buf,
                num_items,
                mode: ScanMode::Inclusive,
                profile_prefix,
                propagate_output: true,
            },
            Some(profiler),
        )
    }

    pub(crate) fn compute_pass_count(&self, num_items: u32) -> u32 {
        self.pipeline.compute_pass_count(num_items)
    }

    pub(crate) fn compute_block_local_pass_count(&self, num_items: u32) -> u32 {
        self.pipeline
            .compute_pass_count(num_items)
            .saturating_sub(u32::from(num_items > 1))
    }

    pub(crate) fn record_block_local_exclusive_scan(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_items: u32,
        profile_prefix: &str,
        profiler: Option<&mut TimestampRecorder>,
    ) -> Result<u32, Error> {
        self.record_scan_with_mode(
            encoder,
            ScanRecording {
                input: input_buf,
                output: output_buf,
                num_items,
                mode: ScanMode::Exclusive,
                profile_prefix,
                propagate_output: false,
            },
            profiler,
        )?;
        Ok(self.pipeline.vt * self.pipeline.block_size)
    }

    pub(crate) fn block_prefix_buffer(&self) -> Option<&wgpu::Buffer> {
        self.scratch.get()
    }

    fn record_scan_with_mode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        recording: ScanRecording<'_>,
        mut profiler: Option<&mut TimestampRecorder>,
    ) -> Result<(), Error> {
        let ScanRecording {
            input,
            output,
            num_items,
            mode,
            profile_prefix,
            propagate_output,
        } = recording;
        if num_items == 0 {
            return Ok(());
        }

        let size_bytes = common::math::checked_byte_size(u64::from(num_items), 4)?;
        common::buffers::validate_buffer(
            input,
            "scan input",
            size_bytes,
            wgpu::BufferUsages::COPY_SRC,
        )?;
        common::buffers::validate_buffer(
            output,
            "scan output",
            size_bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        )?;

        if num_items == 1 {
            match mode {
                ScanMode::Inclusive => {
                    encoder.copy_buffer_to_buffer(input, 0, output, 0, size_bytes);
                }
                ScanMode::Exclusive => encoder.clear_buffer(output, 0, Some(size_bytes)),
            }
            return Ok(());
        }

        self.prepare_scratch(num_items);

        let scratch = self
            .scratch
            .get()
            .expect("scan scratch exists for multi-element inputs");

        struct Level<'a> {
            buf: &'a wgpu::Buffer,
            offset: u64,
            count: u32,
        }

        let mut levels = Vec::new();
        levels.push(Level {
            buf: output,
            offset: 0,
            count: num_items,
        });

        let mut current_scratch_offset = 0u64;

        loop {
            let current = levels.last().unwrap();
            if current.count <= 1 {
                break;
            }

            let items_per_block = self.pipeline.vt * self.pipeline.block_size;

            let aux_count = current.count.div_ceil(items_per_block);
            let aux_size = (aux_count * 4) as u64;
            let aux_offset = crate::common::math::align_to(current_scratch_offset, 256);

            let profile_label = profiler
                .is_some()
                .then(|| format!("{profile_prefix}.level.{}", levels.len() - 1));

            if levels.len() == 1 {
                let scan_pipeline = match mode {
                    ScanMode::Inclusive => &self.pipeline.inclusive_input_scan_pipeline,
                    ScanMode::Exclusive => &self.pipeline.exclusive_input_scan_pipeline,
                };
                self.pipeline.dispatch_input(
                    &self.device,
                    encoder,
                    ScanInputDispatch {
                        pipeline: scan_pipeline,
                        input,
                        data: output,
                        auxiliary: (scratch, aux_offset),
                        num_items: current.count,
                        pass_label: "Prefix Scan",
                        profile_label,
                    },
                    profiler.as_deref_mut(),
                );
            } else {
                self.pipeline.dispatch(
                    &self.device,
                    encoder,
                    ScanDispatch {
                        pipeline: &self.pipeline.inclusive_scan_pipeline,
                        data: (current.buf, current.offset),
                        auxiliary: (scratch, aux_offset),
                        num_items: current.count,
                        pass_label: "Prefix Scan",
                        profile_label,
                    },
                    profiler.as_deref_mut(),
                );
            }

            levels.push(Level {
                buf: scratch,
                offset: aux_offset,
                count: aux_count,
            });
            current_scratch_offset = aux_offset + aux_size;
        }

        let first_add_level = usize::from(!propagate_output);
        for i in (first_add_level..levels.len() - 1).rev() {
            let data_level = &levels[i];
            let aux_level = &levels[i + 1];
            let profile_label = profiler
                .is_some()
                .then(|| format!("{profile_prefix}.add.{i}"));

            self.pipeline.dispatch(
                &self.device,
                encoder,
                ScanDispatch {
                    pipeline: &self.pipeline.add_pipeline,
                    data: (data_level.buf, data_level.offset),
                    auxiliary: (aux_level.buf, aux_level.offset),
                    num_items: data_level.count,
                    pass_label: "Prefix Add",
                    profile_label,
                },
                profiler.as_deref_mut(),
            );
        }

        Ok(())
    }

    fn prepare_scratch(&mut self, num_items: u32) {
        let needed_bytes = self.pipeline.get_scratch_size(num_items);
        self.scratch.ensure(
            &self.device,
            needed_bytes,
            "Scanner Scratch",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
    }
}
