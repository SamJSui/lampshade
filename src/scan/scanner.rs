use super::pipeline::ScanPipeline;
use crate::{Error, common, context::Context};

#[derive(Clone, Copy)]
enum ScanMode {
    Inclusive,
    Exclusive,
}

/// Performs inclusive and exclusive unsigned 32-bit prefix scans on a wgpu device.
pub struct Scanner {
    pipeline: ScanPipeline,
    device: wgpu::Device,
    queue: wgpu::Queue,
    scratch_buffer: Option<wgpu::Buffer>,
    scratch_size_bytes: u64,
}

impl Scanner {
    /// Creates a scanner that submits work through an existing wgpu device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            pipeline: ScanPipeline::new(device),
            device: device.clone(),
            queue: queue.clone(),
            scratch_buffer: None,
            scratch_size_bytes: 0,
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.record_scan_with_mode(&mut encoder, input_buf, output_buf, num_items, mode)?;
        self.queue.submit(Some(encoder.finish()));
        Ok(())
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
            input_buf,
            output_buf,
            num_items,
            ScanMode::Inclusive,
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
            input_buf,
            output_buf,
            num_items,
            ScanMode::Exclusive,
        )
    }

    fn record_scan_with_mode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
        num_items: u32,
        mode: ScanMode,
    ) -> Result<(), Error> {
        if num_items == 0 {
            return Ok(());
        }

        let size_bytes = common::math::checked_byte_size(u64::from(num_items), 4)?;
        common::buffers::validate_buffer(
            input_buf,
            "scan input",
            size_bytes,
            wgpu::BufferUsages::COPY_SRC,
        )?;
        common::buffers::validate_buffer(
            output_buf,
            "scan output",
            size_bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        )?;

        if num_items == 1 {
            match mode {
                ScanMode::Inclusive => {
                    encoder.copy_buffer_to_buffer(input_buf, 0, output_buf, 0, size_bytes);
                }
                ScanMode::Exclusive => encoder.clear_buffer(output_buf, 0, Some(size_bytes)),
            }
            return Ok(());
        }

        encoder.copy_buffer_to_buffer(input_buf, 0, output_buf, 0, size_bytes);

        self.prepare_scratch(num_items);

        let scratch = self
            .scratch_buffer
            .as_ref()
            .expect("scan scratch exists for multi-element inputs");

        struct Level<'a> {
            buf: &'a wgpu::Buffer,
            offset: u64,
            count: u32,
        }

        let mut levels = Vec::new();
        levels.push(Level {
            buf: output_buf,
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

            let scan_pipeline = match (levels.len(), mode) {
                (1, ScanMode::Exclusive) => &self.pipeline.exclusive_scan_pipeline,
                _ => &self.pipeline.inclusive_scan_pipeline,
            };

            self.pipeline.dispatch(
                &self.device,
                encoder,
                scan_pipeline,
                (current.buf, current.offset),
                (scratch, aux_offset),
                current.count,
            );

            levels.push(Level {
                buf: scratch,
                offset: aux_offset,
                count: aux_count,
            });
            current_scratch_offset = aux_offset + aux_size;
        }

        for i in (0..levels.len() - 1).rev() {
            let data_level = &levels[i];
            let aux_level = &levels[i + 1];

            self.pipeline.dispatch(
                &self.device,
                encoder,
                &self.pipeline.add_pipeline,
                (data_level.buf, data_level.offset),
                (aux_level.buf, aux_level.offset),
                data_level.count,
            );
        }

        Ok(())
    }

    fn prepare_scratch(&mut self, num_items: u32) {
        let needed_bytes = self.pipeline.get_scratch_size(num_items);
        if self.scratch_buffer.is_none() || needed_bytes > self.scratch_size_bytes {
            self.scratch_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Scanner Scratch"),
                size: needed_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
            self.scratch_size_bytes = needed_bytes;
        }
    }
}
