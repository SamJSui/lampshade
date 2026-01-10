use super::pipeline::ScanPipeline;
use crate::{common, context::Context};
use std::sync::Arc;

pub struct Scanner {
    pipeline: ScanPipeline,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pub scratch_buffer: Option<wgpu::Buffer>,
    scratch_size_bytes: u64,
}

impl Scanner {
    pub fn new(ctx: &Context) -> Self {
        Self {
            pipeline: ScanPipeline::new(ctx),
            device: Arc::new(ctx.device.clone()),
            queue: Arc::new(ctx.queue.clone()),
            scratch_buffer: None,
            scratch_size_bytes: 0,
        }
    }

    pub async fn scan(&mut self, input: &[u32]) -> Vec<u32> {
        let data_buffer = common::buffers::create_storage_buffer(&self.device, input);
        let dst_buffer =
            common::buffers::create_empty_storage_buffer(&self.device, data_buffer.size());

        self.scan_gpu_to_gpu(&data_buffer, &dst_buffer).await;

        let size_bytes = (input.len() * 4) as u64;
        common::buffers::download_buffer(&self.device, &self.queue, &dst_buffer, size_bytes).await
    }

    pub async fn scan_gpu_to_gpu(&mut self, input_buf: &wgpu::Buffer, output_buf: &wgpu::Buffer) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.record_scan(&mut encoder, input_buf, output_buf);
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn record_scan(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input_buf: &wgpu::Buffer,
        output_buf: &wgpu::Buffer,
    ) {
        let size_bytes = input_buf.size();
        let num_items = (size_bytes / 4) as u32;

        self.prepare_scratch(num_items);

        encoder.copy_buffer_to_buffer(input_buf, 0, output_buf, 0, size_bytes);

        let scratch = self.scratch_buffer.as_ref().unwrap();

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

            let aux_count = (current.count + items_per_block - 1) / items_per_block;
            let aux_size = (aux_count * 4) as u64;
            let aux_offset = crate::common::math::align_to(current_scratch_offset, 256);

            self.pipeline.dispatch(
                &self.device,
                encoder,
                &self.pipeline.scan_pipeline,
                current.buf,
                current.offset,
                scratch,
                aux_offset,
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
                data_level.buf,
                data_level.offset,
                aux_level.buf,
                aux_level.offset,
                data_level.count,
            );
        }
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
