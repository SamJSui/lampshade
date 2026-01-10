use crate::common;
use crate::context::Context;
use crate::scan::Scanner;
use crate::sort::pipeline::SortPipeline;
use std::sync::Arc;

struct SortWorkspace {
    capacity_bytes: u64,
    buf_a: wgpu::Buffer,
    #[allow(dead_code)]
    buf_b: wgpu::Buffer,
    buf_hist: wgpu::Buffer,
    buf_scanned_hist: wgpu::Buffer,
    uniform_buffers: Vec<wgpu::Buffer>,
    bind_groups: Vec<(wgpu::BindGroup, wgpu::BindGroup)>,
}

pub struct Sorter {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    scanner: Scanner,
    pipeline: SortPipeline,
    workspace: Option<SortWorkspace>,
}

impl Sorter {
    pub fn new(ctx: &Context) -> Self {
        Self {
            device: Arc::new(ctx.device.clone()),
            queue: Arc::new(ctx.queue.clone()),
            scanner: Scanner::new(ctx),
            pipeline: SortPipeline::new(ctx),
            workspace: None,
        }
    }

    pub async fn sort(&mut self, input: &[u32]) -> Vec<u32> {
        const GPU_THRESHOLD: usize = 1_000_000;

        if input.len() < GPU_THRESHOLD {
            let mut data = input.to_vec();
            data.sort_unstable();
            return data;
        } else {
            return self.sort_radix(input).await;
        }
    }

    pub async fn sort_radix(&mut self, input: &[u32]) -> Vec<u32> {
        let n = input.len() as u64;
        let n_bytes = n * 4;

        // 1. Allocation
        let need_realloc = if let Some(ws) = &self.workspace {
            ws.capacity_bytes < n_bytes
        } else {
            true
        };

        if need_realloc {
            self.allocate_workspace(n_bytes);
        }

        let ws = self.workspace.as_mut().unwrap();

        self.queue
            .write_buffer(&ws.buf_a, 0, bytemuck::cast_slice(input));

        let items_per_block = (self.pipeline.vt * self.pipeline.block_size) as u64;
        let num_blocks = (n + items_per_block - 1) / items_per_block;

        for i in 0..16 {
            let bit = i * 2;
            let uniform_data = [bit as u32, n as u32, num_blocks as u32, 0];
            self.queue.write_buffer(
                &ws.uniform_buffers[i],
                0,
                bytemuck::cast_slice(&uniform_data),
            );
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Fused Sort"),
            });

        let max_dispatch = 65535;
        let x_groups = if num_blocks as u32 > max_dispatch {
            max_dispatch
        } else {
            num_blocks as u32
        };
        let y_groups = (num_blocks as u32 + max_dispatch - 1) / max_dispatch;

        for i in 0..16 {
            let (reduce_bg, scatter_bg) = &ws.bind_groups[i];

            // A. Reduce
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_pipeline(&self.pipeline.reduce_pipeline);
                cpass.set_bind_group(0, reduce_bg, &[]);
                cpass.dispatch_workgroups(x_groups, y_groups, 1);
            }

            // B. Scan
            self.scanner
                .record_scan(&mut encoder, &ws.buf_hist, &ws.buf_scanned_hist);

            // C. Scatter
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_pipeline(&self.pipeline.scatter_pipeline);
                cpass.set_bind_group(0, scatter_bg, &[]);
                cpass.dispatch_workgroups(x_groups, y_groups, 1);
            }
        }

        self.queue.submit(Some(encoder.finish()));

        common::buffers::download_buffer(&self.device, &self.queue, &ws.buf_a, n_bytes).await
    }

    pub fn sort_resident(&mut self, input: &[u32]) -> &wgpu::Buffer {
        let n = input.len() as u64;
        let n_bytes = n * 4;

        let need_realloc = if let Some(ws) = &self.workspace {
            ws.capacity_bytes < n_bytes
        } else {
            true
        };

        if need_realloc {
            self.allocate_workspace(n_bytes);
        }

        let ws = self.workspace.as_mut().unwrap();

        self.queue
            .write_buffer(&ws.buf_a, 0, bytemuck::cast_slice(input));

        let items_per_block = (self.pipeline.vt * self.pipeline.block_size) as u64;
        let num_blocks = (n + items_per_block - 1) / items_per_block;

        for i in 0..16 {
            let bit = i * 2;
            let uniform_data = [bit as u32, n as u32, num_blocks as u32, 0];
            self.queue.write_buffer(
                &ws.uniform_buffers[i],
                0,
                bytemuck::cast_slice(&uniform_data),
            );
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Fused Sort Resident"),
            });

        let max_dispatch = 65535;
        let x_groups = if num_blocks as u32 > max_dispatch {
            max_dispatch
        } else {
            num_blocks as u32
        };
        let y_groups = (num_blocks as u32 + max_dispatch - 1) / max_dispatch;

        for i in 0..16 {
            let (reduce_bg, scatter_bg) = &ws.bind_groups[i];

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_pipeline(&self.pipeline.reduce_pipeline);
                cpass.set_bind_group(0, reduce_bg, &[]);
                cpass.dispatch_workgroups(x_groups, y_groups, 1);
            }

            self.scanner
                .record_scan(&mut encoder, &ws.buf_hist, &ws.buf_scanned_hist);

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_pipeline(&self.pipeline.scatter_pipeline);
                cpass.set_bind_group(0, scatter_bg, &[]);
                cpass.dispatch_workgroups(x_groups, y_groups, 1);
            }
        }

        self.queue.submit(Some(encoder.finish()));

        &ws.buf_a
    }

    fn allocate_workspace(&mut self, requested_size: u64) {
        let capacity = common::math::align_to(requested_size, 16 * 1024 * 1024);
        let items_per_block = (self.pipeline.vt * self.pipeline.block_size) as u64;
        let max_items = capacity / 4;
        let max_blocks = (max_items + items_per_block - 1) / items_per_block;

        let hist_bytes = max_blocks * 16;
        let hist_bytes_aligned = common::math::align_to(hist_bytes, 256);

        let buf_a = common::buffers::create_empty_storage_buffer(&self.device, capacity);
        let buf_b = common::buffers::create_empty_storage_buffer(&self.device, capacity);
        let buf_hist =
            common::buffers::create_empty_storage_buffer(&self.device, hist_bytes_aligned);
        let buf_scanned_hist =
            common::buffers::create_empty_storage_buffer(&self.device, hist_bytes_aligned);

        let mut uniform_buffers = Vec::with_capacity(16);
        for _ in 0..16 {
            uniform_buffers.push(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sort Uniform"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        let mut bind_groups = Vec::with_capacity(16);
        for i in 0..16 {
            let (source, dest) = if i % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };

            let reduce_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Reduce BG"),
                layout: &self.pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: source.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_hist.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dest.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buffers[i].as_entire_binding(),
                    },
                ],
            });

            let scatter_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Scatter BG"),
                layout: &self.pipeline.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: source.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_scanned_hist.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dest.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buffers[i].as_entire_binding(),
                    },
                ],
            });

            bind_groups.push((reduce_bg, scatter_bg));
        }

        self.workspace = Some(SortWorkspace {
            capacity_bytes: capacity,
            buf_a,
            buf_b,
            buf_hist,
            buf_scanned_hist,
            uniform_buffers,
            bind_groups,
        });
    }
}
