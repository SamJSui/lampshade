use crate::Error;
use crate::common;
use crate::context::Context;
use crate::scan::Scanner;
use crate::sort::pipeline::SortPipeline;

const RADIX_PASSES: usize = 16;
const WORKSPACE_GROWTH_BYTES: u64 = 16 * 1024 * 1024;

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

#[derive(Clone, Copy)]
struct PreparedSort {
    num_items: u32,
    num_blocks: u32,
    size_bytes: u64,
}

pub struct Sorter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    scanner: Scanner,
    pipeline: SortPipeline,
    workspace: Option<SortWorkspace>,
}

impl Sorter {
    pub fn new(ctx: &Context) -> Self {
        Self {
            device: ctx.device.clone(),
            queue: ctx.queue.clone(),
            scanner: Scanner::new(ctx),
            pipeline: SortPipeline::new(ctx),
            workspace: None,
        }
    }

    pub async fn sort(&mut self, input: &[u32]) -> Result<Vec<u32>, Error> {
        const GPU_THRESHOLD: usize = 1_000_000;

        if input.len() < GPU_THRESHOLD {
            let mut data = input.to_vec();
            data.sort_unstable();
            Ok(data)
        } else {
            self.sort_radix(input).await
        }
    }

    pub async fn sort_radix(&mut self, input: &[u32]) -> Result<Vec<u32>, Error> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        let problem = self.prepare_sort(input)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Radix Sort"),
            });
        self.record_radix_passes(&mut encoder, problem)?;
        self.queue.submit(Some(encoder.finish()));

        let output = &self
            .workspace
            .as_ref()
            .expect("sort workspace is prepared")
            .buf_a;
        common::buffers::download_buffer(&self.device, &self.queue, output, problem.size_bytes)
            .await
    }

    pub fn sort_resident(&mut self, input: &[u32]) -> Result<&wgpu::Buffer, Error> {
        let problem = self.prepare_sort(input)?;

        if problem.num_items > 0 {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Radix Sort Without Readback"),
                });
            self.record_radix_passes(&mut encoder, problem)?;
            self.queue.submit(Some(encoder.finish()));
        }

        Ok(&self
            .workspace
            .as_ref()
            .expect("sort workspace is prepared")
            .buf_a)
    }

    fn prepare_sort(&mut self, input: &[u32]) -> Result<PreparedSort, Error> {
        let num_items = common::math::checked_u32(input.len() as u64)?;
        let size_bytes = common::math::checked_byte_size(input.len() as u64, 4)?;
        let required_capacity = size_bytes.max(4);

        let need_realloc = self
            .workspace
            .as_ref()
            .is_none_or(|workspace| workspace.capacity_bytes < required_capacity);
        if need_realloc {
            self.allocate_workspace(required_capacity)?;
        }

        let items_per_block = self.pipeline.vt * self.pipeline.block_size;
        let num_blocks = num_items.div_ceil(items_per_block);
        let workspace = self.workspace.as_ref().expect("sort workspace is prepared");

        if !input.is_empty() {
            self.queue
                .write_buffer(&workspace.buf_a, 0, bytemuck::cast_slice(input));
        }

        for (pass, uniform_buffer) in workspace.uniform_buffers.iter().enumerate() {
            let bit = u32::try_from(pass * 2).expect("radix pass index fits in u32");
            let uniform_data = [bit, num_items, num_blocks, 0];
            self.queue
                .write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&uniform_data));
        }

        Ok(PreparedSort {
            num_items,
            num_blocks,
            size_bytes,
        })
    }

    fn record_radix_passes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        problem: PreparedSort,
    ) -> Result<(), Error> {
        if problem.num_items == 0 {
            return Ok(());
        }

        let max_dispatch = 65_535;
        let x_groups = problem.num_blocks.min(max_dispatch);
        let y_groups = problem.num_blocks.div_ceil(max_dispatch);
        let histogram_items = problem
            .num_blocks
            .checked_mul(4)
            .ok_or(Error::SizeOverflow)?;
        let workspace = self.workspace.as_ref().expect("sort workspace is prepared");

        for (reduce_bind_group, scatter_bind_group) in &workspace.bind_groups {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.pipeline.reduce_pipeline);
                pass.set_bind_group(0, reduce_bind_group, &[]);
                pass.dispatch_workgroups(x_groups, y_groups, 1);
            }

            self.scanner.record_scan(
                encoder,
                &workspace.buf_hist,
                &workspace.buf_scanned_hist,
                histogram_items,
            )?;

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.pipeline.scatter_pipeline);
                pass.set_bind_group(0, scatter_bind_group, &[]);
                pass.dispatch_workgroups(x_groups, y_groups, 1);
            }
        }

        Ok(())
    }

    fn allocate_workspace(&mut self, requested_size: u64) -> Result<(), Error> {
        let capacity = if requested_size <= 4 {
            4
        } else {
            common::math::checked_align_to(requested_size, WORKSPACE_GROWTH_BYTES)?
        };
        let max_buffer_size = self.device.limits().max_buffer_size;
        if capacity > max_buffer_size {
            return Err(Error::BufferLimitExceeded {
                requested: capacity,
                limit: max_buffer_size,
            });
        }

        let items_per_block = u64::from(self.pipeline.vt * self.pipeline.block_size);
        let max_items = capacity / 4;
        let max_blocks = max_items.div_ceil(items_per_block);
        let hist_bytes = common::math::checked_byte_size(max_blocks, 16)?;
        let hist_bytes_aligned = common::math::checked_align_to(hist_bytes, 256)?;

        let buf_a = common::buffers::create_empty_storage_buffer(&self.device, capacity);
        let buf_b = common::buffers::create_empty_storage_buffer(&self.device, capacity);
        let buf_hist =
            common::buffers::create_empty_storage_buffer(&self.device, hist_bytes_aligned);
        let buf_scanned_hist =
            common::buffers::create_empty_storage_buffer(&self.device, hist_bytes_aligned);

        let uniform_buffers = (0..RADIX_PASSES)
            .map(|_| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Sort Uniform"),
                    size: 16,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect::<Vec<_>>();

        let bind_groups = uniform_buffers
            .iter()
            .enumerate()
            .map(|(pass, uniform_buffer)| {
                let (source, destination) = if pass % 2 == 0 {
                    (&buf_a, &buf_b)
                } else {
                    (&buf_b, &buf_a)
                };

                let reduce = self.create_sort_bind_group(
                    "Reduce Bind Group",
                    source,
                    &buf_hist,
                    destination,
                    uniform_buffer,
                );
                let scatter = self.create_sort_bind_group(
                    "Scatter Bind Group",
                    source,
                    &buf_scanned_hist,
                    destination,
                    uniform_buffer,
                );
                (reduce, scatter)
            })
            .collect();

        self.workspace = Some(SortWorkspace {
            capacity_bytes: capacity,
            buf_a,
            buf_b,
            buf_hist,
            buf_scanned_hist,
            uniform_buffers,
            bind_groups,
        });
        Ok(())
    }

    fn create_sort_bind_group(
        &self,
        label: &'static str,
        source: &wgpu::Buffer,
        histogram: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        uniform: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.pipeline.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: histogram.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: destination.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform.as_entire_binding(),
                },
            ],
        })
    }
}
