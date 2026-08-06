use wgpu::util::DeviceExt;

use crate::Error;
use crate::common;
use crate::profiling::{self, GpuProfile, TimestampRecorder};
use crate::scan::Scanner;

use super::eight_bit::EightBitSorter;
use super::pipeline::{RadixVariant, SortItemKind, SortPipeline};

const WORKSPACE_GROWTH_BYTES: u64 = 16 * 1024 * 1024;
const UNIFORM_SIZE_BYTES: u64 = 16;

pub struct RadixSorter {
    implementation: SortImplementation,
}

enum SortImplementation {
    ReduceScan(ReduceScanSorter),
    EightBit(EightBitSorter),
}

impl RadixSorter {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, item_kind: SortItemKind) -> Self {
        Self {
            implementation: SortImplementation::ReduceScan(ReduceScanSorter::new(
                device,
                queue,
                item_kind,
                RadixVariant::Portable,
            )),
        }
    }

    pub fn new_for_adapter(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item_kind: SortItemKind,
        adapter_info: &wgpu::AdapterInfo,
    ) -> Self {
        let variant = RadixVariant::for_adapter(item_kind, adapter_info, device.features());
        let implementation = if variant.uses_eight_bit_pipeline() {
            SortImplementation::EightBit(EightBitSorter::new(device, queue))
        } else {
            SortImplementation::ReduceScan(ReduceScanSorter::new(device, queue, item_kind, variant))
        };
        Self { implementation }
    }

    pub async fn sort_slice<T: bytemuck::Pod>(&mut self, input: &[T]) -> Result<Vec<T>, Error> {
        match &mut self.implementation {
            SortImplementation::ReduceScan(sorter) => sorter.sort_slice(input).await,
            SortImplementation::EightBit(sorter) => sorter.sort_slice(input).await,
        }
    }

    pub fn sort_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        match &mut self.implementation {
            SortImplementation::ReduceScan(sorter) => {
                sorter.sort_gpu_to_gpu(input, output, num_items)
            }
            SortImplementation::EightBit(sorter) => {
                sorter.sort_gpu_to_gpu(input, output, num_items)
            }
        }
    }

    pub async fn profile_sort_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<GpuProfile, Error> {
        match &mut self.implementation {
            SortImplementation::ReduceScan(sorter) => {
                sorter
                    .profile_sort_gpu_to_gpu(input, output, num_items)
                    .await
            }
            SortImplementation::EightBit(sorter) => {
                sorter
                    .profile_sort_gpu_to_gpu(input, output, num_items)
                    .await
            }
        }
    }

    pub fn record_sort(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        match &mut self.implementation {
            SortImplementation::ReduceScan(sorter) => {
                sorter.record_sort(encoder, input, output, num_items)
            }
            SortImplementation::EightBit(sorter) => {
                sorter.record_sort(encoder, input, output, num_items)
            }
        }
    }
}

struct SortWorkspace {
    capacity_bytes: u64,
    scratch: wgpu::Buffer,
    histogram: wgpu::Buffer,
    scanned_histogram: wgpu::Buffer,
}

#[derive(Clone, Copy)]
struct PreparedSort {
    num_items: u32,
    num_blocks: u32,
    size_bytes: u64,
}

struct ReduceScanSorter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    scanner: Scanner,
    pipeline: SortPipeline,
    workspace: Option<SortWorkspace>,
    item_size: u64,
}

impl ReduceScanSorter {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        item_kind: SortItemKind,
        radix_variant: RadixVariant,
    ) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
            scanner: Scanner::new(device, queue),
            pipeline: SortPipeline::new(device, item_kind, radix_variant),
            workspace: None,
            item_size: item_kind.size_bytes(),
        }
    }

    pub async fn sort_slice<T: bytemuck::Pod>(&mut self, input: &[T]) -> Result<Vec<T>, Error> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        assert_eq!(size_of::<T>() as u64, self.item_size);
        let num_items = common::math::checked_u32(input.len() as u64)?;
        let size_bytes = common::math::checked_byte_size(input.len() as u64, self.item_size)?;
        let input_buffer = common::buffers::create_storage_buffer(&self.device, input);
        let output_buffer = common::buffers::create_empty_storage_buffer(&self.device, size_bytes);

        self.sort_gpu_to_gpu(&input_buffer, &output_buffer, num_items)?;
        common::buffers::download_buffer(&self.device, &self.queue, &output_buffer, input.len())
            .await
    }

    pub fn sort_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Radix Sort"),
            });
        self.record_sort(&mut encoder, input, output, num_items)?;
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub fn record_sort(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        let Some(problem) = self.prepare_sort(input, output, num_items)? else {
            return Ok(());
        };
        self.record_radix_passes(encoder, input, output, problem, None)
    }

    pub async fn profile_sort_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<GpuProfile, Error> {
        let Some(problem) = self.prepare_sort(input, output, num_items)? else {
            return Ok(GpuProfile::empty());
        };
        let histogram_items = problem
            .num_blocks
            .checked_mul(self.pipeline.bucket_count)
            .ok_or(Error::SizeOverflow)?;
        let spans_per_radix_pass = self
            .scanner
            .compute_pass_count(histogram_items)
            .checked_add(2)
            .ok_or(Error::SizeOverflow)?;
        let span_count = self
            .pipeline
            .pass_count
            .checked_mul(spans_per_radix_pass)
            .ok_or(Error::SizeOverflow)?;

        let mut profiler = TimestampRecorder::new(&self.device, &self.queue, span_count)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Profiled Radix Sort"),
            });
        self.record_radix_passes(&mut encoder, input, output, problem, Some(&mut profiler))?;
        profiler.resolve(&mut encoder);
        let submission = self.queue.submit(Some(encoder.finish()));
        profiler.read(&self.device, submission).await
    }

    fn prepare_sort(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<Option<PreparedSort>, Error> {
        if num_items == 0 {
            return Ok(None);
        }

        let problem = self.describe_sort(num_items)?;
        common::buffers::validate_buffer(
            input,
            "sort input",
            problem.size_bytes,
            wgpu::BufferUsages::STORAGE,
        )?;
        common::buffers::validate_buffer(
            output,
            "sort output",
            problem.size_bytes,
            wgpu::BufferUsages::STORAGE,
        )?;

        self.ensure_workspace(problem.size_bytes)?;
        Ok(Some(problem))
    }

    fn describe_sort(&self, num_items: u32) -> Result<PreparedSort, Error> {
        let size_bytes = common::math::checked_byte_size(u64::from(num_items), self.item_size)?;
        let items_per_block = self.pipeline.vt * self.pipeline.block_size;
        let num_blocks = num_items.div_ceil(items_per_block);

        Ok(PreparedSort {
            num_items,
            num_blocks,
            size_bytes,
        })
    }

    fn ensure_workspace(&mut self, size_bytes: u64) -> Result<(), Error> {
        let needs_allocation = self
            .workspace
            .as_ref()
            .is_none_or(|workspace| workspace.capacity_bytes < size_bytes);
        if needs_allocation {
            self.allocate_workspace(size_bytes)?;
        }
        Ok(())
    }

    fn record_radix_passes(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        problem: PreparedSort,
        mut profiler: Option<&mut TimestampRecorder>,
    ) -> Result<(), Error> {
        let max_dispatch = 65_535;
        let x_groups = problem.num_blocks.min(max_dispatch);
        let y_groups = problem.num_blocks.div_ceil(max_dispatch);
        let histogram_items = problem
            .num_blocks
            .checked_mul(self.pipeline.bucket_count)
            .ok_or(Error::SizeOverflow)?;

        let workspace = self.workspace.as_ref().expect("sort workspace is prepared");
        let (uniform, uniform_stride) = create_uniform_buffer(
            &self.device,
            problem,
            self.pipeline.bits_per_pass,
            self.pipeline.pass_count,
        );

        for radix_pass in 0..self.pipeline.pass_count {
            let (source, destination) = pass_buffers(radix_pass, input, output, &workspace.scratch);
            let uniform_offset = u64::from(radix_pass) * uniform_stride;
            let reduce_bind_group = create_sort_bind_group(
                &self.device,
                &self.pipeline.bind_group_layout,
                "Reduce Bind Group",
                (source, &workspace.histogram, destination),
                &uniform,
                uniform_offset,
            );
            let scatter_bind_group = create_sort_bind_group(
                &self.device,
                &self.pipeline.bind_group_layout,
                "Scatter Bind Group",
                (source, &workspace.scanned_histogram, destination),
                &uniform,
                uniform_offset,
            );

            let reduce_profile_label = profiler
                .is_some()
                .then(|| format!("radix.{radix_pass:02}.reduce"));
            profiling::record_compute_pass(
                encoder,
                "Radix Histogram Reduce",
                reduce_profile_label,
                profiler.as_deref_mut(),
                |pass| {
                    pass.set_pipeline(&self.pipeline.reduce_pipeline);
                    pass.set_bind_group(0, &reduce_bind_group, &[]);
                    pass.dispatch_workgroups(x_groups, y_groups, 1);
                },
            );
            if let Some(profiler) = profiler.as_deref_mut() {
                self.scanner.record_profiled_scan(
                    encoder,
                    &workspace.histogram,
                    &workspace.scanned_histogram,
                    histogram_items,
                    &format!("radix.{radix_pass:02}.scan"),
                    profiler,
                )?;
            } else {
                self.scanner.record_scan(
                    encoder,
                    &workspace.histogram,
                    &workspace.scanned_histogram,
                    histogram_items,
                )?;
            }
            let scatter_profile_label = profiler
                .is_some()
                .then(|| format!("radix.{radix_pass:02}.scatter"));
            profiling::record_compute_pass(
                encoder,
                "Radix Stable Scatter",
                scatter_profile_label,
                profiler.as_deref_mut(),
                |pass| {
                    pass.set_pipeline(&self.pipeline.scatter_pipeline);
                    pass.set_bind_group(0, &scatter_bind_group, &[]);
                    pass.dispatch_workgroups(x_groups, y_groups, 1);
                },
            );
        }

        Ok(())
    }

    fn allocate_workspace(&mut self, requested_size: u64) -> Result<(), Error> {
        let capacity = workspace_capacity(requested_size)?;
        let limits = self.device.limits();
        let buffer_limit = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));
        if capacity > buffer_limit {
            return Err(Error::BufferLimitExceeded {
                requested: capacity,
                limit: buffer_limit,
            });
        }

        let items_per_block = u64::from(self.pipeline.vt * self.pipeline.block_size);
        let max_items = capacity / self.item_size;
        let max_blocks = max_items.div_ceil(items_per_block);
        let histogram_items = max_blocks
            .checked_mul(u64::from(self.pipeline.bucket_count))
            .ok_or(Error::SizeOverflow)?;
        let histogram_bytes = common::math::checked_byte_size(histogram_items, 4)?;
        let histogram_capacity = common::math::checked_align_to(histogram_bytes, 256)?;

        self.workspace = Some(SortWorkspace {
            capacity_bytes: capacity,
            scratch: common::buffers::create_empty_storage_buffer(&self.device, capacity),
            histogram: common::buffers::create_empty_storage_buffer(
                &self.device,
                histogram_capacity,
            ),
            scanned_histogram: common::buffers::create_empty_storage_buffer(
                &self.device,
                histogram_capacity,
            ),
        });
        Ok(())
    }
}

fn workspace_capacity(requested_size: u64) -> Result<u64, Error> {
    if requested_size < WORKSPACE_GROWTH_BYTES {
        requested_size
            .max(4)
            .checked_next_power_of_two()
            .ok_or(Error::SizeOverflow)
    } else {
        common::math::checked_align_to(requested_size, WORKSPACE_GROWTH_BYTES)
    }
}

fn pass_buffers<'a>(
    radix_pass: u32,
    input: &'a wgpu::Buffer,
    output: &'a wgpu::Buffer,
    scratch: &'a wgpu::Buffer,
) -> (&'a wgpu::Buffer, &'a wgpu::Buffer) {
    if radix_pass == 0 {
        (input, scratch)
    } else if radix_pass.is_multiple_of(2) {
        (output, scratch)
    } else {
        (scratch, output)
    }
}

fn create_uniform_buffer(
    device: &wgpu::Device,
    problem: PreparedSort,
    bits_per_pass: u32,
    pass_count: u32,
) -> (wgpu::Buffer, u64) {
    let uniform_stride =
        u64::from(device.limits().min_uniform_buffer_offset_alignment).max(UNIFORM_SIZE_BYTES);
    let words_per_uniform = (uniform_stride / size_of::<u32>() as u64) as usize;
    let mut data = vec![0_u32; words_per_uniform * pass_count as usize];

    for radix_pass in 0..pass_count as usize {
        let offset = radix_pass * words_per_uniform;
        data[offset..offset + 4].copy_from_slice(&[
            radix_pass as u32 * bits_per_pass,
            problem.num_items,
            problem.num_blocks,
            0,
        ]);
    }

    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sort Uniform"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    (buffer, uniform_stride)
}

fn create_sort_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &'static str,
    buffers: (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer),
    uniform: &wgpu::Buffer,
    uniform_offset: u64,
) -> wgpu::BindGroup {
    let (source, histogram, destination) = buffers;
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
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
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: uniform,
                    offset: uniform_offset,
                    size: wgpu::BufferSize::new(UNIFORM_SIZE_BYTES),
                }),
            },
        ],
    })
}
