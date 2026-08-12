use wgpu::util::DeviceExt;

use crate::{
    CountedSortDispatch, Error, GpuCountPlan, common,
    common::buffers::BufferRange,
    common::runtime::{CommandSession, ProfileSession},
    profiling::{self, GpuProfile, TimestampRecorder},
    scan::Scanner,
};

use super::pipeline::SortItemKind;

const U32_SIZE_BYTES: u64 = size_of::<u32>() as u64;
const UNIFORM_SIZE_BYTES: u64 = 16;
const DISPATCH_ARGS_SIZE_BYTES: u64 = 3 * size_of::<u32>() as u64;
const WORKSPACE_GROWTH_BYTES: u64 = 16 * 1024 * 1024;
const MAX_WORKGROUPS_X: u32 = 65_535;
const FULL_KEY_BITS: u32 = u32::BITS;

pub(super) struct CountedSorter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    scanner: Scanner,
    pipelines: CountedSortPipelines,
    item_size_bytes: u64,
    workspace: Option<CountedSortWorkspace>,
}

struct CountedSortWorkspace {
    capacity_bytes: u64,
    scratch: wgpu::Buffer,
    histogram: wgpu::Buffer,
    scanned_histogram: wgpu::Buffer,
    dispatch_args: wgpu::Buffer,
}

#[derive(Clone, Copy)]
struct CountedProblem {
    capacity_items: u32,
    capacity_blocks: u32,
    histogram_items: u32,
    histogram_bytes: u64,
}

#[derive(Clone, Copy)]
enum DispatchStrategy<'a> {
    PrepareIndirect,
    PreparedIndirect(&'a wgpu::Buffer),
    Capacity,
}

struct CountedSortPipelines {
    prepare_layout: wgpu::BindGroupLayout,
    prepare: wgpu::ComputePipeline,
    sort_layout: wgpu::BindGroupLayout,
    reduce: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    vt: u32,
    block_size: u32,
    item_size_bytes: u64,
}

impl CountedSorter {
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue, item_kind: SortItemKind) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
            scanner: Scanner::new(device, queue),
            pipelines: CountedSortPipelines::new(device, item_kind),
            item_size_bytes: item_kind.size_bytes(),
            workspace: None,
        }
    }

    pub(super) fn reserve(&mut self, capacity: u32) -> Result<(), Error> {
        if capacity == 0 {
            return Ok(());
        }
        let capacity_bytes =
            common::math::checked_byte_size(u64::from(capacity), self.item_size_bytes)?;
        let capacity_blocks = capacity.div_ceil(self.pipelines.items_per_block());
        let histogram_items = capacity_blocks.checked_mul(4).ok_or(Error::SizeOverflow)?;
        let histogram_bytes =
            common::math::checked_byte_size(u64::from(histogram_items), U32_SIZE_BYTES)?;
        self.ensure_workspace(capacity_bytes, histogram_bytes)?;
        self.scanner.reserve(histogram_items);
        Ok(())
    }

    pub(super) fn sort_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        let mut commands = CommandSession::new(&self.device, Some("Counted Radix Sort"));
        self.record_sort(commands.encoder(), input, output, count, capacity, key_bits)?;
        commands.submit(&self.queue);
        Ok(())
    }

    pub(super) fn record_sort(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        self.record_sort_ranges(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(output),
            BufferRange::whole(count),
            capacity,
            key_bits,
        )
    }

    pub(super) fn record_sort_ranges(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        count: BufferRange<'_>,
        capacity: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        validate_key_bits(key_bits)?;
        let Some(problem) = self.prepare(input, output, count, capacity)? else {
            return Ok(());
        };
        let pass_count = pass_count(key_bits);
        self.record_commands(
            encoder,
            input,
            output,
            count,
            problem,
            pass_count,
            DispatchStrategy::PrepareIndirect,
            None,
        )
    }

    pub(super) fn record_sort_with_plan(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        plan: &GpuCountPlan,
        key_bits: u32,
    ) -> Result<(), Error> {
        self.record_sort_ranges_with_plan(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(output),
            plan,
            key_bits,
        )
    }

    pub(super) fn record_sort_ranges_with_plan(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        plan: &GpuCountPlan,
        key_bits: u32,
    ) -> Result<(), Error> {
        validate_key_bits(key_bits)?;
        let count = plan.count();
        let Some(problem) = self.prepare(input, output, count, plan.capacity())? else {
            return Ok(());
        };
        debug_assert_eq!(
            plan.sort_items_per_block(),
            self.pipelines.items_per_block()
        );
        let dispatch = match plan.sort_dispatch() {
            CountedSortDispatch::Indirect => {
                DispatchStrategy::PreparedIndirect(plan.sort_dispatch_args())
            }
            CountedSortDispatch::Capacity => DispatchStrategy::Capacity,
        };
        self.record_commands(
            encoder,
            input,
            output,
            count,
            problem,
            pass_count(key_bits),
            dispatch,
            None,
        )
    }

    pub(super) async fn profile_sort(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
        key_bits: u32,
    ) -> Result<GpuProfile, Error> {
        validate_key_bits(key_bits)?;
        let input = BufferRange::whole(input);
        let output = BufferRange::whole(output);
        let count = BufferRange::whole(count);
        let Some(problem) = self.prepare(input, output, count, capacity)? else {
            return Ok(GpuProfile::empty());
        };
        let pass_count = pass_count(key_bits);
        let spans_per_pass = self
            .scanner
            .compute_pass_count(problem.histogram_items)
            .checked_add(2)
            .ok_or(Error::SizeOverflow)?;
        let span_count = pass_count
            .checked_mul(spans_per_pass)
            .and_then(|count| count.checked_add(1))
            .ok_or(Error::SizeOverflow)?;
        let mut profile = ProfileSession::new(
            &self.device,
            &self.queue,
            span_count,
            "Profiled Counted Radix Sort",
        )?;
        let (encoder, profiler) = profile.recording();
        self.record_commands(
            encoder,
            input,
            output,
            count,
            problem,
            pass_count,
            DispatchStrategy::PrepareIndirect,
            profiler,
        )?;
        profile.finish(&self.device, &self.queue).await
    }

    fn prepare(
        &mut self,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        count: BufferRange<'_>,
        capacity: u32,
    ) -> Result<Option<CountedProblem>, Error> {
        validate_distinct(input, output, count)?;
        count.validate(
            "sort item count",
            U32_SIZE_BYTES,
            wgpu::BufferUsages::STORAGE,
        )?;
        let capacity_bytes =
            common::math::checked_byte_size(u64::from(capacity), self.item_size_bytes)?;
        input.validate("sort input", capacity_bytes, wgpu::BufferUsages::STORAGE)?;
        output.validate("sort output", capacity_bytes, wgpu::BufferUsages::STORAGE)?;
        input.validate_storage_offset(&self.device, "sort input")?;
        output.validate_storage_offset(&self.device, "sort output")?;
        count.validate_storage_offset(&self.device, "sort item count")?;
        if capacity == 0 {
            return Ok(None);
        }

        let items_per_block = self.pipelines.items_per_block();
        let capacity_blocks = capacity.div_ceil(items_per_block);
        let histogram_items = capacity_blocks.checked_mul(4).ok_or(Error::SizeOverflow)?;
        let histogram_bytes =
            common::math::checked_byte_size(u64::from(histogram_items), U32_SIZE_BYTES)?;
        self.ensure_workspace(capacity_bytes, histogram_bytes)?;
        Ok(Some(CountedProblem {
            capacity_items: capacity,
            capacity_blocks,
            histogram_items,
            histogram_bytes,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn record_commands(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        count: BufferRange<'_>,
        problem: CountedProblem,
        pass_count: u32,
        dispatch: DispatchStrategy<'_>,
        mut profiler: Option<&mut TimestampRecorder>,
    ) -> Result<(), Error> {
        let workspace = self
            .workspace
            .as_ref()
            .expect("counted sort workspace is prepared");
        if matches!(dispatch, DispatchStrategy::PrepareIndirect) {
            self.pipelines.prepare_dispatch(
                &self.device,
                encoder,
                count,
                &workspace.dispatch_args,
                problem.capacity_items,
                profiler.as_deref_mut(),
            );
        }
        let dispatch_args = match dispatch {
            DispatchStrategy::PrepareIndirect => &workspace.dispatch_args,
            DispatchStrategy::PreparedIndirect(args) => args,
            DispatchStrategy::Capacity => &workspace.dispatch_args,
        };
        let capacity_dispatch = matches!(dispatch, DispatchStrategy::Capacity);
        let (uniform, uniform_stride) = create_uniform_buffer(
            &self.device,
            problem,
            pass_count,
            self.device.limits().min_uniform_buffer_offset_alignment,
        );
        let mut bind_groups = Vec::with_capacity(pass_count as usize * 2);
        let (capacity_groups_x, capacity_groups_y) = dispatch_dimensions(problem.capacity_blocks);

        for radix_pass in 0..pass_count {
            encoder.clear_buffer(&workspace.histogram, 0, Some(problem.histogram_bytes));
            let (source, destination) = pass_buffers(
                radix_pass,
                pass_count,
                input,
                output,
                BufferRange::whole(&workspace.scratch),
            );
            let uniform_offset = u64::from(radix_pass) * uniform_stride;
            let reduce_bind_group = self.pipelines.create_sort_bind_group(
                &self.device,
                source,
                &workspace.histogram,
                destination,
                &uniform,
                uniform_offset,
                count,
                problem,
                "Counted Sort Reduce Bind Group",
            );
            let scatter_bind_group = self.pipelines.create_sort_bind_group(
                &self.device,
                source,
                &workspace.scanned_histogram,
                destination,
                &uniform,
                uniform_offset,
                count,
                problem,
                "Counted Sort Scatter Bind Group",
            );
            profiling::record_compute_pass(
                encoder,
                "Counted Radix Histogram Reduce",
                profiler
                    .is_some()
                    .then(|| format!("counted.radix.{radix_pass:02}.reduce")),
                profiler.as_deref_mut(),
                |pass| {
                    pass.set_pipeline(&self.pipelines.reduce);
                    pass.set_bind_group(0, &reduce_bind_group, &[]);
                    if capacity_dispatch {
                        pass.dispatch_workgroups(capacity_groups_x, capacity_groups_y, 1);
                    } else {
                        pass.dispatch_workgroups_indirect(dispatch_args, 0);
                    }
                },
            );
            if let Some(profiler) = profiler.as_deref_mut() {
                self.scanner.record_profiled_scan(
                    encoder,
                    &workspace.histogram,
                    &workspace.scanned_histogram,
                    problem.histogram_items,
                    &format!("counted.radix.{radix_pass:02}.scan"),
                    profiler,
                )?;
            } else {
                self.scanner.record_scan(
                    encoder,
                    &workspace.histogram,
                    &workspace.scanned_histogram,
                    problem.histogram_items,
                )?;
            }
            profiling::record_compute_pass(
                encoder,
                "Counted Radix Stable Scatter",
                profiler
                    .is_some()
                    .then(|| format!("counted.radix.{radix_pass:02}.scatter")),
                profiler.as_deref_mut(),
                |pass| {
                    pass.set_pipeline(&self.pipelines.scatter);
                    pass.set_bind_group(0, &scatter_bind_group, &[]);
                    if capacity_dispatch {
                        pass.dispatch_workgroups(capacity_groups_x, capacity_groups_y, 1);
                    } else {
                        pass.dispatch_workgroups_indirect(dispatch_args, 0);
                    }
                },
            );
            bind_groups.push(reduce_bind_group);
            bind_groups.push(scatter_bind_group);
        }
        crate::common::runtime::defer_drop(encoder, (bind_groups, uniform));
        Ok(())
    }

    fn ensure_workspace(
        &mut self,
        requested_bytes: u64,
        requested_histogram_bytes: u64,
    ) -> Result<(), Error> {
        let needs_allocation = self.workspace.as_ref().is_none_or(|workspace| {
            workspace.capacity_bytes < requested_bytes
                || workspace.histogram.size() < requested_histogram_bytes
        });
        if !needs_allocation {
            return Ok(());
        }
        let capacity_bytes = workspace_capacity(requested_bytes)?;
        let histogram_bytes = workspace_capacity(requested_histogram_bytes)?;
        let limit = self
            .device
            .limits()
            .max_buffer_size
            .min(self.device.limits().max_storage_buffer_binding_size);
        for requested in [capacity_bytes, histogram_bytes] {
            if requested > limit {
                return Err(Error::BufferLimitExceeded { requested, limit });
            }
        }
        self.workspace = Some(CountedSortWorkspace {
            capacity_bytes,
            scratch: create_buffer(
                &self.device,
                "Counted Sort Scratch",
                capacity_bytes.max(self.item_size_bytes),
                wgpu::BufferUsages::STORAGE,
            ),
            histogram: create_buffer(
                &self.device,
                "Counted Sort Histogram",
                histogram_bytes,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            ),
            scanned_histogram: create_buffer(
                &self.device,
                "Counted Sort Scanned Histogram",
                histogram_bytes,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            dispatch_args: create_buffer(
                &self.device,
                "Counted Sort Dispatch Arguments",
                DISPATCH_ARGS_SIZE_BYTES,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            ),
        });
        Ok(())
    }
}

impl CountedSortPipelines {
    fn new(device: &wgpu::Device, item_kind: SortItemKind) -> Self {
        let prepare_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Counted Sort Dispatch Preparation Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, false, true),
            ],
        });
        let prepare = common::shader::create_compute_pipeline(
            device,
            &prepare_layout,
            include_str!("counted_prepare.wgsl"),
            "Counted Sort Dispatch Preparation Pipeline",
            "main",
            None,
        );
        let sort_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Counted Sort Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, false, false),
                common::buffers::bind_entry(3, false, true),
                common::buffers::bind_entry(4, true, false),
            ],
        });
        let limits = device.limits();
        let (vt, block_size) = workgroup_config(&limits);
        let source = include_str!("counted.wgsl")
            .replace("{{VT}}", &vt.to_string())
            .replace("{{BLOCK_SIZE}}", &block_size.to_string())
            .replace("{{MAX_WORKGROUPS_X}}", &MAX_WORKGROUPS_X.to_string())
            .replace("{{ITEM_TYPE}}", item_kind.shader_item_type())
            .replace("{{KEY_ACCESS}}", item_kind.shader_key_access());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Counted Sort Shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Counted Sort Pipeline Layout"),
            bind_group_layouts: &[Some(&sort_layout)],
            immediate_size: 0,
        });
        let reduce = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Counted Sort Reduce Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("reduce"),
            compilation_options: Default::default(),
            cache: None,
        });
        let scatter = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Counted Sort Scatter Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("scatter"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            prepare_layout,
            prepare,
            sort_layout,
            reduce,
            scatter,
            vt,
            block_size,
            item_size_bytes: item_kind.size_bytes(),
        }
    }

    const fn items_per_block(&self) -> u32 {
        self.vt * self.block_size
    }

    fn prepare_dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        count: BufferRange<'_>,
        dispatch_args: &wgpu::Buffer,
        capacity_items: u32,
        profiler: Option<&mut TimestampRecorder>,
    ) {
        let config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Counted Sort Dispatch Configuration"),
            contents: bytemuck::cast_slice(&[
                capacity_items,
                self.items_per_block(),
                MAX_WORKGROUPS_X,
                0,
            ]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Counted Sort Dispatch Preparation Bind Group"),
            layout: &self.prepare_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: count.binding(U32_SIZE_BYTES),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: exact_binding(dispatch_args, 0, DISPATCH_ARGS_SIZE_BYTES),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: config.as_entire_binding(),
                },
            ],
        });
        profiling::record_compute_pass(
            encoder,
            "Prepare Counted Sort Dispatch",
            profiler
                .is_some()
                .then(|| "counted.radix.prepare".to_owned()),
            profiler,
            |pass| {
                pass.set_pipeline(&self.prepare);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            },
        );
        crate::common::runtime::defer_drop(encoder, (bind_group, config));
    }

    #[allow(clippy::too_many_arguments)]
    fn create_sort_bind_group(
        &self,
        device: &wgpu::Device,
        input: BufferRange<'_>,
        histogram: &wgpu::Buffer,
        output: BufferRange<'_>,
        uniform: &wgpu::Buffer,
        uniform_offset: u64,
        count: BufferRange<'_>,
        problem: CountedProblem,
        label: &'static str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.sort_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input
                        .binding(u64::from(problem.capacity_items) * self.item_size_bytes),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: exact_binding(histogram, 0, problem.histogram_bytes),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output
                        .binding(u64::from(problem.capacity_items) * self.item_size_bytes),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: uniform,
                        offset: uniform_offset,
                        size: wgpu::BufferSize::new(UNIFORM_SIZE_BYTES),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: count.binding(U32_SIZE_BYTES),
                },
            ],
        })
    }
}

pub(crate) fn items_per_block(limits: &wgpu::Limits) -> u32 {
    let (vt, block_size) = workgroup_config(limits);
    vt * block_size
}

fn workgroup_config(limits: &wgpu::Limits) -> (u32, u32) {
    if limits.max_compute_workgroup_storage_size >= 32_768 {
        (8, 256)
    } else {
        (4, 128)
    }
}

fn create_uniform_buffer(
    device: &wgpu::Device,
    problem: CountedProblem,
    pass_count: u32,
    min_alignment: u32,
) -> (wgpu::Buffer, u64) {
    let stride = u64::from(min_alignment).max(UNIFORM_SIZE_BYTES);
    let words_per_uniform = (stride / U32_SIZE_BYTES) as usize;
    let mut data = vec![0_u32; words_per_uniform * pass_count as usize];
    for radix_pass in 0..pass_count as usize {
        let offset = radix_pass * words_per_uniform;
        data[offset..offset + 4].copy_from_slice(&[
            radix_pass as u32 * 2,
            problem.capacity_items,
            problem.capacity_blocks,
            0,
        ]);
    }
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Counted Sort Uniforms"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    (buffer, stride)
}

fn validate_distinct(
    input: BufferRange<'_>,
    output: BufferRange<'_>,
    count: BufferRange<'_>,
) -> Result<(), Error> {
    for (first, first_name, second, second_name) in [
        (input, "sort input", output, "sort output"),
        (input, "sort input", count, "sort item count"),
        (output, "sort output", count, "sort item count"),
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

fn create_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn exact_binding(buffer: &wgpu::Buffer, offset: u64, size: u64) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer,
        offset,
        size: wgpu::BufferSize::new(size),
    })
}

fn workspace_capacity(requested: u64) -> Result<u64, Error> {
    if requested < WORKSPACE_GROWTH_BYTES {
        requested
            .max(U32_SIZE_BYTES)
            .checked_next_power_of_two()
            .ok_or(Error::SizeOverflow)
    } else {
        common::math::checked_align_to(requested, WORKSPACE_GROWTH_BYTES)
    }
}

fn pass_count(key_bits: u32) -> u32 {
    key_bits.max(1).div_ceil(2)
}

fn dispatch_dimensions(workgroups: u32) -> (u32, u32) {
    (
        workgroups.min(MAX_WORKGROUPS_X),
        workgroups.div_ceil(MAX_WORKGROUPS_X),
    )
}

fn validate_key_bits(key_bits: u32) -> Result<(), Error> {
    if key_bits <= FULL_KEY_BITS {
        Ok(())
    } else {
        Err(Error::InvalidKeyBits { bits: key_bits })
    }
}

fn pass_buffers<'a>(
    radix_pass: u32,
    pass_count: u32,
    input: BufferRange<'a>,
    output: BufferRange<'a>,
    scratch: BufferRange<'a>,
) -> (BufferRange<'a>, BufferRange<'a>) {
    let source = if radix_pass == 0 {
        input
    } else if (pass_count - radix_pass).is_multiple_of(2) {
        output
    } else {
        scratch
    };
    let passes_after = pass_count - radix_pass - 1;
    let destination = if passes_after.is_multiple_of(2) {
        output
    } else {
        scratch
    };
    (source, destination)
}

#[cfg(test)]
mod tests {
    use super::{pass_count, workspace_capacity};

    #[test]
    fn counted_passes_preserve_output_parity() {
        assert_eq!(pass_count(0), 1);
        assert_eq!(pass_count(1), 1);
        assert_eq!(pass_count(2), 1);
        assert_eq!(pass_count(3), 2);
        assert_eq!(pass_count(32), 16);
    }

    #[test]
    fn counted_workspace_never_allocates_zero_bytes() {
        assert_eq!(workspace_capacity(0).unwrap(), 4);
        assert_eq!(workspace_capacity(5).unwrap(), 8);
    }
}
