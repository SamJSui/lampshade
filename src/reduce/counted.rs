use wgpu::util::DeviceExt;

use crate::{
    Error, GpuCountPlan, common,
    common::{
        buffers::BufferRange, runtime::CommandSession, runtime::ProfileSession,
        workspace::ReusableBuffer,
    },
    profiling::{self, GpuProfile, TimestampRecorder},
};

use super::U32Reduction;

const BLOCK_SIZE: u32 = 128;
const ITEMS_PER_THREAD: u32 = 32;
pub(crate) const ITEMS_PER_BLOCK: u32 = BLOCK_SIZE * ITEMS_PER_THREAD;
const MAX_WORKGROUPS_X: u32 = 65_535;
const VALUE_SIZE_BYTES: u64 = size_of::<u32>() as u64;
const PLAN_WORDS: u64 = 2;
const DISPATCH_WORDS: u64 = 3;
const CONFIG_SIZE_BYTES: u64 = 8 * VALUE_SIZE_BYTES;

pub(super) struct CountedReducer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: CountedReductionPipelines,
    scratch_a: ReusableBuffer,
    scratch_b: ReusableBuffer,
    plans: ReusableBuffer,
    dispatch_args: ReusableBuffer,
}

#[derive(Clone, Copy)]
struct CountedProblem {
    capacity_items: u32,
    pass_count: u32,
    plan_stride: u64,
}

struct CountedReductionPipelines {
    prepare_layout: wgpu::BindGroupLayout,
    prepare: wgpu::ComputePipeline,
    reduce_layout: wgpu::BindGroupLayout,
    sum: wgpu::ComputePipeline,
    min: wgpu::ComputePipeline,
    max: wgpu::ComputePipeline,
    identities: wgpu::Buffer,
}

struct CountedDispatch<'a> {
    input: BufferRange<'a>,
    output: BufferRange<'a>,
    input_capacity: u32,
    output_capacity: u32,
    plan_offset: u64,
    args_offset: u64,
    operation: U32Reduction,
    level: u32,
}

impl CountedReducer {
    pub(super) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipelines: CountedReductionPipelines::new(device),
            scratch_a: ReusableBuffer::default(),
            scratch_b: ReusableBuffer::default(),
            plans: ReusableBuffer::default(),
            dispatch_args: ReusableBuffer::default(),
        }
    }

    pub(super) fn reserve(&mut self, capacity: u32) -> Result<(), Error> {
        let pass_count = reduction_pass_count(capacity);
        if pass_count == 0 {
            return Ok(());
        }
        let alignment = u64::from(self.device.limits().min_storage_buffer_offset_alignment);
        let plan_stride = common::math::checked_align_to(
            PLAN_WORDS * VALUE_SIZE_BYTES,
            alignment.max(VALUE_SIZE_BYTES),
        )?;
        self.prepare_workspace(capacity, pass_count, plan_stride)
    }

    pub(super) fn reduce_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
        operation: U32Reduction,
    ) -> Result<(), Error> {
        let mut commands = CommandSession::new(&self.device, Some("Counted Reduction"));
        self.record_reduce(
            commands.encoder(),
            input,
            output,
            count,
            capacity,
            operation,
        )?;
        commands.submit(&self.queue);
        Ok(())
    }

    pub(super) fn record_reduce(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
        operation: U32Reduction,
    ) -> Result<(), Error> {
        self.record_reduce_ranges(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(output),
            BufferRange::whole(count),
            capacity,
            operation,
        )
    }

    pub(super) fn record_reduce_ranges(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        count: BufferRange<'_>,
        capacity: u32,
        operation: U32Reduction,
    ) -> Result<(), Error> {
        let problem = self.prepare(input, output, Some(count), capacity)?;
        self.record_commands(
            encoder,
            input,
            output,
            Some(count),
            problem,
            operation,
            None,
            None,
        )
    }

    pub(super) fn record_reduce_with_plan(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        plan: &GpuCountPlan,
        operation: U32Reduction,
    ) -> Result<(), Error> {
        self.record_reduce_ranges_with_plan(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(output),
            plan,
            operation,
        )
    }

    pub(super) fn record_reduce_ranges_with_plan(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        plan: &GpuCountPlan,
        operation: U32Reduction,
    ) -> Result<(), Error> {
        validate_distinct(input, output, plan.count())?;
        let problem = self.prepare(input, output, None, plan.capacity())?;
        debug_assert_eq!(plan.reduction_pass_count(), problem.pass_count);
        debug_assert_eq!(plan.plan_stride(), problem.plan_stride);
        self.record_commands(
            encoder,
            input,
            output,
            None,
            problem,
            operation,
            Some(plan),
            None,
        )
    }

    pub(super) async fn profile_reduce(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
        operation: U32Reduction,
    ) -> Result<GpuProfile, Error> {
        let input = BufferRange::whole(input);
        let output = BufferRange::whole(output);
        let count = BufferRange::whole(count);
        let problem = self.prepare(input, output, Some(count), capacity)?;
        let span_count = problem
            .pass_count
            .checked_add(u32::from(problem.pass_count > 0))
            .ok_or(Error::SizeOverflow)?;
        let mut profile = ProfileSession::new(
            &self.device,
            &self.queue,
            span_count,
            "Profiled Counted Reduction",
        )?;
        let (encoder, profiler) = profile.recording();
        self.record_commands(
            encoder,
            input,
            output,
            Some(count),
            problem,
            operation,
            None,
            profiler,
        )?;
        profile.finish(&self.device, &self.queue).await
    }

    fn prepare(
        &mut self,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        count: Option<BufferRange<'_>>,
        capacity: u32,
    ) -> Result<CountedProblem, Error> {
        if input.buffer == output.buffer {
            return Err(Error::BufferAlias {
                first: "reduction input",
                second: "reduction output",
            });
        }
        output.validate(
            "reduction output",
            VALUE_SIZE_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        output.validate_storage_offset(&self.device, "reduction output")?;
        if let Some(count) = count {
            validate_distinct(input, output, count)?;
            count.validate(
                "reduction item count",
                VALUE_SIZE_BYTES,
                wgpu::BufferUsages::STORAGE,
            )?;
            count.validate_storage_offset(&self.device, "reduction item count")?;
        }
        let input_bytes = common::math::checked_byte_size(u64::from(capacity), VALUE_SIZE_BYTES)?;
        self.validate_storage_binding_size(input_bytes)?;
        input.validate("reduction input", input_bytes, wgpu::BufferUsages::STORAGE)?;
        input.validate_storage_offset(&self.device, "reduction input")?;

        let pass_count = reduction_pass_count(capacity);
        let alignment = u64::from(self.device.limits().min_storage_buffer_offset_alignment);
        let plan_stride = common::math::checked_align_to(
            PLAN_WORDS * VALUE_SIZE_BYTES,
            alignment.max(VALUE_SIZE_BYTES),
        )?;
        if pass_count > 0 {
            self.prepare_workspace(capacity, pass_count, plan_stride)?;
        }
        Ok(CountedProblem {
            capacity_items: capacity,
            pass_count,
            plan_stride,
        })
    }

    fn prepare_workspace(
        &mut self,
        capacity: u32,
        pass_count: u32,
        plan_stride: u64,
    ) -> Result<(), Error> {
        let first_items = reduction_output_items(capacity);
        if pass_count > 1 {
            let bytes = self.checked_scratch_size(first_items)?;
            self.scratch_a.ensure(
                &self.device,
                bytes,
                "Counted Reduction Scratch A",
                wgpu::BufferUsages::STORAGE,
            );
        }
        if pass_count > 2 {
            let second_items = reduction_output_items(first_items);
            let bytes = self.checked_scratch_size(second_items)?;
            self.scratch_b.ensure(
                &self.device,
                bytes,
                "Counted Reduction Scratch B",
                wgpu::BufferUsages::STORAGE,
            );
        }
        let plan_bytes = u64::from(pass_count)
            .checked_mul(plan_stride)
            .ok_or(Error::SizeOverflow)?;
        let args_bytes = common::math::checked_byte_size(
            u64::from(pass_count) * DISPATCH_WORDS,
            VALUE_SIZE_BYTES,
        )?;
        self.validate_storage_binding_size(plan_bytes)?;
        self.validate_storage_binding_size(args_bytes)?;
        self.plans.ensure(
            &self.device,
            plan_bytes,
            "Counted Reduction Plans",
            wgpu::BufferUsages::STORAGE,
        );
        self.dispatch_args.ensure(
            &self.device,
            args_bytes,
            "Counted Reduction Dispatch Arguments",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        );
        Ok(())
    }

    fn checked_scratch_size(&self, items: u32) -> Result<u64, Error> {
        let bytes = common::math::checked_byte_size(u64::from(items), VALUE_SIZE_BYTES)?;
        self.validate_storage_binding_size(bytes)?;
        Ok(bytes)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_commands(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        count: Option<BufferRange<'_>>,
        problem: CountedProblem,
        operation: U32Reduction,
        prepared_plan: Option<&GpuCountPlan>,
        mut profiler: Option<&mut TimestampRecorder>,
    ) -> Result<(), Error> {
        self.pipelines.record_identity(encoder, output, operation);
        if problem.pass_count == 0 {
            return Ok(());
        }

        let (plans, dispatch_args) = if let Some(plan) = prepared_plan {
            (plan.reduction_plans(), plan.reduction_dispatch_args())
        } else {
            (
                self.plans.get().expect("counted reduction plans exist"),
                self.dispatch_args
                    .get()
                    .expect("counted reduction dispatch arguments exist"),
            )
        };
        if let Some(count) = count {
            self.pipelines.prepare_dispatches(
                &self.device,
                encoder,
                count,
                plans,
                dispatch_args,
                problem,
                profiler.as_deref_mut(),
            );
        } else {
            debug_assert!(prepared_plan.is_some());
        }

        let scratch_a = self.scratch_a.get();
        let scratch_b = self.scratch_b.get();
        let mut current_input = input;
        let mut current_capacity = problem.capacity_items;
        for level in 0..problem.pass_count {
            let output_capacity = reduction_output_items(current_capacity);
            let current_output = if level + 1 == problem.pass_count {
                output
            } else if level.is_multiple_of(2) {
                BufferRange::whole(scratch_a.expect("counted reduction scratch A exists"))
            } else {
                BufferRange::whole(scratch_b.expect("counted reduction scratch B exists"))
            };
            self.pipelines.dispatch(
                &self.device,
                encoder,
                plans,
                dispatch_args,
                CountedDispatch {
                    input: current_input,
                    output: current_output,
                    input_capacity: current_capacity,
                    output_capacity,
                    plan_offset: u64::from(level) * problem.plan_stride,
                    args_offset: u64::from(level) * DISPATCH_WORDS * VALUE_SIZE_BYTES,
                    operation,
                    level,
                },
                profiler.as_deref_mut(),
            );
            current_input = current_output;
            current_capacity = output_capacity;
        }
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

impl CountedReductionPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let prepare_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Counted Reduction Preparation Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, false, false),
                common::buffers::bind_entry(3, false, true),
            ],
        });
        let prepare = common::shader::create_compute_pipeline(
            device,
            &prepare_layout,
            include_str!("counted_prepare.wgsl"),
            "Counted Reduction Preparation Pipeline",
            "main",
            None,
        );
        let reduce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Counted Reduction Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, true, false),
            ],
        });
        let sum = create_pipeline(device, &reduce_layout, U32Reduction::Sum, "lhs + rhs");
        let min = create_pipeline(device, &reduce_layout, U32Reduction::Min, "min(lhs, rhs)");
        let max = create_pipeline(device, &reduce_layout, U32Reduction::Max, "max(lhs, rhs)");
        let identities = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Counted Reduction Identities"),
            contents: bytemuck::cast_slice(&[0_u32, u32::MAX, 0]),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        Self {
            prepare_layout,
            prepare,
            reduce_layout,
            sum,
            min,
            max,
            identities,
        }
    }

    fn record_identity(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output: BufferRange<'_>,
        operation: U32Reduction,
    ) {
        encoder.copy_buffer_to_buffer(
            &self.identities,
            operation.identity_offset(),
            output.buffer,
            output.offset,
            VALUE_SIZE_BYTES,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_dispatches(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        count: BufferRange<'_>,
        plans: &wgpu::Buffer,
        dispatch_args: &wgpu::Buffer,
        problem: CountedProblem,
        profiler: Option<&mut TimestampRecorder>,
    ) {
        let config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Counted Reduction Configuration"),
            contents: bytemuck::cast_slice(&[
                problem.capacity_items,
                problem.pass_count,
                ITEMS_PER_BLOCK,
                (problem.plan_stride / VALUE_SIZE_BYTES) as u32,
                MAX_WORKGROUPS_X,
                0,
                0,
                0,
            ]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        debug_assert_eq!(config.size(), CONFIG_SIZE_BYTES);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Counted Reduction Preparation Bind Group"),
            layout: &self.prepare_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: count.binding(VALUE_SIZE_BYTES),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: plans.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dispatch_args.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: config.as_entire_binding(),
                },
            ],
        });
        profiling::record_compute_pass(
            encoder,
            "Prepare Counted Reduction Dispatches",
            profiler
                .is_some()
                .then(|| "counted.reduction.prepare".to_owned()),
            profiler,
            |pass| {
                pass.set_pipeline(&self.prepare);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        plans: &wgpu::Buffer,
        dispatch_args: &wgpu::Buffer,
        dispatch: CountedDispatch<'_>,
        profiler: Option<&mut TimestampRecorder>,
    ) {
        let input_size =
            wgpu::BufferSize::new(u64::from(dispatch.input_capacity) * VALUE_SIZE_BYTES)
                .expect("counted reduction input capacity is non-zero");
        let output_size =
            wgpu::BufferSize::new(u64::from(dispatch.output_capacity) * VALUE_SIZE_BYTES)
                .expect("counted reduction output capacity is non-zero");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Counted Reduction Dispatch Bind Group"),
            layout: &self.reduce_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: dispatch.input.buffer,
                        offset: dispatch.input.offset,
                        size: Some(input_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: dispatch.output.buffer,
                        offset: dispatch.output.offset,
                        size: Some(output_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: plans,
                        offset: dispatch.plan_offset,
                        size: wgpu::BufferSize::new(PLAN_WORDS * VALUE_SIZE_BYTES),
                    }),
                },
            ],
        });
        let pipeline = match dispatch.operation {
            U32Reduction::Sum => &self.sum,
            U32Reduction::Min => &self.min,
            U32Reduction::Max => &self.max,
        };
        let profile_label = profiler.is_some().then(|| {
            format!(
                "counted.reduction.{}.level.{}",
                dispatch.operation.name(),
                dispatch.level
            )
        });
        profiling::record_compute_pass(
            encoder,
            dispatch.operation.pass_label(),
            profile_label,
            profiler,
            |pass| {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups_indirect(dispatch_args, dispatch.args_offset);
            },
        );
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    operation: U32Reduction,
    combine_expression: &str,
) -> wgpu::ComputePipeline {
    let source = include_str!("counted.wgsl")
        .replace("{{IDENTITY}}", &format!("{}u", operation.identity()))
        .replace("{{COMBINE}}", combine_expression)
        .replace("{{MAX_WORKGROUPS_X}}", &MAX_WORKGROUPS_X.to_string());
    let config = common::shader::ShaderConfig {
        vt: ITEMS_PER_THREAD,
        block_size: BLOCK_SIZE,
    };
    common::shader::create_compute_pipeline(
        device,
        layout,
        &source,
        &format!("Counted U32 {} Reduction Pipeline", operation.name()),
        "main",
        Some(&config),
    )
}

fn validate_distinct(
    input: BufferRange<'_>,
    output: BufferRange<'_>,
    count: BufferRange<'_>,
) -> Result<(), Error> {
    for (first, first_name, second, second_name) in [
        (input, "reduction input", output, "reduction output"),
        (input, "reduction input", count, "reduction item count"),
        (output, "reduction output", count, "reduction item count"),
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

const fn reduction_output_items(input_items: u32) -> u32 {
    input_items.div_ceil(ITEMS_PER_BLOCK)
}

pub(crate) fn reduction_pass_count(mut input_items: u32) -> u32 {
    if input_items == 0 {
        return 0;
    }
    let mut passes = 0;
    loop {
        passes += 1;
        input_items = reduction_output_items(input_items);
        if input_items == 1 {
            return passes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ITEMS_PER_BLOCK, reduction_output_items, reduction_pass_count};

    #[test]
    fn counted_hierarchy_is_capacity_bounded() {
        assert_eq!(reduction_pass_count(0), 0);
        assert_eq!(reduction_pass_count(1), 1);
        assert_eq!(reduction_pass_count(ITEMS_PER_BLOCK), 1);
        assert_eq!(reduction_pass_count(ITEMS_PER_BLOCK + 1), 2);
        assert_eq!(reduction_output_items(ITEMS_PER_BLOCK.saturating_mul(2)), 2);
    }
}
