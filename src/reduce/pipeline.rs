use wgpu::util::DeviceExt;

use crate::{common, common::buffers::BufferRange, profiling};

use super::U32Reduction;

const BLOCK_SIZE: u32 = 128;
const ITEMS_PER_THREAD: u32 = 32;
const MAX_WORKGROUPS_X: u32 = 65_535;
const VALUE_SIZE_BYTES: u64 = size_of::<u32>() as u64;

pub(crate) struct ReductionPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    sum_pipeline: wgpu::ComputePipeline,
    min_pipeline: wgpu::ComputePipeline,
    max_pipeline: wgpu::ComputePipeline,
    min_identity: wgpu::Buffer,
}

pub(crate) struct ReductionDispatch<'a> {
    pub(crate) input: BufferRange<'a>,
    pub(crate) output: BufferRange<'a>,
    pub(crate) input_items: u32,
    pub(crate) output_items: u32,
    pub(crate) operation: U32Reduction,
    pub(crate) level: u32,
}

impl ReductionPipeline {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Reduction Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
            ],
        });
        let sum_pipeline =
            create_pipeline(device, &bind_group_layout, U32Reduction::Sum, "lhs + rhs");
        let min_pipeline = create_pipeline(
            device,
            &bind_group_layout,
            U32Reduction::Min,
            "min(lhs, rhs)",
        );
        let max_pipeline = create_pipeline(
            device,
            &bind_group_layout,
            U32Reduction::Max,
            "max(lhs, rhs)",
        );
        let min_identity = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Minimum Reduction Identity"),
            contents: bytemuck::bytes_of(&u32::MAX),
            usage: wgpu::BufferUsages::COPY_SRC,
        });

        Self {
            bind_group_layout,
            sum_pipeline,
            min_pipeline,
            max_pipeline,
            min_identity,
        }
    }

    pub(crate) const fn output_items(&self, input_items: u32) -> u32 {
        reduction_output_items(input_items)
    }

    pub(crate) fn pass_count(&self, input_items: u32) -> u32 {
        reduction_pass_count(input_items)
    }

    pub(crate) fn record_identity(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output: BufferRange<'_>,
        operation: U32Reduction,
    ) {
        match operation {
            U32Reduction::Min => encoder.copy_buffer_to_buffer(
                &self.min_identity,
                0,
                output.buffer,
                output.offset,
                VALUE_SIZE_BYTES,
            ),
            U32Reduction::Sum | U32Reduction::Max => {
                encoder.clear_buffer(output.buffer, output.offset, Some(VALUE_SIZE_BYTES));
            }
        }
    }

    pub(crate) fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dispatch: ReductionDispatch<'_>,
        profiler: Option<&mut profiling::TimestampRecorder>,
    ) {
        let input_size = wgpu::BufferSize::new(u64::from(dispatch.input_items) * VALUE_SIZE_BYTES)
            .expect("reduction dispatch input is non-empty");
        let output_size =
            wgpu::BufferSize::new(u64::from(dispatch.output_items) * VALUE_SIZE_BYTES)
                .expect("reduction dispatch output is non-empty");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reduction Dispatch Bind Group"),
            layout: &self.bind_group_layout,
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
            ],
        });
        let pipeline = match dispatch.operation {
            U32Reduction::Sum => &self.sum_pipeline,
            U32Reduction::Min => &self.min_pipeline,
            U32Reduction::Max => &self.max_pipeline,
        };
        let (groups_x, groups_y) = dispatch_dimensions(dispatch.output_items);
        let profile_label = profiler.is_some().then(|| {
            format!(
                "reduction.{}.level.{}",
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
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            },
        );
        crate::common::runtime::defer_drop(encoder, bind_group);
    }
}

const fn reduction_output_items(input_items: u32) -> u32 {
    input_items.div_ceil(BLOCK_SIZE * ITEMS_PER_THREAD)
}

fn reduction_pass_count(mut input_items: u32) -> u32 {
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

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    operation: U32Reduction,
    combine_expression: &str,
) -> wgpu::ComputePipeline {
    let source = include_str!("reduce.wgsl")
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
        &format!("U32 {} Reduction Pipeline", operation.name()),
        "main",
        Some(&config),
    )
}

fn dispatch_dimensions(output_items: u32) -> (u32, u32) {
    (
        output_items.min(MAX_WORKGROUPS_X),
        output_items.div_ceil(MAX_WORKGROUPS_X),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        BLOCK_SIZE, ITEMS_PER_THREAD, MAX_WORKGROUPS_X, dispatch_dimensions, reduction_pass_count,
    };

    #[test]
    fn pass_count_matches_the_reduction_hierarchy() {
        let items_per_workgroup = BLOCK_SIZE * ITEMS_PER_THREAD;
        assert_eq!(reduction_pass_count(0), 0);
        assert_eq!(reduction_pass_count(1), 1);
        assert_eq!(reduction_pass_count(items_per_workgroup), 1);
        assert_eq!(reduction_pass_count(items_per_workgroup + 1), 2);
        assert_eq!(
            reduction_pass_count(items_per_workgroup * items_per_workgroup + 1),
            3
        );
    }

    #[test]
    fn dispatch_dimensions_cover_the_two_dimensional_tail() {
        assert_eq!(dispatch_dimensions(1), (1, 1));
        assert_eq!(dispatch_dimensions(MAX_WORKGROUPS_X), (MAX_WORKGROUPS_X, 1));
        assert_eq!(
            dispatch_dimensions(MAX_WORKGROUPS_X + 1),
            (MAX_WORKGROUPS_X, 2)
        );
    }
}
