use wgpu::util::DeviceExt;

use crate::{common, common::buffers::BufferRange, profiling};

const BLOCK_SIZE: u32 = 256;
const PARAMS_SIZE_BYTES: u64 = 16;

#[derive(Clone, Copy)]
pub(crate) enum CompactItemKind {
    Value,
    KeyValue,
}

impl CompactItemKind {
    pub(crate) const fn size_bytes(self) -> u64 {
        match self {
            Self::Value => size_of::<u32>() as u64,
            Self::KeyValue => size_of::<crate::KeyValue>() as u64,
        }
    }

    const fn shader_item_type(self) -> &'static str {
        match self {
            Self::Value => "u32",
            Self::KeyValue => "KeyValue",
        }
    }
}

pub struct CompactPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    max_workgroups_per_dimension: u32,
}

pub struct CompactDispatch<'a> {
    pub input: BufferRange<'a>,
    pub mask: BufferRange<'a>,
    pub offsets: BufferRange<'a>,
    pub block_prefixes: BufferRange<'a>,
    pub output: BufferRange<'a>,
    pub output_count: BufferRange<'a>,
    pub num_items: u32,
    pub scan_items_per_block: u32,
}

impl CompactPipeline {
    pub fn new(device: &wgpu::Device, item_kind: CompactItemKind) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Stream Compaction Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, true, false),
                common::buffers::bind_entry(2, true, false),
                common::buffers::bind_entry(3, false, false),
                common::buffers::bind_entry(4, false, false),
                common::buffers::bind_entry(5, true, false),
                common::buffers::bind_entry(6, false, true),
            ],
        });
        let shader_source =
            include_str!("compact.wgsl").replace("{{ITEM_TYPE}}", item_kind.shader_item_type());
        let pipeline = common::shader::create_compute_pipeline(
            device,
            &bind_group_layout,
            &shader_source,
            "Stream Compaction Pipeline",
            "main",
            None,
        );

        Self {
            bind_group_layout,
            pipeline,
            max_workgroups_per_dimension: device.limits().max_compute_workgroups_per_dimension,
        }
    }

    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dispatch: CompactDispatch<'_>,
        profiler: Option<&mut profiling::TimestampRecorder>,
    ) {
        let workgroups = common::math::calc_groups(dispatch.num_items, BLOCK_SIZE);
        let groups_x = workgroups.min(self.max_workgroups_per_dimension);
        let groups_y = workgroups.div_ceil(self.max_workgroups_per_dimension);
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Stream Compaction Parameters"),
            contents: bytemuck::cast_slice(&[
                dispatch.num_items,
                groups_x,
                dispatch.scan_items_per_block,
                0,
            ]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Stream Compaction Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dispatch.input.binding(dispatch.input.size),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dispatch.mask.binding(dispatch.mask.size),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dispatch.offsets.binding(dispatch.offsets.size),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dispatch.output.binding(dispatch.output.size),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dispatch.output_count.binding(dispatch.output_count.size),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dispatch
                        .block_prefixes
                        .binding(dispatch.block_prefixes.size),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: wgpu::BufferSize::new(PARAMS_SIZE_BYTES),
                    }),
                },
            ],
        });

        profiling::record_compute_pass(
            encoder,
            "Stream Compaction Scatter",
            profiler.is_some().then(|| "compact.scatter".to_owned()),
            profiler,
            |pass| {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            },
        );
    }
}
