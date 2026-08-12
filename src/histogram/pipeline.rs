use crate::{common, profiling};

pub(crate) const MAX_BINS: u32 = 256;
const BLOCK_SIZE: u32 = 256;
const ITEMS_PER_THREAD: u32 = 8;
const PARAMS_SIZE_BYTES: u64 = 16;
const VALUE_SIZE_BYTES: u64 = size_of::<u32>() as u64;

pub(crate) struct HistogramPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    max_workgroups_per_dimension: u32,
}

pub(crate) struct HistogramDispatch<'a> {
    pub(crate) input: &'a wgpu::Buffer,
    pub(crate) output: &'a wgpu::Buffer,
    pub(crate) num_items: u32,
    pub(crate) bin_count: u32,
}

impl HistogramPipeline {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Histogram Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, false, true),
            ],
        });
        let config = common::shader::ShaderConfig {
            vt: ITEMS_PER_THREAD,
            block_size: BLOCK_SIZE,
        };
        let source = include_str!("histogram.wgsl").replace("{{MAX_BINS}}", &MAX_BINS.to_string());
        let pipeline = common::shader::create_compute_pipeline(
            device,
            &bind_group_layout,
            &source,
            "U32 Histogram Pipeline",
            "main",
            Some(&config),
        );

        Self {
            bind_group_layout,
            pipeline,
            max_workgroups_per_dimension: device.limits().max_compute_workgroups_per_dimension,
        }
    }

    pub(crate) fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dispatch: HistogramDispatch<'_>,
        profiler: Option<&mut profiling::TimestampRecorder>,
    ) {
        let items_per_workgroup = BLOCK_SIZE * ITEMS_PER_THREAD;
        let workgroups = common::math::calc_groups(dispatch.num_items, items_per_workgroup);
        let groups_x = workgroups.min(self.max_workgroups_per_dimension);
        let groups_y = workgroups.div_ceil(self.max_workgroups_per_dimension);
        let params_data = [dispatch.num_items, dispatch.bin_count, groups_x, workgroups];
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Histogram Parameters"),
            size: PARAMS_SIZE_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&params_data));

        let input_size = wgpu::BufferSize::new(u64::from(dispatch.num_items) * VALUE_SIZE_BYTES)
            .expect("histogram dispatch input is non-empty");
        let output_size = wgpu::BufferSize::new(u64::from(dispatch.bin_count) * VALUE_SIZE_BYTES)
            .expect("histogram dispatch output is non-empty");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Histogram Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: dispatch.input,
                        offset: 0,
                        size: Some(input_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: dispatch.output,
                        offset: 0,
                        size: Some(output_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
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
            "Histogram Count",
            profiler.is_some().then(|| "histogram.count".to_owned()),
            profiler,
            |pass| {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            },
        );
        crate::common::runtime::defer_drop(encoder, (bind_group, params));
    }
}
