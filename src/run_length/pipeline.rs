use wgpu::util::DeviceExt;

use crate::{common, common::buffers::BufferRange, profiling};

const BLOCK_SIZE: u32 = 256;

pub(crate) struct RunLengthPipeline {
    layout: wgpu::BindGroupLayout,
    mark_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
    dummy_count: wgpu::Buffer,
    max_workgroups_per_dimension: u32,
}

impl RunLengthPipeline {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Run-Length Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, true, false),
                common::buffers::bind_entry(3, false, false),
                common::buffers::bind_entry(4, false, false),
                common::buffers::bind_entry(5, true, false),
                common::buffers::bind_entry(6, false, false),
            ],
        });
        let shader = include_str!("run_length.wgsl");
        let mark_pipeline = common::shader::create_compute_pipeline(
            device,
            &layout,
            shader,
            "Run-Length Head Mark Pipeline",
            "mark_heads",
            None,
        );
        let scatter_pipeline = common::shader::create_compute_pipeline(
            device,
            &layout,
            include_str!("scatter_starts.wgsl"),
            "Run-Length Scatter Pipeline",
            "scatter_starts",
            None,
        );
        let finalize_pipeline = common::shader::create_compute_pipeline(
            device,
            &layout,
            include_str!("run_length_finalize.wgsl"),
            "Run-Length Finalize Pipeline",
            "finalize_lengths",
            None,
        );
        let dummy_count = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Run-Length Fixed-Extent Dummy Count"),
            contents: bytemuck::bytes_of(&u32::MAX),
            usage: wgpu::BufferUsages::STORAGE,
        });
        Self {
            layout,
            mark_pipeline,
            scatter_pipeline,
            finalize_pipeline,
            dummy_count,
            max_workgroups_per_dimension: device.limits().max_compute_workgroups_per_dimension,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_bind_group(
        &self,
        device: &wgpu::Device,
        input: BufferRange<'_>,
        heads: BufferRange<'_>,
        offsets: BufferRange<'_>,
        unique_values: BufferRange<'_>,
        run_lengths: BufferRange<'_>,
        input_count: Option<BufferRange<'_>>,
        run_count: BufferRange<'_>,
    ) -> wgpu::BindGroup {
        let input_count = input_count.unwrap_or_else(|| BufferRange::whole(&self.dummy_count));
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Run-Length Bind Group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.binding(input.size),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: heads.binding(heads.size),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: offsets.binding(offsets.size),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: unique_values.binding(unique_values.size),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: run_lengths.binding(run_lengths.size),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: input_count.binding(size_of::<u32>() as u64),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: run_count.binding(size_of::<u32>() as u64),
                },
            ],
        })
    }

    pub(crate) fn mark(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        capacity_items: u32,
        profiler: Option<&mut profiling::TimestampRecorder>,
    ) {
        let (groups_x, groups_y) = self.dispatch_dimensions(capacity_items);
        profiling::record_compute_pass(
            encoder,
            "Run-Length Head Mark",
            profiler.is_some().then(|| "run_length.mark".to_owned()),
            profiler,
            |pass| {
                pass.set_pipeline(&self.mark_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            },
        );
    }

    pub(crate) fn scatter(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        capacity_items: u32,
        profiler: Option<&mut profiling::TimestampRecorder>,
    ) {
        let (groups_x, groups_y) = self.dispatch_dimensions(capacity_items);
        profiling::record_compute_pass(
            encoder,
            "Run-Length Scatter",
            profiler.is_some().then(|| "run_length.scatter".to_owned()),
            profiler,
            |pass| {
                pass.set_pipeline(&self.scatter_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            },
        );
    }

    pub(crate) fn finalize(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        capacity_items: u32,
        profiler: Option<&mut profiling::TimestampRecorder>,
    ) {
        let (groups_x, groups_y) = self.dispatch_dimensions(capacity_items);
        profiling::record_compute_pass(
            encoder,
            "Run-Length Finalize",
            profiler.is_some().then(|| "run_length.finalize".to_owned()),
            profiler,
            |pass| {
                pass.set_pipeline(&self.finalize_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            },
        );
    }

    fn dispatch_dimensions(&self, capacity_items: u32) -> (u32, u32) {
        let workgroups = common::math::calc_groups(capacity_items, BLOCK_SIZE);
        let groups_x = workgroups.min(self.max_workgroups_per_dimension);
        let groups_y = workgroups.div_ceil(self.max_workgroups_per_dimension);
        (groups_x, groups_y)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn shader_guards_padded_groups_before_index_multiplication() {
        for shader in [
            include_str!("run_length.wgsl"),
            include_str!("scatter_starts.wgsl"),
            include_str!("run_length_finalize.wgsl"),
        ] {
            let guard = shader
                .find("if (flat_group_id >= total_groups)")
                .expect("RLE shader must reject padded 2-D workgroups");
            let multiply = shader
                .find("return flat_group_id * BLOCK_SIZE + local_id.x")
                .expect("RLE shader must flatten valid workgroups");
            assert!(guard < multiply);
        }

        let capacity = u32::MAX;
        let total_groups = capacity / 256 + u32::from(capacity % 256 != 0);
        assert_eq!(total_groups, 16_777_216);
        assert_eq!((total_groups - 1) * 256 + 255, u32::MAX);
    }
}
