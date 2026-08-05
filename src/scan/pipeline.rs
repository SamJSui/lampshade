use crate::{common, profiling};

pub struct ScanDispatch<'a> {
    pub pipeline: &'a wgpu::ComputePipeline,
    pub data: (&'a wgpu::Buffer, u64),
    pub auxiliary: (&'a wgpu::Buffer, u64),
    pub num_items: u32,
    pub pass_label: &'static str,
    pub profile_label: Option<String>,
}

pub struct ScanPipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub inclusive_scan_pipeline: wgpu::ComputePipeline,
    pub exclusive_scan_pipeline: wgpu::ComputePipeline,
    pub add_pipeline: wgpu::ComputePipeline,
    pub vt: u32,
    pub block_size: u32,
}

impl ScanPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scan Layout"),
            entries: &[
                common::buffers::bind_entry(0, false, false),
                common::buffers::bind_entry(1, false, false),
            ],
        });

        let limits = device.limits();
        let max_shared_mem = limits.max_compute_workgroup_storage_size;

        // High End (M3/Desktop): 32KB+ shared mem -> Use VT=8, Block=256
        // Low End (Mobile): <32KB shared mem -> Use VT=4, Block=128 (Lower register pressure)
        let (vt, block_size) = if max_shared_mem >= 32768 {
            (8, 256)
        } else {
            log::warn!("Low-end GPU detected. Downgrading to VT=4.");
            (4, 128)
        };

        let config = common::shader::ShaderConfig { vt, block_size };

        let inclusive_scan_pipeline = common::shader::create_compute_pipeline_with_constants(
            device,
            &bind_group_layout,
            include_str!("scan.wgsl"),
            &format!("Inclusive Scan VT{} Pipeline", vt),
            "main",
            Some(&config),
            &[("EXCLUSIVE", 0.0)],
        );

        let exclusive_scan_pipeline = common::shader::create_compute_pipeline_with_constants(
            device,
            &bind_group_layout,
            include_str!("scan.wgsl"),
            &format!("Exclusive Scan VT{} Pipeline", vt),
            "main",
            Some(&config),
            &[("EXCLUSIVE", 1.0)],
        );

        let add_pipeline = common::shader::create_compute_pipeline(
            device,
            &bind_group_layout,
            include_str!("add.wgsl"),
            &format!("Add VT{} Pipeline", vt),
            "main",
            Some(&config),
        );

        Self {
            bind_group_layout,
            inclusive_scan_pipeline,
            exclusive_scan_pipeline,
            add_pipeline,
            vt,
            block_size,
        }
    }

    pub fn get_scratch_size(&self, num_items: u32) -> u64 {
        let mut size = 0;
        let mut current_items = num_items;

        let items_per_block = self.vt * self.block_size;

        while current_items > 1 {
            let aux_count = current_items.div_ceil(items_per_block);
            let raw_size = (aux_count * 4) as u64;
            let aligned_size = common::math::align_to(raw_size, 256);
            size += aligned_size;
            current_items = aux_count;
        }
        size
    }

    pub fn compute_pass_count(&self, num_items: u32) -> u32 {
        let items_per_block = self.vt * self.block_size;
        let mut levels = 0;
        let mut current_items = num_items;

        while current_items > 1 {
            current_items = current_items.div_ceil(items_per_block);
            levels += 1;
        }

        levels * 2
    }

    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dispatch: ScanDispatch<'_>,
        profiler: Option<&mut profiling::TimestampRecorder>,
    ) {
        let (data_buf, data_off) = dispatch.data;
        let (aux_buf, aux_off) = dispatch.auxiliary;

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scan Dispatch BG"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: data_buf,
                        offset: data_off,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: aux_buf,
                        offset: aux_off,
                        size: None,
                    }),
                },
            ],
        });

        let items_per_block = self.vt * self.block_size;
        let workgroups = common::math::calc_groups(dispatch.num_items, items_per_block);

        let max_dispatch = 65535;
        let x = if workgroups > max_dispatch {
            max_dispatch
        } else {
            workgroups
        };
        let y = workgroups.div_ceil(max_dispatch);

        profiling::record_compute_pass(
            encoder,
            dispatch.pass_label,
            dispatch.profile_label,
            profiler,
            |pass| {
                pass.set_pipeline(dispatch.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(x, y, 1);
            },
        );
    }
}
