use crate::common;

pub struct ScanPipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub scan_pipeline: wgpu::ComputePipeline,
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

        let scan_pipeline = common::shader::create_compute_pipeline(
            device,
            &bind_group_layout,
            include_str!("scan.wgsl"),
            &format!("Scan VT{} Pipeline", vt),
            "main",
            Some(&config),
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
            scan_pipeline,
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

    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        data: (&wgpu::Buffer, u64),
        auxiliary: (&wgpu::Buffer, u64),
        num_items: u32,
    ) {
        let (data_buf, data_off) = data;
        let (aux_buf, aux_off) = auxiliary;

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

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        cpass.set_pipeline(pipeline);
        cpass.set_bind_group(0, &bg, &[]);

        let items_per_block = self.vt * self.block_size;
        let workgroups = common::math::calc_groups(num_items, items_per_block);

        let max_dispatch = 65535;
        let x = if workgroups > max_dispatch {
            max_dispatch
        } else {
            workgroups
        };
        let y = workgroups.div_ceil(max_dispatch);

        cpass.dispatch_workgroups(x, y, 1);
    }
}
