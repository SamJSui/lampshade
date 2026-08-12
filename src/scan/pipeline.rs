use crate::{common, profiling};

pub struct ScanDispatch<'a> {
    pub pipeline: &'a wgpu::ComputePipeline,
    pub data: (&'a wgpu::Buffer, u64),
    pub auxiliary: (&'a wgpu::Buffer, u64),
    pub num_items: u32,
    pub pass_label: &'static str,
    pub profile_label: Option<String>,
}

pub struct ScanInputDispatch<'a> {
    pub pipeline: &'a wgpu::ComputePipeline,
    pub input: (&'a wgpu::Buffer, u64),
    pub data: (&'a wgpu::Buffer, u64),
    pub auxiliary: (&'a wgpu::Buffer, u64),
    pub num_items: u32,
    pub pass_label: &'static str,
    pub profile_label: Option<String>,
}

pub struct ScanPipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub input_bind_group_layout: wgpu::BindGroupLayout,
    pub inclusive_scan_pipeline: wgpu::ComputePipeline,
    pub inclusive_input_scan_pipeline: wgpu::ComputePipeline,
    pub exclusive_input_scan_pipeline: wgpu::ComputePipeline,
    pub add_pipeline: wgpu::ComputePipeline,
    pub vt: u32,
    pub block_size: u32,
}

impl ScanPipeline {
    pub fn new(device: &wgpu::Device, allow_subgroups: bool) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scan Layout"),
            entries: &[
                common::buffers::bind_entry(0, false, false),
                common::buffers::bind_entry(1, false, false),
            ],
        });
        let input_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Scan Input Layout"),
                entries: &[
                    common::buffers::bind_entry(0, true, false),
                    common::buffers::bind_entry(1, false, false),
                    common::buffers::bind_entry(2, false, false),
                ],
            });

        let limits = device.limits();
        let max_shared_mem = limits.max_compute_workgroup_storage_size;
        let subgroups_enabled =
            allow_subgroups && device.features().contains(wgpu::Features::SUBGROUP);

        // Subgroup scans use one coalesced item per lane. The portable fallback
        // amortizes its shared-memory scan across several items per thread.
        let (vt, block_size) = if subgroups_enabled {
            (1, 256)
        } else if max_shared_mem >= 32768 {
            (8, 256)
        } else {
            log::warn!("Low-end GPU detected. Downgrading to VT=4.");
            (4, 128)
        };
        let (scan_shader, input_scan_shader, pipeline_kind) = if subgroups_enabled {
            (
                include_str!("scan_subgroup.wgsl"),
                include_str!("scan_input_subgroup.wgsl"),
                "Subgroup",
            )
        } else {
            (
                include_str!("scan.wgsl"),
                include_str!("scan_input.wgsl"),
                "Portable",
            )
        };

        let config = common::shader::ShaderConfig { vt, block_size };

        let inclusive_scan_pipeline = common::shader::create_compute_pipeline_with_constants(
            device,
            &bind_group_layout,
            scan_shader,
            &format!("Inclusive {pipeline_kind} Scan VT{vt} Pipeline"),
            "main",
            Some(&config),
            &[("EXCLUSIVE", 0.0)],
        );

        let inclusive_input_scan_pipeline = common::shader::create_compute_pipeline_with_constants(
            device,
            &input_bind_group_layout,
            input_scan_shader,
            &format!("Inclusive Input {pipeline_kind} Scan VT{vt} Pipeline"),
            "main",
            Some(&config),
            &[("EXCLUSIVE", 0.0)],
        );

        let exclusive_input_scan_pipeline = common::shader::create_compute_pipeline_with_constants(
            device,
            &input_bind_group_layout,
            input_scan_shader,
            &format!("Exclusive Input {pipeline_kind} Scan VT{vt} Pipeline"),
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
            input_bind_group_layout,
            inclusive_scan_pipeline,
            inclusive_input_scan_pipeline,
            exclusive_input_scan_pipeline,
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

        let items_per_block = self.vt * self.block_size;
        let workgroups = common::math::calc_groups(dispatch.num_items, items_per_block);
        let data_size = wgpu::BufferSize::new(u64::from(dispatch.num_items) * 4)
            .expect("scan dispatch data range is non-empty");
        let auxiliary_size = wgpu::BufferSize::new(u64::from(workgroups) * 4)
            .expect("scan dispatch auxiliary range is non-empty");

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scan Dispatch BG"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: data_buf,
                        offset: data_off,
                        size: Some(data_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: aux_buf,
                        offset: aux_off,
                        size: Some(auxiliary_size),
                    }),
                },
            ],
        });

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
        crate::common::runtime::defer_drop(encoder, bg);
    }

    pub fn dispatch_input(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        dispatch: ScanInputDispatch<'_>,
        profiler: Option<&mut profiling::TimestampRecorder>,
    ) {
        let (input, input_offset) = dispatch.input;
        let (data, data_offset) = dispatch.data;
        let (auxiliary, auxiliary_offset) = dispatch.auxiliary;
        let items_per_block = self.vt * self.block_size;
        let workgroups = common::math::calc_groups(dispatch.num_items, items_per_block);
        let data_size = wgpu::BufferSize::new(u64::from(dispatch.num_items) * 4)
            .expect("scan input range is non-empty");
        let auxiliary_size = wgpu::BufferSize::new(u64::from(workgroups) * 4)
            .expect("scan input auxiliary range is non-empty");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scan Input Dispatch BG"),
            layout: &self.input_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: input,
                        offset: input_offset,
                        size: Some(data_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: data,
                        offset: data_offset,
                        size: Some(data_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: auxiliary,
                        offset: auxiliary_offset,
                        size: Some(auxiliary_size),
                    }),
                },
            ],
        });

        let max_dispatch = 65535;
        let x = workgroups.min(max_dispatch);
        let y = workgroups.div_ceil(max_dispatch);
        profiling::record_compute_pass(
            encoder,
            dispatch.pass_label,
            dispatch.profile_label,
            profiler,
            |pass| {
                pass.set_pipeline(dispatch.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(x, y, 1);
            },
        );
        crate::common::runtime::defer_drop(encoder, bind_group);
    }
}
