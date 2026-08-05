use crate::common;

#[derive(Clone, Copy)]
pub enum SortItemKind {
    Key,
    KeyValue,
}

impl SortItemKind {
    pub const fn size_bytes(self) -> u64 {
        match self {
            Self::Key => 4,
            Self::KeyValue => 8,
        }
    }

    fn shader_item_type(self) -> &'static str {
        match self {
            Self::Key => "u32",
            Self::KeyValue => "KeyValue",
        }
    }

    fn shader_key_access(self) -> &'static str {
        match self {
            Self::Key => "item",
            Self::KeyValue => "item.key",
        }
    }
}

pub struct SortPipeline {
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub reduce_pipeline: wgpu::ComputePipeline,
    pub scatter_pipeline: wgpu::ComputePipeline,
    pub vt: u32,
    pub block_size: u32,
}

impl SortPipeline {
    pub fn new(device: &wgpu::Device, item_kind: SortItemKind) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fused Sort Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),  // Input
                common::buffers::bind_entry(1, false, false), // Hist
                common::buffers::bind_entry(2, false, false), // Output
                common::buffers::bind_entry(3, false, true),  // Uniforms
            ],
        });

        let limits = device.limits();
        let max_shared_mem = limits.max_compute_workgroup_storage_size;

        let (vt, block_size) = if max_shared_mem >= 32768 {
            (8, 256) // M3 / Desktop
        } else {
            (4, 128) // Mobile
        };

        let raw_shader = include_str!("sort.wgsl");
        let final_source = raw_shader
            .replace("{{VT}}", &vt.to_string())
            .replace("{{BLOCK_SIZE}}", &block_size.to_string())
            .replace("{{ITEM_TYPE}}", item_kind.shader_item_type())
            .replace("{{KEY_ACCESS}}", item_kind.shader_key_access());

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fused Sort Shader"),
            source: wgpu::ShaderSource::Wgsl(final_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fused Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let reduce_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reduce Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main_reduce"),
            compilation_options: Default::default(),
            cache: None,
        });

        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Scatter Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main_scatter"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            bind_group_layout,
            reduce_pipeline,
            scatter_pipeline,
            vt,
            block_size,
        }
    }
}
