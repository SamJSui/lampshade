pub struct ShaderConfig {
    pub vt: u32,
    pub block_size: u32,
}

/// Compiles a shader, optionally performing string replacement for constants
pub fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    shader_source: &str,
    label: &str,
    entry_point: &str, // NEW ARGUMENT
    config: Option<&ShaderConfig>,
) -> wgpu::ComputePipeline {
    let final_source = if let Some(cfg) = config {
        shader_source
            .replace("{{VT}}", &cfg.vt.to_string())
            .replace("{{BLOCK_SIZE}}", &cfg.block_size.to_string())
    } else {
        shader_source.to_string()
    };

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(final_source.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[layout],
        immediate_size: 0,
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}
