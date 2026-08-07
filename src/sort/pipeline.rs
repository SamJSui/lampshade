use crate::common;

#[derive(Clone, Copy, PartialEq, Eq)]
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
    pub bits_per_pass: u32,
    pub bucket_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadixVariant {
    Portable,
    NvidiaVulkanWide,
    NvidiaVulkanSubgroup,
}

impl RadixVariant {
    pub fn for_adapter(
        item_kind: SortItemKind,
        adapter_info: &wgpu::AdapterInfo,
        enabled_features: wgpu::Features,
    ) -> Self {
        Self::for_hardware(
            item_kind,
            adapter_info.backend,
            adapter_info.vendor,
            adapter_info.device_type,
            adapter_info.subgroup_min_size,
            adapter_info.subgroup_max_size,
            enabled_features.contains(wgpu::Features::SUBGROUP),
        )
    }

    fn for_hardware(
        item_kind: SortItemKind,
        backend: wgpu::Backend,
        vendor: u32,
        device_type: wgpu::DeviceType,
        subgroup_min_size: u32,
        subgroup_max_size: u32,
        subgroups_enabled: bool,
    ) -> Self {
        if item_kind == SortItemKind::KeyValue
            && backend == wgpu::Backend::Vulkan
            && vendor == 0x10de
            && device_type == wgpu::DeviceType::DiscreteGpu
        {
            if subgroups_enabled && subgroup_min_size == 32 && subgroup_max_size == 32 {
                Self::NvidiaVulkanSubgroup
            } else {
                Self::NvidiaVulkanWide
            }
        } else {
            Self::Portable
        }
    }

    pub const fn uses_eight_bit_pipeline(self) -> bool {
        matches!(self, Self::NvidiaVulkanSubgroup)
    }

    fn bits_per_pass(self) -> u32 {
        match self {
            Self::Portable => 2,
            Self::NvidiaVulkanWide => 4,
            Self::NvidiaVulkanSubgroup => {
                unreachable!("8-bit radix uses its dedicated pipeline")
            }
        }
    }

    fn shader_source(self) -> &'static str {
        match self {
            Self::Portable => include_str!("sort.wgsl"),
            Self::NvidiaVulkanWide => include_str!("sort_wide.wgsl"),
            Self::NvidiaVulkanSubgroup => {
                unreachable!("8-bit radix uses its dedicated shader")
            }
        }
    }
}

impl SortPipeline {
    pub fn new(
        device: &wgpu::Device,
        item_kind: SortItemKind,
        radix_variant: RadixVariant,
    ) -> Self {
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

        let bits_per_pass = radix_variant.bits_per_pass();
        let bucket_count = 1 << bits_per_pass;
        let bucket_group_count = bucket_count / 4;
        let raw_shader = radix_variant.shader_source();
        let local_histogram_size = block_size * bucket_group_count;
        let final_source = raw_shader
            .replace("{{VT}}", &vt.to_string())
            .replace("{{BLOCK_SIZE}}", &block_size.to_string())
            .replace("{{RADIX_BITS}}", &bits_per_pass.to_string())
            .replace("{{RADIX_BUCKETS}}", &bucket_count.to_string())
            .replace("{{RADIX_BUCKET_GROUPS}}", &bucket_group_count.to_string())
            .replace(
                "{{LOCAL_HISTOGRAM_SIZE}}",
                &local_histogram_size.to_string(),
            )
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
            bits_per_pass,
            bucket_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RadixVariant, SortItemKind};

    #[test]
    fn selects_subgroup_radix_only_for_compatible_nvidia_vulkan_key_value_items() {
        assert_eq!(
            RadixVariant::for_hardware(
                SortItemKind::KeyValue,
                wgpu::Backend::Vulkan,
                0x10de,
                wgpu::DeviceType::DiscreteGpu,
                32,
                32,
                true,
            ),
            RadixVariant::NvidiaVulkanSubgroup
        );
        assert_eq!(
            RadixVariant::for_hardware(
                SortItemKind::KeyValue,
                wgpu::Backend::Vulkan,
                0x10de,
                wgpu::DeviceType::DiscreteGpu,
                32,
                32,
                false,
            ),
            RadixVariant::NvidiaVulkanWide
        );
        assert_eq!(
            RadixVariant::for_hardware(
                SortItemKind::Key,
                wgpu::Backend::Vulkan,
                0x10de,
                wgpu::DeviceType::DiscreteGpu,
                32,
                32,
                true,
            ),
            RadixVariant::Portable
        );
        assert_eq!(
            RadixVariant::for_hardware(
                SortItemKind::KeyValue,
                wgpu::Backend::Dx12,
                0x10de,
                wgpu::DeviceType::DiscreteGpu,
                32,
                32,
                true,
            ),
            RadixVariant::Portable
        );
        assert_eq!(
            RadixVariant::for_hardware(
                SortItemKind::KeyValue,
                wgpu::Backend::Vulkan,
                0x1002,
                wgpu::DeviceType::DiscreteGpu,
                32,
                64,
                true,
            ),
            RadixVariant::Portable
        );
    }
}
