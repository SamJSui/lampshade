use crate::common::{self, capabilities::AdapterCapabilities};

const EIGHT_BIT_BLOCK_SIZE: u32 = 256;
const EIGHT_BIT_WORKGROUP_STORAGE_BYTES: u32 = 16_388;
const WIDE_RADIX_BUCKET_GROUPS: u32 = 4;
const WIDE_RADIX_BYTES_PER_BUCKET_GROUP: u32 = 16;

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

    pub(super) fn shader_item_type(self) -> &'static str {
        match self {
            Self::Key => "u32",
            Self::KeyValue => "KeyValue",
        }
    }

    pub(super) fn shader_key_access(self) -> &'static str {
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
    IntelVulkanWide,
    NvidiaVulkanWide,
    NvidiaVulkanSubgroup,
}

impl RadixVariant {
    pub fn for_adapter(
        item_kind: SortItemKind,
        adapter_info: &wgpu::AdapterInfo,
        enabled_features: wgpu::Features,
        device_limits: &wgpu::Limits,
    ) -> Self {
        Self::for_capabilities(
            item_kind,
            AdapterCapabilities::from_adapter(adapter_info, enabled_features, device_limits),
        )
    }

    fn for_capabilities(item_kind: SortItemKind, capabilities: AdapterCapabilities) -> Self {
        let is_vulkan_key_value =
            item_kind == SortItemKind::KeyValue && capabilities.backend == wgpu::Backend::Vulkan;
        if !is_vulkan_key_value {
            return Self::Portable;
        }

        if capabilities.vendor == 0x10de
            && capabilities.subgroups_enabled
            && capabilities.subgroup_min_size == 32
            && capabilities.subgroup_max_size == 32
            && capabilities.supports_workgroup(
                EIGHT_BIT_BLOCK_SIZE,
                EIGHT_BIT_BLOCK_SIZE,
                EIGHT_BIT_WORKGROUP_STORAGE_BYTES,
            )
        {
            Self::NvidiaVulkanSubgroup
        } else if capabilities.vendor == 0x10de
            && capabilities.device_type == wgpu::DeviceType::DiscreteGpu
        {
            Self::NvidiaVulkanWide
        } else if capabilities.vendor == 0x8086 && supports_wide_radix(capabilities) {
            Self::IntelVulkanWide
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
            Self::IntelVulkanWide | Self::NvidiaVulkanWide => 4,
            Self::NvidiaVulkanSubgroup => {
                unreachable!("8-bit radix uses its dedicated pipeline")
            }
        }
    }

    fn shader_source(self) -> &'static str {
        match self {
            Self::Portable => include_str!("sort.wgsl"),
            Self::IntelVulkanWide | Self::NvidiaVulkanWide => include_str!("sort_wide.wgsl"),
            Self::NvidiaVulkanSubgroup => {
                unreachable!("8-bit radix uses its dedicated shader")
            }
        }
    }
}

fn supports_wide_radix(capabilities: AdapterCapabilities) -> bool {
    let (_, block_size) = radix_workgroup_config(capabilities.max_compute_workgroup_storage_size);
    let storage_bytes = block_size * WIDE_RADIX_BUCKET_GROUPS * WIDE_RADIX_BYTES_PER_BUCKET_GROUP;
    capabilities.supports_workgroup(block_size, block_size, storage_bytes)
}

fn radix_workgroup_config(max_workgroup_storage_size: u32) -> (u32, u32) {
    if max_workgroup_storage_size >= 32_768 {
        (8, 256)
    } else {
        (4, 128)
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
        let (vt, block_size) = radix_workgroup_config(limits.max_compute_workgroup_storage_size);

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
            bind_group_layouts: &[Some(&bind_group_layout)],
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
    use crate::common::capabilities::AdapterCapabilities;

    use super::{RadixVariant, SortItemKind};

    #[test]
    fn selects_radix_variant_from_adapter_capabilities() {
        let select = |item_kind, backend, vendor, device_type, min, max, feature, limits| {
            let workgroup_limit = if limits { u32::MAX } else { 0 };
            RadixVariant::for_capabilities(
                item_kind,
                AdapterCapabilities {
                    backend,
                    vendor,
                    device_type,
                    subgroup_min_size: min,
                    subgroup_max_size: max,
                    subgroups_enabled: feature,
                    max_compute_invocations_per_workgroup: workgroup_limit,
                    max_compute_workgroup_size_x: workgroup_limit,
                    max_compute_workgroup_storage_size: workgroup_limit,
                },
            )
        };

        for device_type in [
            wgpu::DeviceType::DiscreteGpu,
            wgpu::DeviceType::IntegratedGpu,
        ] {
            assert_eq!(
                select(
                    SortItemKind::KeyValue,
                    wgpu::Backend::Vulkan,
                    0x10de,
                    device_type,
                    32,
                    32,
                    true,
                    true,
                ),
                RadixVariant::NvidiaVulkanSubgroup
            );
        }
        assert_eq!(
            select(
                SortItemKind::KeyValue,
                wgpu::Backend::Vulkan,
                0x10de,
                wgpu::DeviceType::DiscreteGpu,
                32,
                32,
                false,
                true,
            ),
            RadixVariant::NvidiaVulkanWide
        );
        for (device_type, expected) in [
            (
                wgpu::DeviceType::DiscreteGpu,
                RadixVariant::NvidiaVulkanWide,
            ),
            (wgpu::DeviceType::IntegratedGpu, RadixVariant::Portable),
        ] {
            assert_eq!(
                select(
                    SortItemKind::KeyValue,
                    wgpu::Backend::Vulkan,
                    0x10de,
                    device_type,
                    32,
                    32,
                    true,
                    false,
                ),
                expected
            );
        }
        assert_eq!(
            select(
                SortItemKind::Key,
                wgpu::Backend::Vulkan,
                0x10de,
                wgpu::DeviceType::DiscreteGpu,
                32,
                32,
                true,
                true,
            ),
            RadixVariant::Portable
        );
        assert_eq!(
            select(
                SortItemKind::Key,
                wgpu::Backend::Vulkan,
                0x8086,
                wgpu::DeviceType::IntegratedGpu,
                8,
                32,
                true,
                true,
            ),
            RadixVariant::Portable
        );
        assert_eq!(
            select(
                SortItemKind::KeyValue,
                wgpu::Backend::Dx12,
                0x8086,
                wgpu::DeviceType::IntegratedGpu,
                8,
                32,
                true,
                true,
            ),
            RadixVariant::Portable
        );
        assert_eq!(
            select(
                SortItemKind::KeyValue,
                wgpu::Backend::Dx12,
                0x10de,
                wgpu::DeviceType::DiscreteGpu,
                32,
                32,
                true,
                true,
            ),
            RadixVariant::Portable
        );
        assert_eq!(
            select(
                SortItemKind::KeyValue,
                wgpu::Backend::Vulkan,
                0x1002,
                wgpu::DeviceType::DiscreteGpu,
                32,
                64,
                true,
                true,
            ),
            RadixVariant::Portable
        );

        assert_eq!(
            select(
                SortItemKind::KeyValue,
                wgpu::Backend::Vulkan,
                0x8086,
                wgpu::DeviceType::IntegratedGpu,
                8,
                32,
                true,
                true,
            ),
            RadixVariant::IntelVulkanWide
        );
        assert_eq!(
            select(
                SortItemKind::KeyValue,
                wgpu::Backend::Vulkan,
                0x8086,
                wgpu::DeviceType::IntegratedGpu,
                8,
                32,
                true,
                false,
            ),
            RadixVariant::Portable
        );
        assert_eq!(
            select(
                SortItemKind::KeyValue,
                wgpu::Backend::Metal,
                0x106b,
                wgpu::DeviceType::IntegratedGpu,
                4,
                64,
                true,
                true,
            ),
            RadixVariant::Portable
        );
    }
}
