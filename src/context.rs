use crate::Error;
use wgpu::{AdapterInfo, Backends, Device, Instance, MemoryHints, Queue, RequestAdapterOptions};

pub struct Context {
    pub adapter_info: AdapterInfo,
    pub device: Device,
    pub queue: Queue,
}

impl Context {
    /// Creates a context with the optional GPU features Lampshade can use safely.
    ///
    /// Timestamp queries are not enabled by this convenience constructor on
    /// Apple Metal or integrated NVIDIA Vulkan. Those paths have produced
    /// incomplete timestamps or corrupted repeated compute dispatches.
    /// Profiling methods consequently return [`Error::TimestampQueriesUnsupported`]
    /// for this context instead of reporting misleading durations.
    pub async fn init() -> Result<Self, Error> {
        let descriptor = wgpu::InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        }
        .with_env();
        let instance = Instance::new(descriptor);

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        let adapter_info = adapter.get_info();
        let required_features = context_features(&adapter_info, adapter.features());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Context Device"),
                required_features,
                required_limits: adapter.limits(),
                memory_hints: MemoryHints::Performance,
                ..Default::default()
            })
            .await?;

        Ok(Self {
            adapter_info,
            device,
            queue,
        })
    }
}

fn context_features(
    adapter_info: &wgpu::AdapterInfo,
    supported_features: wgpu::Features,
) -> wgpu::Features {
    let mut optional_features = wgpu::Features::empty();
    if reliable_subgroup_scan(adapter_info) {
        optional_features |= wgpu::Features::SUBGROUP;
    }
    // Apple Metal stage-boundary timestamp queries can leave the final query
    // pair unwritten. Do not expose incomplete profiling through Context.
    let apple_metal = adapter_info.backend == wgpu::Backend::Metal
        && (adapter_info.vendor == 0x106b || adapter_info.name.starts_with("Apple "));
    if !apple_metal && reliable_optional_compute_features(adapter_info) {
        optional_features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    supported_features & optional_features
}

pub(crate) fn reliable_subgroup_scan(adapter_info: &wgpu::AdapterInfo) -> bool {
    reliable_optional_compute_features(adapter_info)
}

fn reliable_optional_compute_features(adapter_info: &wgpu::AdapterInfo) -> bool {
    !(adapter_info.backend == wgpu::Backend::Vulkan
        && adapter_info.vendor == 0x10de
        && adapter_info.device_type == wgpu::DeviceType::IntegratedGpu)
}

#[cfg(test)]
mod tests {
    use super::{context_features, reliable_subgroup_scan};

    fn adapter(name: &str, vendor: u32, backend: wgpu::Backend) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.to_owned(),
            vendor,
            device: 0,
            device_type: wgpu::DeviceType::Other,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend,
            subgroup_min_size: 0,
            subgroup_max_size: 0,
            transient_saves_memory: false,
        }
    }

    #[test]
    fn disables_unreliable_apple_metal_timestamps_only() {
        let supported = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SUBGROUP;
        let apple_vendor = context_features(
            &adapter("Metal device", 0x106b, wgpu::Backend::Metal),
            supported,
        );
        assert!(!apple_vendor.contains(wgpu::Features::TIMESTAMP_QUERY));
        assert!(apple_vendor.contains(wgpu::Features::SUBGROUP));

        let apple_name =
            context_features(&adapter("Apple M3 Pro", 0, wgpu::Backend::Metal), supported);
        assert!(!apple_name.contains(wgpu::Features::TIMESTAMP_QUERY));

        let intel_metal = context_features(
            &adapter("Intel GPU", 0x8086, wgpu::Backend::Metal),
            supported,
        );
        assert!(intel_metal.contains(wgpu::Features::TIMESTAMP_QUERY));

        let apple_vulkan = context_features(
            &adapter("Apple GPU", 0x106b, wgpu::Backend::Vulkan),
            supported,
        );
        assert!(apple_vulkan.contains(wgpu::Features::TIMESTAMP_QUERY));
    }

    #[test]
    fn disables_subgroups_on_integrated_nvidia_vulkan() {
        let mut orin = adapter("Orin (nvgpu)", 0x10de, wgpu::Backend::Vulkan);
        orin.device_type = wgpu::DeviceType::IntegratedGpu;
        let enabled = context_features(
            &orin,
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SUBGROUP,
        );

        assert!(!reliable_subgroup_scan(&orin));
        assert!(!enabled.contains(wgpu::Features::SUBGROUP));
        assert!(!enabled.contains(wgpu::Features::TIMESTAMP_QUERY));
    }
}
