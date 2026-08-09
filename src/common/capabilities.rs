/// Adapter and device capabilities used to select private kernel backends.
///
/// Keeping this snapshot separate from a primitive's policy makes two things
/// explicit: what the device can do, and what implementation we choose to run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AdapterCapabilities {
    pub(crate) backend: wgpu::Backend,
    pub(crate) vendor: u32,
    pub(crate) device_type: wgpu::DeviceType,
    pub(crate) subgroup_min_size: u32,
    pub(crate) subgroup_max_size: u32,
    pub(crate) subgroups_enabled: bool,
    pub(crate) max_compute_invocations_per_workgroup: u32,
    pub(crate) max_compute_workgroup_size_x: u32,
    pub(crate) max_compute_workgroup_storage_size: u32,
}

impl AdapterCapabilities {
    pub(crate) fn from_adapter(
        adapter_info: &wgpu::AdapterInfo,
        enabled_features: wgpu::Features,
        device_limits: &wgpu::Limits,
    ) -> Self {
        Self {
            backend: adapter_info.backend,
            vendor: adapter_info.vendor,
            device_type: adapter_info.device_type,
            subgroup_min_size: adapter_info.subgroup_min_size,
            subgroup_max_size: adapter_info.subgroup_max_size,
            subgroups_enabled: enabled_features.contains(wgpu::Features::SUBGROUP),
            max_compute_invocations_per_workgroup: device_limits
                .max_compute_invocations_per_workgroup,
            max_compute_workgroup_size_x: device_limits.max_compute_workgroup_size_x,
            max_compute_workgroup_storage_size: device_limits.max_compute_workgroup_storage_size,
        }
    }

    pub(crate) const fn supports_workgroup(
        self,
        invocations: u32,
        size_x: u32,
        storage_bytes: u32,
    ) -> bool {
        self.max_compute_invocations_per_workgroup >= invocations
            && self.max_compute_workgroup_size_x >= size_x
            && self.max_compute_workgroup_storage_size >= storage_bytes
    }
}
