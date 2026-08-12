use std::sync::Mutex;

use wgpu::util::DeviceExt;

use crate::common::buffers::BufferRange;
use crate::{Context, Error, common};

use super::eight_bit::NativeKeyValueSoaSorter;
use super::key_value_sorter::KeyValueSorter;
use super::pipeline::{
    EIGHT_BIT_BLOCK_SIZE, EIGHT_BIT_WORKGROUP_STORAGE_BYTES, RadixVariant, SortItemKind,
};

const U32_SIZE_BYTES: u64 = size_of::<u32>() as u64;
const KEY_VALUE_SIZE_BYTES: u64 = 2 * U32_SIZE_BYTES;
const BRIDGE_BLOCK_SIZE: u32 = 256;
const MAX_WORKGROUPS_X: u32 = 65_535;
const WORKSPACE_GROWTH_BYTES: u64 = 16 * 1024 * 1024;

/// Device features and limits that enable Lampshade's accelerated SoA sorter.
///
/// When `accelerated` is false, request the application's normal features and
/// limits; [`KeyValueSoaSorter`] will use its portable backend.
#[derive(Clone, Debug)]
pub struct KeyValueSoaRequirements {
    additional_features: wgpu::Features,
    min_compute_invocations: u32,
    min_compute_workgroup_size_x: u32,
    min_compute_workgroup_storage_size: u32,
    /// Whether requesting these additions enables the native SoA backend.
    pub accelerated: bool,
}

impl KeyValueSoaRequirements {
    /// Adds the optional features required by the native SoA backend.
    pub fn features(&self, existing: wgpu::Features) -> wgpu::Features {
        existing | self.additional_features
    }

    /// Raises only the limits required by the native SoA backend.
    ///
    /// All unrelated application limits, including storage-buffer sizes, are
    /// preserved.
    pub fn limits(&self, mut existing: wgpu::Limits) -> wgpu::Limits {
        existing.max_compute_invocations_per_workgroup = existing
            .max_compute_invocations_per_workgroup
            .max(self.min_compute_invocations);
        existing.max_compute_workgroup_size_x = existing
            .max_compute_workgroup_size_x
            .max(self.min_compute_workgroup_size_x);
        existing.max_compute_workgroup_storage_size = existing
            .max_compute_workgroup_storage_size
            .max(self.min_compute_workgroup_storage_size);
        existing
    }
}

/// Stable in-place radix sort for separate `u32` key and value buffers.
///
/// [`Self::new`] selects the accelerated native-SoA implementation when the
/// enabled device supports it and otherwise bridges through Lampshade's
/// portable key/value radix sorter. Keys and values remain in separate caller
/// buffers in both cases.
pub struct KeyValueSoaSorter {
    backend: SoaBackend,
    fixed: Option<FixedPlan>,
}

enum SoaBackend {
    Native(Box<NativeKeyValueSoaSorter>),
    Portable(Box<PortableKeyValueSoaSorter>),
}

struct FixedPlan {
    keys: wgpu::Buffer,
    values: wgpu::Buffer,
    count: wgpu::Buffer,
    num_items: u32,
}

impl KeyValueSoaSorter {
    /// Returns the additions to a default device request that enable the
    /// accelerated native-SoA backend on this adapter.
    pub fn requirements(adapter: &wgpu::Adapter) -> KeyValueSoaRequirements {
        let adapter_info = adapter.get_info();
        let adapter_limits = adapter.limits();
        let accelerated = RadixVariant::for_adapter(
            SortItemKind::KeyValue,
            &adapter_info,
            adapter.features(),
            &adapter_limits,
        )
        .uses_eight_bit_pipeline();
        KeyValueSoaRequirements {
            additional_features: if accelerated {
                wgpu::Features::SUBGROUP
            } else {
                wgpu::Features::empty()
            },
            min_compute_invocations: if accelerated { EIGHT_BIT_BLOCK_SIZE } else { 0 },
            min_compute_workgroup_size_x: if accelerated { EIGHT_BIT_BLOCK_SIZE } else { 0 },
            min_compute_workgroup_storage_size: if accelerated {
                EIGHT_BIT_WORKGROUP_STORAGE_BYTES
            } else {
                0
            },
            accelerated,
        }
    }

    /// Creates an adapter-selected sorter from an existing device and queue.
    ///
    /// The native backend is selected only if the device was created with all
    /// requirements returned by [`Self::requirements`].
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let adapter_info = device.adapter_info();
        let backend =
            if let Some(sorter) = NativeKeyValueSoaSorter::new_for_adapter(device, &adapter_info) {
                SoaBackend::Native(Box::new(sorter))
            } else {
                SoaBackend::Portable(Box::new(PortableKeyValueSoaSorter::new_for_adapter(
                    device,
                    queue,
                    &adapter_info,
                )))
            };
        Self {
            backend,
            fixed: None,
        }
    }

    /// Creates a sorter that always uses the portable SoA bridge.
    pub fn new_portable(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            backend: SoaBackend::Portable(Box::new(PortableKeyValueSoaSorter::new_portable(
                device, queue,
            ))),
            fixed: None,
        }
    }

    /// Creates an adapter-selected sorter from Lampshade's convenience context.
    pub fn from_context(context: &Context) -> Self {
        Self::new(&context.device, &context.queue)
    }

    /// Creates the native backend only, returning `None` when unsupported.
    ///
    /// New code should generally use [`Self::new`] so unsupported devices fall
    /// back transparently.
    pub fn new_native_for_adapter(
        device: &wgpu::Device,
        adapter_info: &wgpu::AdapterInfo,
    ) -> Option<Self> {
        NativeKeyValueSoaSorter::new_for_adapter(device, adapter_info).map(|sorter| Self {
            backend: SoaBackend::Native(Box::new(sorter)),
            fixed: None,
        })
    }

    /// Native-only constructor retained for 0.10 API compatibility.
    pub fn new_for_adapter(
        device: &wgpu::Device,
        adapter_info: &wgpu::AdapterInfo,
    ) -> Option<Self> {
        Self::new_native_for_adapter(device, adapter_info)
    }

    /// Returns true when this instance uses the validated native SoA backend.
    pub const fn is_accelerated(&self) -> bool {
        matches!(&self.backend, SoaBackend::Native(_))
    }

    /// Reserves internal storage for fixed or GPU-counted sorts up to `capacity`.
    ///
    /// Call a `prepare_*` method afterward to bind a particular buffer set.
    pub fn reserve(&mut self, capacity: u32) -> Result<(), Error> {
        self.fixed = None;
        match &mut self.backend {
            SoaBackend::Native(sorter) => sorter.reserve(capacity),
            SoaBackend::Portable(sorter) => sorter.reserve(capacity),
        }
    }

    /// Prepares a reusable fixed-count binding plan.
    ///
    /// `keys` and `values` require `STORAGE` usage, must be distinct buffers,
    /// and must each contain at least `num_items` `u32` elements.
    pub fn prepare_sort(
        &mut self,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        let count = self
            .fixed
            .as_ref()
            .and_then(|plan| (plan.num_items == num_items).then(|| plan.count.clone()));
        let count = count.unwrap_or_else(|| match &self.backend {
            SoaBackend::Native(sorter) => {
                sorter
                    .device()
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Fixed SoA Sort Count"),
                        contents: bytemuck::bytes_of(&num_items),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            }
            SoaBackend::Portable(sorter) => {
                sorter
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Fixed SoA Sort Count"),
                        contents: bytemuck::bytes_of(&num_items),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            }
        });
        self.prepare_counted_from_word(keys, values, &count, 0, num_items)?;
        self.fixed = Some(FixedPlan {
            keys: keys.clone(),
            values: values.clone(),
            count,
            num_items,
        });
        Ok(())
    }

    /// Records a fixed-count stable in-place SoA sort.
    ///
    /// The first call for a buffer pair prepares its reusable bindings. Call
    /// [`Self::prepare_sort`] during setup to keep that work out of recording.
    pub fn record_sort(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        if !self.fixed_matches(keys, values, num_items) {
            self.prepare_sort(keys, values, num_items)?;
        }
        self.record_reserved_sort(encoder, keys, values, num_items)
    }

    /// Records a previously prepared fixed-count sort.
    ///
    /// On the native backend this performs no buffer, bind-group, or pipeline
    /// creation. The portable bridge may still prepare transient state inside
    /// its general-purpose radix sorter.
    pub fn record_reserved_sort(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        let plan = self.fixed.as_ref().ok_or(Error::BufferTooSmall {
            name: "SoA fixed sort plan",
            required: 1,
            actual: 0,
        })?;
        if plan.keys != *keys || plan.values != *values || plan.num_items != num_items {
            return Err(Error::BufferTooSmall {
                name: "SoA fixed sort plan",
                required: 1,
                actual: 0,
            });
        }
        self.record_reserved_sort_counted_from_word(
            encoder,
            keys,
            values,
            &plan.count,
            0,
            num_items,
        )
    }

    /// Records a stable sort of a GPU-counted prefix using count word zero.
    ///
    /// The GPU count is clamped to `capacity`. Keys and values beyond that
    /// prefix are unspecified. All three buffers require `STORAGE` usage and
    /// must have distinct handles.
    pub fn record_sort_counted(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
    ) -> Result<(), Error> {
        self.record_sort_counted_from_word(encoder, keys, values, count, 0, capacity)
    }

    /// Records a stable GPU-counted prefix sort from a metadata buffer.
    ///
    /// `count_word` is a `u32` index within `count`, not a byte offset. The
    /// selected count is clamped to `capacity`; output beyond it is unspecified.
    #[allow(clippy::too_many_arguments)]
    pub fn record_sort_counted_from_word(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        count: &wgpu::Buffer,
        count_word: u32,
        capacity: u32,
    ) -> Result<(), Error> {
        self.fixed = None;
        match &mut self.backend {
            SoaBackend::Native(sorter) => sorter
                .record_sort_counted_from_word(encoder, keys, values, count, count_word, capacity),
            SoaBackend::Portable(sorter) => sorter
                .record_sort_counted_from_word(encoder, keys, values, count, count_word, capacity),
        }
    }

    /// Prepares buffers and binding state for a GPU-counted sort.
    ///
    /// Preparing a counted sort invalidates any fixed-count plan held by this
    /// sorter instance.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_counted_from_word(
        &mut self,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        count: &wgpu::Buffer,
        count_word: u32,
        capacity: u32,
    ) -> Result<(), Error> {
        self.fixed = None;
        match &mut self.backend {
            SoaBackend::Native(sorter) => {
                sorter.prepare_counted_from_word(keys, values, count, count_word, capacity)
            }
            SoaBackend::Portable(sorter) => {
                sorter.prepare_counted_from_word(keys, values, count, count_word, capacity)
            }
        }
    }

    /// Records a previously prepared GPU-counted sort.
    ///
    /// Native recording is allocation-free. The portable backend preserves
    /// the same command-ordering contract but may build transient state inside
    /// the general-purpose radix sorter.
    #[allow(clippy::too_many_arguments)]
    pub fn record_reserved_sort_counted_from_word(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        count: &wgpu::Buffer,
        count_word: u32,
        capacity: u32,
    ) -> Result<(), Error> {
        match &self.backend {
            SoaBackend::Native(sorter) => sorter.record_reserved_sort_counted_from_word(
                encoder, keys, values, count, count_word, capacity,
            ),
            SoaBackend::Portable(sorter) => sorter.record_reserved_sort_counted_from_word(
                encoder, keys, values, count, count_word, capacity,
            ),
        }
    }

    fn fixed_matches(&self, keys: &wgpu::Buffer, values: &wgpu::Buffer, num_items: u32) -> bool {
        self.fixed.as_ref().is_some_and(|plan| {
            plan.keys == *keys && plan.values == *values && plan.num_items == num_items
        })
    }
}

struct PortableKeyValueSoaSorter {
    device: wgpu::Device,
    sorter: Mutex<KeyValueSorter>,
    pack_layout: wgpu::BindGroupLayout,
    unpack_layout: wgpu::BindGroupLayout,
    pack: wgpu::ComputePipeline,
    unpack: wgpu::ComputePipeline,
    workspace: Option<PortableWorkspace>,
    bindings: Option<PortableBindings>,
}

struct PortableWorkspace {
    capacity_bytes: u64,
    packed_input: wgpu::Buffer,
    packed_output: wgpu::Buffer,
    clamped_count: wgpu::Buffer,
}

struct PortableBindings {
    keys: wgpu::Buffer,
    values: wgpu::Buffer,
    count: wgpu::Buffer,
    count_word: u32,
    capacity: u32,
    workspace_capacity: u64,
    _config: wgpu::Buffer,
    pack: wgpu::BindGroup,
    unpack: wgpu::BindGroup,
}

impl PortableKeyValueSoaSorter {
    fn new_for_adapter(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter_info: &wgpu::AdapterInfo,
    ) -> Self {
        Self::with_sorter(
            device,
            KeyValueSorter::new_for_adapter(device, queue, adapter_info),
        )
    }

    fn new_portable(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self::with_sorter(device, KeyValueSorter::new(device, queue))
    }

    fn with_sorter(device: &wgpu::Device, sorter: KeyValueSorter) -> Self {
        let pack_layout = bridge_layout(device, "SoA Pack Layout", true);
        let unpack_layout = bridge_layout(device, "SoA Unpack Layout", false);
        let pack = bridge_pipeline(
            device,
            &pack_layout,
            include_str!("soa_pack.wgsl"),
            "SoA Pack Pipeline",
        );
        let unpack = bridge_pipeline(
            device,
            &unpack_layout,
            include_str!("soa_unpack.wgsl"),
            "SoA Unpack Pipeline",
        );
        Self {
            device: device.clone(),
            sorter: Mutex::new(sorter),
            pack_layout,
            unpack_layout,
            pack,
            unpack,
            workspace: None,
            bindings: None,
        }
    }

    fn reserve(&mut self, capacity: u32) -> Result<(), Error> {
        if capacity == 0 {
            return Ok(());
        }
        let packed_bytes =
            common::math::checked_byte_size(u64::from(capacity), KEY_VALUE_SIZE_BYTES)?;
        self.ensure_workspace(packed_bytes)?;
        self.sorter
            .get_mut()
            .expect("portable SoA sorter lock is not poisoned")
            .reserve_counted(capacity)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_sort_counted_from_word(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        count: &wgpu::Buffer,
        count_word: u32,
        capacity: u32,
    ) -> Result<(), Error> {
        self.prepare_counted_from_word(keys, values, count, count_word, capacity)?;
        self.record_reserved_sort_counted_from_word(
            encoder, keys, values, count, count_word, capacity,
        )
    }

    fn prepare_counted_from_word(
        &mut self,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        count: &wgpu::Buffer,
        count_word: u32,
        capacity: u32,
    ) -> Result<(), Error> {
        let Some((keys_range, values_range, count_range, count_bytes)) =
            self.validate_inputs(keys, values, count, count_word, capacity)?
        else {
            self.bindings = None;
            return Ok(());
        };
        self.reserve(capacity)?;
        if self.bindings_match(keys, values, count, count_word, capacity) {
            return Ok(());
        }
        let workspace = self
            .workspace
            .as_ref()
            .expect("portable SoA workspace is prepared");
        let config = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Portable SoA Bridge Configuration"),
                contents: bytemuck::cast_slice(&[capacity, count_word, 0_u32, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let pack = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Portable SoA Pack Bind Group"),
            layout: &self.pack_layout,
            entries: &[
                bridge_entry(0, keys_range.binding(u64::from(capacity) * U32_SIZE_BYTES)),
                bridge_entry(
                    1,
                    values_range.binding(u64::from(capacity) * U32_SIZE_BYTES),
                ),
                bridge_entry(2, workspace.packed_input.as_entire_binding()),
                bridge_entry(3, count_range.binding(count_bytes)),
                bridge_entry(4, config.as_entire_binding()),
                bridge_entry(5, workspace.clamped_count.as_entire_binding()),
            ],
        });
        let unpack = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Portable SoA Unpack Bind Group"),
            layout: &self.unpack_layout,
            entries: &[
                bridge_entry(0, workspace.packed_output.as_entire_binding()),
                bridge_entry(1, keys_range.binding(u64::from(capacity) * U32_SIZE_BYTES)),
                bridge_entry(
                    2,
                    values_range.binding(u64::from(capacity) * U32_SIZE_BYTES),
                ),
                bridge_entry(3, count_range.binding(count_bytes)),
                bridge_entry(4, config.as_entire_binding()),
            ],
        });
        self.bindings = Some(PortableBindings {
            keys: keys.clone(),
            values: values.clone(),
            count: count.clone(),
            count_word,
            capacity,
            workspace_capacity: workspace.capacity_bytes,
            _config: config,
            pack,
            unpack,
        });
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn record_reserved_sort_counted_from_word(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        count: &wgpu::Buffer,
        count_word: u32,
        capacity: u32,
    ) -> Result<(), Error> {
        if self
            .validate_inputs(keys, values, count, count_word, capacity)?
            .is_none()
        {
            return Ok(());
        }
        if !self.bindings_match(keys, values, count, count_word, capacity) {
            return Err(Error::BufferTooSmall {
                name: "SoA sort binding plan",
                required: 1,
                actual: 0,
            });
        }
        let workspace = self
            .workspace
            .as_ref()
            .expect("portable SoA workspace is prepared");
        let bindings = self
            .bindings
            .as_ref()
            .expect("portable SoA bindings are prepared");
        let (groups_x, groups_y) = bridge_dispatch(capacity);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Portable SoA Pack"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack);
            pass.set_bind_group(0, &bindings.pack, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        self.sorter
            .lock()
            .expect("portable SoA sorter lock is not poisoned")
            .record_sort_counted(
                encoder,
                &workspace.packed_input,
                &workspace.packed_output,
                &workspace.clamped_count,
                capacity,
            )?;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Portable SoA Unpack"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.unpack);
            pass.set_bind_group(0, &bindings.unpack, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        }
        Ok(())
    }

    fn validate_inputs<'a>(
        &self,
        keys: &'a wgpu::Buffer,
        values: &'a wgpu::Buffer,
        count: &'a wgpu::Buffer,
        count_word: u32,
        capacity: u32,
    ) -> Result<Option<(BufferRange<'a>, BufferRange<'a>, BufferRange<'a>, u64)>, Error> {
        for (first, first_name, second, second_name) in [
            (keys, "sort keys", values, "sort values"),
            (keys, "sort keys", count, "sort item count"),
            (values, "sort values", count, "sort item count"),
        ] {
            if first == second {
                return Err(Error::BufferAlias {
                    first: first_name,
                    second: second_name,
                });
            }
        }
        let count_bytes = u64::from(count_word)
            .checked_add(1)
            .and_then(|words| words.checked_mul(U32_SIZE_BYTES))
            .ok_or(Error::SizeOverflow)?;
        let item_bytes = common::math::checked_byte_size(u64::from(capacity), U32_SIZE_BYTES)?;
        let packed_bytes =
            common::math::checked_byte_size(u64::from(capacity), KEY_VALUE_SIZE_BYTES)?;
        let keys = BufferRange::whole(keys);
        let values = BufferRange::whole(values);
        let count = BufferRange::whole(count);
        keys.validate("sort keys", item_bytes, wgpu::BufferUsages::STORAGE)?;
        values.validate("sort values", item_bytes, wgpu::BufferUsages::STORAGE)?;
        count.validate("sort item count", count_bytes, wgpu::BufferUsages::STORAGE)?;
        for (range, name, size) in [
            (keys, "sort keys", item_bytes),
            (values, "sort values", item_bytes),
            (count, "sort item count", count_bytes),
        ] {
            range.validate_storage_offset(&self.device, name)?;
            range.validate_storage_binding_size(&self.device, size)?;
        }
        common::buffers::validate_storage_binding_size(&self.device, packed_bytes)?;
        if capacity == 0 {
            Ok(None)
        } else {
            Ok(Some((keys, values, count, count_bytes)))
        }
    }

    fn ensure_workspace(&mut self, requested_bytes: u64) -> Result<(), Error> {
        if self
            .workspace
            .as_ref()
            .is_some_and(|workspace| workspace.capacity_bytes >= requested_bytes)
        {
            return Ok(());
        }
        let capacity_bytes = workspace_capacity(requested_bytes)?;
        common::buffers::validate_storage_binding_size(&self.device, capacity_bytes)?;
        self.workspace = Some(PortableWorkspace {
            capacity_bytes,
            packed_input: common::buffers::create_empty_storage_buffer(
                &self.device,
                capacity_bytes,
            ),
            packed_output: common::buffers::create_empty_storage_buffer(
                &self.device,
                capacity_bytes,
            ),
            clamped_count: common::buffers::create_empty_storage_buffer(
                &self.device,
                U32_SIZE_BYTES,
            ),
        });
        self.bindings = None;
        Ok(())
    }

    fn bindings_match(
        &self,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        count: &wgpu::Buffer,
        count_word: u32,
        capacity: u32,
    ) -> bool {
        let workspace_capacity = self
            .workspace
            .as_ref()
            .map_or(0, |workspace| workspace.capacity_bytes);
        self.bindings.as_ref().is_some_and(|bindings| {
            bindings.keys == *keys
                && bindings.values == *values
                && bindings.count == *count
                && bindings.count_word == count_word
                && bindings.capacity == capacity
                && bindings.workspace_capacity == workspace_capacity
        })
    }
}

fn bridge_layout(device: &wgpu::Device, label: &'static str, pack: bool) -> wgpu::BindGroupLayout {
    let mut entries = vec![
        common::buffers::bind_entry(0, true, false),
        common::buffers::bind_entry(1, pack, false),
        common::buffers::bind_entry(2, false, false),
        common::buffers::bind_entry(3, true, false),
        common::buffers::bind_entry(4, false, true),
    ];
    if pack {
        entries.push(common::buffers::bind_entry(5, false, false));
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn bridge_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source: &str,
    label: &'static str,
) -> wgpu::ComputePipeline {
    common::shader::create_compute_pipeline(device, layout, source, label, "main", None)
}

fn bridge_entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

fn bridge_dispatch(capacity: u32) -> (u32, u32) {
    let groups = capacity.div_ceil(BRIDGE_BLOCK_SIZE);
    let groups_x = groups.min(MAX_WORKGROUPS_X);
    (groups_x, groups.div_ceil(groups_x))
}

fn workspace_capacity(requested: u64) -> Result<u64, Error> {
    if requested < WORKSPACE_GROWTH_BYTES {
        requested
            .max(KEY_VALUE_SIZE_BYTES)
            .checked_next_power_of_two()
            .ok_or(Error::SizeOverflow)
    } else {
        common::math::checked_align_to(requested, WORKSPACE_GROWTH_BYTES)
    }
}

#[cfg(test)]
mod tests {
    use super::{BRIDGE_BLOCK_SIZE, KeyValueSoaRequirements, bridge_dispatch, workspace_capacity};

    #[test]
    fn bridge_dispatch_covers_large_capacities() {
        assert_eq!(bridge_dispatch(1), (1, 1));
        assert_eq!(bridge_dispatch(BRIDGE_BLOCK_SIZE * 65_535), (65_535, 1));
        assert_eq!(bridge_dispatch(BRIDGE_BLOCK_SIZE * 65_535 + 1), (65_535, 2));
    }

    #[test]
    fn bridge_workspace_never_allocates_zero_bytes() {
        assert_eq!(workspace_capacity(0).unwrap(), 8);
        assert_eq!(workspace_capacity(9).unwrap(), 16);
    }

    #[test]
    fn requirements_merge_without_lowering_unrelated_application_limits() {
        let requirements = KeyValueSoaRequirements {
            additional_features: wgpu::Features::SUBGROUP,
            min_compute_invocations: 256,
            min_compute_workgroup_size_x: 256,
            min_compute_workgroup_storage_size: 16_388,
            accelerated: true,
        };
        let application_limits = wgpu::Limits {
            max_buffer_size: 987_654_321,
            max_compute_invocations_per_workgroup: 128,
            max_compute_workgroup_size_x: 128,
            max_compute_workgroup_storage_size: 16_384,
            ..wgpu::Limits::default()
        };

        let merged_features = requirements.features(wgpu::Features::TIMESTAMP_QUERY);
        let merged_limits = requirements.limits(application_limits);

        assert!(merged_features.contains(wgpu::Features::TIMESTAMP_QUERY));
        assert!(merged_features.contains(wgpu::Features::SUBGROUP));
        assert_eq!(merged_limits.max_buffer_size, 987_654_321);
        assert_eq!(merged_limits.max_compute_invocations_per_workgroup, 256);
        assert_eq!(merged_limits.max_compute_workgroup_size_x, 256);
        assert_eq!(merged_limits.max_compute_workgroup_storage_size, 16_388);
    }
}
