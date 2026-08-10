use wgpu::util::DeviceExt;

use crate::{
    Error, common,
    common::buffers::BufferRange,
    reduce::counted::{ITEMS_PER_BLOCK as REDUCTION_ITEMS_PER_BLOCK, reduction_pass_count},
    sort::counted::items_per_block as sort_items_per_block,
};

const VALUE_SIZE_BYTES: u64 = size_of::<u32>() as u64;
const DISPATCH_WORDS: u64 = 3;
const PLAN_WORDS: u64 = 2;
const CONFIG_WORDS: u64 = 8;
const MAX_WORKGROUPS_X: u32 = 65_535;

/// How a [`GpuCountPlan`] schedules counted radix-sort workgroups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CountedSortDispatch {
    /// Reads dispatch dimensions from the GPU count.
    ///
    /// This is the safe default for unknown or sparse output sizes because its
    /// radix reduce/scatter launches cover only selected values. The histogram
    /// scan remains capacity-sized.
    #[default]
    Indirect,
    /// Dispatches for the full capacity and lets inactive workgroups return early.
    ///
    /// This can avoid repeated indirect-dispatch overhead when the selected
    /// count is usually dense, but wastes work when few values are selected.
    Capacity,
}

/// Reusable GPU metadata for multiple primitives consuming one resident count.
///
/// A plan is tied to one count buffer and one capacity. It clones the ref-counted
/// wgpu buffer handle, not its contents. Record [`Self::record_prepare`] after
/// the GPU producer writes the count, then pass the plan to counted sort and
/// reduction in the same command encoder. This shares one preparation dispatch
/// across both consumers and avoids a host count readback.
pub struct GpuCountPlan {
    count: wgpu::Buffer,
    count_offset: u64,
    capacity: u32,
    sort_dispatch: CountedSortDispatch,
    sort_items_per_block: u32,
    reduction_pass_count: u32,
    plan_stride: u64,
    pipeline: wgpu::ComputePipeline,
    prepare_bind_group: wgpu::BindGroup,
    _config: wgpu::Buffer,
    sort_dispatch_args: wgpu::Buffer,
    reduction_plans: wgpu::Buffer,
    reduction_dispatch_args: wgpu::Buffer,
}

impl GpuCountPlan {
    /// Creates reusable metadata using count-proportional sort dispatch.
    pub fn new(device: &wgpu::Device, count: &wgpu::Buffer, capacity: u32) -> Result<Self, Error> {
        Self::new_with_sort_dispatch(device, count, capacity, CountedSortDispatch::Indirect)
    }

    /// Creates reusable metadata with an explicit counted-sort dispatch strategy.
    pub fn new_with_sort_dispatch(
        device: &wgpu::Device,
        count: &wgpu::Buffer,
        capacity: u32,
        sort_dispatch: CountedSortDispatch,
    ) -> Result<Self, Error> {
        Self::new_with_count_range(device, BufferRange::whole(count), capacity, sort_dispatch)
    }

    pub(crate) fn new_with_count_range(
        device: &wgpu::Device,
        count: BufferRange<'_>,
        capacity: u32,
        sort_dispatch: CountedSortDispatch,
    ) -> Result<Self, Error> {
        count.validate(
            "GPU item count",
            VALUE_SIZE_BYTES,
            wgpu::BufferUsages::STORAGE,
        )?;
        count.validate_storage_offset(device, "GPU item count")?;
        let limits = device.limits();
        let sort_items_per_block = sort_items_per_block(&limits);
        let reduction_pass_count = reduction_pass_count(capacity);
        let alignment =
            u64::from(device.limits().min_storage_buffer_offset_alignment).max(VALUE_SIZE_BYTES);
        let plan_stride = common::math::checked_align_to(PLAN_WORDS * VALUE_SIZE_BYTES, alignment)?;
        let plan_bytes = u64::from(reduction_pass_count.max(1))
            .checked_mul(plan_stride)
            .ok_or(Error::SizeOverflow)?;
        let reduction_args_bytes = common::math::checked_byte_size(
            u64::from(reduction_pass_count.max(1)) * DISPATCH_WORDS,
            VALUE_SIZE_BYTES,
        )?;
        validate_storage_size(device, plan_bytes)?;
        validate_storage_size(device, reduction_args_bytes)?;

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU Count Plan Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, false, false),
                common::buffers::bind_entry(3, false, false),
                common::buffers::bind_entry(4, false, true),
            ],
        });
        let pipeline = common::shader::create_compute_pipeline(
            device,
            &layout,
            include_str!("count_prepare.wgsl"),
            "GPU Count Plan Preparation Pipeline",
            "main",
            None,
        );
        let config = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU Count Plan Configuration"),
            contents: bytemuck::cast_slice(&[
                capacity,
                sort_items_per_block,
                reduction_pass_count,
                REDUCTION_ITEMS_PER_BLOCK,
                (plan_stride / VALUE_SIZE_BYTES) as u32,
                MAX_WORKGROUPS_X,
                0,
                0,
            ]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        debug_assert_eq!(config.size(), CONFIG_WORDS * VALUE_SIZE_BYTES);
        let sort_dispatch_args = create_buffer(
            device,
            "GPU Count Plan Sort Dispatch",
            DISPATCH_WORDS * VALUE_SIZE_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        );
        let reduction_plans = create_buffer(
            device,
            "GPU Count Plan Reduction Levels",
            plan_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let reduction_dispatch_args = create_buffer(
            device,
            "GPU Count Plan Reduction Dispatches",
            reduction_args_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
        );
        let prepare_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU Count Plan Preparation Bind Group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: count.binding(VALUE_SIZE_BYTES),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sort_dispatch_args.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: reduction_plans.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: reduction_dispatch_args.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: config.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            count: count.buffer.clone(),
            count_offset: count.offset,
            capacity,
            sort_dispatch,
            sort_items_per_block,
            reduction_pass_count,
            plan_stride,
            pipeline,
            prepare_bind_group,
            _config: config,
            sort_dispatch_args,
            reduction_plans,
            reduction_dispatch_args,
        })
    }

    /// Records one dispatch that clamps the associated count and prepares all consumers.
    ///
    /// Recording does not submit or wait; command order makes later plan
    /// consumers observe the generated metadata.
    pub fn record_prepare(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.capacity == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Prepare GPU Count Plan"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.prepare_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Returns the maximum number of items covered by this plan.
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    pub(crate) fn count(&self) -> BufferRange<'_> {
        BufferRange {
            buffer: &self.count,
            offset: self.count_offset,
            size: VALUE_SIZE_BYTES,
        }
    }

    pub(crate) const fn sort_dispatch(&self) -> CountedSortDispatch {
        self.sort_dispatch
    }

    pub(crate) const fn sort_items_per_block(&self) -> u32 {
        self.sort_items_per_block
    }

    pub(crate) const fn reduction_pass_count(&self) -> u32 {
        self.reduction_pass_count
    }

    pub(crate) const fn plan_stride(&self) -> u64 {
        self.plan_stride
    }

    pub(crate) const fn sort_dispatch_args(&self) -> &wgpu::Buffer {
        &self.sort_dispatch_args
    }

    pub(crate) const fn reduction_plans(&self) -> &wgpu::Buffer {
        &self.reduction_plans
    }

    pub(crate) const fn reduction_dispatch_args(&self) -> &wgpu::Buffer {
        &self.reduction_dispatch_args
    }
}

fn validate_storage_size(device: &wgpu::Device, requested: u64) -> Result<(), Error> {
    let limits = device.limits();
    let limit = limits
        .max_buffer_size
        .min(limits.max_storage_buffer_binding_size);
    if requested > limit {
        Err(Error::BufferLimitExceeded { requested, limit })
    } else {
        Ok(())
    }
}

fn create_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}
