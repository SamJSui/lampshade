use std::sync::{Arc, Mutex};

use crate::{
    KeyValueField, U32Predicate,
    common::{self, buffers::BufferRange},
    profiling,
};
use wgpu::util::DeviceExt;

const BLOCK_SIZE: u32 = 256;
const PARAMS_SIZE_BYTES: u64 = 32;

#[derive(Clone, Copy)]
pub(crate) enum PredicateItemKind {
    Value,
    KeyValue,
}

impl PredicateItemKind {
    pub(crate) const fn size_bytes(self) -> u64 {
        match self {
            Self::Value => size_of::<u32>() as u64,
            Self::KeyValue => size_of::<crate::KeyValue>() as u64,
        }
    }

    const fn shader_item_type(self) -> &'static str {
        match self {
            Self::Value => "u32",
            Self::KeyValue => "KeyValue",
        }
    }

    const fn shader_value_expression(self) -> &'static str {
        match self {
            Self::Value => "item",
            Self::KeyValue => "select(item.key, item.value, params.field == 1u)",
        }
    }
}

pub(crate) struct PredicatePipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    max_workgroups_per_dimension: u32,
    params: Arc<Mutex<ParameterPool>>,
}

#[derive(Default)]
struct ParameterPool {
    slots: Vec<ParameterSlot>,
}

struct ParameterSlot {
    buffer: wgpu::Buffer,
    in_use: bool,
}

struct ParameterLease {
    pool: Arc<Mutex<ParameterPool>>,
    slot: usize,
}

impl Drop for ParameterLease {
    fn drop(&mut self) {
        let mut pool = self
            .pool
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        pool.slots[self.slot].in_use = false;
    }
}

pub(crate) struct PredicateDispatch<'a> {
    pub input: BufferRange<'a>,
    pub mask: BufferRange<'a>,
    pub num_items: u32,
    pub field: KeyValueField,
    pub predicate: U32Predicate,
}

impl PredicatePipeline {
    pub(crate) fn new(device: &wgpu::Device, item_kind: PredicateItemKind) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Predicate Mask Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, false, true),
            ],
        });
        let shader_source = include_str!("predicate.wgsl")
            .replace("{{ITEM_TYPE}}", item_kind.shader_item_type())
            .replace("{{VALUE_EXPRESSION}}", item_kind.shader_value_expression());
        let pipeline = common::shader::create_compute_pipeline(
            device,
            &bind_group_layout,
            &shader_source,
            "Predicate Mask Pipeline",
            "main",
            None,
        );

        Self {
            bind_group_layout,
            pipeline,
            max_workgroups_per_dimension: device.limits().max_compute_workgroups_per_dimension,
            params: Arc::new(Mutex::new(ParameterPool::default())),
        }
    }

    pub(crate) fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dispatch: PredicateDispatch<'_>,
        profiler: Option<&mut profiling::TimestampRecorder>,
    ) {
        let workgroups = common::math::calc_groups(dispatch.num_items, BLOCK_SIZE);
        let groups_x = workgroups.min(self.max_workgroups_per_dimension);
        let groups_y = workgroups.div_ceil(self.max_workgroups_per_dimension);
        let (operation, lower, upper) = dispatch.predicate.encode();
        let params_data = [
            dispatch.num_items,
            groups_x,
            operation,
            dispatch.field.encode(),
            lower,
            upper,
            0_u32,
            0,
        ];
        let (params, lease) =
            self.acquire_params(device, queue, bytemuck::cast_slice(&params_data));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Predicate Mask Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dispatch.input.binding(dispatch.input.size),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dispatch.mask.binding(dispatch.mask.size),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params,
                        offset: 0,
                        size: wgpu::BufferSize::new(PARAMS_SIZE_BYTES),
                    }),
                },
            ],
        });
        profiling::record_compute_pass(
            encoder,
            "Predicate Mask",
            profiler.is_some().then(|| "predicate.mask".to_owned()),
            profiler,
            |pass| {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups_x, groups_y, 1);
            },
        );
        // Keep transient bindings alive through asynchronous execution. This
        // explicit ownership also avoids premature handle reuse on the Jetson
        // Vulkan path while still releasing resources when this submission ends.
        encoder.on_submitted_work_done(move || drop((bind_group, lease)));
    }

    fn acquire_params(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        contents: &[u8],
    ) -> (wgpu::Buffer, ParameterLease) {
        let mut pool = self
            .params
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let slot = pool.slots.iter().position(|slot| !slot.in_use);
        let slot = match slot {
            Some(slot) => {
                pool.slots[slot].in_use = true;
                queue.write_buffer(&pool.slots[slot].buffer, 0, contents);
                slot
            }
            None => {
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Predicate Mask Parameters"),
                    contents,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                pool.slots.push(ParameterSlot {
                    buffer,
                    in_use: true,
                });
                pool.slots.len() - 1
            }
        };
        let buffer = pool.slots[slot].buffer.clone();
        drop(pool);
        (
            buffer,
            ParameterLease {
                pool: Arc::clone(&self.params),
                slot,
            },
        )
    }
}
