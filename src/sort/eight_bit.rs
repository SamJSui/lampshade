use wgpu::util::DeviceExt;

use crate::Error;
use crate::common;
use crate::common::runtime::{CommandSession, ProfileSession};
use crate::profiling::{self, GpuProfile, TimestampRecorder};

const BLOCK_SIZE: u32 = 256;
const ITEMS_PER_THREAD: u32 = 7;
const ITEMS_PER_TILE: u32 = BLOCK_SIZE * ITEMS_PER_THREAD;
const RADIX_BITS: u32 = 8;
const BUCKET_COUNT: u32 = 256;
const MAX_PASS_COUNT: u32 = u32::BITS / RADIX_BITS;
const TILE_COUNTER_COUNT: u64 = MAX_PASS_COUNT as u64;
const MAX_HISTOGRAM_GROUPS: u32 = 2048;
const MAX_PACKED_COUNT: u32 = 0x0fff_ffff;
const ITEM_SIZE_BYTES: u64 = 8;
const UNIFORM_SIZE_BYTES: u64 = 32;
const DISPATCH_ARGS_SIZE_BYTES: u64 = MAX_PASS_COUNT as u64 * 3 * 4;
const WORKSPACE_GROWTH_BYTES: u64 = 16 * 1024 * 1024;

struct EightBitWorkspace {
    capacity_bytes: u64,
    scratch: wgpu::Buffer,
    histogram: wgpu::Buffer,
    offsets: wgpu::Buffer,
    partition_state: wgpu::Buffer,
    dispatch_args: wgpu::Buffer,
}

#[derive(Clone, Copy)]
struct PreparedSort {
    num_items: u32,
    num_tiles: u32,
}

struct CachedBindings {
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    num_items: u32,
    pass_count: u32,
    workspace_capacity: u64,
    _uniform: wgpu::Buffer,
    histogram: wgpu::BindGroup,
    prefix: wgpu::BindGroup,
    scatter: Vec<wgpu::BindGroup>,
}

pub struct EightBitSorter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: EightBitPipelines,
    workspace: Option<EightBitWorkspace>,
    cached_bindings: Option<CachedBindings>,
}

impl EightBitSorter {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        debug_assert!(device.features().contains(wgpu::Features::SUBGROUP));
        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipelines: EightBitPipelines::new(device),
            workspace: None,
            cached_bindings: None,
        }
    }

    pub async fn sort_slice<T: bytemuck::Pod>(
        &mut self,
        input: &[T],
        key_bits: u32,
    ) -> Result<Vec<T>, Error> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        assert_eq!(size_of::<T>() as u64, ITEM_SIZE_BYTES);
        let num_items = common::math::checked_u32(input.len() as u64)?;
        let size_bytes = common::math::checked_byte_size(input.len() as u64, ITEM_SIZE_BYTES)?;
        let input_buffer = common::buffers::create_storage_buffer(&self.device, input);
        let output_buffer = common::buffers::create_empty_storage_buffer(&self.device, size_bytes);

        self.sort_gpu_to_gpu(&input_buffer, &output_buffer, num_items, key_bits)?;
        common::buffers::download_buffer(&self.device, &self.queue, &output_buffer, input.len())
            .await
    }

    pub fn sort_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        let mut commands = CommandSession::new(&self.device, Some("8-bit Radix Sort"));
        self.record_sort(commands.encoder(), input, output, num_items, key_bits)?;
        commands.submit(&self.queue);
        Ok(())
    }

    pub fn record_sort(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        let Some(problem) = self.prepare_sort(input, output, num_items)? else {
            return Ok(());
        };
        let pass_count = pass_count_for_key_bits(key_bits);
        self.record_commands(encoder, input, output, problem, pass_count, None)
    }

    pub async fn profile_sort_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        key_bits: u32,
    ) -> Result<GpuProfile, Error> {
        let Some(problem) = self.prepare_sort(input, output, num_items)? else {
            return Ok(GpuProfile::empty());
        };

        let pass_count = pass_count_for_key_bits(key_bits);
        let mut profile = ProfileSession::new(
            &self.device,
            &self.queue,
            pass_count + 2,
            "Profiled 8-bit Radix Sort",
        )?;
        let (encoder, profiler) = profile.recording();
        self.record_commands(encoder, input, output, problem, pass_count, profiler)?;
        profile.finish(&self.device, &self.queue).await
    }

    fn prepare_sort(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<Option<PreparedSort>, Error> {
        if num_items == 0 {
            return Ok(None);
        }
        if input == output {
            return Err(Error::BufferAlias {
                first: "sort input",
                second: "sort output",
            });
        }
        let max_items =
            element_count_limit(self.device.limits().max_compute_workgroups_per_dimension);
        if num_items > max_items {
            return Err(Error::RadixElementCountLimitExceeded {
                count: num_items,
                limit: max_items,
            });
        }

        let size_bytes = common::math::checked_byte_size(u64::from(num_items), ITEM_SIZE_BYTES)?;
        common::buffers::validate_buffer(
            input,
            "sort input",
            size_bytes,
            wgpu::BufferUsages::STORAGE,
        )?;
        common::buffers::validate_buffer(
            output,
            "sort output",
            size_bytes,
            wgpu::BufferUsages::STORAGE,
        )?;
        self.ensure_workspace(size_bytes)?;

        Ok(Some(PreparedSort {
            num_items,
            num_tiles: num_items.div_ceil(ITEMS_PER_TILE),
        }))
    }

    fn record_commands(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        problem: PreparedSort,
        pass_count: u32,
        mut profiler: Option<&mut TimestampRecorder>,
    ) -> Result<(), Error> {
        self.ensure_bindings(input, output, problem, pass_count);
        let workspace = self.workspace.as_ref().expect("sort workspace is prepared");
        let bindings = self
            .cached_bindings
            .as_ref()
            .expect("sort bindings are prepared");

        encoder.clear_buffer(&workspace.histogram, 0, None);
        encoder.clear_buffer(&workspace.partition_state, 0, None);

        let histogram_groups = problem.num_tiles.min(MAX_HISTOGRAM_GROUPS);
        profiling::record_compute_pass(
            encoder,
            "8-bit Radix Histogram",
            profiler.is_some().then(|| "radix.histogram".to_owned()),
            profiler.as_deref_mut(),
            |pass| {
                pass.set_pipeline(&self.pipelines.histogram);
                pass.set_bind_group(0, &bindings.histogram, &[]);
                pass.dispatch_workgroups(histogram_groups, 1, 1);
            },
        );
        profiling::record_compute_pass(
            encoder,
            "8-bit Radix Prefix",
            profiler.is_some().then(|| "radix.prefix".to_owned()),
            profiler.as_deref_mut(),
            |pass| {
                pass.set_pipeline(&self.pipelines.prefix);
                pass.set_bind_group(0, &bindings.prefix, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            },
        );

        for (radix_pass, bind_group) in bindings.scatter.iter().enumerate() {
            profiling::record_compute_pass(
                encoder,
                "8-bit Radix Scatter",
                profiler
                    .is_some()
                    .then(|| format!("radix.{radix_pass:02}.scatter")),
                profiler.as_deref_mut(),
                |pass| {
                    pass.set_pipeline(&self.pipelines.scatter);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups_indirect(
                        &workspace.dispatch_args,
                        radix_pass as u64 * 3 * 4,
                    );
                },
            );
        }
        Ok(())
    }

    fn ensure_bindings(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        problem: PreparedSort,
        pass_count: u32,
    ) {
        let workspace = self.workspace.as_ref().expect("sort workspace is prepared");
        let matches = self.cached_bindings.as_ref().is_some_and(|bindings| {
            bindings.input == *input
                && bindings.output == *output
                && bindings.num_items == problem.num_items
                && bindings.pass_count == pass_count
                && bindings.workspace_capacity == workspace.capacity_bytes
        });
        if matches {
            return;
        }

        let (uniform, uniform_stride) = create_uniform_buffer(&self.device, problem, pass_count);
        let histogram = self.pipelines.create_histogram_bind_group(
            &self.device,
            input,
            &workspace.histogram,
            &uniform,
        );
        let prefix = self.pipelines.create_prefix_bind_group(
            &self.device,
            &workspace.histogram,
            &workspace.offsets,
            &workspace.dispatch_args,
            &uniform,
        );
        let scatter = (0..pass_count)
            .map(|pass| {
                let (source, destination) =
                    pass_buffers(pass, pass_count, input, output, &workspace.scratch);
                self.pipelines.create_scatter_bind_group(
                    &self.device,
                    source,
                    destination,
                    workspace,
                    &uniform,
                    u64::from(pass) * uniform_stride,
                )
            })
            .collect();
        self.cached_bindings = Some(CachedBindings {
            input: input.clone(),
            output: output.clone(),
            num_items: problem.num_items,
            pass_count,
            workspace_capacity: workspace.capacity_bytes,
            _uniform: uniform,
            histogram,
            prefix,
            scatter,
        });
    }

    fn ensure_workspace(&mut self, requested_size: u64) -> Result<(), Error> {
        let needs_allocation = self
            .workspace
            .as_ref()
            .is_none_or(|workspace| workspace.capacity_bytes < requested_size);
        if !needs_allocation {
            return Ok(());
        }

        let capacity = workspace_capacity(requested_size)?;
        let max_items = capacity / ITEM_SIZE_BYTES;
        let max_tiles = max_items.div_ceil(u64::from(ITEMS_PER_TILE));
        let partition_entries = max_tiles
            .checked_mul(u64::from(BUCKET_COUNT))
            .and_then(|entries| entries.checked_add(TILE_COUNTER_COUNT))
            .ok_or(Error::SizeOverflow)?;
        let partition_bytes = common::math::checked_align_to(
            common::math::checked_byte_size(partition_entries, 4)?,
            256,
        )?;
        let limits = self.device.limits();
        let buffer_limit = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));
        for requested in [capacity, partition_bytes] {
            if requested > buffer_limit {
                return Err(Error::BufferLimitExceeded {
                    requested,
                    limit: buffer_limit,
                });
            }
        }

        let digit_table_bytes = u64::from(MAX_PASS_COUNT * BUCKET_COUNT) * 4;
        self.workspace = Some(EightBitWorkspace {
            capacity_bytes: capacity,
            scratch: common::buffers::create_empty_storage_buffer(&self.device, capacity),
            histogram: common::buffers::create_empty_storage_buffer(
                &self.device,
                digit_table_bytes,
            ),
            offsets: common::buffers::create_empty_storage_buffer(&self.device, digit_table_bytes),
            partition_state: common::buffers::create_empty_storage_buffer(
                &self.device,
                partition_bytes,
            ),
            dispatch_args: self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("8-bit Radix Dispatch Arguments"),
                size: DISPATCH_ARGS_SIZE_BYTES,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
        });
        self.cached_bindings = None;
        Ok(())
    }
}

struct EightBitPipelines {
    histogram_layout: wgpu::BindGroupLayout,
    prefix_layout: wgpu::BindGroupLayout,
    scatter_layout: wgpu::BindGroupLayout,
    histogram: wgpu::ComputePipeline,
    prefix: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
}

impl EightBitPipelines {
    fn new(device: &wgpu::Device) -> Self {
        let histogram_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("8-bit Histogram Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, false, true),
            ],
        });
        let prefix_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("8-bit Prefix Layout"),
            entries: &[
                common::buffers::bind_entry(0, false, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, false, false),
                common::buffers::bind_entry(3, false, true),
            ],
        });
        let scatter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("8-bit Scatter Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, true, false),
                common::buffers::bind_entry(3, false, false),
                common::buffers::bind_entry(4, false, true),
            ],
        });

        let histogram_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("8-bit Histogram Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("histogram_8bit.wgsl").into()),
        });
        let prefix_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("8-bit Prefix Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("prefix_8bit.wgsl").into()),
        });
        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("8-bit Scatter Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scatter_8bit.wgsl").into()),
        });

        Self {
            histogram: create_pipeline(
                device,
                "8-bit Histogram Pipeline",
                &histogram_layout,
                &histogram_shader,
                "main_histogram",
            ),
            prefix: create_pipeline(
                device,
                "8-bit Prefix Pipeline",
                &prefix_layout,
                &prefix_shader,
                "main_prefix",
            ),
            scatter: create_pipeline(
                device,
                "8-bit Scatter Pipeline",
                &scatter_layout,
                &scatter_shader,
                "main_scatter",
            ),
            histogram_layout,
            prefix_layout,
            scatter_layout,
        }
    }

    fn create_histogram_bind_group(
        &self,
        device: &wgpu::Device,
        input: &wgpu::Buffer,
        histogram: &wgpu::Buffer,
        uniform: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        create_bind_group(
            device,
            &self.histogram_layout,
            "8-bit Histogram Bind Group",
            &[
                entire_buffer(0, input),
                entire_buffer(1, histogram),
                uniform_binding(2, uniform, 0),
            ],
        )
    }

    fn create_prefix_bind_group(
        &self,
        device: &wgpu::Device,
        histogram: &wgpu::Buffer,
        offsets: &wgpu::Buffer,
        dispatch_args: &wgpu::Buffer,
        uniform: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        create_bind_group(
            device,
            &self.prefix_layout,
            "8-bit Prefix Bind Group",
            &[
                entire_buffer(0, histogram),
                entire_buffer(1, offsets),
                entire_buffer(2, dispatch_args),
                uniform_binding(3, uniform, 0),
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_scatter_bind_group(
        &self,
        device: &wgpu::Device,
        source: &wgpu::Buffer,
        destination: &wgpu::Buffer,
        workspace: &EightBitWorkspace,
        uniform: &wgpu::Buffer,
        uniform_offset: u64,
    ) -> wgpu::BindGroup {
        create_bind_group(
            device,
            &self.scatter_layout,
            "8-bit Scatter Bind Group",
            &[
                entire_buffer(0, source),
                entire_buffer(1, destination),
                entire_buffer(2, &workspace.offsets),
                entire_buffer(3, &workspace.partition_state),
                uniform_binding(4, uniform, uniform_offset),
            ],
        )
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    bind_group_layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
    entry_point: &'static str,
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bind_group_layout],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn create_uniform_buffer(
    device: &wgpu::Device,
    problem: PreparedSort,
    pass_count: u32,
) -> (wgpu::Buffer, u64) {
    let stride =
        u64::from(device.limits().min_uniform_buffer_offset_alignment).max(UNIFORM_SIZE_BYTES);
    let words_per_record = (stride / size_of::<u32>() as u64) as usize;
    let mut data = vec![0_u32; words_per_record * pass_count as usize];
    for radix_pass in 0..pass_count as usize {
        let offset = radix_pass * words_per_record;
        data[offset..offset + 5].copy_from_slice(&[
            problem.num_items,
            problem.num_tiles,
            radix_pass as u32 + 1,
            radix_pass as u32 * RADIX_BITS,
            pass_count,
        ]);
    }
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("8-bit Sort Uniform"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    (buffer, stride)
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    label: &'static str,
    entries: &[wgpu::BindGroupEntry<'_>],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries,
    })
}

fn entire_buffer(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn uniform_binding(binding: u32, buffer: &wgpu::Buffer, offset: u64) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset,
            size: wgpu::BufferSize::new(UNIFORM_SIZE_BYTES),
        }),
    }
}

fn pass_buffers<'a>(
    radix_pass: u32,
    pass_count: u32,
    input: &'a wgpu::Buffer,
    output: &'a wgpu::Buffer,
    scratch: &'a wgpu::Buffer,
) -> (&'a wgpu::Buffer, &'a wgpu::Buffer) {
    let (source_slot, destination_slot) = pass_buffer_slots(radix_pass, pass_count);
    let source = match source_slot {
        BufferSlot::Input => input,
        BufferSlot::Output => output,
        BufferSlot::Scratch => scratch,
    };
    let destination = match destination_slot {
        BufferSlot::Input => unreachable!("sort input is never a pass destination"),
        BufferSlot::Output => output,
        BufferSlot::Scratch => scratch,
    };
    (source, destination)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BufferSlot {
    Input,
    Output,
    Scratch,
}

fn pass_buffer_slots(radix_pass: u32, pass_count: u32) -> (BufferSlot, BufferSlot) {
    debug_assert!(radix_pass < pass_count);
    let source = if radix_pass == 0 {
        BufferSlot::Input
    } else if (pass_count - radix_pass).is_multiple_of(2) {
        BufferSlot::Output
    } else {
        BufferSlot::Scratch
    };
    let passes_after = pass_count - radix_pass - 1;
    let destination = if passes_after.is_multiple_of(2) {
        BufferSlot::Output
    } else {
        BufferSlot::Scratch
    };
    (source, destination)
}

fn pass_count_for_key_bits(key_bits: u32) -> u32 {
    debug_assert!(key_bits <= u32::BITS);
    key_bits.max(1).div_ceil(RADIX_BITS)
}

fn element_count_limit(max_compute_workgroups_per_dimension: u32) -> u32 {
    max_compute_workgroups_per_dimension
        .saturating_mul(ITEMS_PER_TILE)
        .min(MAX_PACKED_COUNT)
}

fn workspace_capacity(requested_size: u64) -> Result<u64, Error> {
    if requested_size < WORKSPACE_GROWTH_BYTES {
        requested_size
            .max(ITEM_SIZE_BYTES)
            .checked_next_power_of_two()
            .ok_or(Error::SizeOverflow)
    } else {
        common::math::checked_align_to(requested_size, WORKSPACE_GROWTH_BYTES)
    }
}

#[cfg(test)]
mod tests {
    use super::{BufferSlot, element_count_limit, pass_buffer_slots, pass_count_for_key_bits};

    #[test]
    fn maps_key_widths_to_active_byte_passes() {
        for (key_bits, expected) in [
            (0, 1),
            (1, 1),
            (8, 1),
            (9, 2),
            (16, 2),
            (17, 3),
            (24, 3),
            (25, 4),
            (32, 4),
        ] {
            assert_eq!(pass_count_for_key_bits(key_bits), expected);
        }
    }

    #[test]
    fn routes_every_pass_count_to_the_caller_output() {
        use BufferSlot::{Input, Output, Scratch};

        let expected = [
            vec![(Input, Output)],
            vec![(Input, Scratch), (Scratch, Output)],
            vec![(Input, Output), (Output, Scratch), (Scratch, Output)],
            vec![
                (Input, Scratch),
                (Scratch, Output),
                (Output, Scratch),
                (Scratch, Output),
            ],
        ];
        for (pass_count, expected_routes) in (1..=4).zip(expected) {
            let actual: Vec<_> = (0..pass_count)
                .map(|radix_pass| pass_buffer_slots(radix_pass, pass_count))
                .collect();
            assert_eq!(actual, expected_routes);
        }
    }

    #[test]
    fn caps_elements_at_the_one_dimensional_dispatch_limit() {
        assert_eq!(element_count_limit(1), 1_792);
        assert_eq!(element_count_limit(65_535), 117_438_720);
        assert_eq!(element_count_limit(u32::MAX), 0x0fff_ffff);
    }
}
