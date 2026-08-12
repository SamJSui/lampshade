use wgpu::util::DeviceExt;

use crate::Error;
use crate::common;
use crate::common::buffers::BufferRange;
use crate::common::runtime::{CommandSession, ProfileSession};
use crate::profiling::{self, GpuProfile, TimestampRecorder};

use super::pipeline::{RadixVariant, SortItemKind};

const BLOCK_SIZE: u32 = 256;
const ITEMS_PER_THREAD: u32 = 7;
const ITEMS_PER_TILE: u32 = BLOCK_SIZE * ITEMS_PER_THREAD;
const RADIX_BITS: u32 = 8;
const BUCKET_COUNT: u32 = 256;
const MAX_PASS_COUNT: u32 = u32::BITS / RADIX_BITS;
const TILE_COUNTER_COUNT: u64 = MAX_PASS_COUNT as u64;
const MAX_HISTOGRAM_GROUPS: u32 = 2048;
const MAX_PACKED_COUNT: u32 = 0x0fff_ffff;
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

struct CountedEightBitState {
    prepare_layout: wgpu::BindGroupLayout,
    prepare: wgpu::ComputePipeline,
    uniforms: wgpu::Buffer,
}

struct SoaWorkspace {
    capacity_bytes: u64,
    scratch_keys: wgpu::Buffer,
    scratch_values: wgpu::Buffer,
    histogram: wgpu::Buffer,
    offsets: wgpu::Buffer,
    partition_state: wgpu::Buffer,
    dispatch_args: wgpu::Buffer,
}

struct SoaBindings {
    keys: wgpu::Buffer,
    values: wgpu::Buffer,
    count: wgpu::Buffer,
    count_word: u32,
    capacity: u32,
    workspace_capacity: u64,
    _params: wgpu::Buffer,
    prepare: wgpu::BindGroup,
    histogram: wgpu::BindGroup,
    prefix: wgpu::BindGroup,
    scatter: Vec<wgpu::BindGroup>,
}

/// Records a stable, GPU-counted sort of separate u32 key and value buffers.
///
/// This backend is available only when Lampshade's validated 8-bit subgroup
/// route is supported. The supplied key and value buffers are sorted in place.
pub(super) struct NativeKeyValueSoaSorter {
    device: wgpu::Device,
    histogram_layout: wgpu::BindGroupLayout,
    prefix_layout: wgpu::BindGroupLayout,
    scatter_layout: wgpu::BindGroupLayout,
    histogram: wgpu::ComputePipeline,
    prefix: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    counted: CountedEightBitState,
    workspace: Option<SoaWorkspace>,
    bindings: Option<SoaBindings>,
}

#[derive(Clone, Copy)]
struct PreparedSort {
    num_items: u32,
    num_tiles: u32,
    size_bytes: u64,
}

#[derive(Clone, Copy)]
struct PreparedCountedSort {
    capacity_bytes: u64,
    count_bytes: u64,
    pass_count: u32,
}

struct CachedBindings {
    input: wgpu::Buffer,
    input_offset: u64,
    output: wgpu::Buffer,
    output_offset: u64,
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
    item_size_bytes: u64,
    workspace: Option<EightBitWorkspace>,
    cached_bindings: Option<CachedBindings>,
    counted: Option<CountedEightBitState>,
}

impl EightBitSorter {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, item_kind: SortItemKind) -> Self {
        debug_assert!(device.features().contains(wgpu::Features::SUBGROUP));
        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipelines: EightBitPipelines::new(device, item_kind),
            item_size_bytes: item_kind.size_bytes(),
            workspace: None,
            cached_bindings: None,
            counted: None,
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

        assert_eq!(size_of::<T>() as u64, self.item_size_bytes);
        let num_items = common::math::checked_u32(input.len() as u64)?;
        let size_bytes = common::math::checked_byte_size(input.len() as u64, self.item_size_bytes)?;
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
        self.record_sort_ranges(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(output),
            num_items,
            key_bits,
        )
    }

    pub(crate) fn record_sort_ranges(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        num_items: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        let Some(problem) = self.prepare_sort(input, output, num_items)? else {
            return Ok(());
        };
        let input = BufferRange {
            size: problem.size_bytes,
            ..input
        };
        let output = BufferRange {
            size: problem.size_bytes,
            ..output
        };
        let pass_count = pass_count_for_key_bits(key_bits);
        self.record_commands(encoder, input, output, problem, pass_count, None)
    }

    pub(crate) fn sort_counted_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        count_word: u32,
        capacity: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        let mut commands = CommandSession::new(&self.device, Some("8-bit Counted Radix Sort"));
        self.record_sort_counted(
            commands.encoder(),
            input,
            output,
            count,
            count_word,
            capacity,
            key_bits,
        )?;
        commands.submit(&self.queue);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_sort_counted(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        count_word: u32,
        capacity: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        self.record_sort_counted_ranges(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(output),
            BufferRange::whole(count),
            count_word,
            capacity,
            key_bits,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_sort_counted_ranges(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        count: BufferRange<'_>,
        count_word: u32,
        capacity: u32,
        key_bits: u32,
    ) -> Result<(), Error> {
        let Some(problem) =
            self.prepare_counted_sort(input, output, count, count_word, capacity, key_bits)?
        else {
            return Ok(());
        };

        let input = BufferRange {
            size: problem.capacity_bytes,
            ..input
        };
        let output = BufferRange {
            size: problem.capacity_bytes,
            ..output
        };
        let count = BufferRange {
            size: problem.count_bytes,
            ..count
        };
        self.record_counted_commands(
            encoder,
            input,
            output,
            count,
            count_word,
            capacity,
            problem.pass_count,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn profile_sort_counted(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        count_word: u32,
        capacity: u32,
        key_bits: u32,
    ) -> Result<GpuProfile, Error> {
        let input = BufferRange::whole(input);
        let output = BufferRange::whole(output);
        let count = BufferRange::whole(count);
        let Some(problem) =
            self.prepare_counted_sort(input, output, count, count_word, capacity, key_bits)?
        else {
            return Ok(GpuProfile::empty());
        };
        let input = BufferRange {
            size: problem.capacity_bytes,
            ..input
        };
        let output = BufferRange {
            size: problem.capacity_bytes,
            ..output
        };
        let count = BufferRange {
            size: problem.count_bytes,
            ..count
        };
        let mut profile = ProfileSession::new(
            &self.device,
            &self.queue,
            problem.pass_count + 3,
            "Profiled 8-bit Counted Radix Sort",
        )?;
        let (encoder, profiler) = profile.recording();
        self.record_counted_commands(
            encoder,
            input,
            output,
            count,
            count_word,
            capacity,
            problem.pass_count,
            profiler,
        )?;
        profile.finish(&self.device, &self.queue).await
    }

    fn prepare_counted_sort(
        &mut self,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        count: BufferRange<'_>,
        count_word: u32,
        capacity: u32,
        key_bits: u32,
    ) -> Result<Option<PreparedCountedSort>, Error> {
        if key_bits > u32::BITS {
            return Err(Error::InvalidKeyBits { bits: key_bits });
        }
        validate_counted_distinct(input, output, count)?;
        let count_bytes = u64::from(count_word)
            .checked_add(1)
            .and_then(|words| words.checked_mul(size_of::<u32>() as u64))
            .ok_or(Error::SizeOverflow)?;
        count.validate("sort item count", count_bytes, wgpu::BufferUsages::STORAGE)?;
        count.validate_storage_offset(&self.device, "sort item count")?;
        count.validate_storage_binding_size(&self.device, count_bytes)?;
        let capacity_bytes =
            common::math::checked_byte_size(u64::from(capacity), self.item_size_bytes)?;
        input.validate("sort input", capacity_bytes, wgpu::BufferUsages::STORAGE)?;
        output.validate("sort output", capacity_bytes, wgpu::BufferUsages::STORAGE)?;
        input.validate_storage_offset(&self.device, "sort input")?;
        output.validate_storage_offset(&self.device, "sort output")?;
        for range in [input, output] {
            range.validate_storage_binding_size(&self.device, capacity_bytes)?;
        }
        if capacity == 0 {
            return Ok(None);
        }
        let max_items =
            element_count_limit(self.device.limits().max_compute_workgroups_per_dimension);
        if capacity > max_items {
            return Err(Error::RadixElementCountLimitExceeded {
                count: capacity,
                limit: max_items,
            });
        }
        self.ensure_workspace(capacity_bytes)?;
        self.ensure_counted_state();
        Ok(Some(PreparedCountedSort {
            capacity_bytes,
            count_bytes,
            pass_count: pass_count_for_key_bits(key_bits),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn record_counted_commands(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        count: BufferRange<'_>,
        count_word: u32,
        capacity: u32,
        pass_count: u32,
        mut profiler: Option<&mut TimestampRecorder>,
    ) -> Result<(), Error> {
        let workspace = self.workspace.as_ref().expect("sort workspace is prepared");
        let counted = self
            .counted
            .as_ref()
            .expect("counted sort pipeline is prepared");
        let params_data = [capacity, pass_count, count_word, 0, 0, 0, 0, 0];
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("8-bit Counted Sort Parameters"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let prepare = counted.create_prepare_bind_group(&self.device, count, &params);
        let histogram = self.pipelines.create_histogram_bind_group(
            &self.device,
            input,
            &workspace.histogram,
            &counted.uniforms,
        );
        let prefix = self.pipelines.create_prefix_bind_group(
            &self.device,
            &workspace.histogram,
            &workspace.offsets,
            &workspace.dispatch_args,
            &counted.uniforms,
        );
        let uniform_stride = u64::from(self.device.limits().min_uniform_buffer_offset_alignment)
            .max(UNIFORM_SIZE_BYTES);
        let scatter: Vec<_> = (0..pass_count)
            .map(|pass| {
                let (source, destination) =
                    pass_buffers(pass, pass_count, input, output, &workspace.scratch);
                self.pipelines.create_scatter_bind_group(
                    &self.device,
                    source,
                    destination,
                    workspace,
                    &counted.uniforms,
                    u64::from(pass) * uniform_stride,
                )
            })
            .collect();

        encoder.clear_buffer(&workspace.histogram, 0, None);
        encoder.clear_buffer(&workspace.partition_state, 0, None);
        profiling::record_compute_pass(
            encoder,
            "8-bit Counted Radix Preparation",
            profiler
                .is_some()
                .then(|| "counted.radix.prepare".to_owned()),
            profiler.as_deref_mut(),
            |pass| {
                pass.set_pipeline(&counted.prepare);
                pass.set_bind_group(0, &prepare, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            },
        );
        profiling::record_compute_pass(
            encoder,
            "8-bit Counted Radix Histogram",
            profiler
                .is_some()
                .then(|| "counted.radix.histogram".to_owned()),
            profiler.as_deref_mut(),
            |pass| {
                pass.set_pipeline(&self.pipelines.histogram);
                pass.set_bind_group(0, &histogram, &[]);
                pass.dispatch_workgroups(
                    capacity.div_ceil(ITEMS_PER_TILE).min(MAX_HISTOGRAM_GROUPS),
                    1,
                    1,
                );
            },
        );
        profiling::record_compute_pass(
            encoder,
            "8-bit Counted Radix Prefix",
            profiler
                .is_some()
                .then(|| "counted.radix.prefix".to_owned()),
            profiler.as_deref_mut(),
            |pass| {
                pass.set_pipeline(&self.pipelines.prefix);
                pass.set_bind_group(0, &prefix, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            },
        );
        for (radix_pass, bind_group) in scatter.iter().enumerate() {
            profiling::record_compute_pass(
                encoder,
                "8-bit Counted Radix Scatter",
                profiler
                    .is_some()
                    .then(|| format!("counted.radix.{radix_pass:02}.scatter")),
                profiler.as_deref_mut(),
                |pass| {
                    pass.set_pipeline(&self.pipelines.scatter);
                    pass.set_bind_group(0, bind_group, &[]);
                    pass.dispatch_workgroups_indirect(
                        &workspace.dispatch_args,
                        radix_pass as u64 * 3 * size_of::<u32>() as u64,
                    );
                },
            );
        }
        encoder.on_submitted_work_done(move || {
            drop((params, prepare, histogram, prefix, scatter));
        });
        Ok(())
    }

    pub async fn profile_sort_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        key_bits: u32,
    ) -> Result<GpuProfile, Error> {
        let input = BufferRange::whole(input);
        let output = BufferRange::whole(output);
        let Some(problem) = self.prepare_sort(input, output, num_items)? else {
            return Ok(GpuProfile::empty());
        };
        let input = BufferRange {
            size: problem.size_bytes,
            ..input
        };
        let output = BufferRange {
            size: problem.size_bytes,
            ..output
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
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        num_items: u32,
    ) -> Result<Option<PreparedSort>, Error> {
        if num_items == 0 {
            return Ok(None);
        }
        if input.buffer == output.buffer {
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

        let size_bytes =
            common::math::checked_byte_size(u64::from(num_items), self.item_size_bytes)?;
        input.validate("sort input", size_bytes, wgpu::BufferUsages::STORAGE)?;
        output.validate("sort output", size_bytes, wgpu::BufferUsages::STORAGE)?;
        input.validate_storage_offset(&self.device, "sort input")?;
        output.validate_storage_offset(&self.device, "sort output")?;
        self.ensure_workspace(size_bytes)?;

        Ok(Some(PreparedSort {
            num_items,
            num_tiles: num_items.div_ceil(ITEMS_PER_TILE),
            size_bytes,
        }))
    }

    fn record_commands(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
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
        let keepalive = (
            bindings._uniform.clone(),
            bindings.histogram.clone(),
            bindings.prefix.clone(),
            bindings.scatter.clone(),
        );
        encoder.on_submitted_work_done(move || drop(keepalive));
        Ok(())
    }

    fn ensure_bindings(
        &mut self,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        problem: PreparedSort,
        pass_count: u32,
    ) {
        let workspace = self.workspace.as_ref().expect("sort workspace is prepared");
        let matches = self.cached_bindings.as_ref().is_some_and(|bindings| {
            bindings.input == *input.buffer
                && bindings.input_offset == input.offset
                && bindings.output == *output.buffer
                && bindings.output_offset == output.offset
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
            input: input.buffer.clone(),
            input_offset: input.offset,
            output: output.buffer.clone(),
            output_offset: output.offset,
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

        let capacity = workspace_capacity(requested_size, self.item_size_bytes)?;
        let max_items = capacity / self.item_size_bytes;
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
            .min(limits.max_storage_buffer_binding_size);
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

    fn ensure_counted_state(&mut self) {
        if self.counted.is_some() {
            return;
        }
        self.counted = Some(CountedEightBitState::new(&self.device));
    }

    pub(crate) fn reserve(&mut self, capacity: u32) -> Result<(), Error> {
        if capacity == 0 {
            return Ok(());
        }
        let size_bytes =
            common::math::checked_byte_size(u64::from(capacity), self.item_size_bytes)?;
        self.ensure_workspace(size_bytes)
    }

    pub(crate) fn reserve_counted(&mut self, capacity: u32) -> Result<(), Error> {
        self.ensure_counted_state();
        self.reserve(capacity)
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

impl CountedEightBitState {
    fn new(device: &wgpu::Device) -> Self {
        let prepare_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("8-bit Counted Preparation Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, false, true),
            ],
        });
        let source = include_str!("counted_prepare_8bit.wgsl").replace(
            "{{UNIFORM_STRIDE_WORDS}}",
            &(u64::from(device.limits().min_uniform_buffer_offset_alignment)
                .max(UNIFORM_SIZE_BYTES)
                / size_of::<u32>() as u64)
                .to_string(),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("8-bit Counted Preparation Shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let prepare = create_pipeline(
            device,
            "8-bit Counted Preparation Pipeline",
            &prepare_layout,
            &shader,
            "main",
        );
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("8-bit Counted Sort Uniforms"),
            size: u64::from(device.limits().min_uniform_buffer_offset_alignment)
                .max(UNIFORM_SIZE_BYTES)
                * u64::from(MAX_PASS_COUNT),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        Self {
            prepare_layout,
            prepare,
            uniforms,
        }
    }

    fn create_prepare_bind_group(
        &self,
        device: &wgpu::Device,
        count: BufferRange<'_>,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        create_bind_group(
            device,
            &self.prepare_layout,
            "8-bit Counted Preparation Bind Group",
            &[
                range_buffer(0, count),
                entire_buffer(1, &self.uniforms),
                uniform_binding(2, params, 0),
            ],
        )
    }
}

impl NativeKeyValueSoaSorter {
    pub(super) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Creates the native separate-buffer sorter when the adapter and enabled
    /// device features satisfy the validated 8-bit subgroup contract.
    pub fn new_for_adapter(
        device: &wgpu::Device,
        adapter_info: &wgpu::AdapterInfo,
    ) -> Option<Self> {
        let variant = RadixVariant::for_adapter(
            SortItemKind::KeyValue,
            adapter_info,
            device.features(),
            &device.limits(),
        );
        variant.uses_eight_bit_pipeline().then(|| Self::new(device))
    }

    fn new(device: &wgpu::Device) -> Self {
        let histogram_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("8-bit SoA Histogram Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, false, true),
            ],
        });
        let prefix_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("8-bit SoA Prefix Layout"),
            entries: &[
                common::buffers::bind_entry(0, false, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, false, false),
                common::buffers::bind_entry(3, false, true),
            ],
        });
        let scatter_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("8-bit SoA Scatter Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, true, false),
                common::buffers::bind_entry(2, false, false),
                common::buffers::bind_entry(3, false, false),
                common::buffers::bind_entry(4, true, false),
                common::buffers::bind_entry(5, false, false),
                common::buffers::bind_entry(6, false, true),
            ],
        });
        let histogram_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("8-bit SoA Histogram Shader"),
            source: wgpu::ShaderSource::Wgsl(
                item_shader(include_str!("histogram_8bit.wgsl"), SortItemKind::Key).into(),
            ),
        });
        let prefix_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("8-bit SoA Prefix Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("prefix_8bit.wgsl").into()),
        });
        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("8-bit SoA Scatter Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("scatter_8bit_soa.wgsl").into()),
        });
        Self {
            histogram: create_pipeline(
                device,
                "8-bit SoA Histogram Pipeline",
                &histogram_layout,
                &histogram_shader,
                "main_histogram",
            ),
            prefix: create_pipeline(
                device,
                "8-bit SoA Prefix Pipeline",
                &prefix_layout,
                &prefix_shader,
                "main_prefix",
            ),
            scatter: create_pipeline(
                device,
                "8-bit SoA Scatter Pipeline",
                &scatter_layout,
                &scatter_shader,
                "main_scatter",
            ),
            histogram_layout,
            prefix_layout,
            scatter_layout,
            counted: CountedEightBitState::new(device),
            device: device.clone(),
            workspace: None,
            bindings: None,
        }
    }

    /// Prepares all buffers required to record a sort up to the given capacity.
    pub fn reserve(&mut self, capacity: u32) -> Result<(), Error> {
        if capacity == 0 {
            return Ok(());
        }
        let capacity_bytes = common::math::checked_byte_size(u64::from(capacity), 4)?;
        self.ensure_workspace(capacity_bytes)
    }

    /// Records a counted separate-buffer sort using a u32 count word within an
    /// aligned GPU metadata buffer.
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
        let count_bytes = u64::from(count_word)
            .checked_add(1)
            .and_then(|words| words.checked_mul(4))
            .ok_or(Error::SizeOverflow)?;
        let count_range = BufferRange::whole(count);
        count_range.validate("sort item count", count_bytes, wgpu::BufferUsages::STORAGE)?;
        count_range.validate_storage_binding_size(&self.device, count_bytes)?;
        self.record_sort_counted_impl(encoder, keys, values, count_range, count_word, capacity)
    }

    /// Records a previously reserved counted separate-buffer sort without
    /// allocating workspace or compiling pipelines.
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
        let capacity_bytes = common::math::checked_byte_size(u64::from(capacity), 4)?;
        let Some(_) = self.validate_inputs(keys, values, count, count_word, capacity)? else {
            return Ok(());
        };
        let workspace = self.workspace.as_ref().ok_or(Error::BufferTooSmall {
            name: "sort workspace",
            required: capacity_bytes,
            actual: 0,
        })?;
        if workspace.capacity_bytes < capacity_bytes {
            return Err(Error::BufferTooSmall {
                name: "sort workspace",
                required: capacity_bytes,
                actual: workspace.capacity_bytes,
            });
        }
        if !self.bindings_match(keys, values, count, count_word, capacity) {
            return Err(Error::BufferTooSmall {
                name: "sort binding plan",
                required: 1,
                actual: 0,
            });
        }
        self.record_commands(encoder, capacity)
    }

    /// Prepares a reusable binding plan for allocation-free command recording.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_counted_from_word(
        &mut self,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        count: &wgpu::Buffer,
        count_word: u32,
        capacity: u32,
    ) -> Result<(), Error> {
        let capacity_bytes = common::math::checked_byte_size(u64::from(capacity), 4)?;
        if capacity != 0 {
            self.ensure_workspace(capacity_bytes)?;
        }
        let Some((keys, values, count)) =
            self.validate_inputs(keys, values, count, count_word, capacity)?
        else {
            return Ok(());
        };
        self.ensure_bindings(keys, values, count, count_word, capacity);
        Ok(())
    }

    fn record_sort_counted_impl(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        count: BufferRange<'_>,
        count_word: u32,
        capacity: u32,
    ) -> Result<(), Error> {
        let capacity_bytes = common::math::checked_byte_size(u64::from(capacity), 4)?;
        if capacity != 0 {
            self.ensure_workspace(capacity_bytes)?;
        }
        let Some((keys, values, count)) =
            self.validate_inputs(keys, values, count.buffer, count_word, capacity)?
        else {
            return Ok(());
        };
        self.ensure_bindings(keys, values, count, count_word, capacity);
        self.record_commands(encoder, capacity)
    }

    fn validate_inputs<'a>(
        &self,
        keys: &'a wgpu::Buffer,
        values: &'a wgpu::Buffer,
        count_buffer: &'a wgpu::Buffer,
        count_word: u32,
        capacity: u32,
    ) -> Result<Option<(BufferRange<'a>, BufferRange<'a>, BufferRange<'a>)>, Error> {
        let count_bytes = u64::from(count_word)
            .checked_add(1)
            .and_then(|words| words.checked_mul(4))
            .ok_or(Error::SizeOverflow)?;
        let count = BufferRange {
            buffer: count_buffer,
            offset: 0,
            size: count_bytes,
        };
        count.validate("sort item count", count_bytes, wgpu::BufferUsages::STORAGE)?;
        for (first, first_name, second, second_name) in [
            (keys, "sort keys", values, "sort values"),
            (keys, "sort keys", count_buffer, "sort item count"),
            (values, "sort values", count_buffer, "sort item count"),
        ] {
            if first == second {
                return Err(Error::BufferAlias {
                    first: first_name,
                    second: second_name,
                });
            }
        }
        let capacity_bytes = common::math::checked_byte_size(u64::from(capacity), 4)?;
        let keys = BufferRange::whole(keys);
        let values = BufferRange::whole(values);
        keys.validate("sort keys", capacity_bytes, wgpu::BufferUsages::STORAGE)?;
        values.validate("sort values", capacity_bytes, wgpu::BufferUsages::STORAGE)?;
        for (range, name, size) in [
            (keys, "sort keys", capacity_bytes),
            (values, "sort values", capacity_bytes),
            (count, "sort item count", count_bytes),
        ] {
            range.validate_storage_offset(&self.device, name)?;
            range.validate_storage_binding_size(&self.device, size)?;
        }
        if capacity == 0 {
            return Ok(None);
        }
        let max_items =
            element_count_limit(self.device.limits().max_compute_workgroups_per_dimension);
        if capacity > max_items {
            return Err(Error::RadixElementCountLimitExceeded {
                count: capacity,
                limit: max_items,
            });
        }
        Ok(Some((
            BufferRange {
                size: capacity_bytes,
                ..keys
            },
            BufferRange {
                size: capacity_bytes,
                ..values
            },
            count,
        )))
    }

    fn ensure_bindings(
        &mut self,
        keys: BufferRange<'_>,
        values: BufferRange<'_>,
        count: BufferRange<'_>,
        count_word: u32,
        capacity: u32,
    ) {
        if self.bindings_match(
            keys.buffer,
            values.buffer,
            count.buffer,
            count_word,
            capacity,
        ) {
            return;
        }
        let workspace = self.workspace.as_ref().expect("SoA workspace is prepared");
        let params_data = [capacity, MAX_PASS_COUNT, count_word, 0, 0, 0, 0, 0];
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("8-bit SoA Counted Parameters"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let prepare = self
            .counted
            .create_prepare_bind_group(&self.device, count, &params);
        let histogram = create_bind_group(
            &self.device,
            &self.histogram_layout,
            "8-bit SoA Histogram Bind Group",
            &[
                range_buffer(0, keys),
                entire_buffer(1, &workspace.histogram),
                uniform_binding(2, &self.counted.uniforms, 0),
            ],
        );
        let prefix = create_bind_group(
            &self.device,
            &self.prefix_layout,
            "8-bit SoA Prefix Bind Group",
            &[
                entire_buffer(0, &workspace.histogram),
                entire_buffer(1, &workspace.offsets),
                entire_buffer(2, &workspace.dispatch_args),
                uniform_binding(3, &self.counted.uniforms, 0),
            ],
        );
        let uniform_stride = u64::from(self.device.limits().min_uniform_buffer_offset_alignment)
            .max(UNIFORM_SIZE_BYTES);
        let scatter: Vec<_> = (0..MAX_PASS_COUNT)
            .map(|radix_pass| {
                let (source_keys, source_values, destination_keys, destination_values) =
                    soa_pass_buffers(radix_pass, keys, values, workspace);
                create_bind_group(
                    &self.device,
                    &self.scatter_layout,
                    "8-bit SoA Scatter Bind Group",
                    &[
                        range_buffer(0, source_keys),
                        range_buffer(1, source_values),
                        range_buffer(2, destination_keys),
                        range_buffer(3, destination_values),
                        entire_buffer(4, &workspace.offsets),
                        entire_buffer(5, &workspace.partition_state),
                        uniform_binding(
                            6,
                            &self.counted.uniforms,
                            u64::from(radix_pass) * uniform_stride,
                        ),
                    ],
                )
            })
            .collect();

        self.bindings = Some(SoaBindings {
            keys: keys.buffer.clone(),
            values: values.buffer.clone(),
            count: count.buffer.clone(),
            count_word,
            capacity,
            workspace_capacity: workspace.capacity_bytes,
            _params: params,
            prepare,
            histogram,
            prefix,
            scatter,
        });
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

    fn record_commands(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        capacity: u32,
    ) -> Result<(), Error> {
        let workspace = self.workspace.as_ref().expect("SoA workspace is prepared");
        let bindings = self
            .bindings
            .as_ref()
            .expect("SoA sort bindings are prepared");
        encoder.clear_buffer(&workspace.histogram, 0, None);
        encoder.clear_buffer(&workspace.partition_state, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("8-bit SoA Counted Preparation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.counted.prepare);
        pass.set_bind_group(0, &bindings.prepare, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("8-bit SoA Histogram"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.histogram);
        pass.set_bind_group(0, &bindings.histogram, &[]);
        pass.dispatch_workgroups(
            capacity.div_ceil(ITEMS_PER_TILE).min(MAX_HISTOGRAM_GROUPS),
            1,
            1,
        );
        drop(pass);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("8-bit SoA Prefix"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.prefix);
        pass.set_bind_group(0, &bindings.prefix, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        drop(pass);

        for (radix_pass, bind_group) in bindings.scatter.iter().enumerate() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("8-bit SoA Scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.scatter);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups_indirect(
                &workspace.dispatch_args,
                radix_pass as u64 * 3 * size_of::<u32>() as u64,
            );
        }
        Ok(())
    }

    fn ensure_workspace(&mut self, requested_bytes: u64) -> Result<(), Error> {
        if self
            .workspace
            .as_ref()
            .is_some_and(|workspace| workspace.capacity_bytes >= requested_bytes)
        {
            return Ok(());
        }
        let capacity_bytes = workspace_capacity(requested_bytes, 4)?;
        let max_items = capacity_bytes / 4;
        let max_tiles = max_items.div_ceil(u64::from(ITEMS_PER_TILE));
        let partition_entries = max_tiles
            .checked_mul(u64::from(BUCKET_COUNT))
            .and_then(|entries| entries.checked_add(TILE_COUNTER_COUNT))
            .ok_or(Error::SizeOverflow)?;
        let partition_bytes = common::math::checked_align_to(
            common::math::checked_byte_size(partition_entries, 4)?,
            256,
        )?;
        for requested in [capacity_bytes, partition_bytes] {
            common::buffers::validate_storage_binding_size(&self.device, requested)?;
        }
        let digit_table_bytes = u64::from(MAX_PASS_COUNT * BUCKET_COUNT) * 4;
        self.workspace = Some(SoaWorkspace {
            capacity_bytes,
            scratch_keys: common::buffers::create_empty_storage_buffer(
                &self.device,
                capacity_bytes,
            ),
            scratch_values: common::buffers::create_empty_storage_buffer(
                &self.device,
                capacity_bytes,
            ),
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
                label: Some("8-bit SoA Dispatch Arguments"),
                size: DISPATCH_ARGS_SIZE_BYTES,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
        });
        self.bindings = None;
        Ok(())
    }
}

fn soa_pass_buffers<'a>(
    radix_pass: u32,
    keys: BufferRange<'a>,
    values: BufferRange<'a>,
    workspace: &'a SoaWorkspace,
) -> (
    BufferRange<'a>,
    BufferRange<'a>,
    BufferRange<'a>,
    BufferRange<'a>,
) {
    let scratch_keys = BufferRange {
        buffer: &workspace.scratch_keys,
        offset: 0,
        size: keys.size,
    };
    let scratch_values = BufferRange {
        buffer: &workspace.scratch_values,
        offset: 0,
        size: values.size,
    };
    if radix_pass.is_multiple_of(2) {
        (keys, values, scratch_keys, scratch_values)
    } else {
        (scratch_keys, scratch_values, keys, values)
    }
}

impl EightBitPipelines {
    fn new(device: &wgpu::Device, item_kind: SortItemKind) -> Self {
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

        let histogram_source = item_shader(include_str!("histogram_8bit.wgsl"), item_kind);
        let scatter_source = item_shader(include_str!("scatter_8bit.wgsl"), item_kind);
        let histogram_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("8-bit Histogram Shader"),
            source: wgpu::ShaderSource::Wgsl(histogram_source.into()),
        });
        let prefix_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("8-bit Prefix Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("prefix_8bit.wgsl").into()),
        });
        let scatter_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("8-bit Scatter Shader"),
            source: wgpu::ShaderSource::Wgsl(scatter_source.into()),
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
        input: BufferRange<'_>,
        histogram: &wgpu::Buffer,
        uniform: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        create_bind_group(
            device,
            &self.histogram_layout,
            "8-bit Histogram Bind Group",
            &[
                range_buffer(0, input),
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
        source: BufferRange<'_>,
        destination: BufferRange<'_>,
        workspace: &EightBitWorkspace,
        uniform: &wgpu::Buffer,
        uniform_offset: u64,
    ) -> wgpu::BindGroup {
        create_bind_group(
            device,
            &self.scatter_layout,
            "8-bit Scatter Bind Group",
            &[
                range_buffer(0, source),
                range_buffer(1, destination),
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
        bind_group_layouts: &[Some(bind_group_layout)],
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

fn range_buffer(binding: u32, range: BufferRange<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: range.binding(range.size),
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
    input: BufferRange<'a>,
    output: BufferRange<'a>,
    scratch: &'a wgpu::Buffer,
) -> (BufferRange<'a>, BufferRange<'a>) {
    let scratch = BufferRange {
        buffer: scratch,
        offset: 0,
        size: input.size,
    };
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

fn validate_counted_distinct(
    input: BufferRange<'_>,
    output: BufferRange<'_>,
    count: BufferRange<'_>,
) -> Result<(), Error> {
    for (first, first_name, second, second_name) in [
        (input, "sort input", output, "sort output"),
        (input, "sort input", count, "sort item count"),
        (output, "sort output", count, "sort item count"),
    ] {
        if first.buffer == second.buffer {
            return Err(Error::BufferAlias {
                first: first_name,
                second: second_name,
            });
        }
    }
    Ok(())
}

fn element_count_limit(max_compute_workgroups_per_dimension: u32) -> u32 {
    max_compute_workgroups_per_dimension
        .saturating_mul(ITEMS_PER_TILE)
        .min(MAX_PACKED_COUNT)
}

fn workspace_capacity(requested_size: u64, item_size_bytes: u64) -> Result<u64, Error> {
    if requested_size < WORKSPACE_GROWTH_BYTES {
        requested_size
            .max(item_size_bytes)
            .checked_next_power_of_two()
            .ok_or(Error::SizeOverflow)
    } else {
        common::math::checked_align_to(requested_size, WORKSPACE_GROWTH_BYTES)
    }
}

fn item_shader(source: &str, item_kind: SortItemKind) -> String {
    source
        .replace("{{ITEM_TYPE}}", item_kind.shader_item_type())
        .replace("{{KEY_MEMBER}}", item_kind.shader_key_member())
}

#[cfg(test)]
mod tests {
    use super::{
        BufferSlot, SortItemKind, element_count_limit, item_shader, pass_buffer_slots,
        pass_count_for_key_bits,
    };

    #[test]
    fn specializes_eight_bit_item_shaders_without_leaving_placeholders() {
        let template = include_str!("scatter_8bit.wgsl");
        let key_shader = item_shader(template, SortItemKind::Key);
        assert!(key_shader.contains("var<storage, read> input: array<u32>;"));
        assert!(key_shader.contains("sorted_items[subgroup_id] = lane_prefix + count;"));
        assert!(!key_shader.contains("{{"));

        let key_value_shader = item_shader(template, SortItemKind::KeyValue);
        assert!(key_value_shader.contains("var<storage, read> input: array<KeyValue>;"));
        assert!(key_value_shader.contains("sorted_items[subgroup_id].key = lane_prefix + count;"));
        assert!(!key_value_shader.contains("{{"));
    }

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
