use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::{
    Context, Error, GpuProfile, KeyValue, common,
    common::{
        buffers::BufferRange,
        runtime::{CommandSession, ProfileSession},
        workspace::ReusableBuffer,
    },
    profiling::TimestampRecorder,
};

use super::pipeline::{ArgminPipelines, output_items, pass_count};

const PAIR_SIZE_BYTES: u64 = size_of::<KeyValue>() as u64;
const COUNT_SIZE_BYTES: u64 = size_of::<u32>() as u64;
const IDENTITY: KeyValue = KeyValue::new(u32::MAX, u32::MAX);

struct CachedFixedBindings {
    input: wgpu::Buffer,
    input_offset: u64,
    output: wgpu::Buffer,
    output_offset: u64,
    num_items: u32,
    resources: Arc<PassResources>,
}

struct CachedCountedBindings {
    input: wgpu::Buffer,
    input_offset: u64,
    output: wgpu::Buffer,
    output_offset: u64,
    count: wgpu::Buffer,
    count_offset: u64,
    capacity: u32,
    resources: Arc<PassResources>,
}

struct PassResources {
    passes: Vec<PassBinding>,
}

struct PassBinding {
    bind_group: wgpu::BindGroup,
    _params: wgpu::Buffer,
    output_items: u32,
}

/// Selects the lexicographically smallest `(key, value)` record on the GPU.
///
/// A smaller key always wins. Equal keys are resolved by the smaller value,
/// which makes the result deterministic without sorting the remaining records.
/// Empty fixed or GPU-counted inputs write `(u32::MAX, u32::MAX)`.
pub struct ArgminByKey {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: ArgminPipelines,
    identity: wgpu::Buffer,
    scratch_a: ReusableBuffer,
    scratch_b: ReusableBuffer,
    scratch_a_capacity: u64,
    scratch_b_capacity: u64,
    fixed_bindings: Option<CachedFixedBindings>,
    counted_bindings: Option<CachedCountedBindings>,
}

impl ArgminByKey {
    /// Creates an argmin selector over an existing wgpu device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let identity = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Argmin-by-Key Identity"),
            contents: bytemuck::bytes_of(&IDENTITY),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        Self {
            device: device.clone(),
            queue: queue.clone(),
            pipelines: ArgminPipelines::new(device),
            identity,
            scratch_a: ReusableBuffer::default(),
            scratch_b: ReusableBuffer::default(),
            scratch_a_capacity: 0,
            scratch_b_capacity: 0,
            fixed_bindings: None,
            counted_bindings: None,
        }
    }

    /// Creates an argmin selector from Lampshade's convenience context.
    pub fn from_context(context: &Context) -> Self {
        Self::new(&context.device, &context.queue)
    }

    /// Returns the required size of the caller-owned `KeyValue` output buffer.
    pub const fn output_buffer_size() -> u64 {
        PAIR_SIZE_BYTES
    }

    /// Uploads records, selects their lexicographic minimum, and downloads it.
    pub async fn argmin(&mut self, input: &[KeyValue]) -> Result<KeyValue, Error> {
        if input.is_empty() {
            return Ok(IDENTITY);
        }
        let num_items = common::math::checked_u32(input.len() as u64)?;
        let input_bytes = common::math::checked_byte_size(u64::from(num_items), PAIR_SIZE_BYTES)?;
        common::buffers::validate_storage_binding_size(&self.device, input_bytes)?;
        let input_buffer = common::buffers::create_storage_buffer(&self.device, input);
        let output_buffer =
            common::buffers::create_empty_storage_buffer(&self.device, PAIR_SIZE_BYTES);
        self.argmin_gpu_to_gpu(&input_buffer, &output_buffer, num_items)?;
        let result = common::buffers::download_buffer::<KeyValue>(
            &self.device,
            &self.queue,
            &output_buffer,
            1,
        )
        .await?;
        Ok(result[0])
    }

    /// Selects one record from a fixed-length caller-owned GPU buffer.
    ///
    /// `input` requires `STORAGE`. `output` must be a distinct buffer with at
    /// least eight bytes and `STORAGE | COPY_DST`; `COPY_DST` stores the empty
    /// identity.
    pub fn argmin_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        let mut commands = CommandSession::new(&self.device, Some("Argmin by Key"));
        self.record_argmin(commands.encoder(), input, output, num_items)?;
        commands.submit(&self.queue);
        Ok(())
    }

    /// Selects one record from a prefix whose length remains GPU-resident.
    ///
    /// The count is clamped to `capacity`. All three buffers must be distinct.
    pub fn argmin_counted_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
    ) -> Result<(), Error> {
        let mut commands = CommandSession::new(&self.device, Some("Counted Argmin by Key"));
        self.record_argmin_counted(commands.encoder(), input, output, count, capacity)?;
        commands.submit(&self.queue);
        Ok(())
    }

    /// Records a fixed-length argmin without submitting or waiting.
    pub fn record_argmin(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<(), Error> {
        self.record_argmin_ranges(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(output),
            num_items,
            None,
        )
    }

    pub(crate) fn record_argmin_ranges(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        num_items: u32,
        profiler: Option<&mut TimestampRecorder>,
    ) -> Result<(), Error> {
        self.validate_fixed(input, output, num_items)?;
        if num_items == 0 {
            self.record_identity(encoder, output);
            return Ok(());
        }
        self.prepare_scratch(num_items)?;
        let resources = self.fixed_resources(input, output, num_items);
        self.record_resources(encoder, &resources, false, profiler);
        Ok(())
    }

    /// Records a capacity-bounded argmin without submitting or waiting.
    pub fn record_argmin_counted(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
    ) -> Result<(), Error> {
        self.record_argmin_counted_ranges(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(output),
            BufferRange::whole(count),
            capacity,
            None,
        )
    }

    pub(crate) fn record_argmin_counted_ranges(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        count: BufferRange<'_>,
        capacity: u32,
        profiler: Option<&mut TimestampRecorder>,
    ) -> Result<(), Error> {
        self.validate_counted(input, output, count, capacity)?;
        if capacity == 0 {
            self.record_identity(encoder, output);
            return Ok(());
        }
        self.prepare_scratch(capacity)?;
        let resources = self.counted_resources(input, output, count, capacity);
        self.record_resources(encoder, &resources, true, profiler);
        Ok(())
    }

    /// Profiles a fixed-length argmin using GPU timestamps.
    pub async fn profile_argmin_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
    ) -> Result<GpuProfile, Error> {
        let mut profile = ProfileSession::new(
            &self.device,
            &self.queue,
            pass_count(num_items),
            "Profiled Argmin by Key",
        )?;
        let (encoder, profiler) = profile.recording();
        self.record_argmin_ranges(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(output),
            num_items,
            profiler,
        )?;
        profile.finish(&self.device, &self.queue).await
    }

    /// Profiles a capacity-bounded argmin using GPU timestamps.
    pub async fn profile_argmin_counted_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
    ) -> Result<GpuProfile, Error> {
        let mut profile = ProfileSession::new(
            &self.device,
            &self.queue,
            pass_count(capacity),
            "Profiled Counted Argmin by Key",
        )?;
        let (encoder, profiler) = profile.recording();
        self.record_argmin_counted_ranges(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(output),
            BufferRange::whole(count),
            capacity,
            profiler,
        )?;
        profile.finish(&self.device, &self.queue).await
    }

    pub(crate) fn reserve(&mut self, capacity: u32) -> Result<(), Error> {
        if capacity > 0 {
            self.prepare_scratch(capacity)?;
        }
        Ok(())
    }

    fn validate_fixed(
        &self,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        num_items: u32,
    ) -> Result<(), Error> {
        validate_distinct(input, output)?;
        validate_output(&self.device, output)?;
        if num_items == 0 {
            return Ok(());
        }
        validate_input(&self.device, input, num_items)
    }

    fn validate_counted(
        &self,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        count: BufferRange<'_>,
        capacity: u32,
    ) -> Result<(), Error> {
        validate_distinct(input, output)?;
        if input.buffer == count.buffer {
            return Err(Error::BufferAlias {
                first: "argmin input",
                second: "argmin item count",
            });
        }
        if output.buffer == count.buffer {
            return Err(Error::BufferAlias {
                first: "argmin output",
                second: "argmin item count",
            });
        }
        validate_output(&self.device, output)?;
        count.validate(
            "argmin item count",
            COUNT_SIZE_BYTES,
            wgpu::BufferUsages::STORAGE,
        )?;
        count.validate_storage_offset(&self.device, "argmin item count")?;
        if capacity == 0 {
            return Ok(());
        }
        validate_input(&self.device, input, capacity)
    }

    fn prepare_scratch(&mut self, capacity: u32) -> Result<(), Error> {
        let first = output_items(capacity);
        if first <= 1 {
            return Ok(());
        }
        let first_bytes = common::math::checked_byte_size(u64::from(first), PAIR_SIZE_BYTES)?;
        common::buffers::validate_storage_binding_size(&self.device, first_bytes)?;
        if first_bytes > self.scratch_a_capacity {
            self.scratch_a.ensure(
                &self.device,
                first_bytes,
                "Argmin Scratch A",
                wgpu::BufferUsages::STORAGE,
            );
            self.scratch_a_capacity = first_bytes;
            self.invalidate_bindings();
        }

        let second = output_items(first);
        if second > 1 {
            let second_bytes = common::math::checked_byte_size(u64::from(second), PAIR_SIZE_BYTES)?;
            common::buffers::validate_storage_binding_size(&self.device, second_bytes)?;
            if second_bytes > self.scratch_b_capacity {
                self.scratch_b.ensure(
                    &self.device,
                    second_bytes,
                    "Argmin Scratch B",
                    wgpu::BufferUsages::STORAGE,
                );
                self.scratch_b_capacity = second_bytes;
                self.invalidate_bindings();
            }
        }
        Ok(())
    }

    fn invalidate_bindings(&mut self) {
        self.fixed_bindings = None;
        self.counted_bindings = None;
    }

    fn fixed_resources(
        &mut self,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        num_items: u32,
    ) -> Arc<PassResources> {
        let matches = self.fixed_bindings.as_ref().is_some_and(|cached| {
            &cached.input == input.buffer
                && cached.input_offset == input.offset
                && &cached.output == output.buffer
                && cached.output_offset == output.offset
                && cached.num_items == num_items
        });
        if !matches {
            let resources = Arc::new(self.create_fixed_resources(input, output, num_items));
            self.fixed_bindings = Some(CachedFixedBindings {
                input: input.buffer.clone(),
                input_offset: input.offset,
                output: output.buffer.clone(),
                output_offset: output.offset,
                num_items,
                resources,
            });
        }
        Arc::clone(
            &self
                .fixed_bindings
                .as_ref()
                .expect("fixed argmin bindings are initialized")
                .resources,
        )
    }

    fn counted_resources(
        &mut self,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        count: BufferRange<'_>,
        capacity: u32,
    ) -> Arc<PassResources> {
        let matches = self.counted_bindings.as_ref().is_some_and(|cached| {
            &cached.input == input.buffer
                && cached.input_offset == input.offset
                && &cached.output == output.buffer
                && cached.output_offset == output.offset
                && &cached.count == count.buffer
                && cached.count_offset == count.offset
                && cached.capacity == capacity
        });
        if !matches {
            let resources = Arc::new(self.create_counted_resources(input, output, count, capacity));
            self.counted_bindings = Some(CachedCountedBindings {
                input: input.buffer.clone(),
                input_offset: input.offset,
                output: output.buffer.clone(),
                output_offset: output.offset,
                count: count.buffer.clone(),
                count_offset: count.offset,
                capacity,
                resources,
            });
        }
        Arc::clone(
            &self
                .counted_bindings
                .as_ref()
                .expect("counted argmin bindings are initialized")
                .resources,
        )
    }

    fn create_fixed_resources(
        &self,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        num_items: u32,
    ) -> PassResources {
        self.create_resources(input, output, None, num_items)
    }

    fn create_counted_resources(
        &self,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        count: BufferRange<'_>,
        capacity: u32,
    ) -> PassResources {
        self.create_resources(input, output, Some(count), capacity)
    }

    fn create_resources(
        &self,
        input: BufferRange<'_>,
        output: BufferRange<'_>,
        count: Option<BufferRange<'_>>,
        capacity: u32,
    ) -> PassResources {
        let mut passes = Vec::with_capacity(pass_count(capacity) as usize);
        let mut current_input = input;
        let mut current_capacity = capacity;
        let mut write_to_a = true;
        let mut level = 0;

        loop {
            let next_capacity = output_items(current_capacity);
            let current_output = if next_capacity == 1 {
                BufferRange {
                    buffer: output.buffer,
                    offset: output.offset,
                    size: PAIR_SIZE_BYTES,
                }
            } else {
                let scratch = if write_to_a {
                    self.scratch_a
                        .get()
                        .expect("first argmin scratch is prepared")
                } else {
                    self.scratch_b
                        .get()
                        .expect("second argmin scratch is prepared")
                };
                BufferRange {
                    buffer: scratch,
                    offset: 0,
                    size: u64::from(next_capacity) * PAIR_SIZE_BYTES,
                }
            };
            let params_words = match count {
                Some(_) => [capacity, level, next_capacity, 0],
                None => [current_capacity, 0, 0, 0],
            };
            let params = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Argmin Pass Parameters"),
                    contents: bytemuck::cast_slice(&params_words),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let layout = if count.is_some() {
                &self.pipelines.counted_layout
            } else {
                &self.pipelines.fixed_layout
            };
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: current_input.binding(u64::from(current_capacity) * PAIR_SIZE_BYTES),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: current_output.binding(u64::from(next_capacity) * PAIR_SIZE_BYTES),
                },
            ];
            if let Some(count) = count {
                entries.push(wgpu::BindGroupEntry {
                    binding: 2,
                    resource: count.binding(COUNT_SIZE_BYTES),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                });
            } else {
                entries.push(wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                });
            }
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Argmin Pass Bind Group"),
                layout,
                entries: &entries,
            });
            passes.push(PassBinding {
                bind_group,
                _params: params,
                output_items: next_capacity,
            });
            if next_capacity == 1 {
                break;
            }
            current_input = current_output;
            current_capacity = next_capacity;
            write_to_a = !write_to_a;
            level += 1;
        }
        PassResources { passes }
    }

    fn record_resources(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        resources: &Arc<PassResources>,
        counted: bool,
        mut profiler: Option<&mut TimestampRecorder>,
    ) {
        for (level, pass) in resources.passes.iter().enumerate() {
            self.pipelines.record_pass(
                encoder,
                &pass.bind_group,
                pass.output_items,
                level as u32,
                counted,
                profiler.as_deref_mut(),
            );
        }
        let resources = Arc::clone(resources);
        crate::common::runtime::defer_drop(encoder, resources);
    }

    fn record_identity(&self, encoder: &mut wgpu::CommandEncoder, output: BufferRange<'_>) {
        encoder.copy_buffer_to_buffer(
            &self.identity,
            0,
            output.buffer,
            output.offset,
            PAIR_SIZE_BYTES,
        );
    }
}

fn validate_distinct(input: BufferRange<'_>, output: BufferRange<'_>) -> Result<(), Error> {
    if input.buffer == output.buffer {
        Err(Error::BufferAlias {
            first: "argmin input",
            second: "argmin output",
        })
    } else {
        Ok(())
    }
}

fn validate_output(device: &wgpu::Device, output: BufferRange<'_>) -> Result<(), Error> {
    common::buffers::validate_storage_binding_size(device, PAIR_SIZE_BYTES)?;
    output.validate(
        "argmin output",
        PAIR_SIZE_BYTES,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    )?;
    output.validate_storage_offset(device, "argmin output")
}

fn validate_input(device: &wgpu::Device, input: BufferRange<'_>, items: u32) -> Result<(), Error> {
    let bytes = common::math::checked_byte_size(u64::from(items), PAIR_SIZE_BYTES)?;
    common::buffers::validate_storage_binding_size(device, bytes)?;
    input.validate("argmin input", bytes, wgpu::BufferUsages::STORAGE)?;
    input.validate_storage_offset(device, "argmin input")
}

#[cfg(test)]
mod tests {
    use super::IDENTITY;

    #[test]
    fn identity_and_parameter_layout_match_wgsl() {
        assert_eq!(IDENTITY.key, u32::MAX);
        assert_eq!(IDENTITY.value, u32::MAX);
        assert_eq!(size_of::<[u32; 4]>(), 16);
    }
}
