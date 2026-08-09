use crate::{
    Error, common,
    common::{runtime::CommandSession, runtime::ProfileSession, workspace::ReusableBuffer},
    context::Context,
    profiling::{GpuProfile, TimestampRecorder},
};

use super::{
    U32Reduction,
    pipeline::{ReductionDispatch, ReductionPipeline},
};

const VALUE_SIZE_BYTES: u64 = size_of::<u32>() as u64;

/// Reduces unsigned 32-bit values to one sum, minimum, or maximum.
///
/// Sum uses wrapping `u32` addition. Empty inputs return the operation's
/// identity: `0` for sum and maximum, and [`u32::MAX`] for minimum.
pub struct Reducer {
    pipeline: ReductionPipeline,
    device: wgpu::Device,
    queue: wgpu::Queue,
    scratch_a: ReusableBuffer,
    scratch_b: ReusableBuffer,
}

impl Reducer {
    /// Creates a reducer that submits work through an existing device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            pipeline: ReductionPipeline::new(device),
            device: device.clone(),
            queue: queue.clone(),
            scratch_a: ReusableBuffer::default(),
            scratch_b: ReusableBuffer::default(),
        }
    }

    /// Creates a reducer from the crate's optional convenience context.
    pub fn from_context(context: &Context) -> Self {
        Self::new(&context.device, &context.queue)
    }

    /// Returns the required size of the caller-owned scalar output buffer.
    pub const fn output_buffer_size() -> u64 {
        VALUE_SIZE_BYTES
    }

    /// Uploads values, computes their wrapping sum, and downloads the scalar.
    pub async fn sum(&mut self, input: &[u32]) -> Result<u32, Error> {
        self.reduce(input, U32Reduction::Sum).await
    }

    /// Uploads values, computes their minimum, and downloads the scalar.
    pub async fn min(&mut self, input: &[u32]) -> Result<u32, Error> {
        self.reduce(input, U32Reduction::Min).await
    }

    /// Uploads values, computes their maximum, and downloads the scalar.
    pub async fn max(&mut self, input: &[u32]) -> Result<u32, Error> {
        self.reduce(input, U32Reduction::Max).await
    }

    /// Uploads values, applies one reduction, and downloads the scalar.
    pub async fn reduce(&mut self, input: &[u32], operation: U32Reduction) -> Result<u32, Error> {
        if input.is_empty() {
            return Ok(operation.identity());
        }

        let num_items = common::math::checked_u32(input.len() as u64)?;
        let input_bytes = common::math::checked_byte_size(u64::from(num_items), VALUE_SIZE_BYTES)?;
        self.validate_storage_binding_size(input_bytes)?;
        let input_buffer = common::buffers::create_storage_buffer(&self.device, input);
        let output_buffer =
            common::buffers::create_empty_storage_buffer(&self.device, VALUE_SIZE_BYTES);
        self.reduce_gpu_to_gpu(&input_buffer, &output_buffer, num_items, operation)?;
        let output =
            common::buffers::download_buffer::<u32>(&self.device, &self.queue, &output_buffer, 1)
                .await?;
        Ok(output[0])
    }

    /// Reduces a caller-owned GPU buffer and submits the work immediately.
    ///
    /// `input` requires `STORAGE`. `output` must be a distinct buffer of at
    /// least four bytes with `STORAGE | COPY_DST`; the latter usage stores
    /// empty-input identities.
    pub fn reduce_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        operation: U32Reduction,
    ) -> Result<(), Error> {
        let mut commands = CommandSession::new(&self.device, None);
        self.record_reduce(commands.encoder(), input, output, num_items, operation)?;
        commands.submit(&self.queue);
        Ok(())
    }

    /// Records a reduction without submitting or waiting for the work.
    ///
    /// Buffer requirements match [`Self::reduce_gpu_to_gpu`].
    ///
    /// Multiple calls may reuse this reducer's scratch buffers in one encoder;
    /// wgpu preserves the recorded pass order.
    pub fn record_reduce(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        operation: U32Reduction,
    ) -> Result<(), Error> {
        self.record_commands(encoder, input, output, num_items, operation, None)
    }

    /// Profiles a caller-owned GPU reduction using GPU timestamps.
    pub async fn profile_reduce_gpu_to_gpu(
        &mut self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        operation: U32Reduction,
    ) -> Result<GpuProfile, Error> {
        let span_count = self.pipeline.pass_count(num_items);
        let label = if num_items == 0 {
            "Profiled Empty Reduction"
        } else {
            "Profiled Reduction"
        };
        let mut profile = ProfileSession::new(&self.device, &self.queue, span_count, label)?;
        let (encoder, profiler) = profile.recording();
        self.record_commands(encoder, input, output, num_items, operation, profiler)?;
        profile.finish(&self.device, &self.queue).await
    }

    fn record_commands(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        operation: U32Reduction,
        mut profiler: Option<&mut TimestampRecorder>,
    ) -> Result<(), Error> {
        if input == output {
            return Err(Error::BufferAlias {
                first: "reduction input",
                second: "reduction output",
            });
        }
        common::buffers::validate_buffer(
            output,
            "reduction output",
            VALUE_SIZE_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        if num_items == 0 {
            self.pipeline.record_identity(encoder, output, operation);
            return Ok(());
        }

        let input_bytes = common::math::checked_byte_size(u64::from(num_items), VALUE_SIZE_BYTES)?;
        self.validate_storage_binding_size(input_bytes)?;
        common::buffers::validate_buffer(
            input,
            "reduction input",
            input_bytes,
            wgpu::BufferUsages::STORAGE,
        )?;
        self.prepare_scratch(num_items)?;

        let scratch_a = self.scratch_a.get();
        let scratch_b = self.scratch_b.get();
        let mut current_input = input;
        let mut current_items = num_items;
        let mut write_to_a = true;
        let mut level = 0;

        loop {
            let output_items = self.pipeline.output_items(current_items);
            let current_output = if output_items == 1 {
                output
            } else if write_to_a {
                scratch_a.expect("first reduction scratch exists")
            } else {
                scratch_b.expect("second reduction scratch exists")
            };
            self.pipeline.dispatch(
                &self.device,
                encoder,
                ReductionDispatch {
                    input: current_input,
                    output: current_output,
                    input_items: current_items,
                    output_items,
                    operation,
                    level,
                },
                profiler.as_deref_mut(),
            );

            if output_items == 1 {
                return Ok(());
            }
            current_input = current_output;
            current_items = output_items;
            write_to_a = !write_to_a;
            level += 1;
        }
    }

    fn prepare_scratch(&mut self, num_items: u32) -> Result<(), Error> {
        let first_items = self.pipeline.output_items(num_items);
        if first_items <= 1 {
            return Ok(());
        }
        self.ensure_scratch_a(first_items)?;

        let second_items = self.pipeline.output_items(first_items);
        if second_items > 1 {
            self.ensure_scratch_b(second_items)?;
        }
        Ok(())
    }

    fn ensure_scratch_a(&mut self, items: u32) -> Result<(), Error> {
        let size = self.checked_scratch_size(items)?;
        self.scratch_a.ensure(
            &self.device,
            size,
            "Reduction Scratch A",
            wgpu::BufferUsages::STORAGE,
        );
        Ok(())
    }

    fn ensure_scratch_b(&mut self, items: u32) -> Result<(), Error> {
        let size = self.checked_scratch_size(items)?;
        self.scratch_b.ensure(
            &self.device,
            size,
            "Reduction Scratch B",
            wgpu::BufferUsages::STORAGE,
        );
        Ok(())
    }

    fn checked_scratch_size(&self, items: u32) -> Result<u64, Error> {
        let requested = common::math::checked_byte_size(u64::from(items), VALUE_SIZE_BYTES)?;
        self.validate_storage_binding_size(requested)?;
        Ok(requested)
    }

    fn validate_storage_binding_size(&self, requested: u64) -> Result<(), Error> {
        let limits = self.device.limits();
        let limit = effective_storage_binding_limit(
            limits.max_buffer_size,
            limits.max_storage_buffer_binding_size,
        );
        if requested > limit {
            return Err(Error::BufferLimitExceeded { requested, limit });
        }
        Ok(())
    }
}

fn effective_storage_binding_limit(
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
) -> u64 {
    max_buffer_size.min(max_storage_buffer_binding_size)
}

#[cfg(test)]
mod tests {
    use super::effective_storage_binding_limit;

    #[test]
    fn storage_binding_limit_uses_the_stricter_device_limit() {
        assert_eq!(effective_storage_binding_limit(1_024, 512), 512);
        assert_eq!(effective_storage_binding_limit(256, 512), 256);
    }
}
