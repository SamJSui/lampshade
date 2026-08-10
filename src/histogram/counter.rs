use crate::{
    Error, common,
    common::runtime::{CommandSession, ProfileSession},
    context::Context,
    profiling::{GpuProfile, TimestampRecorder},
};

use super::pipeline::{HistogramDispatch, HistogramPipeline, MAX_BINS};

const VALUE_SIZE_BYTES: u64 = size_of::<u32>() as u64;

/// Counts `u32` values into a bounded set of bins.
///
/// Input values in `0..bin_count` increment their corresponding bin. Values
/// outside that range are ignored. The portable implementation supports at
/// most [`Self::MAX_BINS`] bins so every workgroup can privatize its counters
/// before merging them into the caller-owned output.
pub struct Histogram {
    pipeline: HistogramPipeline,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Histogram {
    /// Maximum number of bins supported by the portable kernel.
    pub const MAX_BINS: u32 = MAX_BINS;

    /// Creates a histogram primitive for an existing device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            pipeline: HistogramPipeline::new(device),
            device: device.clone(),
            queue: queue.clone(),
        }
    }

    /// Creates a histogram primitive from the crate's convenience context.
    pub fn from_context(context: &Context) -> Self {
        Self::new(&context.device, &context.queue)
    }

    /// Returns the required output allocation size for `bin_count` counters.
    pub fn output_buffer_size(bin_count: u32) -> Result<u64, Error> {
        validate_bin_count(bin_count)?;
        common::math::checked_byte_size(u64::from(bin_count), VALUE_SIZE_BYTES)
    }

    /// Uploads values, counts them on the GPU, and downloads the bins.
    pub async fn histogram(&self, input: &[u32], bin_count: u32) -> Result<Vec<u32>, Error> {
        let output_bytes = Self::output_buffer_size(bin_count)?;
        if input.is_empty() {
            return Ok(vec![0; bin_count as usize]);
        }

        let num_items = common::math::checked_u32(input.len() as u64)?;
        let input_bytes = common::math::checked_byte_size(u64::from(num_items), VALUE_SIZE_BYTES)?;
        self.validate_storage_binding_size(input_bytes)?;
        let input_buffer = common::buffers::create_storage_buffer(&self.device, input);
        let output_buffer =
            common::buffers::create_empty_storage_buffer(&self.device, output_bytes);
        self.histogram_gpu_to_gpu(&input_buffer, &output_buffer, num_items, bin_count)?;
        common::buffers::download_buffer(
            &self.device,
            &self.queue,
            &output_buffer,
            bin_count as usize,
        )
        .await
    }

    /// Counts values in caller-owned GPU buffers and submits immediately.
    ///
    /// `input` requires `STORAGE`. `output` requires `STORAGE | COPY_DST` and
    /// [`Self::output_buffer_size`] bytes. The output is cleared before counts
    /// are accumulated, and the buffers must be distinct.
    pub fn histogram_gpu_to_gpu(
        &self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        bin_count: u32,
    ) -> Result<(), Error> {
        let mut commands = CommandSession::new(&self.device, None);
        self.record_histogram(commands.encoder(), input, output, num_items, bin_count)?;
        commands.submit(&self.queue);
        Ok(())
    }

    /// Records histogram counting without submitting or waiting.
    ///
    /// Buffer requirements match [`Self::histogram_gpu_to_gpu`]. Multiple
    /// calls can be composed in one command encoder; each call clears its own
    /// output before dispatching.
    pub fn record_histogram(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        bin_count: u32,
    ) -> Result<(), Error> {
        self.record_commands(encoder, input, output, num_items, bin_count, None)
    }

    /// Profiles caller-owned GPU histogram counting using timestamp queries.
    pub async fn profile_histogram_gpu_to_gpu(
        &self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        bin_count: u32,
    ) -> Result<GpuProfile, Error> {
        let span_count = u32::from(num_items > 0);
        let mut profile =
            ProfileSession::new(&self.device, &self.queue, span_count, "Profiled Histogram")?;
        let (encoder, profiler) = profile.recording();
        self.record_commands(encoder, input, output, num_items, bin_count, profiler)?;
        profile.finish(&self.device, &self.queue).await
    }

    fn record_commands(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        num_items: u32,
        bin_count: u32,
        profiler: Option<&mut TimestampRecorder>,
    ) -> Result<(), Error> {
        if input == output {
            return Err(Error::BufferAlias {
                first: "histogram input",
                second: "histogram output",
            });
        }

        let output_bytes = Self::output_buffer_size(bin_count)?;
        common::buffers::validate_buffer(
            output,
            "histogram output",
            output_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )?;
        if num_items == 0 {
            encoder.clear_buffer(output, 0, Some(output_bytes));
            return Ok(());
        }

        let input_bytes = common::math::checked_byte_size(u64::from(num_items), VALUE_SIZE_BYTES)?;
        self.validate_storage_binding_size(input_bytes)?;
        common::buffers::validate_buffer(
            input,
            "histogram input",
            input_bytes,
            wgpu::BufferUsages::STORAGE,
        )?;
        encoder.clear_buffer(output, 0, Some(output_bytes));
        self.pipeline.dispatch(
            &self.device,
            &self.queue,
            encoder,
            HistogramDispatch {
                input,
                output,
                num_items,
                bin_count,
            },
            profiler,
        );
        Ok(())
    }

    fn validate_storage_binding_size(&self, requested: u64) -> Result<(), Error> {
        let limits = self.device.limits();
        let limit = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size);
        if requested > limit {
            return Err(Error::BufferLimitExceeded { requested, limit });
        }
        Ok(())
    }
}

fn validate_bin_count(bin_count: u32) -> Result<(), Error> {
    if (1..=MAX_BINS).contains(&bin_count) {
        Ok(())
    } else {
        Err(Error::InvalidHistogramBinCount {
            bins: bin_count,
            max: MAX_BINS,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_the_portable_bin_range() {
        assert!(validate_bin_count(1).is_ok());
        assert!(validate_bin_count(MAX_BINS).is_ok());
        assert!(matches!(
            validate_bin_count(0),
            Err(Error::InvalidHistogramBinCount { bins: 0, .. })
        ));
        assert!(matches!(
            validate_bin_count(MAX_BINS + 1),
            Err(Error::InvalidHistogramBinCount { .. })
        ));
    }
}
