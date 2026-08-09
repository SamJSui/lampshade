use crate::{
    Error,
    profiling::{GpuProfile, TimestampRecorder},
};

/// One command encoder that will be submitted as a unit.
///
/// This private type standardizes immediate primitive execution while leaving
/// caller-owned `record_*` composition completely under the caller's control.
pub(crate) struct CommandSession {
    encoder: wgpu::CommandEncoder,
}

impl CommandSession {
    pub(crate) fn new(device: &wgpu::Device, label: Option<&str>) -> Self {
        Self {
            encoder: device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label }),
        }
    }

    pub(crate) fn encoder(&mut self) -> &mut wgpu::CommandEncoder {
        &mut self.encoder
    }

    pub(crate) fn submit(self, queue: &wgpu::Queue) -> wgpu::SubmissionIndex {
        queue.submit(Some(self.encoder.finish()))
    }
}

/// A command session with optional timestamp-query storage.
///
/// A zero span count still records and waits for commands, but avoids creating
/// an invalid zero-sized query allocation. That preserves trivial operations
/// such as clearing an empty compaction count buffer.
pub(crate) struct ProfileSession {
    commands: CommandSession,
    profiler: Option<TimestampRecorder>,
}

impl ProfileSession {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        span_count: u32,
        label: &'static str,
    ) -> Result<Self, Error> {
        Ok(Self {
            commands: CommandSession::new(device, Some(label)),
            profiler: (span_count > 0)
                .then(|| TimestampRecorder::new(device, queue, span_count))
                .transpose()?,
        })
    }

    pub(crate) fn recording(
        &mut self,
    ) -> (&mut wgpu::CommandEncoder, Option<&mut TimestampRecorder>) {
        (self.commands.encoder(), self.profiler.as_mut())
    }

    pub(crate) async fn finish(
        mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<GpuProfile, Error> {
        let Some(profiler) = self.profiler.take() else {
            let submission = self.commands.submit(queue);
            device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })?;
            return Ok(GpuProfile::empty());
        };

        profiler.resolve(self.commands.encoder());
        let submission = self.commands.submit(queue);
        profiler.read(device, submission).await
    }
}
