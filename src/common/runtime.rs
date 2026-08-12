use crate::{
    Error,
    profiling::{GpuProfile, TimestampRecorder},
};

#[cfg(target_arch = "wasm32")]
use std::{
    any::Any,
    cell::{Cell, RefCell},
    collections::BTreeMap,
};

#[cfg(target_arch = "wasm32")]
thread_local! {
    static NEXT_KEEPALIVE_ID: Cell<u64> = const { Cell::new(0) };
    static SUBMISSION_KEEPALIVES: RefCell<BTreeMap<u64, Box<dyn Any>>> =
        RefCell::new(BTreeMap::new());
}

#[cfg(target_arch = "wasm32")]
struct SubmissionKeepaliveToken(u64);

#[cfg(target_arch = "wasm32")]
impl Drop for SubmissionKeepaliveToken {
    fn drop(&mut self) {
        SUBMISSION_KEEPALIVES.with(|keepalives| {
            let removed = keepalives.borrow_mut().remove(&self.0);
            debug_assert!(removed.is_some());
        });
    }
}

/// Retain transient command resources until their encoded work has completed.
///
/// Native wgpu handles are `Send`, so the completion callback can own them
/// directly. WebGPU handles are intentionally thread-bound; on wasm the
/// callback instead owns a numeric token and releases the resources from
/// thread-local storage on the browser's event-loop thread.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn defer_drop<T: Send + 'static>(encoder: &wgpu::CommandEncoder, value: T) {
    encoder.on_submitted_work_done(move || drop(value));
}

#[cfg(target_arch = "wasm32")]
pub(crate) fn defer_drop<T: 'static>(encoder: &wgpu::CommandEncoder, value: T) {
    let id = NEXT_KEEPALIVE_ID.with(|next| {
        let id = next.get();
        next.set(
            id.checked_add(1)
                .expect("submission keepalive id exhausted"),
        );
        id
    });
    SUBMISSION_KEEPALIVES.with(|keepalives| {
        let previous = keepalives.borrow_mut().insert(id, Box::new(value));
        debug_assert!(previous.is_none());
    });
    let token = SubmissionKeepaliveToken(id);
    encoder.on_submitted_work_done(move || drop(token));
}

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
