use std::time::Duration;

use futures::channel::oneshot;

use crate::Error;

const TIMESTAMP_SIZE_BYTES: u64 = size_of::<u64>() as u64;

/// GPU time measured for one labeled compute dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuTimestampSpan {
    pub label: String,
    pub duration: Duration,
}

/// Timestamp-query measurements for one primitive invocation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GpuProfile {
    /// Time from the beginning of the first dispatch through the end of the last dispatch.
    pub gpu_elapsed: Duration,
    /// Sum of the measured compute-pass durations.
    pub dispatch_time: Duration,
    pub spans: Vec<GpuTimestampSpan>,
}

impl GpuProfile {
    pub(crate) fn empty() -> Self {
        Self::default()
    }
}

pub(crate) struct TimestampRecorder {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    labels: Vec<String>,
    span_capacity: u32,
    timestamp_period_ns: f64,
}

impl TimestampRecorder {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        span_capacity: u32,
    ) -> Result<Self, Error> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Err(Error::TimestampQueriesUnsupported);
        }

        let query_count = span_capacity.checked_mul(2).ok_or(Error::SizeOverflow)?;
        let size_bytes = u64::from(query_count)
            .checked_mul(TIMESTAMP_SIZE_BYTES)
            .ok_or(Error::SizeOverflow)?;

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Primitive Timestamp Queries"),
            ty: wgpu::QueryType::Timestamp,
            count: query_count,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primitive Timestamp Resolve"),
            size: size_bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primitive Timestamp Readback"),
            size: size_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            labels: Vec::with_capacity(span_capacity as usize),
            span_capacity,
            timestamp_period_ns: f64::from(queue.get_timestamp_period()),
        })
    }

    fn reserve(&mut self, label: String) -> wgpu::ComputePassTimestampWrites<'_> {
        assert!(
            self.labels.len() < self.span_capacity as usize,
            "timestamp span capacity must match the recorded dispatch count"
        );
        let beginning = self.labels.len() as u32 * 2;
        self.labels.push(label);

        wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(beginning),
            end_of_pass_write_index: Some(beginning + 1),
        }
    }

    pub(crate) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let query_count = self.labels.len() as u32 * 2;
        let size_bytes = u64::from(query_count) * TIMESTAMP_SIZE_BYTES;
        encoder.resolve_query_set(&self.query_set, 0..query_count, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            size_bytes,
        );
    }

    pub(crate) async fn read(
        self,
        device: &wgpu::Device,
        submission: wgpu::SubmissionIndex,
    ) -> Result<GpuProfile, Error> {
        let size_bytes = self.labels.len() as u64 * 2 * TIMESTAMP_SIZE_BYTES;
        let slice = self.readback_buffer.slice(..size_bytes);
        let (sender, receiver) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })?;
        receiver.await.map_err(|_| Error::ReadbackChannelClosed)??;

        let timestamps: Vec<u64> = {
            let data = slice.get_mapped_range()?;
            bytemuck::cast_slice(&data).to_vec()
        };
        self.readback_buffer.unmap();

        build_profile(self.labels, &timestamps, self.timestamp_period_ns)
    }
}

pub(crate) fn record_compute_pass(
    encoder: &mut wgpu::CommandEncoder,
    pass_label: &'static str,
    profile_label: Option<String>,
    profiler: Option<&mut TimestampRecorder>,
    record: impl FnOnce(&mut wgpu::ComputePass<'_>),
) {
    let timestamp_writes = profiler
        .map(|profiler| profiler.reserve(profile_label.unwrap_or_else(|| pass_label.to_owned())));
    let descriptor = wgpu::ComputePassDescriptor {
        label: Some(pass_label),
        timestamp_writes,
    };
    let mut pass = encoder.begin_compute_pass(&descriptor);
    record(&mut pass);
}

fn build_profile(
    labels: Vec<String>,
    timestamps: &[u64],
    period_ns: f64,
) -> Result<GpuProfile, Error> {
    let has_nonzero_timestamp = timestamps.iter().any(|timestamp| *timestamp != 0);
    let mut previous_end = None;
    let mut spans = Vec::with_capacity(labels.len());
    for (label, timestamps) in labels.into_iter().zip(timestamps.chunks_exact(2)) {
        let beginning = timestamps[0];
        let end = timestamps[1];
        let unavailable = has_nonzero_timestamp && beginning == 0 && end == 0;
        let non_monotonic =
            end < beginning || previous_end.is_some_and(|previous_end| beginning < previous_end);
        if unavailable || non_monotonic {
            return Err(Error::TimestampQueryResultUnavailable { label });
        }
        spans.push(GpuTimestampSpan {
            label,
            duration: ticks_to_duration(end - beginning, period_ns),
        });
        previous_end = Some(end);
    }
    let dispatch_time = spans
        .iter()
        .map(|span| span.duration)
        .fold(Duration::ZERO, |total, duration| total + duration);
    let gpu_elapsed = match (timestamps.first(), timestamps.last()) {
        (Some(first), Some(last)) => ticks_to_duration(last.saturating_sub(*first), period_ns),
        _ => Duration::ZERO,
    };

    Ok(GpuProfile {
        gpu_elapsed,
        dispatch_time,
        spans,
    })
}

fn ticks_to_duration(ticks: u64, period_ns: f64) -> Duration {
    Duration::from_secs_f64(ticks as f64 * period_ns / 1_000_000_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_labeled_profile_from_timestamp_pairs() {
        let profile = build_profile(
            vec!["reduce".to_owned(), "scatter".to_owned()],
            &[10, 20, 25, 45],
            2.0,
        )
        .expect("valid timestamps should build a profile");

        assert_eq!(profile.spans.len(), 2);
        assert_eq!(profile.spans[0].duration, Duration::from_nanos(20));
        assert_eq!(profile.spans[1].duration, Duration::from_nanos(40));
        assert_eq!(profile.dispatch_time, Duration::from_nanos(60));
        assert_eq!(profile.gpu_elapsed, Duration::from_nanos(70));
    }

    #[test]
    fn rejects_unwritten_or_non_monotonic_timestamp_pairs() {
        let unwritten = build_profile(
            vec!["prepare".to_owned(), "reduce".to_owned()],
            &[10, 20, 0, 0],
            1.0,
        );
        assert!(matches!(
            unwritten,
            Err(Error::TimestampQueryResultUnavailable { label }) if label == "reduce"
        ));

        let non_monotonic = build_profile(
            vec!["prepare".to_owned(), "reduce".to_owned()],
            &[10, 20, 15, 25],
            1.0,
        );
        assert!(matches!(
            non_monotonic,
            Err(Error::TimestampQueryResultUnavailable { label }) if label == "reduce"
        ));
    }
}
