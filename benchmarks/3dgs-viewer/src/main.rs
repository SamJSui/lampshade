use std::{
    error::Error,
    time::{Duration, Instant},
};

use glam::{UVec2, Vec3};
use lampshade::KeyValueSoaSorter;
use wgpu_3dgs_viewer::{
    Camera, CameraPod, RadixSorter, Viewer,
    core::{GaussianPodWithShSingleCov3dSingleConfigs, Gaussians, GaussiansSource},
};

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 1024;

type G = GaussianPodWithShSingleCov3dSingleConfigs;

#[derive(Clone, Copy)]
struct Measurement {
    wall: Duration,
    gpu: Duration,
}

struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    timestamp_period_ns: f64,
}

impl GpuTimer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("3DGS frame timestamp queries"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3DGS frame timestamp resolve"),
            size: 2 * size_of::<u64>() as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("3DGS frame timestamp readback"),
            size: 2 * size_of::<u64>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            query_set,
            resolve_buffer,
            readback_buffer,
            timestamp_period_ns: f64::from(queue.get_timestamp_period()),
        }
    }

    fn measure(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        record: impl FnOnce(&mut wgpu::CommandEncoder) -> Result<(), Box<dyn Error>>,
    ) -> Result<Measurement, Box<dyn Error>> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("3DGS frame benchmark encoder"),
        });
        encoder.write_timestamp(&self.query_set, 0);
        record(&mut encoder)?;
        encoder.write_timestamp(&self.query_set, 1);
        encoder.resolve_query_set(&self.query_set, 0..2, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.readback_buffer,
            0,
            2 * size_of::<u64>() as u64,
        );

        let command_buffer = encoder.finish();
        let start = Instant::now();
        let submission = queue.submit([command_buffer]);
        let slice = self.readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })?;
        let wall = start.elapsed();
        receiver.recv()??;
        let timestamps = {
            let mapped = slice.get_mapped_range()?;
            let values = bytemuck::cast_slice::<u8, u64>(&mapped);
            [values[0], values[1]]
        };
        self.readback_buffer.unmap();
        if timestamps[1] < timestamps[0] {
            return Err("non-monotonic GPU timestamp result".into());
        }
        let gpu = Duration::from_secs_f64(
            (timestamps[1] - timestamps[0]) as f64 * self.timestamp_period_ns / 1_000_000_000.0,
        );
        Ok(Measurement { wall, gpu })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn Error>> {
    let model_path = std::env::var("LAMPSHADE_3DGS_MODEL")
        .map_err(|_| "set LAMPSHADE_3DGS_MODEL to a .ply or .spz scene")?;
    let warmups = env_usize("LAMPSHADE_3DGS_WARMUPS", 20)?;
    let samples = env_usize("LAMPSHADE_3DGS_SAMPLES", 31)?;
    if warmups == 0 || samples == 0 {
        return Err("warmups and samples must both be positive".into());
    }
    let candidate_first = std::env::var("LAMPSHADE_3DGS_FIRST")
        .is_ok_and(|value| value.eq_ignore_ascii_case("candidate"));

    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None).await?;
    let adapter_info = adapter.get_info();
    let requirements = KeyValueSoaSorter::requirements(&adapter);
    if !requirements.accelerated {
        return Err(format!(
            "Lampshade does not advertise its accelerated SoA path for {} ({:?})",
            adapter_info.name, adapter_info.backend
        )
        .into());
    }

    let timestamp_features =
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    if !adapter.features().contains(timestamp_features) {
        return Err("adapter does not support encoder-level GPU timestamp queries".into());
    }
    let required_features = requirements.features(timestamp_features);
    let required_limits = requirements.limits(adapter.limits());
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Lampshade 3DGS benchmark device"),
            required_features,
            required_limits,
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await?;

    let gaussians = [GaussiansSource::Ply, GaussiansSource::Spz]
        .into_iter()
        .find_map(|source| Gaussians::read_from_file(&model_path, source).ok())
        .ok_or_else(|| format!("could not read {model_path:?} as PLY or SPZ"))?;
    let capacity = u32::try_from(gaussians.len())?;
    if capacity == 0 {
        return Err("scene contains no Gaussians".into());
    }

    let mut viewer = Viewer::<G>::new(&device, wgpu::TextureFormat::Rgba8Unorm, &gaussians)?;
    if !viewer.uses_lampshade_sorter() {
        return Err("production Viewer did not select Lampshade".into());
    }
    let mut sorter = RadixSorter::new(
        &device,
        &viewer.gaussians_depth_buffer,
        &viewer.indirect_indices_buffer,
    );

    let camera = Camera {
        yaw: 0.1,
        pitch: 0.1,
        ..Camera::new(0.1..1e4, 60_f32.to_radians())
    };
    viewer.update_camera_with_pod(&queue, &CameraPod::new(&camera, UVec2::new(WIDTH, HEIGHT)));
    viewer.update_model_transform(&queue, Vec3::ZERO, glam::Quat::IDENTITY, Vec3::ONE);
    let target = target(&device);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let timer = GpuTimer::new(&device, &queue);

    let before = validate_pair(&device, &queue, &mut viewer, &mut sorter, &view, &target)?;
    if before.non_background_pixels == 0 {
        return Err("validation rendered only the black clear color".into());
    }

    for iteration in 0..warmups {
        let candidate_goes_first = (iteration % 2 == 0) == candidate_first;
        if candidate_goes_first {
            let _ = measure_candidate(&timer, &device, &queue, &mut viewer, &mut sorter, &view)?;
            let _ = measure_baseline(&timer, &device, &queue, &mut viewer, &mut sorter, &view)?;
        } else {
            let _ = measure_baseline(&timer, &device, &queue, &mut viewer, &mut sorter, &view)?;
            let _ = measure_candidate(&timer, &device, &queue, &mut viewer, &mut sorter, &view)?;
        }
    }

    let mut baseline = Vec::with_capacity(samples);
    let mut candidate = Vec::with_capacity(samples);
    for sample in 0..samples {
        let candidate_goes_first = (sample % 2 == 0) == candidate_first;
        if candidate_goes_first {
            candidate.push(measure_candidate(
                &timer,
                &device,
                &queue,
                &mut viewer,
                &mut sorter,
                &view,
            )?);
            baseline.push(measure_baseline(
                &timer,
                &device,
                &queue,
                &mut viewer,
                &mut sorter,
                &view,
            )?);
        } else {
            baseline.push(measure_baseline(
                &timer,
                &device,
                &queue,
                &mut viewer,
                &mut sorter,
                &view,
            )?);
            candidate.push(measure_candidate(
                &timer,
                &device,
                &queue,
                &mut viewer,
                &mut sorter,
                &view,
            )?);
        }
    }

    let after = validate_pair(&device, &queue, &mut viewer, &mut sorter, &view, &target)?;
    if before.baseline_pixels != after.baseline_pixels {
        return Err("baseline output changed across the timed run".into());
    }

    print_report(
        &adapter_info,
        &model_path,
        capacity,
        warmups,
        samples,
        candidate_first,
        before.non_background_pixels,
        &baseline,
        &candidate,
        timer.timestamp_period_ns,
    );
    Ok(())
}

struct Validation {
    baseline_pixels: Vec<u8>,
    non_background_pixels: usize,
}

fn validate_pair(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    viewer: &mut Viewer<G>,
    sorter: &mut RadixSorter,
    view: &wgpu::TextureView,
    texture: &wgpu::Texture,
) -> Result<Validation, Box<dyn Error>> {
    let baseline_pixels = render_and_read(device, queue, texture, |encoder| {
        record_path(encoder, viewer, sorter, view, false)
    })?;
    let candidate_pixels = render_and_read(device, queue, texture, |encoder| {
        record_path(encoder, viewer, sorter, view, true)
    })?;
    if baseline_pixels != candidate_pixels {
        let mismatch = baseline_pixels
            .iter()
            .zip(&candidate_pixels)
            .position(|(left, right)| left != right)
            .unwrap_or(baseline_pixels.len().min(candidate_pixels.len()));
        return Err(format!("baseline/candidate RGBA mismatch at byte {mismatch}").into());
    }
    let non_background_pixels = baseline_pixels
        .chunks_exact(4)
        .filter(|pixel| *pixel != [0, 0, 0, 255])
        .count();
    Ok(Validation {
        baseline_pixels,
        non_background_pixels,
    })
}

fn render_and_read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    record: impl FnOnce(&mut wgpu::CommandEncoder) -> Result<(), Box<dyn Error>>,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let bytes_per_row = WIDTH * 4;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("3DGS validation readback"),
        size: u64::from(bytes_per_row) * u64::from(HEIGHT),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("3DGS validation encoder"),
    });
    record(&mut encoder)?;
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(HEIGHT),
            },
        },
        texture.size(),
    );
    let submission = queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    })?;
    receiver.recv()??;
    let pixels = {
        let mapped = slice.get_mapped_range()?;
        mapped.to_vec()
    };
    readback.unmap();
    Ok(pixels)
}

fn measure_baseline(
    timer: &GpuTimer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    viewer: &mut Viewer<G>,
    sorter: &mut RadixSorter,
    view: &wgpu::TextureView,
) -> Result<Measurement, Box<dyn Error>> {
    timer.measure(device, queue, |encoder| {
        record_path(encoder, viewer, sorter, view, false)
    })
}

fn measure_candidate(
    timer: &GpuTimer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    viewer: &mut Viewer<G>,
    sorter: &mut RadixSorter,
    view: &wgpu::TextureView,
) -> Result<Measurement, Box<dyn Error>> {
    timer.measure(device, queue, |encoder| {
        record_path(encoder, viewer, sorter, view, true)
    })
}

fn record_path(
    encoder: &mut wgpu::CommandEncoder,
    viewer: &mut Viewer<G>,
    embedded: &mut RadixSorter,
    view: &wgpu::TextureView,
    candidate: bool,
) -> Result<(), Box<dyn Error>> {
    if !candidate {
        std::mem::swap(&mut viewer.radix_sorter, embedded);
    }
    assert_eq!(viewer.uses_lampshade_sorter(), candidate);
    // Both arms use the public production API on the same model/buffers.
    viewer.render(encoder, view);
    if !candidate {
        std::mem::swap(&mut viewer.radix_sorter, embedded);
    }
    Ok(())
}

fn target(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("3DGS benchmark render target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

#[allow(clippy::too_many_arguments)]
fn print_report(
    adapter: &wgpu::AdapterInfo,
    model_path: &str,
    splats: u32,
    warmups: usize,
    samples: usize,
    candidate_first: bool,
    non_background_pixels: usize,
    baseline: &[Measurement],
    candidate: &[Measurement],
    timestamp_period_ns: f64,
) {
    let baseline_wall = durations(baseline, |value| value.wall);
    let candidate_wall = durations(candidate, |value| value.wall);
    let baseline_gpu = durations(baseline, |value| value.gpu);
    let candidate_gpu = durations(candidate, |value| value.gpu);
    let baseline_wall_median = median_ms(&baseline_wall);
    let candidate_wall_median = median_ms(&candidate_wall);
    let baseline_gpu_median = median_ms(&baseline_gpu);
    let candidate_gpu_median = median_ms(&candidate_gpu);

    println!("benchmark=lampshade-3dgs-production-v2");
    println!(
        "adapter_name={:?} vendor=0x{:04x} device=0x{:04x} device_type={:?} backend={:?}",
        adapter.name, adapter.vendor, adapter.device, adapter.device_type, adapter.backend
    );
    println!(
        "driver={:?} driver_info={:?}",
        adapter.driver, adapter.driver_info
    );
    println!("model={model_path:?} splats={splats} resolution={WIDTH}x{HEIGHT}");
    println!(
        "warmups={warmups} samples={samples} first={} timestamp_period_ns={timestamp_period_ns}",
        if candidate_first {
            "candidate"
        } else {
            "baseline"
        }
    );
    println!(
        "validation=byte-identical-before-and-after non_background_pixels={non_background_pixels} lampshade_accelerated=true"
    );
    println!(
        "scope=preprocess+sort+render; command_encoding=excluded; wall=instrumented-serialized-submit-to-completion; gpu=encoder-timestamps"
    );
    println!(
        "wall_baseline_median_ms={baseline_wall_median:.6} wall_candidate_median_ms={candidate_wall_median:.6} wall_reduction_pct={:.3} wall_speedup={:.4}",
        reduction_pct(baseline_wall_median, candidate_wall_median),
        baseline_wall_median / candidate_wall_median,
    );
    println!(
        "gpu_baseline_median_ms={baseline_gpu_median:.6} gpu_candidate_median_ms={candidate_gpu_median:.6} gpu_reduction_pct={:.3} gpu_speedup={:.4}",
        reduction_pct(baseline_gpu_median, candidate_gpu_median),
        baseline_gpu_median / candidate_gpu_median,
    );
    println!("raw_wall_baseline_ms={}", format_samples(&baseline_wall));
    println!("raw_wall_candidate_ms={}", format_samples(&candidate_wall));
    println!("raw_gpu_baseline_ms={}", format_samples(&baseline_gpu));
    println!("raw_gpu_candidate_ms={}", format_samples(&candidate_gpu));
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    Ok(std::env::var(name).map_or(Ok(default), |value| value.parse())?)
}

fn durations(
    measurements: &[Measurement],
    field: impl Fn(Measurement) -> Duration,
) -> Vec<Duration> {
    measurements.iter().copied().map(field).collect()
}

fn median_ms(samples: &[Duration]) -> f64 {
    let mut values: Vec<_> = samples
        .iter()
        .map(|duration| duration.as_secs_f64() * 1_000.0)
        .collect();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn reduction_pct(baseline_ms: f64, candidate_ms: f64) -> f64 {
    (1.0 - candidate_ms / baseline_ms) * 100.0
}

fn format_samples(samples: &[Duration]) -> String {
    samples
        .iter()
        .map(|duration| format!("{:.6}", duration.as_secs_f64() * 1_000.0))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_handles_odd_and_even_samples() {
        let samples = [7, 1, 6, 2, 5, 3, 4].map(Duration::from_millis);
        assert_eq!(median_ms(&samples), 4.0);
        assert_eq!(median_ms(&samples[..4]), 4.0);
        assert_eq!(median_ms(&samples[..1]), 7.0);
    }
}
