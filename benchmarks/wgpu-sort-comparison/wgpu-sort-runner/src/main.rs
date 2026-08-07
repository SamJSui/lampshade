use std::hint::black_box;
use std::num::NonZeroU32;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;
use wgpu_sort::{GPUSorter, SortBuffers};
use wgpu_sort_benchmark_common::{
    AdapterMetadata, BenchmarkConfig, BenchmarkMode, BenchmarkRun, GeneratorMetadata, LogicalInput,
    SCHEMA_VERSION, WGPU_SORT_REVISION, median, wgpu_sort_pinned_memory,
};

fn main() {
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("wgpu_sort comparison runner failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = BenchmarkConfig::from_env()?;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
        .await
        .ok_or("no compatible wgpu_sort adapter found")?;
    let info = adapter.get_info();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("wgpu_sort Comparison Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
            },
            None,
        )
        .await?;
    let subgroup_size = wgpu_sort::utils::guess_workgroup_size(&device, &queue)
        .await
        .ok_or("wgpu_sort could not find a working subgroup size")?;
    let sorter = GPUSorter::new(&device, subgroup_size);
    let logical = LogicalInput::generate(config.items, config.workload);
    let expected = logical.stable_sorted_pairs();
    let sort_buffers = sorter.create_sort_buffers(
        &device,
        NonZeroU32::new(config.items).expect("config rejects zero items"),
    );
    let resident_input = matches!(config.mode, BenchmarkMode::Resident)
        .then(|| ResidentInput::new(&device, &logical));

    let (keys, values) = match config.mode {
        BenchmarkMode::Resident => {
            restore_input(
                &device,
                &queue,
                &sort_buffers,
                resident_input
                    .as_ref()
                    .expect("resident mode retains its input backup"),
            )?;
            submit_sort(&device, &queue, &sorter, &sort_buffers)?;
            read_results(&device, &queue, &sort_buffers, config.items)?
        }
        BenchmarkMode::RoundTrip => {
            upload_sort_and_read(&device, &queue, &sorter, &sort_buffers, &logical)?
        }
    };
    if !keys.into_iter().zip(values).eq(expected.iter().copied()) {
        return Err(format!(
            "wgpu_sort {} path did not match the stable CPU reference",
            config.mode.as_str()
        )
        .into());
    }
    drop(expected);

    let resident_buffers = match config.mode {
        BenchmarkMode::Resident => Some(sort_buffers),
        BenchmarkMode::RoundTrip => None,
    };

    let warmup_started = Instant::now();
    let mut warmups_completed = 0;
    while warmups_completed < config.warmups
        || warmup_started.elapsed() < Duration::from_millis(config.warmup_ms)
    {
        if matches!(config.mode, BenchmarkMode::Resident) {
            restore_input(
                &device,
                &queue,
                resident_buffers
                    .as_ref()
                    .expect("resident mode retains its buffers"),
                resident_input
                    .as_ref()
                    .expect("resident mode retains its input backup"),
            )?;
        }
        run_once(
            config.mode,
            &device,
            &queue,
            &sorter,
            resident_buffers.as_ref(),
            &logical,
        )?;
        warmups_completed += 1;
    }

    let mut samples_ms = Vec::with_capacity(config.samples as usize);
    for _ in 0..config.samples {
        if matches!(config.mode, BenchmarkMode::Resident) {
            restore_input(
                &device,
                &queue,
                resident_buffers
                    .as_ref()
                    .expect("resident mode retains its buffers"),
                resident_input
                    .as_ref()
                    .expect("resident mode retains its input backup"),
            )?;
        }
        let start = Instant::now();
        run_once(
            config.mode,
            &device,
            &queue,
            &sorter,
            resident_buffers.as_ref(),
            &logical,
        )?;
        samples_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    let median_ms = median(&samples_ms);
    let run = BenchmarkRun {
        schema_version: SCHEMA_VERSION,
        implementation: "wgpu_sort".into(),
        implementation_version: runtime_metadata("WGPU_SORT_BENCH_IMPLEMENTATION_VERSION", "git"),
        implementation_revision: runtime_metadata(
            "WGPU_SORT_BENCH_IMPLEMENTATION_REVISION",
            WGPU_SORT_REVISION,
        ),
        wgpu_version: "0.20.1".into(),
        adapter: AdapterMetadata {
            name: info.name,
            vendor: info.vendor,
            device: info.device,
            device_type: format!("{:?}", info.device_type),
            backend: format!("{:?}", info.backend),
            driver: info.driver,
            driver_info: info.driver_info,
            subgroup_min_size: Some(subgroup_size),
            subgroup_max_size: Some(subgroup_size),
        },
        config: config.clone(),
        generator: GeneratorMetadata::current(),
        correctness_checked: true,
        samples_ms,
        median_ms,
        throughput_pairs_per_second: f64::from(config.items) / (median_ms / 1_000.0),
        memory: wgpu_sort_pinned_memory(config.items),
    };
    println!("{}", serde_json::to_string(&run)?);
    Ok(())
}

fn run_once(
    mode: BenchmarkMode,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sorter: &GPUSorter,
    resident_buffers: Option<&SortBuffers>,
    input: &LogicalInput,
) -> Result<(), Box<dyn std::error::Error>> {
    match mode {
        BenchmarkMode::Resident => submit_sort(
            device,
            queue,
            sorter,
            resident_buffers.expect("resident mode retains its buffers"),
        )?,
        BenchmarkMode::RoundTrip => {
            let buffers = sorter.create_sort_buffers(
                device,
                NonZeroU32::new(input.keys.len() as u32).expect("input is nonempty"),
            );
            let (keys, values) = upload_sort_and_read(device, queue, sorter, &buffers, input)?;
            black_box((keys, values));
        }
    }
    Ok(())
}

fn restore_input(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffers: &SortBuffers,
    input: &ResidentInput,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_sort Input Restore Encoder"),
    });
    encoder.copy_buffer_to_buffer(&input.keys, 0, buffers.keys(), 0, input.size);
    encoder.copy_buffer_to_buffer(&input.values, 0, buffers.values(), 0, input.size);
    let submission = queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
    Ok(())
}

struct ResidentInput {
    keys: wgpu::Buffer,
    values: wgpu::Buffer,
    size: u64,
}

impl ResidentInput {
    fn new(device: &wgpu::Device, input: &LogicalInput) -> Self {
        let keys = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_sort Resident Key Backup"),
            contents: bytemuck::cast_slice(&input.keys),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let values = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wgpu_sort Resident Value Backup"),
            contents: bytemuck::cast_slice(&input.values),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        Self {
            size: input.keys.len() as u64 * size_of::<u32>() as u64,
            keys,
            values,
        }
    }
}

fn submit_sort(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sorter: &GPUSorter,
    buffers: &SortBuffers,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_sort Comparison Encoder"),
    });
    sorter.sort(&mut encoder, queue, buffers, None);
    let submission = queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
    Ok(())
}

fn upload_sort_and_read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sorter: &GPUSorter,
    buffers: &SortBuffers,
    input: &LogicalInput,
) -> Result<(Vec<u32>, Vec<u32>), Box<dyn std::error::Error>> {
    let size = input.keys.len() as u64 * size_of::<u32>() as u64;
    let key_readback = readback_buffer(device, "wgpu_sort Key Readback", size);
    let value_readback = readback_buffer(device, "wgpu_sort Value Readback", size);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_sort Round Trip Encoder"),
    });
    wgpu_sort::utils::upload_to_buffer(&mut encoder, buffers.keys(), device, &input.keys);
    wgpu_sort::utils::upload_to_buffer(&mut encoder, buffers.values(), device, &input.values);
    sorter.sort(&mut encoder, queue, buffers, None);
    encoder.copy_buffer_to_buffer(buffers.keys(), 0, &key_readback, 0, size);
    encoder.copy_buffer_to_buffer(buffers.values(), 0, &value_readback, 0, size);
    let submission = queue.submit([encoder.finish()]);
    map_two(device, submission, &key_readback, &value_readback)
}

fn read_results(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffers: &SortBuffers,
    items: u32,
) -> Result<(Vec<u32>, Vec<u32>), Box<dyn std::error::Error>> {
    let size = u64::from(items) * size_of::<u32>() as u64;
    let key_readback = readback_buffer(device, "wgpu_sort Key Validation", size);
    let value_readback = readback_buffer(device, "wgpu_sort Value Validation", size);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("wgpu_sort Validation Readback"),
    });
    encoder.copy_buffer_to_buffer(buffers.keys(), 0, &key_readback, 0, size);
    encoder.copy_buffer_to_buffer(buffers.values(), 0, &value_readback, 0, size);
    let submission = queue.submit([encoder.finish()]);
    map_two(device, submission, &key_readback, &value_readback)
}

fn map_two(
    device: &wgpu::Device,
    submission: wgpu::SubmissionIndex,
    keys: &wgpu::Buffer,
    values: &wgpu::Buffer,
) -> Result<(Vec<u32>, Vec<u32>), Box<dyn std::error::Error>> {
    let key_slice = keys.slice(..);
    let value_slice = values.slice(..);
    let (key_sender, key_receiver) = mpsc::channel();
    let (value_sender, value_receiver) = mpsc::channel();
    key_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = key_sender.send(result);
    });
    value_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = value_sender.send(result);
    });
    device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
    key_receiver.recv()??;
    value_receiver.recv()??;
    let key_result = bytemuck::cast_slice(&key_slice.get_mapped_range()).to_vec();
    let value_result = bytemuck::cast_slice(&value_slice.get_mapped_range()).to_vec();
    keys.unmap();
    values.unmap();
    Ok((key_result, value_result))
}

fn readback_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn runtime_metadata(name: &str, fallback: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| fallback.into())
}
