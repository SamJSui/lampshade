use std::{
    hint::black_box,
    sync::mpsc,
    time::{Duration, Instant},
};

use lampshade::{Context, KeyValue, KeyValueSorter};
use wgpu::util::DeviceExt;
use wgpu_sort_benchmark_common::{
    AdapterMetadata, BenchmarkConfig, BenchmarkMode, BenchmarkRun, GeneratorMetadata, LogicalInput,
    MemoryEstimate, SCHEMA_VERSION, Workload, lampshade_eight_bit_memory, median,
};

fn main() {
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("Lampshade comparison runner failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = BenchmarkConfig::from_env()?;
    let context = Context::init().await?;
    let logical = LogicalInput::generate(config.items, config.workload);
    let input: Vec<_> = logical
        .keys
        .iter()
        .copied()
        .zip(logical.values.iter().copied())
        .map(|(key, value)| KeyValue::new(key, value))
        .collect();
    let expected: Vec<_> = logical
        .stable_sorted_pairs()
        .into_iter()
        .map(|(key, value)| KeyValue::new(key, value))
        .collect();
    let key_bits = workload_key_bits(config.workload);

    let mut sorter = KeyValueSorter::from_context(&context);
    let resident_buffers = match config.mode {
        BenchmarkMode::Resident => {
            let gpu_input = context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Comparison Input"),
                    contents: bytemuck::cast_slice(&input),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let gpu_output = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Comparison Output"),
                size: gpu_input.size(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            sorter.sort_gpu_to_gpu_with_key_bits(
                &gpu_input,
                &gpu_output,
                config.items,
                key_bits,
            )?;
            wait_for_gpu(&context.device)?;
            let actual = read_buffer::<KeyValue>(
                &context.device,
                &context.queue,
                &gpu_output,
                config.items,
            )?;
            if actual != expected {
                return Err("resident sort did not match the stable CPU reference".into());
            }
            Some((gpu_input, gpu_output))
        }
        BenchmarkMode::RoundTrip => {
            let actual = sorter.sort_with_key_bits(&input, key_bits).await?;
            if actual != expected {
                return Err("round-trip sort did not match the stable CPU reference".into());
            }
            None
        }
    };
    drop(expected);

    let warmup_started = Instant::now();
    let mut warmups_completed = 0;
    while warmups_completed < config.warmups
        || warmup_started.elapsed() < Duration::from_millis(config.warmup_ms)
    {
        run_once(
            config.mode,
            &context,
            &mut sorter,
            &input,
            resident_buffers.as_ref(),
            key_bits,
        )
        .await?;
        warmups_completed += 1;
    }

    let mut samples_ms = Vec::with_capacity(config.samples as usize);
    for _ in 0..config.samples {
        let start = Instant::now();
        run_once(
            config.mode,
            &context,
            &mut sorter,
            &input,
            resident_buffers.as_ref(),
            key_bits,
        )
        .await?;
        samples_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }

    let median_ms = median(&samples_ms);
    let memory = memory_estimate(&context, config.items);
    let run = BenchmarkRun {
        schema_version: SCHEMA_VERSION,
        implementation: "lampshade".into(),
        implementation_version: runtime_metadata(
            "WGPU_SORT_BENCH_IMPLEMENTATION_VERSION",
            "working-tree",
        ),
        implementation_revision: runtime_metadata(
            "WGPU_SORT_BENCH_IMPLEMENTATION_REVISION",
            "working-tree",
        ),
        wgpu_version: "28.0.0".into(),
        adapter: adapter_metadata(&context),
        config: config.clone(),
        generator: GeneratorMetadata::current(),
        correctness_checked: true,
        samples_ms,
        median_ms,
        throughput_pairs_per_second: f64::from(config.items) / (median_ms / 1_000.0),
        memory,
    };
    println!("{}", serde_json::to_string(&run)?);
    Ok(())
}

async fn run_once(
    mode: BenchmarkMode,
    context: &Context,
    sorter: &mut KeyValueSorter,
    input: &[KeyValue],
    resident_buffers: Option<&(wgpu::Buffer, wgpu::Buffer)>,
    key_bits: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    match mode {
        BenchmarkMode::Resident => {
            let (gpu_input, gpu_output) =
                resident_buffers.expect("resident mode retains its input and output buffers");
            sorter.sort_gpu_to_gpu_with_key_bits(
                gpu_input,
                gpu_output,
                input.len() as u32,
                key_bits,
            )?;
            wait_for_gpu(&context.device)?;
        }
        BenchmarkMode::RoundTrip => {
            let output = sorter.sort_with_key_bits(input, key_bits).await?;
            black_box(output);
        }
    }
    Ok(())
}

const fn workload_key_bits(workload: Workload) -> u32 {
    match workload {
        Workload::Bounded16 => 16,
        Workload::FullWidth => 32,
    }
}

fn read_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    item_count: u32,
) -> Result<Vec<T>, Box<dyn std::error::Error>> {
    let size = u64::from(item_count) * size_of::<T>() as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Comparison Readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    let submission = queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    })?;
    receiver.recv()??;
    let values = bytemuck::cast_slice(&slice.get_mapped_range()?).to_vec();
    staging.unmap();
    Ok(values)
}

fn wait_for_gpu(device: &wgpu::Device) -> Result<(), wgpu::PollError> {
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    })?;
    Ok(())
}

fn adapter_metadata(context: &Context) -> AdapterMetadata {
    let info = &context.adapter_info;
    AdapterMetadata {
        name: info.name.clone(),
        vendor: info.vendor,
        device: info.device,
        device_type: format!("{:?}", info.device_type),
        backend: format!("{:?}", info.backend),
        driver: info.driver.clone(),
        driver_info: info.driver_info.clone(),
        subgroup_min_size: Some(info.subgroup_min_size),
        subgroup_max_size: Some(info.subgroup_max_size),
    }
}

fn memory_estimate(context: &Context, items: u32) -> MemoryEstimate {
    let info = &context.adapter_info;
    let uses_eight_bit = info.backend == wgpu::Backend::Vulkan
        && info.vendor == 0x10de
        && info.device_type == wgpu::DeviceType::DiscreteGpu
        && info.subgroup_min_size == 32
        && info.subgroup_max_size == 32
        && context.device.features().contains(wgpu::Features::SUBGROUP);
    if uses_eight_bit {
        lampshade_eight_bit_memory(
            items,
            u64::from(context.device.limits().min_uniform_buffer_offset_alignment),
        )
    } else {
        MemoryEstimate::unavailable(
            "portable/wide workspace formula is not modeled by this comparison runner",
        )
    }
}

fn runtime_metadata(name: &str, fallback: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| fallback.into())
}
