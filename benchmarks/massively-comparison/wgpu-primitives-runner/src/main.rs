use std::hint::black_box;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use massively_benchmark_common::{
    AdapterMetadata, BenchmarkConfig, BenchmarkRun, GeneratorMetadata, SCHEMA_VERSION, SortInput,
    Workload, generate_compact, generate_scan, median, public_buffer_memory, runtime_metadata,
    validate_compact, validate_exclusive_scan,
};
use wgpu::util::DeviceExt;
use wgpu_primitives::{Compactor, Context, KeyValue, KeyValueSorter, Scanner};

type AnyError = Box<dyn std::error::Error>;

fn main() {
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("wgpu-primitives Massively comparison runner failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), AnyError> {
    let config = BenchmarkConfig::from_env()?;
    let context = Context::init().await?;
    let (samples_ms, output_items) = match config.workload {
        Workload::SortBounded16 | Workload::SortFullWidth => run_sort(&context, &config)?,
        Workload::ExclusiveScan => run_scan(&context, &config)?,
        Workload::Compact50 => run_compact(&context, &config)?,
    };
    let median_ms = median(&samples_ms);
    let run = BenchmarkRun {
        schema_version: SCHEMA_VERSION,
        implementation: "wgpu-primitives".into(),
        implementation_version: runtime_metadata(
            "MASSIVELY_BENCH_IMPLEMENTATION_VERSION",
            "working-tree",
        ),
        implementation_revision: runtime_metadata(
            "MASSIVELY_BENCH_IMPLEMENTATION_REVISION",
            "working-tree",
        ),
        runtime_stack: "wgpu 28.0.0".into(),
        adapter: adapter_metadata(&context),
        config: config.clone(),
        generator: GeneratorMetadata::current(),
        timing_boundary: "public resident GPU API call through device completion; excludes upload, readback, and validation".into(),
        output_allocation: "caller-owned output and reusable primitive workspace allocated before timing".into(),
        correctness_checked: true,
        samples_ms,
        median_ms,
        throughput_items_per_second: f64::from(config.items) / (median_ms / 1_000.0),
        output_items,
        memory: public_buffer_memory("wgpu-primitives", config.workload, config.items),
    };
    println!("{}", serde_json::to_string(&run)?);
    Ok(())
}

fn run_sort(context: &Context, config: &BenchmarkConfig) -> Result<(Vec<f64>, u32), AnyError> {
    let logical = SortInput::generate(config.items, config.workload);
    let input: Vec<_> = logical
        .keys
        .iter()
        .copied()
        .zip(logical.values.iter().copied())
        .map(|(key, value)| KeyValue::new(key, value))
        .collect();
    let gpu_input = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Massively Comparison Sort Input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let gpu_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Massively Comparison Sort Output"),
        size: gpu_input.size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let key_bits = match config.workload {
        Workload::SortBounded16 => 16,
        Workload::SortFullWidth => 32,
        _ => unreachable!(),
    };
    let mut sorter = KeyValueSorter::from_context(context);

    sorter.sort_gpu_to_gpu_with_key_bits(&gpu_input, &gpu_output, config.items, key_bits)?;
    wait_for_gpu(&context.device)?;
    let actual =
        read_buffer::<KeyValue>(&context.device, &context.queue, &gpu_output, config.items)?;
    for (position, pair) in actual.iter().enumerate() {
        let original = pair.value as usize;
        if original >= logical.keys.len() || logical.keys[original] != pair.key {
            return Err(format!("sort key/value association mismatch at output {position}").into());
        }
    }
    let output_values: Vec<_> = actual.iter().map(|pair| pair.value).collect();
    logical.validate_values(&output_values)?;
    drop(actual);
    drop(output_values);

    let samples = warm_and_sample(config, || {
        sorter.sort_gpu_to_gpu_with_key_bits(&gpu_input, &gpu_output, config.items, key_bits)?;
        wait_for_gpu(&context.device)?;
        Ok(())
    })?;
    Ok((samples, config.items))
}

fn run_scan(context: &Context, config: &BenchmarkConfig) -> Result<(Vec<f64>, u32), AnyError> {
    let input = generate_scan(config.items);
    let gpu_input = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Massively Comparison Scan Input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let gpu_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Massively Comparison Scan Output"),
        size: gpu_input.size(),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut scanner = Scanner::from_context(context);

    scanner.scan_exclusive_gpu_to_gpu(&gpu_input, &gpu_output, config.items)?;
    wait_for_gpu(&context.device)?;
    let actual = read_buffer::<u32>(&context.device, &context.queue, &gpu_output, config.items)?;
    validate_exclusive_scan(&input, &actual)?;
    drop(actual);

    let samples = warm_and_sample(config, || {
        scanner.scan_exclusive_gpu_to_gpu(&gpu_input, &gpu_output, config.items)?;
        wait_for_gpu(&context.device)?;
        Ok(())
    })?;
    Ok((samples, config.items))
}

fn run_compact(context: &Context, config: &BenchmarkConfig) -> Result<(Vec<f64>, u32), AnyError> {
    let (input, mask) = generate_compact(config.items);
    let gpu_input = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Massively Comparison Compact Input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let gpu_mask = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Massively Comparison Compact Mask"),
            contents: bytemuck::cast_slice(&mask),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let gpu_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Massively Comparison Compact Output"),
        size: gpu_input.size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let gpu_count = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Massively Comparison Compact Count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut compactor = Compactor::from_context(context);

    compactor.compact_gpu_to_gpu(&gpu_input, &gpu_mask, &gpu_output, &gpu_count, config.items)?;
    wait_for_gpu(&context.device)?;
    let count = read_buffer::<u32>(&context.device, &context.queue, &gpu_count, 1)?[0];
    let actual = read_buffer::<u32>(&context.device, &context.queue, &gpu_output, count)?;
    validate_compact(&input, &mask, &actual)?;
    drop(actual);

    let samples = warm_and_sample(config, || {
        compactor.compact_gpu_to_gpu(
            &gpu_input,
            &gpu_mask,
            &gpu_output,
            &gpu_count,
            config.items,
        )?;
        wait_for_gpu(&context.device)?;
        Ok(())
    })?;
    Ok((samples, count))
}

fn warm_and_sample<F>(config: &BenchmarkConfig, mut run_once: F) -> Result<Vec<f64>, AnyError>
where
    F: FnMut() -> Result<(), AnyError>,
{
    let warmup_started = Instant::now();
    let mut completed = 0;
    while completed < config.warmups
        || warmup_started.elapsed() < Duration::from_millis(config.warmup_ms)
    {
        run_once()?;
        completed += 1;
    }
    let mut samples_ms = Vec::with_capacity(config.samples as usize);
    for _ in 0..config.samples {
        let start = Instant::now();
        run_once()?;
        samples_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    black_box(&samples_ms);
    Ok(samples_ms)
}

fn read_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    item_count: u32,
) -> Result<Vec<T>, AnyError> {
    if item_count == 0 {
        return Ok(Vec::new());
    }
    let size = u64::from(item_count) * size_of::<T>() as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Massively Comparison Readback"),
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
    let values = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
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
