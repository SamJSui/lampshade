use std::{
    hint::black_box,
    sync::mpsc,
    time::{Duration, Instant},
};

use lampshade::{Compactor, Context, KeyValue, KeyValueSorter, Reducer, Scanner, U32Reduction};
use massively_benchmark_common::{
    AdapterMetadata, BenchmarkConfig, BenchmarkRun, GeneratorMetadata, SCHEMA_VERSION, SortInput,
    Workload, generate_compact, generate_reduction, generate_scan, median, public_buffer_memory,
    runtime_metadata, validate_compact, validate_exclusive_scan, validate_reduction_sum,
};
use wgpu::util::DeviceExt;

type AnyError = Box<dyn std::error::Error>;

trait IntoMappedRange {
    fn into_mapped_range(self) -> Result<wgpu::BufferView, AnyError>;
}

impl IntoMappedRange for wgpu::BufferView {
    fn into_mapped_range(self) -> Result<wgpu::BufferView, AnyError> {
        Ok(self)
    }
}

impl<E> IntoMappedRange for Result<wgpu::BufferView, E>
where
    E: std::error::Error + 'static,
{
    fn into_mapped_range(self) -> Result<wgpu::BufferView, AnyError> {
        self.map_err(Into::into)
    }
}

fn main() {
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("Primitive comparison runner failed: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), AnyError> {
    let config = BenchmarkConfig::from_env()?;
    let context = Context::init().await?;
    let (samples_ms, output_items) = match config.workload {
        Workload::ReduceSum => run_reduce(&context, &config)?,
        Workload::SortBounded16 | Workload::SortFullWidth => run_sort(&context, &config)?,
        Workload::SortCountedFullWidth => run_counted_sort(&context, &config)?,
        Workload::ExclusiveScan => run_scan(&context, &config)?,
        Workload::Compact50 => run_compact(&context, &config)?,
    };
    let median_ms = median(&samples_ms);
    let run = BenchmarkRun {
        schema_version: SCHEMA_VERSION,
        implementation: runtime_metadata("MASSIVELY_BENCH_IMPLEMENTATION_NAME", "lampshade"),
        implementation_version: runtime_metadata(
            "MASSIVELY_BENCH_IMPLEMENTATION_VERSION",
            "working-tree",
        ),
        implementation_revision: runtime_metadata(
            "MASSIVELY_BENCH_IMPLEMENTATION_REVISION",
            "working-tree",
        ),
        runtime_stack: runtime_metadata(
            "MASSIVELY_BENCH_RUNTIME_STACK",
            "wgpu 29.0.4; wgpu-core 29.0.4; wgpu-hal 29.0.4; wgpu-types 29.0.4",
        ),
        adapter: adapter_metadata(&context),
        config: config.clone(),
        generator: GeneratorMetadata::current(),
        timing_boundary: match config.workload {
            Workload::ReduceSum => "resident GPU input through returned host scalar; excludes upload and validation",
            _ => "public resident GPU API call through device completion; excludes upload, readback, and validation",
        }.into(),
        output_allocation: match config.workload {
            Workload::ReduceSum => "caller-owned scalar output; timed readback allocates one staging buffer per call",
            _ => "caller-owned output and reusable primitive workspace allocated before timing",
        }.into(),
        correctness_checked: true,
        samples_ms,
        median_ms,
        throughput_items_per_second: f64::from(config.items) / (median_ms / 1_000.0),
        output_items,
        memory: public_buffer_memory("lampshade", config.workload, config.items),
    };
    println!("{}", serde_json::to_string(&run)?);
    Ok(())
}

fn run_reduce(context: &Context, config: &BenchmarkConfig) -> Result<(Vec<f64>, u32), AnyError> {
    let input = generate_reduction(config.items);
    let gpu_input = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Massively Comparison Reduction Input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let gpu_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Massively Comparison Reduction Output"),
        size: Reducer::output_buffer_size(),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut reducer = Reducer::from_context(context);

    let actual = reduce_to_host(context, &mut reducer, &gpu_input, &gpu_output, config.items)?;
    validate_reduction_sum(&input, actual)?;

    let samples = warm_and_sample(config, || {
        let value = reduce_to_host(context, &mut reducer, &gpu_input, &gpu_output, config.items)?;
        black_box(value);
        Ok(())
    })?;
    if std::env::var_os("MASSIVELY_BENCH_PROFILE_REDUCTION_PHASES").is_some() {
        profile_reduction_readback(context, &mut reducer, &gpu_input, &gpu_output, config.items)?;
    }
    Ok((samples, 1))
}

#[derive(Default)]
struct ReductionReadbackPhases {
    total_ms: f64,
    allocation_ms: f64,
    encoding_ms: f64,
    submission_ms: f64,
    map_request_ms: f64,
    poll_ms: f64,
    receive_and_read_ms: f64,
}

fn profile_reduction_readback(
    context: &Context,
    reducer: &mut Reducer,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    items: u32,
) -> Result<(), AnyError> {
    let reusable = create_reduction_readback_buffer(context);
    let mut allocated_samples = Vec::with_capacity(11);
    let mut reused_samples = Vec::with_capacity(11);

    for sample in 0..15 {
        let (_, allocated) = reduce_to_host_timed(context, reducer, input, output, items, None)?;
        let (_, reused) =
            reduce_to_host_timed(context, reducer, input, output, items, Some(&reusable))?;
        if sample >= 4 {
            allocated_samples.push(allocated);
            reused_samples.push(reused);
        }
    }

    print_reduction_phases("allocated", &allocated_samples);
    print_reduction_phases("reused", &reused_samples);
    let mut gpu_elapsed_samples = Vec::with_capacity(7);
    let mut dispatch_samples = Vec::with_capacity(7);
    for _ in 0..7 {
        let profile = pollster::block_on(reducer.profile_reduce_gpu_to_gpu(
            input,
            output,
            items,
            U32Reduction::Sum,
        ))?;
        gpu_elapsed_samples.push(profile.gpu_elapsed.as_secs_f64() * 1_000.0);
        dispatch_samples.push(profile.dispatch_time.as_secs_f64() * 1_000.0);
    }
    eprintln!(
        "reduction_gpu_phases gpu_elapsed_ms={:.4} dispatch_ms={:.4}",
        median(&gpu_elapsed_samples),
        median(&dispatch_samples),
    );
    Ok(())
}

fn print_reduction_phases(label: &str, samples: &[ReductionReadbackPhases]) {
    let field_median = |field: fn(&ReductionReadbackPhases) -> f64| {
        median(&samples.iter().map(field).collect::<Vec<_>>())
    };
    eprintln!(
        "reduction_readback_phases variant={label} total_ms={:.4} allocation_ms={:.4} encoding_ms={:.4} submission_ms={:.4} map_request_ms={:.4} poll_ms={:.4} receive_and_read_ms={:.4}",
        field_median(|sample| sample.total_ms),
        field_median(|sample| sample.allocation_ms),
        field_median(|sample| sample.encoding_ms),
        field_median(|sample| sample.submission_ms),
        field_median(|sample| sample.map_request_ms),
        field_median(|sample| sample.poll_ms),
        field_median(|sample| sample.receive_and_read_ms),
    );
}

fn create_reduction_readback_buffer(context: &Context) -> wgpu::Buffer {
    context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Massively Comparison Reduction Readback"),
        size: Reducer::output_buffer_size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn reduce_to_host(
    context: &Context,
    reducer: &mut Reducer,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    items: u32,
) -> Result<u32, AnyError> {
    let staging = create_reduction_readback_buffer(context);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    reducer.record_reduce(&mut encoder, input, output, items, U32Reduction::Sum)?;
    encoder.copy_buffer_to_buffer(output, 0, &staging, 0, Reducer::output_buffer_size());
    let submission = context.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    context.device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    })?;
    receiver.recv()??;
    let value = {
        let mapped = slice.get_mapped_range().into_mapped_range()?;
        bytemuck::cast_slice::<u8, u32>(&mapped)[0]
    };
    staging.unmap();
    Ok(value)
}

fn reduce_to_host_timed(
    context: &Context,
    reducer: &mut Reducer,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    items: u32,
    reusable_staging: Option<&wgpu::Buffer>,
) -> Result<(u32, ReductionReadbackPhases), AnyError> {
    let total_started = Instant::now();
    let allocation_started = Instant::now();
    let owned_staging = reusable_staging
        .is_none()
        .then(|| create_reduction_readback_buffer(context));
    let staging = reusable_staging.unwrap_or_else(|| {
        owned_staging
            .as_ref()
            .expect("owned reduction readback buffer exists")
    });
    let allocation_ms = allocation_started.elapsed().as_secs_f64() * 1_000.0;

    let encoding_started = Instant::now();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    reducer.record_reduce(&mut encoder, input, output, items, U32Reduction::Sum)?;
    encoder.copy_buffer_to_buffer(output, 0, staging, 0, Reducer::output_buffer_size());
    let encoding_ms = encoding_started.elapsed().as_secs_f64() * 1_000.0;

    let submission_started = Instant::now();
    let submission = context.queue.submit([encoder.finish()]);
    let submission_ms = submission_started.elapsed().as_secs_f64() * 1_000.0;

    let map_request_started = Instant::now();
    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let map_request_ms = map_request_started.elapsed().as_secs_f64() * 1_000.0;

    let poll_started = Instant::now();
    context.device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    })?;
    let poll_ms = poll_started.elapsed().as_secs_f64() * 1_000.0;

    let receive_started = Instant::now();
    receiver.recv()??;
    let value = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range().into_mapped_range()?)[0];
    staging.unmap();
    let receive_and_read_ms = receive_started.elapsed().as_secs_f64() * 1_000.0;
    let phases = ReductionReadbackPhases {
        total_ms: total_started.elapsed().as_secs_f64() * 1_000.0,
        allocation_ms,
        encoding_ms,
        submission_ms,
        map_request_ms,
        poll_ms,
        receive_and_read_ms,
    };
    Ok((value, phases))
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

fn run_counted_sort(
    context: &Context,
    config: &BenchmarkConfig,
) -> Result<(Vec<f64>, u32), AnyError> {
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
            label: Some("Release Regression Counted Sort Input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let gpu_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Release Regression Counted Sort Output"),
        size: gpu_input.size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let gpu_count = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Release Regression Counted Sort Count"),
            contents: bytemuck::bytes_of(&config.items),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let mut sorter = KeyValueSorter::from_context(context);

    sorter.sort_counted_gpu_to_gpu(&gpu_input, &gpu_output, &gpu_count, config.items)?;
    wait_for_gpu(&context.device)?;
    let actual =
        read_buffer::<KeyValue>(&context.device, &context.queue, &gpu_output, config.items)?;
    validate_key_values(&logical, &actual)?;
    drop(actual);

    let samples = warm_and_sample(config, || {
        sorter.sort_counted_gpu_to_gpu(&gpu_input, &gpu_output, &gpu_count, config.items)?;
        wait_for_gpu(&context.device)?;
        Ok(())
    })?;
    Ok((samples, config.items))
}

fn validate_key_values(logical: &SortInput, actual: &[KeyValue]) -> Result<(), AnyError> {
    for (position, pair) in actual.iter().enumerate() {
        let original = pair.value as usize;
        if original >= logical.keys.len() || logical.keys[original] != pair.key {
            return Err(format!("sort key/value association mismatch at output {position}").into());
        }
    }
    let output_values: Vec<_> = actual.iter().map(|pair| pair.value).collect();
    logical.validate_values(&output_values)?;
    Ok(())
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
    let values = bytemuck::cast_slice(&slice.get_mapped_range().into_mapped_range()?).to_vec();
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
