use std::hint::black_box;
use std::time::{Duration, Instant};

use cubecl::prelude::*;
use cubecl::wgpu::{AutoGraphicsApi, RuntimeOptions, WgpuDevice, WgpuRuntime, init_setup};
use massively::{Executor, vector};
use massively_benchmark_common::{
    AdapterMetadata, BenchmarkConfig, BenchmarkRun, GeneratorMetadata, MASSIVELY_REVISION,
    MASSIVELY_VERSION, SCHEMA_VERSION, SortInput, Workload, generate_compact, generate_scan,
    median, public_buffer_memory, validate_compact, validate_exclusive_scan,
};

type AnyError = Box<dyn std::error::Error>;

struct Add;

#[cubecl::cube]
impl massively::op::ReductionOp<u32> for Add {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Massively comparison runner failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), AnyError> {
    let config = BenchmarkConfig::from_env()?;
    let device = WgpuDevice::DefaultDevice;
    let setup = init_setup::<AutoGraphicsApi>(&device, RuntimeOptions::default());
    let adapter = adapter_metadata(&setup.adapter.get_info());
    let exec = Executor::<WgpuRuntime>::new(device);

    let (samples_ms, output_items) = match config.workload {
        Workload::SortBounded16 | Workload::SortFullWidth => run_sort(&exec, &config)?,
        Workload::ExclusiveScan => run_scan(&exec, &config)?,
        Workload::Compact50 => run_compact(&exec, &config)?,
    };
    let median_ms = median(&samples_ms);
    let run = BenchmarkRun {
        schema_version: SCHEMA_VERSION,
        implementation: "massively".into(),
        implementation_version: MASSIVELY_VERSION.into(),
        implementation_revision: MASSIVELY_REVISION.into(),
        runtime_stack: "wgpu 30.0.0 / CubeCL 0.11.0-pre.1".into(),
        adapter,
        config: config.clone(),
        generator: GeneratorMetadata::current(),
        timing_boundary: "public resident GPU API call through executor synchronization; excludes upload, readback, and validation".into(),
        output_allocation: "public algorithm allocates an owned output per call; CubeCL allocator may recycle storage".into(),
        correctness_checked: true,
        samples_ms,
        median_ms,
        throughput_items_per_second: f64::from(config.items) / (median_ms / 1_000.0),
        output_items,
        memory: public_buffer_memory("massively", config.workload, config.items),
    };
    println!("{}", serde_json::to_string(&run)?);
    Ok(())
}

fn run_sort(
    exec: &Executor<WgpuRuntime>,
    config: &BenchmarkConfig,
) -> Result<(Vec<f64>, u32), AnyError> {
    let logical = SortInput::generate(config.items, config.workload);
    let keys = exec.to_device(&logical.keys);
    let values = exec.to_device(&logical.values);

    let output = vector::radix_sort_by_key(exec, keys.slice(..), values.slice(..))?;
    let actual = exec.to_host(&output)?;
    logical.validate_values(&actual)?;
    drop(actual);
    drop(output);

    let samples = warm_and_sample(config, || {
        let output = vector::radix_sort_by_key(exec, keys.slice(..), values.slice(..))?;
        exec.sync()?;
        black_box(output);
        Ok(())
    })?;
    Ok((samples, config.items))
}

fn run_scan(
    exec: &Executor<WgpuRuntime>,
    config: &BenchmarkConfig,
) -> Result<(Vec<f64>, u32), AnyError> {
    let input = generate_scan(config.items);
    let device_input = exec.to_device(&input);

    let output = vector::exclusive_scan(exec, device_input.slice(..), 0_u32, Add)?;
    let actual = exec.to_host(&output)?;
    validate_exclusive_scan(&input, &actual)?;
    drop(actual);
    drop(output);

    let samples = warm_and_sample(config, || {
        let output = vector::exclusive_scan(exec, device_input.slice(..), 0_u32, Add)?;
        exec.sync()?;
        black_box(output);
        Ok(())
    })?;
    Ok((samples, config.items))
}

fn run_compact(
    exec: &Executor<WgpuRuntime>,
    config: &BenchmarkConfig,
) -> Result<(Vec<f64>, u32), AnyError> {
    let (input, mask) = generate_compact(config.items);
    let device_input = exec.to_device(&input);
    let device_mask = exec.to_device(&mask);

    let output = vector::copy_where(exec, device_input.slice(..), device_mask.slice(..))?;
    let actual = exec.to_host(&output)?;
    validate_compact(&input, &mask, &actual)?;
    let output_items = actual.len() as u32;
    drop(actual);
    drop(output);

    let samples = warm_and_sample(config, || {
        let output = vector::copy_where(exec, device_input.slice(..), device_mask.slice(..))?;
        exec.sync()?;
        black_box(output);
        Ok(())
    })?;
    Ok((samples, output_items))
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
    Ok(samples_ms)
}

fn adapter_metadata(info: &wgpu::AdapterInfo) -> AdapterMetadata {
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
