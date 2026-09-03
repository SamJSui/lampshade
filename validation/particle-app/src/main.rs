use std::{
    env,
    error::Error,
    fmt, io,
    sync::mpsc,
    time::{Duration, Instant},
};

use lampshade::{
    Context, GpuCountPlan, KeyValue, KeyValueCompactor, KeyValueField, KeyValueSorter,
    MaskGenerator, U32Predicate,
    pipeline::{GpuCount, GpuSlice, GpuSliceMut, Primitives, SortOptions, WorkspaceRequirements},
};
use serde::Serialize;
use wgpu::util::DeviceExt;

type AnyError = Box<dyn Error + Send + Sync>;

const KEY_BITS: u32 = 16;
const VISIBILITY: U32Predicate = U32Predicate::BetweenInclusive {
    min: 0x4000,
    max: 0xbfff,
};

fn main() -> Result<(), AnyError> {
    let config = Config::parse()?;
    pollster::block_on(run(config))
}

async fn run(config: Config) -> Result<(), AnyError> {
    let device_start = Instant::now();
    let instance = wgpu::Instance::new(
        wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        }
        .with_env(),
    );
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .map_err(|error| io::Error::other(format!("failed to request adapter: {error}")))?;
    let adapter_info = adapter.get_info();
    let optional_features = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SUBGROUP;
    let required_features = adapter.features() & optional_features;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Standalone Particle Consumer Device"),
            required_features,
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .map_err(|error| io::Error::other(format!("failed to request device: {error}")))?;
    let device_initialization_ms = elapsed_ms(device_start.elapsed());

    let particles = particle_input(config.items);
    let expected_selected = particles
        .iter()
        .filter(|item| (0x4000..=0xbfff).contains(&item.key))
        .count();
    let buffers = ParticleBuffers::new(&device, &particles);
    let setup = Engine::build(
        config.mode,
        &device,
        &queue,
        &adapter_info,
        &buffers,
        config.items,
    )?;
    let mut engine = setup.engine;

    for _ in 0..config.warmups {
        let command = record_command(&device, &mut engine, &buffers, config.items)?;
        submit_and_wait(&device, &queue, command)?;
    }

    let mut recording_ms = Vec::with_capacity(config.iterations as usize);
    let mut submission_wait_ms = Vec::with_capacity(config.iterations as usize);
    let mut total_ms = Vec::with_capacity(config.iterations as usize);
    for _ in 0..config.iterations {
        let total_start = Instant::now();
        let record_start = Instant::now();
        let command = record_command(&device, &mut engine, &buffers, config.items)?;
        recording_ms.push(elapsed_ms(record_start.elapsed()));
        submission_wait_ms.push(elapsed_ms(submit_and_wait(&device, &queue, command)?));
        total_ms.push(elapsed_ms(total_start.elapsed()));
    }

    let selected = validate_final_output(
        &device,
        &queue,
        &mut engine,
        &buffers,
        config.items,
        expected_selected,
    )?;
    let record_bytes = u64::from(config.items) * size_of::<KeyValue>() as u64;
    let mask_bytes = u64::from(config.items) * size_of::<u32>() as u64;
    let report = Report {
        schema_version: 1,
        mode: config.mode.to_string(),
        adapter: AdapterReport {
            name: adapter_info.name,
            vendor: adapter_info.vendor,
            device: adapter_info.device,
            device_type: format!("{:?}", adapter_info.device_type),
            backend: format!("{:?}", adapter_info.backend),
            driver: adapter_info.driver,
            driver_info: adapter_info.driver_info,
        },
        config: ConfigReport {
            items: config.items,
            selected_items: selected,
            warmups: config.warmups,
            iterations: config.iterations,
            key_bits: KEY_BITS,
        },
        timings_ms: TimingReport {
            device_initialization: device_initialization_ms,
            primitive_construction: elapsed_ms(setup.construction),
            workspace_reservation: elapsed_ms(setup.reservation),
            command_recording: Summary::new(&recording_ms),
            submission_through_completion: Summary::new(&submission_wait_ms),
            record_submit_completion: Summary::new(&total_ms),
        },
        memory: MemoryReport {
            resident_application_buffers: record_bytes * 3 + mask_bytes + size_of::<u32>() as u64,
            validation_readback: 8 + record_bytes,
            internal_primitive_workspace: None,
            internal_workspace_note: "wgpu does not expose portable physical-allocation or peak-driver-memory telemetry",
        },
        contracts: ContractReport {
            application_owns_wgpu_device: true,
            public_api_only: true,
            gpu_resident_count_between_operations: true,
            one_submission_per_measured_iteration: true,
            validation_uses_one_final_map: true,
        },
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Config {
    mode: Mode,
    items: u32,
    warmups: u32,
    iterations: u32,
}

impl Config {
    fn parse() -> Result<Self, AnyError> {
        let mut config = Self {
            mode: Mode::Typed,
            items: 1_000_000,
            warmups: 3,
            iterations: 10,
        };
        let mut args = env::args().skip(1);
        while let Some(argument) = args.next() {
            match argument.as_str() {
                "--mode" => config.mode = Mode::parse(&next_value(&mut args, "--mode")?)?,
                "--items" => config.items = parse_u32(&next_value(&mut args, "--items")?)?,
                "--warmups" => config.warmups = parse_u32(&next_value(&mut args, "--warmups")?)?,
                "--iterations" => {
                    config.iterations = parse_u32(&next_value(&mut args, "--iterations")?)?
                }
                "--help" | "-h" => {
                    println!(
                        "usage: particle-app [--mode typed|raw] [--items N] [--warmups N] [--iterations N]"
                    );
                    std::process::exit(0);
                }
                _ => return Err(io::Error::other(format!("unknown argument: {argument}")).into()),
            }
        }
        if config.items == 0 {
            return Err(io::Error::other("--items must be greater than zero").into());
        }
        if config.iterations == 0 {
            return Err(io::Error::other("--iterations must be greater than zero").into());
        }
        Ok(config)
    }
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, AnyError> {
    args.next()
        .ok_or_else(|| io::Error::other(format!("{name} requires a value")).into())
}

fn parse_u32(value: &str) -> Result<u32, AnyError> {
    value
        .replace('_', "")
        .parse::<u32>()
        .map_err(|error| io::Error::other(format!("invalid integer {value:?}: {error}")).into())
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Typed,
    Raw,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, AnyError> {
        match value {
            "typed" => Ok(Self::Typed),
            "raw" => Ok(Self::Raw),
            _ => Err(io::Error::other("--mode must be typed or raw").into()),
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Typed => formatter.write_str("typed"),
            Self::Raw => formatter.write_str("raw"),
        }
    }
}

struct ParticleBuffers {
    input: wgpu::Buffer,
    mask: wgpu::Buffer,
    compacted: wgpu::Buffer,
    sorted: wgpu::Buffer,
    count: wgpu::Buffer,
}

impl ParticleBuffers {
    fn new(device: &wgpu::Device, particles: &[KeyValue]) -> Self {
        let capacity = particles.len() as u32;
        let record_bytes = u64::from(capacity) * size_of::<KeyValue>() as u64;
        let mask_bytes = u64::from(capacity) * size_of::<u32>() as u64;
        Self {
            input: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Consumer Particle Input"),
                contents: bytemuck::cast_slice(particles),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            mask: storage_buffer_with_usage(
                device,
                "Consumer Particle Mask",
                mask_bytes,
                wgpu::BufferUsages::COPY_SRC,
            ),
            compacted: storage_buffer(device, "Consumer Compacted Particles", record_bytes),
            sorted: storage_buffer_with_usage(
                device,
                "Consumer Sorted Particles",
                record_bytes,
                wgpu::BufferUsages::COPY_SRC,
            ),
            count: storage_buffer_with_usage(
                device,
                "Consumer Particle Count",
                size_of::<u32>() as u64,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            ),
        }
    }
}

struct EngineSetup {
    engine: Engine,
    construction: Duration,
    reservation: Duration,
}

enum Engine {
    Typed(Box<Primitives>),
    Raw(Box<RawEngine>),
}

struct RawEngine {
    generator: MaskGenerator,
    compactor: KeyValueCompactor,
    sorter: KeyValueSorter,
    count_plan: GpuCountPlan,
}

impl Engine {
    fn build(
        mode: Mode,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        adapter_info: &wgpu::AdapterInfo,
        buffers: &ParticleBuffers,
        capacity: u32,
    ) -> Result<EngineSetup, AnyError> {
        match mode {
            Mode::Typed => {
                let construction_start = Instant::now();
                let mut primitives = Primitives::new_for_adapter(device, queue, adapter_info);
                let construction = construction_start.elapsed();
                let reservation_start = Instant::now();
                primitives.reserve_workspace(
                    WorkspaceRequirements::new(capacity)
                        .predicate()
                        .compact_key_values()
                        .counted_key_value_sort(),
                )?;
                primitives.reserve_count(GpuCount::new(&buffers.count)?, capacity)?;
                Ok(EngineSetup {
                    engine: Self::Typed(Box::new(primitives)),
                    construction,
                    reservation: reservation_start.elapsed(),
                })
            }
            Mode::Raw => {
                let construction_start = Instant::now();
                let context = Context {
                    adapter_info: adapter_info.clone(),
                    device: device.clone(),
                    queue: queue.clone(),
                };
                let generator = MaskGenerator::from_context(&context);
                let compactor = KeyValueCompactor::from_context(&context);
                let sorter = KeyValueSorter::from_context(&context);
                let count_plan = GpuCountPlan::new(device, &buffers.count, capacity)?;
                Ok(EngineSetup {
                    engine: Self::Raw(Box::new(RawEngine {
                        generator,
                        compactor,
                        sorter,
                        count_plan,
                    })),
                    construction: construction_start.elapsed(),
                    reservation: Duration::ZERO,
                })
            }
        }
    }

    fn record(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        buffers: &ParticleBuffers,
        capacity: u32,
    ) -> Result<(), lampshade::Error> {
        match self {
            Self::Typed(primitives) => {
                let input = GpuSlice::from_range(&buffers.input, 0..capacity)?;
                let mask = GpuSliceMut::from_range(&buffers.mask, 0..capacity)?;
                let compacted = GpuSliceMut::from_range(&buffers.compacted, 0..capacity)?;
                let sorted = GpuSliceMut::from_range(&buffers.sorted, 0..capacity)?;
                let count = GpuCount::new(&buffers.count)?;
                let mut recorder = primitives.record(encoder);
                let mask = recorder.mask_key_values(input, mask, KeyValueField::Key, VISIBILITY)?;
                let visible = recorder.compact_key_values(input, mask, compacted, count)?;
                recorder.sort_by_key(visible, sorted, SortOptions::default().key_bits(KEY_BITS))?;
            }
            Self::Raw(raw) => {
                raw.generator.record_key_value_mask(
                    encoder,
                    &buffers.input,
                    &buffers.mask,
                    capacity,
                    KeyValueField::Key,
                    VISIBILITY,
                )?;
                raw.compactor.record_compact(
                    encoder,
                    &buffers.input,
                    &buffers.mask,
                    &buffers.compacted,
                    &buffers.count,
                    capacity,
                )?;
                raw.count_plan.record_prepare(encoder);
                raw.sorter.record_sort_with_count_plan_and_key_bits(
                    encoder,
                    &buffers.compacted,
                    &buffers.sorted,
                    &raw.count_plan,
                    KEY_BITS,
                )?;
            }
        }
        Ok(())
    }
}

fn record_command(
    device: &wgpu::Device,
    engine: &mut Engine,
    buffers: &ParticleBuffers,
    capacity: u32,
) -> Result<wgpu::CommandBuffer, lampshade::Error> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Standalone Particle Pipeline"),
    });
    engine.record(&mut encoder, buffers, capacity)?;
    Ok(encoder.finish())
}

fn submit_and_wait(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    command: wgpu::CommandBuffer,
) -> Result<Duration, AnyError> {
    let start = Instant::now();
    let submission = queue.submit([command]);
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    })?;
    Ok(start.elapsed())
}

fn validate_final_output(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    engine: &mut Engine,
    buffers: &ParticleBuffers,
    capacity: u32,
    expected_selected: usize,
) -> Result<u32, AnyError> {
    let record_bytes = u64::from(capacity) * size_of::<KeyValue>() as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Standalone Particle Validation Readback"),
        size: 8 + record_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Standalone Particle Validation"),
    });
    engine.record(&mut encoder, buffers, capacity)?;
    encoder.copy_buffer_to_buffer(&buffers.count, 0, &readback, 0, size_of::<u32>() as u64);
    encoder.copy_buffer_to_buffer(&buffers.sorted, 0, &readback, 8, record_bytes);
    let submission = queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    })?;
    receiver
        .recv()
        .map_err(|error| io::Error::other(format!("validation channel closed: {error}")))?
        .map_err(|error| io::Error::other(format!("validation map failed: {error}")))?;

    {
        let mapped = slice.get_mapped_range().map_err(|error| {
            io::Error::other(format!("validation mapped range failed: {error}"))
        })?;
        let selected = bytemuck::cast_slice::<u8, u32>(&mapped[..4])[0] as usize;
        if selected != expected_selected {
            return Err(io::Error::other(format!(
                "selected count mismatch: GPU {selected}, CPU {expected_selected}"
            ))
            .into());
        }
        let records: &[KeyValue] = bytemuck::cast_slice(&mapped[8..8 + selected * 8]);
        for (index, record) in records.iter().enumerate() {
            if !(0x4000..=0xbfff).contains(&record.key) {
                return Err(io::Error::other(format!(
                    "record {index} failed visibility predicate: {}",
                    record.key
                ))
                .into());
            }
        }
        for (index, pair) in records.windows(2).enumerate() {
            if pair[0].key > pair[1].key {
                return Err(io::Error::other(format!(
                    "keys are not sorted at records {index} and {}",
                    index + 1
                ))
                .into());
            }
            if pair[0].key == pair[1].key && pair[0].value > pair[1].value {
                return Err(io::Error::other(format!(
                    "stable payload order changed at records {index} and {}",
                    index + 1
                ))
                .into());
            }
        }
        drop(mapped);
        readback.unmap();
        Ok(selected as u32)
    }
}

fn particle_input(items: u32) -> Vec<KeyValue> {
    let mut state = 0x5041_5254_u32;
    (0..items)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            KeyValue::new(state & 0xffff, index)
        })
        .collect()
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    storage_buffer_with_usage(device, label, size, wgpu::BufferUsages::empty())
}

fn storage_buffer_with_usage(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    extra_usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | extra_usage,
        mapped_at_creation: false,
    })
}

fn elapsed_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    mode: String,
    adapter: AdapterReport,
    config: ConfigReport,
    timings_ms: TimingReport,
    memory: MemoryReport,
    contracts: ContractReport,
}

#[derive(Serialize)]
struct AdapterReport {
    name: String,
    vendor: u32,
    device: u32,
    device_type: String,
    backend: String,
    driver: String,
    driver_info: String,
}

#[derive(Serialize)]
struct ConfigReport {
    items: u32,
    selected_items: u32,
    warmups: u32,
    iterations: u32,
    key_bits: u32,
}

#[derive(Serialize)]
struct TimingReport {
    device_initialization: f64,
    primitive_construction: f64,
    workspace_reservation: f64,
    command_recording: Summary,
    submission_through_completion: Summary,
    record_submit_completion: Summary,
}

#[derive(Serialize)]
struct Summary {
    median: f64,
    minimum: f64,
    maximum: f64,
}

impl Summary {
    fn new(samples: &[f64]) -> Self {
        let mut ordered = samples.to_vec();
        ordered.sort_by(f64::total_cmp);
        let middle = ordered.len() / 2;
        let median = if ordered.len().is_multiple_of(2) {
            (ordered[middle - 1] + ordered[middle]) / 2.0
        } else {
            ordered[middle]
        };
        Self {
            median,
            minimum: ordered[0],
            maximum: ordered[ordered.len() - 1],
        }
    }
}

#[derive(Serialize)]
struct MemoryReport {
    resident_application_buffers: u64,
    validation_readback: u64,
    internal_primitive_workspace: Option<u64>,
    internal_workspace_note: &'static str,
}

#[derive(Serialize)]
struct ContractReport {
    application_owns_wgpu_device: bool,
    public_api_only: bool,
    gpu_resident_count_between_operations: bool,
    one_submission_per_measured_iteration: bool,
    validation_uses_one_final_map: bool,
}
