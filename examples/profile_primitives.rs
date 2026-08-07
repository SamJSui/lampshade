use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;
use wgpu_primitives::{Context, GpuProfile, KeyValue, KeyValueSorter, Scanner, Sorter};

const DEFAULT_INPUT_SIZES: [usize; 3] = [1_000_000, 10_000_000, 100_000_000];
const DEFAULT_SAMPLES: usize = 5;
const DEFAULT_WARMUP_MS: u64 = 1_000;
const DEFAULT_CASES: [ProfileCase; 4] = [
    ProfileCase::Scan,
    ProfileCase::KeySort,
    ProfileCase::KeyValueBounded16,
    ProfileCase::KeyValueFullWidth,
];

#[derive(Clone, Copy)]
enum ProfileCase {
    Scan,
    KeySort,
    KeyValueBounded16,
    KeyValueFullWidth,
}

struct ProfileConfig {
    samples: usize,
    warmup: Duration,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::init().await?;
    if !context
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
    {
        return Err("selected adapter does not support timestamp queries".into());
    }

    let sizes = input_sizes()?;
    let cases = profile_cases()?;
    let config = profile_config()?;
    println!(
        "adapter={:?} vendor={} device={} device_type={:?} backend={:?} driver={:?} driver_info={:?} subgroup_min={} subgroup_max={} samples={} warmup_ms={}",
        context.adapter_info.name,
        context.adapter_info.vendor,
        context.adapter_info.device,
        context.adapter_info.device_type,
        context.adapter_info.backend,
        context.adapter_info.driver,
        context.adapter_info.driver_info,
        context.adapter_info.subgroup_min_size,
        context.adapter_info.subgroup_max_size,
        config.samples,
        config.warmup.as_millis(),
    );
    println!(
        "primitive,items,resident_wall_median_ms,gpu_elapsed_median_ms,dispatch_median_ms,inter_pass_gap_ms,wall_minus_gpu_ms"
    );

    for item_count in sizes {
        for case in &cases {
            match case {
                ProfileCase::Scan => profile_scan(&context, item_count, &config).await?,
                ProfileCase::KeySort => profile_key_sort(&context, item_count, &config).await?,
                ProfileCase::KeyValueBounded16 => {
                    profile_key_value_sort(&context, item_count, &config, false).await?
                }
                ProfileCase::KeyValueFullWidth => {
                    profile_key_value_sort(&context, item_count, &config, true).await?
                }
            }
        }
    }

    Ok(())
}

async fn profile_scan(
    context: &Context,
    item_count: usize,
    config: &ProfileConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let input: Vec<u32> = (0..item_count as u32)
        .map(|value| value ^ 0xA5A5_A5A5)
        .collect();
    let gpu_input = create_input(context, "Profile Scan Input", &input);
    let gpu_output = create_output(context, "Profile Scan Output", gpu_input.size());
    let mut scanner = Scanner::from_context(context);

    warm_up(
        config.warmup,
        || scanner.scan_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32),
        context,
    )?;
    let wall = measure_wall(
        config.samples,
        || scanner.scan_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32),
        context,
    )?;
    let _ = scanner
        .profile_scan_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32)
        .await?;
    let mut profiles = Vec::with_capacity(config.samples);
    for _ in 0..config.samples {
        profiles.push(
            scanner
                .profile_scan_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32)
                .await?,
        );
    }
    report("scan", item_count, wall, &profiles);
    Ok(())
}

async fn profile_key_sort(
    context: &Context,
    item_count: usize,
    config: &ProfileConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let input = deterministic_keys(item_count);
    let gpu_input = create_input(context, "Profile Sort Input", &input);
    let gpu_output = create_output(context, "Profile Sort Output", gpu_input.size());
    let mut sorter = Sorter::from_context(context);

    warm_up(
        config.warmup,
        || sorter.sort_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32),
        context,
    )?;
    let wall = measure_wall(
        config.samples,
        || sorter.sort_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32),
        context,
    )?;
    let _ = sorter
        .profile_sort_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32)
        .await?;
    let mut profiles = Vec::with_capacity(config.samples);
    for _ in 0..config.samples {
        profiles.push(
            sorter
                .profile_sort_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32)
                .await?,
        );
    }
    report("key_sort", item_count, wall, &profiles);
    Ok(())
}

async fn profile_key_value_sort(
    context: &Context,
    item_count: usize,
    config: &ProfileConfig,
    full_width: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let input: Vec<_> = deterministic_keys(item_count)
        .into_iter()
        .enumerate()
        .map(|(index, key)| {
            let key = if full_width { key } else { key & 0xffff };
            KeyValue::new(key, index as u32)
        })
        .collect();
    let gpu_input = create_input(context, "Profile Key-Value Input", &input);
    let gpu_output = create_output(context, "Profile Key-Value Output", gpu_input.size());
    let mut sorter = KeyValueSorter::from_context(context);

    warm_up(
        config.warmup,
        || sorter.sort_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32),
        context,
    )?;
    let wall = measure_wall(
        config.samples,
        || sorter.sort_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32),
        context,
    )?;
    let _ = sorter
        .profile_sort_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32)
        .await?;
    let mut profiles = Vec::with_capacity(config.samples);
    for _ in 0..config.samples {
        profiles.push(
            sorter
                .profile_sort_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32)
                .await?,
        );
    }
    let primitive = if full_width {
        "key_value_sort_full_width"
    } else {
        "key_value_sort_bounded16"
    };
    report(primitive, item_count, wall, &profiles);
    Ok(())
}

fn measure_wall(
    samples: usize,
    mut submit: impl FnMut() -> Result<(), wgpu_primitives::Error>,
    context: &Context,
) -> Result<Duration, wgpu_primitives::Error> {
    let mut durations = Vec::with_capacity(samples);
    for _ in 0..samples {
        let started = Instant::now();
        submit()?;
        wait_for_gpu(&context.device)?;
        durations.push(started.elapsed());
    }
    Ok(median(durations))
}

fn warm_up(
    minimum_duration: Duration,
    mut submit: impl FnMut() -> Result<(), wgpu_primitives::Error>,
    context: &Context,
) -> Result<(), wgpu_primitives::Error> {
    let started = Instant::now();
    loop {
        submit()?;
        wait_for_gpu(&context.device)?;
        if started.elapsed() >= minimum_duration {
            break;
        }
    }
    Ok(())
}

fn report(primitive: &str, item_count: usize, wall: Duration, profiles: &[GpuProfile]) {
    let gpu_elapsed = median(profiles.iter().map(|profile| profile.gpu_elapsed).collect());
    let dispatch_time = median(
        profiles
            .iter()
            .map(|profile| profile.dispatch_time)
            .collect(),
    );
    let inter_pass_gap = gpu_elapsed.saturating_sub(dispatch_time);
    let wall_minus_gpu = wall.saturating_sub(gpu_elapsed);
    println!(
        "{primitive},{item_count},{:.3},{:.3},{:.3},{:.3},{:.3}",
        milliseconds(wall),
        milliseconds(gpu_elapsed),
        milliseconds(dispatch_time),
        milliseconds(inter_pass_gap),
        milliseconds(wall_minus_gpu),
    );

    let mut stages: BTreeMap<&str, Vec<Duration>> = BTreeMap::new();
    let mut spans: BTreeMap<String, Vec<Duration>> = BTreeMap::new();
    for profile in profiles {
        let mut sample_stages: BTreeMap<&str, Duration> = BTreeMap::new();
        for span in &profile.spans {
            *sample_stages.entry(stage(&span.label)).or_default() += span.duration;
            spans
                .entry(span.label.clone())
                .or_default()
                .push(span.duration);
        }
        for (stage, duration) in sample_stages {
            stages.entry(stage).or_default().push(duration);
        }
    }
    let stage_medians: Vec<_> = stages
        .into_iter()
        .map(|(stage, durations)| (stage, median(durations)))
        .collect();
    let stage_total = stage_medians
        .iter()
        .map(|(_, duration)| *duration)
        .sum::<Duration>();
    for (stage, median_duration) in stage_medians {
        let percent = median_duration.as_secs_f64() / stage_total.as_secs_f64() * 100.0;
        println!(
            "stage,{primitive},{item_count},{stage},{:.3},{percent:.1}%",
            milliseconds(median_duration)
        );
    }
    for (label, durations) in spans {
        println!(
            "span,{primitive},{item_count},{label},{:.3}",
            milliseconds(median(durations))
        );
    }
}

fn stage(label: &str) -> &'static str {
    if label.ends_with(".histogram") {
        "histogram"
    } else if label.ends_with(".prefix") {
        "prefix"
    } else if label.ends_with(".reduce") {
        "reduce"
    } else if label.ends_with(".scatter") {
        "scatter"
    } else if label.contains(".scan.") {
        "histogram_scan"
    } else if label.contains(".level.") {
        "scan"
    } else {
        "add"
    }
}

fn median(mut durations: Vec<Duration>) -> Duration {
    durations.sort_unstable();
    let middle = durations.len() / 2;
    if durations.len().is_multiple_of(2) {
        (durations[middle - 1] + durations[middle]) / 2
    } else {
        durations[middle]
    }
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn deterministic_keys(item_count: usize) -> Vec<u32> {
    let mut state = 0x9E37_79B9_u32 ^ item_count as u32;
    (0..item_count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        })
        .collect()
}

fn create_input<T: bytemuck::Pod>(
    context: &Context,
    label: &'static str,
    input: &[T],
) -> wgpu::Buffer {
    context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
}

fn create_output(context: &Context, label: &'static str, size: u64) -> wgpu::Buffer {
    context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn wait_for_gpu(device: &wgpu::Device) -> Result<(), wgpu_primitives::Error> {
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    })?;
    Ok(())
}

fn input_sizes() -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let Some(raw) = std::env::var_os("WGPU_PRIMITIVES_PROFILE_ITEMS") else {
        return Ok(DEFAULT_INPUT_SIZES.to_vec());
    };
    raw.to_string_lossy()
        .split(',')
        .map(|value| Ok(value.trim().replace('_', "").parse()?))
        .collect()
}

fn profile_cases() -> Result<Vec<ProfileCase>, Box<dyn std::error::Error>> {
    let Some(raw) = std::env::var_os("WGPU_PRIMITIVES_PROFILE_CASES") else {
        return Ok(DEFAULT_CASES.to_vec());
    };
    let cases: Vec<_> = raw
        .to_string_lossy()
        .split(',')
        .map(|value| match value.trim() {
            "scan" => Ok(ProfileCase::Scan),
            "key_sort" => Ok(ProfileCase::KeySort),
            "key_value_bounded16" => Ok(ProfileCase::KeyValueBounded16),
            "key_value_full_width" => Ok(ProfileCase::KeyValueFullWidth),
            value => Err(format!(
                "unknown WGPU_PRIMITIVES_PROFILE_CASES value {value:?}"
            )),
        })
        .collect::<Result<_, _>>()?;
    if cases.is_empty() {
        return Err("WGPU_PRIMITIVES_PROFILE_CASES must not be empty".into());
    }
    Ok(cases)
}

fn profile_config() -> Result<ProfileConfig, Box<dyn std::error::Error>> {
    let samples = std::env::var("WGPU_PRIMITIVES_PROFILE_SAMPLES")
        .unwrap_or_else(|_| DEFAULT_SAMPLES.to_string())
        .parse()?;
    if samples == 0 {
        return Err("WGPU_PRIMITIVES_PROFILE_SAMPLES must be greater than zero".into());
    }
    let warmup_ms = std::env::var("WGPU_PRIMITIVES_PROFILE_WARMUP_MS")
        .unwrap_or_else(|_| DEFAULT_WARMUP_MS.to_string())
        .parse()?;
    Ok(ProfileConfig {
        samples,
        warmup: Duration::from_millis(warmup_ms),
    })
}
