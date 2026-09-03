mod support;

use std::sync::mpsc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lampshade::{
    Context, GpuCountPlan, KeyValue, KeyValueCompactor, KeyValueField, KeyValueSorter,
    MaskGenerator, U32Predicate,
    pipeline::{GpuCount, GpuSlice, GpuSliceMut, Primitives, SortOptions, WorkspaceRequirements},
};
use wgpu::util::DeviceExt;

const KEY_BITS: u32 = 16;
const VISIBILITY: U32Predicate = U32Predicate::BetweenInclusive {
    min: 0x4000,
    max: 0xbfff,
};

fn benchmark_particle_pipeline(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("failed to create benchmark runtime");
    let context = runtime
        .block_on(Context::init())
        .expect("failed to create wgpu context");
    support::report_adapter(&context);
    let mut group = c.benchmark_group("particle_pipeline");

    for item_count in configured_items() {
        support::configure_group(&mut group, item_count);
        if let Some(seconds) = configured_seconds("WGPU_PARTICLE_PIPELINE_MEASUREMENT_SECONDS") {
            group.measurement_time(std::time::Duration::from_secs(seconds));
        }
        if let Some(seconds) = configured_seconds("WGPU_PARTICLE_PIPELINE_WARMUP_SECONDS") {
            group.warm_up_time(std::time::Duration::from_secs(seconds));
        }
        let input = particle_input(item_count);
        let input_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Particle Benchmark Input"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let expected = expected_visible(&input);

        {
            let buffers = ParticleBuffers::new(&context.device, item_count as u32, "Raw");
            let generator = MaskGenerator::from_context(&context);
            let mut compactor = KeyValueCompactor::from_context(&context);
            let mut sorter = KeyValueSorter::from_context(&context);
            let count_plan = GpuCountPlan::new(&context.device, &buffers.count, item_count as u32)
                .expect("failed to create raw count plan");
            run_raw(
                &context,
                &generator,
                &mut compactor,
                &mut sorter,
                &count_plan,
                &input_buffer,
                &buffers,
                item_count as u32,
            );
            validate(&context, &buffers, &expected);
            group.bench_with_input(
                BenchmarkId::new("raw_explicit", item_count),
                &item_count,
                |b, &_| {
                    b.iter(|| {
                        run_raw(
                            &context,
                            &generator,
                            &mut compactor,
                            &mut sorter,
                            &count_plan,
                            &input_buffer,
                            &buffers,
                            item_count as u32,
                        );
                    });
                },
            );
        }

        {
            let buffers = ParticleBuffers::new(&context.device, item_count as u32, "Typed");
            let mut primitives = Primitives::from_context(&context);
            primitives
                .reserve_workspace(
                    WorkspaceRequirements::new(item_count as u32)
                        .predicate()
                        .compact_key_values()
                        .counted_key_value_sort(),
                )
                .expect("failed to reserve typed particle workspace");
            primitives
                .reserve_count(
                    GpuCount::new(&buffers.count).expect("invalid typed count"),
                    item_count as u32,
                )
                .expect("failed to reserve typed count metadata");
            run_typed(
                &context,
                &mut primitives,
                &input_buffer,
                &buffers,
                item_count as u32,
            );
            validate(&context, &buffers, &expected);
            group.bench_with_input(
                BenchmarkId::new("typed_recorder", item_count),
                &item_count,
                |b, &_| {
                    b.iter(|| {
                        run_typed(
                            &context,
                            &mut primitives,
                            &input_buffer,
                            &buffers,
                            item_count as u32,
                        );
                    });
                },
            );
        }
    }
    group.finish();
}

struct ParticleBuffers {
    mask: wgpu::Buffer,
    compacted: wgpu::Buffer,
    sorted: wgpu::Buffer,
    count: wgpu::Buffer,
}

impl ParticleBuffers {
    fn new(device: &wgpu::Device, capacity: u32, prefix: &str) -> Self {
        let record_bytes = u64::from(capacity) * size_of::<KeyValue>() as u64;
        let mask_bytes = u64::from(capacity) * size_of::<u32>() as u64;
        Self {
            mask: storage_buffer(
                device,
                &format!("{prefix} Particle Mask"),
                mask_bytes,
                wgpu::BufferUsages::COPY_SRC,
            ),
            compacted: storage_buffer(
                device,
                &format!("{prefix} Compacted Particles"),
                record_bytes,
                wgpu::BufferUsages::empty(),
            ),
            sorted: storage_buffer(
                device,
                &format!("{prefix} Sorted Particles"),
                record_bytes,
                wgpu::BufferUsages::COPY_SRC,
            ),
            count: storage_buffer(
                device,
                &format!("{prefix} Particle Count"),
                size_of::<u32>() as u64,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_raw(
    context: &Context,
    generator: &MaskGenerator,
    compactor: &mut KeyValueCompactor,
    sorter: &mut KeyValueSorter,
    count_plan: &GpuCountPlan,
    input: &wgpu::Buffer,
    buffers: &ParticleBuffers,
    capacity: u32,
) {
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Raw Particle Pipeline"),
        });
    generator
        .record_key_value_mask(
            &mut encoder,
            input,
            &buffers.mask,
            capacity,
            KeyValueField::Key,
            VISIBILITY,
        )
        .expect("failed to record raw predicate");
    compactor
        .record_compact(
            &mut encoder,
            input,
            &buffers.mask,
            &buffers.compacted,
            &buffers.count,
            capacity,
        )
        .expect("failed to record raw compaction");
    count_plan.record_prepare(&mut encoder);
    sorter
        .record_sort_with_count_plan_and_key_bits(
            &mut encoder,
            &buffers.compacted,
            &buffers.sorted,
            count_plan,
            KEY_BITS,
        )
        .expect("failed to record raw counted sort");
    context.queue.submit([encoder.finish()]);
    support::wait_for_gpu(&context.device);
}

fn run_typed(
    context: &Context,
    primitives: &mut Primitives,
    input_buffer: &wgpu::Buffer,
    buffers: &ParticleBuffers,
    capacity: u32,
) {
    let input = GpuSlice::from_range(input_buffer, 0..capacity).expect("invalid typed input");
    let mask = GpuSliceMut::from_range(&buffers.mask, 0..capacity).expect("invalid typed mask");
    let compacted = GpuSliceMut::from_range(&buffers.compacted, 0..capacity)
        .expect("invalid typed compaction output");
    let sorted =
        GpuSliceMut::from_range(&buffers.sorted, 0..capacity).expect("invalid typed sort output");
    let count = GpuCount::new(&buffers.count).expect("invalid typed count");
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Typed Particle Pipeline"),
        });
    {
        let mut recorder = primitives.record(&mut encoder);
        let mask = recorder
            .mask_key_values(input, mask, KeyValueField::Key, VISIBILITY)
            .expect("failed to record typed predicate");
        let visible = recorder
            .compact_key_values(input, mask, compacted, count)
            .expect("failed to record typed compaction");
        recorder
            .sort_by_key(visible, sorted, SortOptions::default().key_bits(KEY_BITS))
            .expect("failed to record typed counted sort");
    }
    context.queue.submit([encoder.finish()]);
    support::wait_for_gpu(&context.device);
}

fn particle_input(item_count: usize) -> Vec<KeyValue> {
    support::seeded_input(item_count, 0x5041_5254)
        .into_iter()
        .enumerate()
        .map(|(index, key)| KeyValue::new(key & 0xffff, index as u32))
        .collect()
}

fn expected_visible(input: &[KeyValue]) -> Vec<KeyValue> {
    let mut expected: Vec<_> = input
        .iter()
        .copied()
        .filter(|item| (0x4000..=0xbfff).contains(&item.key))
        .collect();
    expected.sort_by_key(|item| item.key);
    expected
}

fn validate(context: &Context, buffers: &ParticleBuffers, expected: &[KeyValue]) {
    let staging_size = 8 + buffers.sorted.size();
    let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Particle Benchmark Validation"),
        size: staging_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(&buffers.count, 0, &staging, 0, size_of::<u32>() as u64);
    encoder.copy_buffer_to_buffer(
        &buffers.sorted,
        0,
        &staging,
        8,
        expected.len() as u64 * size_of::<KeyValue>() as u64,
    );
    let submission = context.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("failed to wait for particle validation");
    receiver
        .recv()
        .expect("particle validation channel closed")
        .expect("failed to map particle validation");
    {
        let mapped = slice
            .get_mapped_range()
            .expect("particle validation mapped range unavailable");
        let selected = bytemuck::cast_slice::<u8, u32>(&mapped[..4])[0] as usize;
        let records: &[KeyValue] = bytemuck::cast_slice(&mapped[8..8 + expected.len() * 8]);
        assert_eq!(selected, expected.len());
        assert_eq!(records, expected);
    }
    staging.unmap();
}

fn storage_buffer(
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

fn configured_items() -> Vec<usize> {
    std::env::var("WGPU_PARTICLE_PIPELINE_ITEMS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|part| {
                    part.trim()
                        .replace('_', "")
                        .parse::<usize>()
                        .expect("WGPU_PARTICLE_PIPELINE_ITEMS contains an invalid integer")
                })
                .collect()
        })
        .unwrap_or_else(|| support::INPUT_SIZES[1..3].to_vec())
}

fn configured_seconds(name: &str) -> Option<u64> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("{name} must be an integer number of seconds"))
        })
        .filter(|seconds| *seconds > 0)
}

criterion_group!(benches, benchmark_particle_pipeline);
criterion_main!(benches);
