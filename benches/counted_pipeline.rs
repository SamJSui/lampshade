mod support;

use std::{hint::black_box, sync::mpsc};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lampshade::{
    Compactor, Context, CountedSortDispatch, GpuCountPlan, Reducer, Sorter, U32Reduction,
};
use wgpu::util::DeviceExt;

const DEFAULT_SELECTIVITIES: [u32; 3] = [10, 50, 90];

fn benchmark_counted_pipeline(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("failed to create benchmark runtime");
    let context = runtime
        .block_on(Context::init())
        .expect("failed to create wgpu context");
    support::report_adapter(&context);
    let mut group = c.benchmark_group("gpu_counted_pipeline");

    for item_count in configured_items() {
        support::configure_group(&mut group, item_count);
        if let Some(seconds) = configured_seconds("WGPU_COUNTED_PIPELINE_MEASUREMENT_SECONDS") {
            group.measurement_time(std::time::Duration::from_secs(seconds));
        }
        if let Some(seconds) = configured_seconds("WGPU_COUNTED_PIPELINE_WARMUP_SECONDS") {
            group.warm_up_time(std::time::Duration::from_secs(seconds));
        }
        let input = support::seeded_input(item_count, 0xC01D_ED50);
        let gpu_input = initialized_storage(&context.device, "Pipeline Input", &input);
        let compacted = storage_output(
            &context.device,
            "Pipeline Compacted",
            gpu_input.size(),
            wgpu::BufferUsages::empty(),
        );
        let sorted = storage_output(
            &context.device,
            "Pipeline Sorted",
            gpu_input.size(),
            wgpu::BufferUsages::COPY_SRC,
        );
        let count = storage_output(
            &context.device,
            "Pipeline Count",
            size_of::<u32>() as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        let sum = storage_output(
            &context.device,
            "Pipeline Sum",
            Reducer::output_buffer_size(),
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        );
        let count_staging = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pipeline Count Readback"),
            size: size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut compactor = Compactor::from_context(&context);
        let mut sorter = Sorter::from_context(&context);
        let mut reducer = Reducer::from_context(&context);
        let indirect_plan = GpuCountPlan::new(&context.device, &count, item_count as u32)
            .expect("failed to create GPU count plan");
        let capacity_plan = GpuCountPlan::new_with_sort_dispatch(
            &context.device,
            &count,
            item_count as u32,
            CountedSortDispatch::Capacity,
        )
        .expect("failed to create capacity-dispatch GPU count plan");

        for selectivity in configured_selectivities() {
            let mask: Vec<_> = (0..item_count)
                .map(|index| u32::from(index % 100 < selectivity as usize))
                .collect();
            let gpu_mask = context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Pipeline Selection Mask"),
                    contents: bytemuck::cast_slice(&mask),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });
            let mut expected: Vec<_> = input
                .iter()
                .zip(&mask)
                .filter_map(|(&value, &keep)| (keep == 1).then_some(value))
                .collect();
            expected.sort_unstable();
            let expected_sum = expected
                .iter()
                .fold(0_u32, |total, value| total.wrapping_add(*value));

            run_counted(
                &context,
                &mut compactor,
                &mut sorter,
                &mut reducer,
                &indirect_plan,
                &gpu_input,
                &gpu_mask,
                &compacted,
                &count,
                &sorted,
                &sum,
                item_count as u32,
            );
            validate_outputs(&context, &count, &sorted, &sum, &expected, expected_sum);

            run_counted(
                &context,
                &mut compactor,
                &mut sorter,
                &mut reducer,
                &capacity_plan,
                &gpu_input,
                &gpu_mask,
                &compacted,
                &count,
                &sorted,
                &sum,
                item_count as u32,
            );
            validate_outputs(&context, &count, &sorted, &sum, &expected, expected_sum);

            let host_count = run_host_synchronized(
                &context,
                &mut compactor,
                &mut sorter,
                &mut reducer,
                &gpu_input,
                &gpu_mask,
                &compacted,
                &count,
                &count_staging,
                &sorted,
                &sum,
                item_count as u32,
            );
            assert_eq!(host_count as usize, expected.len());
            validate_outputs(&context, &count, &sorted, &sum, &expected, expected_sum);

            group.bench_with_input(
                BenchmarkId::new(format!("gpu_indirect_{selectivity}pct"), item_count),
                &item_count,
                |b, &_| {
                    b.iter(|| {
                        run_counted(
                            &context,
                            &mut compactor,
                            &mut sorter,
                            &mut reducer,
                            &indirect_plan,
                            &gpu_input,
                            &gpu_mask,
                            &compacted,
                            &count,
                            &sorted,
                            &sum,
                            item_count as u32,
                        );
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("gpu_capacity_{selectivity}pct"), item_count),
                &item_count,
                |b, &_| {
                    b.iter(|| {
                        run_counted(
                            &context,
                            &mut compactor,
                            &mut sorter,
                            &mut reducer,
                            &capacity_plan,
                            &gpu_input,
                            &gpu_mask,
                            &compacted,
                            &count,
                            &sorted,
                            &sum,
                            item_count as u32,
                        );
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("host_count_readback_{selectivity}pct"), item_count),
                &item_count,
                |b, &_| {
                    b.iter(|| {
                        black_box(run_host_synchronized(
                            &context,
                            &mut compactor,
                            &mut sorter,
                            &mut reducer,
                            &gpu_input,
                            &gpu_mask,
                            &compacted,
                            &count,
                            &count_staging,
                            &sorted,
                            &sum,
                            item_count as u32,
                        ));
                    });
                },
            );
        }
    }
    group.finish();
}

#[allow(clippy::too_many_arguments)]
fn run_counted(
    context: &Context,
    compactor: &mut Compactor,
    sorter: &mut Sorter,
    reducer: &mut Reducer,
    count_plan: &GpuCountPlan,
    input: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    compacted: &wgpu::Buffer,
    count: &wgpu::Buffer,
    sorted: &wgpu::Buffer,
    sum: &wgpu::Buffer,
    capacity: u32,
) {
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU-Counted Pipeline"),
        });
    compactor
        .record_compact(&mut encoder, input, mask, compacted, count, capacity)
        .expect("failed to record compaction");
    count_plan.record_prepare(&mut encoder);
    sorter
        .record_sort_with_count_plan(&mut encoder, compacted, sorted, count_plan)
        .expect("failed to record counted sort");
    reducer
        .record_reduce_with_count_plan(&mut encoder, sorted, sum, count_plan, U32Reduction::Sum)
        .expect("failed to record counted reduction");
    context.queue.submit([encoder.finish()]);
    support::wait_for_gpu(&context.device);
}

#[allow(clippy::too_many_arguments)]
fn run_host_synchronized(
    context: &Context,
    compactor: &mut Compactor,
    sorter: &mut Sorter,
    reducer: &mut Reducer,
    input: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    compacted: &wgpu::Buffer,
    count: &wgpu::Buffer,
    count_staging: &wgpu::Buffer,
    sorted: &wgpu::Buffer,
    sum: &wgpu::Buffer,
    capacity: u32,
) -> u32 {
    let mut count_encoder =
        context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Host-Synchronized Compaction"),
            });
    compactor
        .record_compact(&mut count_encoder, input, mask, compacted, count, capacity)
        .expect("failed to record compaction");
    count_encoder.copy_buffer_to_buffer(count, 0, count_staging, 0, size_of::<u32>() as u64);
    let count_submission = context.queue.submit([count_encoder.finish()]);
    let selected = map_scalar(&context.device, count_staging, count_submission);

    let mut work_encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Host-Synchronized Sort and Reduction"),
        });
    sorter
        .record_sort(&mut work_encoder, compacted, sorted, selected)
        .expect("failed to record fixed-length sort");
    reducer
        .record_reduce(&mut work_encoder, sorted, sum, selected, U32Reduction::Sum)
        .expect("failed to record fixed-length reduction");
    context.queue.submit([work_encoder.finish()]);
    support::wait_for_gpu(&context.device);
    selected
}

fn map_scalar(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    submission: wgpu::SubmissionIndex,
) -> u32 {
    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("failed to wait for count readback");
    receiver
        .recv()
        .expect("count readback channel closed")
        .expect("count readback mapping failed");
    let value = {
        let mapped = slice.get_mapped_range().expect("count staging is mapped");
        bytemuck::cast_slice::<u8, u32>(&mapped)[0]
    };
    staging.unmap();
    value
}

fn validate_outputs(
    context: &Context,
    count: &wgpu::Buffer,
    sorted: &wgpu::Buffer,
    sum: &wgpu::Buffer,
    expected: &[u32],
    expected_sum: u32,
) {
    assert_eq!(read_u32(context, count, 1), [expected.len() as u32]);
    assert_eq!(read_u32(context, sorted, expected.len()), expected);
    assert_eq!(read_u32(context, sum, 1), [expected_sum]);
}

fn read_u32(context: &Context, source: &wgpu::Buffer, len: usize) -> Vec<u32> {
    if len == 0 {
        return Vec::new();
    }
    let size = (len * size_of::<u32>()) as u64;
    let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Pipeline Validation Readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
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
        .expect("failed to wait for validation readback");
    receiver
        .recv()
        .expect("validation readback channel closed")
        .expect("validation readback mapping failed");
    let values = {
        let mapped = slice
            .get_mapped_range()
            .expect("validation staging is mapped");
        bytemuck::cast_slice(&mapped).to_vec()
    };
    staging.unmap();
    values
}

fn initialized_storage(device: &wgpu::Device, label: &'static str, data: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn storage_output(
    device: &wgpu::Device,
    label: &'static str,
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
    csv_env("WGPU_COUNTED_PIPELINE_ITEMS").unwrap_or_else(|| support::INPUT_SIZES[1..].to_vec())
}

fn configured_selectivities() -> Vec<u32> {
    let values = csv_env("WGPU_COUNTED_PIPELINE_SELECTIVITIES").unwrap_or_else(|| {
        DEFAULT_SELECTIVITIES
            .into_iter()
            .map(|value| value as usize)
            .collect()
    });
    values
        .into_iter()
        .map(|value| u32::try_from(value).expect("selectivity exceeds u32"))
        .inspect(|value| assert!(*value <= 100, "selectivity must be between 0 and 100"))
        .collect()
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

fn csv_env(name: &str) -> Option<Vec<usize>> {
    std::env::var(name).ok().map(|value| {
        let parsed: Vec<_> = value
            .split(',')
            .map(|part| {
                part.trim()
                    .replace('_', "")
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("{name} contains an invalid integer"))
            })
            .collect();
        assert!(!parsed.is_empty(), "{name} must contain at least one value");
        parsed
    })
}

criterion_group!(benches, benchmark_counted_pipeline);
criterion_main!(benches);
