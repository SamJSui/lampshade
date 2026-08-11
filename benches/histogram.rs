mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lampshade::{Context, Histogram};
use wgpu::util::DeviceExt;

const BIN_COUNT: u32 = 256;
const HISTOGRAM_SEED: u64 = 0xA11C_E5ED;

fn benchmark_histogram(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let context = runtime.block_on(Context::init()).unwrap();
    let histogram = Histogram::from_context(&context);
    support::report_adapter(&context);

    let mut group = c.benchmark_group("histogram");
    for item_count in support::INPUT_SIZES {
        support::configure_group(&mut group, item_count);
        let input: Vec<_> = support::seeded_input(item_count, HISTOGRAM_SEED)
            .into_iter()
            .map(|value| value % BIN_COUNT)
            .collect();
        let gpu_input = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Histogram Benchmark Input"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let gpu_output = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Histogram Benchmark Output"),
            size: Histogram::output_buffer_size(BIN_COUNT).unwrap(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        group.bench_with_input(
            BenchmarkId::new("cpu_scalar", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    let mut bins = [0_u32; BIN_COUNT as usize];
                    for &value in black_box(&input) {
                        bins[value as usize] += 1;
                    }
                    black_box(bins);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gpu_round_trip", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    black_box(
                        pollster::block_on(histogram.histogram(black_box(&input), BIN_COUNT))
                            .expect("GPU round-trip histogram failed"),
                    );
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gpu_resident", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    histogram
                        .histogram_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32, BIN_COUNT)
                        .expect("GPU-resident histogram failed");
                    support::wait_for_gpu(&context.device);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_histogram);
criterion_main!(benches);
