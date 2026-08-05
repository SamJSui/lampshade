mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use wgpu::util::DeviceExt;
use wgpu_primitives::{Context, Scanner};

const SCAN_SEED: u64 = 0x5CA1;

fn scan_on_cpu(input: &[u32], output: &mut [u32]) {
    let mut sum = 0_u32;
    for (value, prefix) in input.iter().zip(output) {
        sum = sum.wrapping_add(*value);
        *prefix = sum;
    }
}

fn exclusive_scan_on_cpu(input: &[u32], output: &mut [u32]) {
    let mut sum = 0_u32;
    for (value, prefix) in input.iter().zip(output) {
        *prefix = sum;
        sum = sum.wrapping_add(*value);
    }
}

fn benchmark_scan(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let context = runtime.block_on(Context::init()).unwrap();
    let mut scanner = Scanner::from_context(&context);
    support::report_adapter(&context);

    let mut group = c.benchmark_group("prefix_scan");

    for item_count in support::INPUT_SIZES {
        support::configure_group(&mut group, item_count);
        let input = support::seeded_input(item_count, SCAN_SEED);
        let mut cpu_output = vec![0_u32; item_count];
        let gpu_input = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Scan Benchmark Input"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let gpu_output = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Scan Benchmark Output"),
            size: gpu_input.size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        group.bench_with_input(
            BenchmarkId::new("cpu_scalar", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    scan_on_cpu(black_box(&input), &mut cpu_output);
                    black_box(&cpu_output);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cpu_scalar_exclusive", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    exclusive_scan_on_cpu(black_box(&input), &mut cpu_output);
                    black_box(&cpu_output);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gpu_round_trip", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    let output = pollster::block_on(scanner.scan(&input))
                        .expect("GPU round-trip scan failed");
                    black_box(output);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gpu_round_trip_exclusive", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    let output = pollster::block_on(scanner.scan_exclusive(&input))
                        .expect("GPU round-trip exclusive scan failed");
                    black_box(output);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gpu_resident", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    scanner
                        .scan_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32)
                        .expect("GPU-resident scan failed");
                    support::wait_for_gpu(&context.device);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gpu_resident_exclusive", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    scanner
                        .scan_exclusive_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32)
                        .expect("GPU-resident exclusive scan failed");
                    support::wait_for_gpu(&context.device);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_scan);
criterion_main!(benches);
