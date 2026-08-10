mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rayon::prelude::*;
use wgpu::util::DeviceExt;
use wgpu_primitives::{Context, Sorter};

const SORT_SEED: u64 = 0x5077;

fn benchmark_sort(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let context = runtime.block_on(Context::init()).unwrap();
    let mut sorter = Sorter::from_context(&context);
    support::report_adapter(&context);

    let mut group = c.benchmark_group("radix_sort");

    for item_count in support::INPUT_SIZES {
        support::configure_group(&mut group, item_count);
        let input = support::seeded_input(item_count, SORT_SEED);
        let gpu_input = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sort Benchmark Input"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let gpu_output = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sort Benchmark Output"),
            size: gpu_input.size(),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let gpu_count = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Counted Sort Benchmark Length"),
                contents: bytemuck::cast_slice(&[item_count as u32]),
                usage: wgpu::BufferUsages::STORAGE,
            });

        group.bench_with_input(
            BenchmarkId::new("cpu_rayon", item_count),
            &item_count,
            |b, &_| {
                b.iter_with_setup(
                    || input.clone(),
                    |mut values| {
                        values.par_sort_unstable();
                        black_box(values);
                    },
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gpu_round_trip", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    let output = pollster::block_on(sorter.sort(&input))
                        .expect("GPU round-trip sort failed");
                    black_box(output);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("gpu_resident", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    sorter
                        .sort_gpu_to_gpu(&gpu_input, &gpu_output, item_count as u32)
                        .expect("GPU-resident sort failed");
                    support::wait_for_gpu(&context.device);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gpu_resident_counted", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    sorter
                        .sort_counted_gpu_to_gpu(
                            &gpu_input,
                            &gpu_output,
                            &gpu_count,
                            item_count as u32,
                        )
                        .expect("GPU-counted resident sort failed");
                    support::wait_for_gpu(&context.device);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_sort);
criterion_main!(benches);
