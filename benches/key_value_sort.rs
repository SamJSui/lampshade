mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rayon::prelude::*;
use wgpu::util::DeviceExt;
use wgpu_primitives::{Context, KeyValue, KeyValueSorter};

const SORT_SEED: u64 = 0x4B56;

fn benchmark_key_value_sort(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let context = runtime.block_on(Context::init()).unwrap();
    let mut sorter = KeyValueSorter::from_context(&context);
    support::report_adapter(&context);

    let mut group = c.benchmark_group("key_value_radix_sort");

    for item_count in support::INPUT_SIZES {
        support::configure_group(&mut group, item_count);
        let input: Vec<_> = support::seeded_input(item_count, SORT_SEED)
            .into_iter()
            .enumerate()
            .map(|(index, key)| KeyValue::new(key & 0xffff, index as u32))
            .collect();
        let gpu_input = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Key-Value Sort Benchmark Input"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let gpu_output = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Key-Value Sort Benchmark Output"),
            size: gpu_input.size(),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        group.bench_with_input(
            BenchmarkId::new("cpu_rayon_stable", item_count),
            &item_count,
            |b, &_| {
                b.iter_with_setup(
                    || input.clone(),
                    |mut items| {
                        items.par_sort_by_key(|item| item.key);
                        black_box(items);
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
                        .expect("GPU round-trip key-value sort failed");
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
                        .expect("GPU-resident key-value sort failed");
                    support::wait_for_gpu(&context.device);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_key_value_sort);
criterion_main!(benches);
