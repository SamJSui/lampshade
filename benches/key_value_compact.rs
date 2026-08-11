mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lampshade::{Context, KeyValue, KeyValueCompactor};
use wgpu::util::DeviceExt;

const SELECTIVITY_PERCENTAGES: [u32; 5] = [0, 10, 50, 90, 100];

fn compact_on_cpu(input: &[KeyValue], mask: &[u32], output: &mut Vec<KeyValue>) {
    output.clear();
    output.extend(
        input
            .iter()
            .zip(mask)
            .filter_map(|(&item, &keep)| (keep == 1).then_some(item)),
    );
}

fn benchmark_key_value_compaction(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let context = runtime.block_on(Context::init()).unwrap();
    let mut compactor = KeyValueCompactor::from_context(&context);
    support::report_adapter(&context);
    let mut group = c.benchmark_group("key_value_stream_compaction");

    for item_count in support::INPUT_SIZES {
        support::configure_group(&mut group, item_count);
        let input: Vec<_> = support::seeded_input(item_count, 0x0C0A_0AC7)
            .into_iter()
            .enumerate()
            .map(|(index, key)| KeyValue::new(key, index as u32))
            .collect();
        let gpu_input = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Key-Value Compaction Benchmark Input"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let gpu_output = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Key-Value Compaction Benchmark Output"),
            size: gpu_input.size(),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let gpu_count = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Key-Value Compaction Benchmark Count"),
            size: size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        for selectivity in SELECTIVITY_PERCENTAGES {
            let mask: Vec<u32> = (0..item_count)
                .map(|index| u32::from(index % 100 < selectivity as usize))
                .collect();
            let gpu_mask = context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Key-Value Compaction Benchmark Mask"),
                    contents: bytemuck::cast_slice(&mask),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });
            let mut cpu_output = Vec::with_capacity(item_count);

            group.bench_with_input(
                BenchmarkId::new(format!("cpu_scalar_{selectivity}pct"), item_count),
                &item_count,
                |b, &_| {
                    b.iter(|| {
                        compact_on_cpu(black_box(&input), black_box(&mask), &mut cpu_output);
                        black_box(&cpu_output);
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("gpu_resident_{selectivity}pct"), item_count),
                &item_count,
                |b, &_| {
                    b.iter(|| {
                        compactor
                            .compact_gpu_to_gpu(
                                &gpu_input,
                                &gpu_mask,
                                &gpu_output,
                                &gpu_count,
                                item_count as u32,
                            )
                            .expect("GPU-resident key-value compaction failed");
                        support::wait_for_gpu(&context.device);
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_key_value_compaction);
criterion_main!(benches);
