mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use wgpu::util::DeviceExt;
use wgpu_primitives::{Context, Reducer, U32Reduction};

const REDUCTION_SEED: u64 = 0x005E_D0CE;

fn benchmark_reduction(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let context = runtime.block_on(Context::init()).unwrap();
    let mut reducer = Reducer::from_context(&context);
    support::report_adapter(&context);

    let mut group = c.benchmark_group("reduction");
    for item_count in support::INPUT_SIZES {
        support::configure_group(&mut group, item_count);
        let input = support::seeded_input(item_count, REDUCTION_SEED);
        let gpu_input = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Reduction Benchmark Input"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let gpu_output = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reduction Benchmark Output"),
            size: Reducer::output_buffer_size(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        group.bench_with_input(
            BenchmarkId::new("cpu_scalar_sum", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    black_box(
                        input
                            .iter()
                            .fold(0_u32, |sum, value| sum.wrapping_add(black_box(*value))),
                    );
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gpu_round_trip_sum", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    black_box(
                        pollster::block_on(reducer.sum(black_box(&input)))
                            .expect("GPU round-trip reduction failed"),
                    );
                });
            },
        );

        for operation in [U32Reduction::Sum, U32Reduction::Min, U32Reduction::Max] {
            group.bench_with_input(
                BenchmarkId::new(
                    format!("gpu_resident_{}", operation_name(operation)),
                    item_count,
                ),
                &item_count,
                |b, &_| {
                    b.iter(|| {
                        reducer
                            .reduce_gpu_to_gpu(
                                &gpu_input,
                                &gpu_output,
                                item_count as u32,
                                operation,
                            )
                            .expect("GPU-resident reduction failed");
                        support::wait_for_gpu(&context.device);
                    });
                },
            );
        }
    }
    group.finish();
}

const fn operation_name(operation: U32Reduction) -> &'static str {
    match operation {
        U32Reduction::Sum => "sum",
        U32Reduction::Min => "min",
        U32Reduction::Max => "max",
    }
}

criterion_group!(benches, benchmark_reduction);
criterion_main!(benches);
