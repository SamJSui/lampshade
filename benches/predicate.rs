mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use wgpu::util::DeviceExt;
use wgpu_primitives::{Compactor, Context, MaskGenerator, U32Predicate};

const PREDICATE_THRESHOLD: u32 = 1_u32 << 31;

fn mask_on_cpu(input: &[u32], output: &mut [u32]) {
    for (&value, flag) in input.iter().zip(output) {
        *flag = u32::from(value < PREDICATE_THRESHOLD);
    }
}

fn benchmark_predicate(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let context = runtime.block_on(Context::init()).unwrap();
    let generator = MaskGenerator::from_context(&context);
    let mut compactor = Compactor::from_context(&context);
    support::report_adapter(&context);
    let mut group = c.benchmark_group("predicate_mask");

    for item_count in support::INPUT_SIZES {
        support::configure_group(&mut group, item_count);
        let input = support::seeded_input(item_count, 0x50ED_1CA7);
        let gpu_input = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Predicate Benchmark Input"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let gpu_mask = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Predicate Benchmark Mask"),
            size: MaskGenerator::mask_buffer_size(item_count as u32)
                .expect("predicate mask size overflow"),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let gpu_output = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Predicate Compaction Benchmark Output"),
            size: gpu_input.size(),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let gpu_count = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Predicate Compaction Benchmark Count"),
            size: size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut cpu_mask = vec![0_u32; item_count];
        let predicate = U32Predicate::LessThan(PREDICATE_THRESHOLD);

        group.bench_with_input(
            BenchmarkId::new("cpu_scalar", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    mask_on_cpu(black_box(&input), black_box(&mut cpu_mask));
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gpu_resident", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    generator
                        .mask_gpu_to_gpu(&gpu_input, &gpu_mask, item_count as u32, predicate)
                        .expect("GPU-resident predicate failed");
                    support::wait_for_gpu(&context.device);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("gpu_resident_then_compact", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    let mut encoder = context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    generator
                        .record_mask(
                            &mut encoder,
                            &gpu_input,
                            &gpu_mask,
                            item_count as u32,
                            predicate,
                        )
                        .expect("predicate recording failed");
                    compactor
                        .record_compact(
                            &mut encoder,
                            &gpu_input,
                            &gpu_mask,
                            &gpu_output,
                            &gpu_count,
                            item_count as u32,
                        )
                        .expect("compaction recording failed");
                    context.queue.submit(Some(encoder.finish()));
                    support::wait_for_gpu(&context.device);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_predicate);
criterion_main!(benches);
