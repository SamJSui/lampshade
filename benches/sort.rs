use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rayon::prelude::*;
use wgpu::util::DeviceExt;
use wgpu_primitives::context::Context;
use wgpu_primitives::sort::Sorter;

fn benchmark_sorts(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let ctx = rt.block_on(Context::init()).unwrap();
    let mut sorter = Sorter::from_context(&ctx);

    let mut group = c.benchmark_group("Sort");

    let inputs = [100_000, 1_000_000, 10_000_000, 100_000_000];

    for &n in &inputs {
        group.throughput(Throughput::Elements(n as u64));
        let data: Vec<u32> = (0..n).map(|_| rand::random()).collect();
        let gpu_input = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sort Benchmark Input"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let gpu_output = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sort Benchmark Output"),
            size: (n * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        if n >= 10_000_000 {
            group.sample_size(10);
            group.measurement_time(std::time::Duration::from_secs(15));
        } else {
            group.sample_size(30);
            group.measurement_time(std::time::Duration::from_secs(5));
        }

        // 1. CPU (Rayon) - The Baseline
        group.bench_with_input(BenchmarkId::new("CPU (Rayon)", n), &n, |b, &_| {
            b.iter_with_setup(|| data.clone(), |mut input| input.par_sort_unstable());
        });

        // 2. GPU (Round Trip) - The "Utility" use case
        // Includes: Upload -> Sort -> Download
        group.bench_with_input(BenchmarkId::new("GPU (Round Trip)", n), &n, |b, &_| {
            b.iter(|| {
                pollster::block_on(sorter.sort(&data)).expect("GPU round-trip sort failed");
            });
        });

        // 3. GPU (Buffer to Buffer) - The composable pipeline use case
        // Excludes upload and download; input and output remain on the GPU.
        group.bench_with_input(
            BenchmarkId::new("GPU (Buffer to Buffer)", n),
            &n,
            |b, &_| {
                b.iter(|| {
                    sorter
                        .sort_gpu_to_gpu(&gpu_input, &gpu_output, n as u32)
                        .expect("GPU buffer-to-buffer sort failed");
                    // Force GPU to finish execution to measure raw throughput
                    ctx.device
                        .poll(wgpu::PollType::Wait {
                            submission_index: None,
                            timeout: None,
                        })
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_sorts);
criterion_main!(benches);
