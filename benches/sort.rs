use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rayon::prelude::*;
use wgpu_algorithms::context::Context;
use wgpu_algorithms::sort::Sorter;

fn benchmark_sorts(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let ctx = rt.block_on(Context::init()).unwrap();
    let mut my_sorter = Sorter::new(&ctx);

    let mut group = c.benchmark_group("Sort");

    let inputs = [100_000, 1_000_000, 10_000_000, 100_000_000];

    for &n in &inputs {
        group.throughput(Throughput::Elements(n as u64));
        let data: Vec<u32> = (0..n).map(|_| rand::random()).collect();

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
                pollster::block_on(my_sorter.sort_radix(&data));
            });
        });

        // 3. GPU (Resident) - The "Pipeline" use case
        // Includes: Upload -> Sort
        // Excludes: Download (The result stays on VRAM)
        group.bench_with_input(BenchmarkId::new("GPU (Resident)", n), &n, |b, &_| {
            b.iter(|| {
                my_sorter.sort_resident(&data);
                // Force GPU to finish execution to measure raw throughput
                ctx.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_sorts);
criterion_main!(benches);
