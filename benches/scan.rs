use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use wgpu::util::DeviceExt;
use wgpu_algorithms::{Context, Scanner};

fn benchmark_scan(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let ctx = rt.block_on(Context::init()).unwrap();
    let mut scanner = Scanner::new(&ctx);

    let mut group = c.benchmark_group("Prefix Scan");

    let n = 100_000_000;
    let input: Vec<u32> = (0..n).map(|_| rand::random()).collect();

    group.throughput(Throughput::Elements(n as u64));
    group.sample_size(10);

    let src = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scan Input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    let dst = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scan Output"),
        size: src.size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    group.bench_function("GPU Scan (Resident)", |b| {
        b.iter(|| {
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            scanner.record_scan(&mut encoder, &src, &dst);

            ctx.queue.submit(Some(encoder.finish()));

            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_scan);
criterion_main!(benches);
