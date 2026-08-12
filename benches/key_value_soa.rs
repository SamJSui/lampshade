mod support;

use std::mem::size_of;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use futures::channel::oneshot;
use lampshade::{Context, KeyValueSoaSorter};
use wgpu::util::DeviceExt;

fn benchmark_key_value_soa(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().expect("benchmark runtime");
    let context = runtime
        .block_on(Context::init())
        .expect("benchmark GPU context");
    support::report_adapter(&context);

    let mut group = c.benchmark_group("key_value_soa_sort");
    for item_count in support::INPUT_SIZES
        .into_iter()
        .filter(|&items| (1_000_000..=10_000_000).contains(&items))
    {
        support::configure_group(&mut group, item_count);
        let keys = support::seeded_input(item_count, 0x50A5_0A11);
        let values: Vec<_> = (0..item_count as u32).collect();
        let mut expected: Vec<_> = keys.iter().copied().zip(values.iter().copied()).collect();
        expected.sort_by_key(|&(key, _)| key);

        let (fixed_keys, fixed_values) = buffers(&context, &keys, &values, "Fixed");
        let mut fixed = KeyValueSoaSorter::from_context(&context);
        fixed
            .prepare_sort(&fixed_keys, &fixed_values, item_count as u32)
            .expect("fixed SoA benchmark plan");
        submit_fixed(
            &context,
            &fixed,
            &fixed_keys,
            &fixed_values,
            item_count as u32,
        );
        validate_output(&runtime, &context, &fixed_keys, &fixed_values, &expected);

        group.bench_with_input(
            BenchmarkId::new("fixed", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    submit_fixed(
                        &context,
                        &fixed,
                        &fixed_keys,
                        &fixed_values,
                        item_count as u32,
                    );
                });
            },
        );

        let (counted_keys, counted_values) = buffers(&context, &keys, &values, "Counted");
        let count = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SoA Benchmark Count"),
                contents: bytemuck::bytes_of(&(item_count as u32)),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let mut counted = KeyValueSoaSorter::from_context(&context);
        counted
            .prepare_counted_from_word(&counted_keys, &counted_values, &count, 0, item_count as u32)
            .expect("counted SoA benchmark plan");
        submit_counted(
            &context,
            &counted,
            &counted_keys,
            &counted_values,
            &count,
            item_count as u32,
        );
        validate_output(
            &runtime,
            &context,
            &counted_keys,
            &counted_values,
            &expected,
        );

        group.bench_with_input(
            BenchmarkId::new("gpu_counted", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    submit_counted(
                        &context,
                        &counted,
                        &counted_keys,
                        &counted_values,
                        &count,
                        item_count as u32,
                    );
                });
            },
        );

        let (portable_keys, portable_values) = buffers(&context, &keys, &values, "Portable");
        let mut portable = KeyValueSoaSorter::new_portable(&context.device, &context.queue);
        portable
            .prepare_sort(&portable_keys, &portable_values, item_count as u32)
            .expect("portable SoA benchmark plan");
        submit_fixed(
            &context,
            &portable,
            &portable_keys,
            &portable_values,
            item_count as u32,
        );
        validate_output(
            &runtime,
            &context,
            &portable_keys,
            &portable_values,
            &expected,
        );

        group.bench_with_input(
            BenchmarkId::new("portable_fixed", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    submit_fixed(
                        &context,
                        &portable,
                        &portable_keys,
                        &portable_values,
                        item_count as u32,
                    );
                });
            },
        );
    }
    group.finish();
}

fn submit_fixed(
    context: &Context,
    sorter: &KeyValueSoaSorter,
    keys: &wgpu::Buffer,
    values: &wgpu::Buffer,
    item_count: u32,
) {
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    sorter
        .record_reserved_sort(&mut encoder, keys, values, item_count)
        .expect("fixed SoA benchmark recording");
    context.queue.submit(Some(encoder.finish()));
    support::wait_for_gpu(&context.device);
}

fn submit_counted(
    context: &Context,
    sorter: &KeyValueSoaSorter,
    keys: &wgpu::Buffer,
    values: &wgpu::Buffer,
    count: &wgpu::Buffer,
    item_count: u32,
) {
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    sorter
        .record_reserved_sort_counted_from_word(&mut encoder, keys, values, count, 0, item_count)
        .expect("counted SoA benchmark recording");
    context.queue.submit(Some(encoder.finish()));
    support::wait_for_gpu(&context.device);
}

fn validate_output(
    runtime: &tokio::runtime::Runtime,
    context: &Context,
    keys: &wgpu::Buffer,
    values: &wgpu::Buffer,
    expected: &[(u32, u32)],
) {
    let actual_keys = runtime.block_on(read_pod::<u32>(context, keys, expected.len()));
    let actual_values = runtime.block_on(read_pod::<u32>(context, values, expected.len()));
    assert_eq!(
        actual_keys
            .into_iter()
            .zip(actual_values)
            .collect::<Vec<_>>(),
        expected
    );
}

async fn read_pod<T: bytemuck::Pod>(
    context: &Context,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Vec<T> {
    let size = (len * size_of::<T>()) as u64;
    let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("SoA Benchmark Validation Readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    context.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    support::wait_for_gpu(&context.device);
    receiver
        .await
        .expect("benchmark validation callback")
        .expect("benchmark validation map");
    let mapped = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    result
}

fn buffers(
    context: &Context,
    keys: &[u32],
    values: &[u32],
    kind: &str,
) -> (wgpu::Buffer, wgpu::Buffer) {
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    (
        context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{kind} SoA Benchmark Keys")),
                contents: bytemuck::cast_slice(keys),
                usage,
            }),
        context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{kind} SoA Benchmark Values")),
                contents: bytemuck::cast_slice(values),
                usage,
            }),
    )
}

criterion_group!(benches, benchmark_key_value_soa);
criterion_main!(benches);
