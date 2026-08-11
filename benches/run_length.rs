#[allow(dead_code)]
mod support;

use std::hint::black_box;
use std::sync::mpsc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lampshade::{Context, RunLengthEncoder, RunLengthOutputBuffers};
use wgpu::util::DeviceExt;

const RUN_LENGTHS: [u32; 3] = [1, 8, 256];

fn cpu_encode(input: &[u32], values: &mut Vec<u32>, lengths: &mut Vec<u32>) {
    values.clear();
    lengths.clear();
    for &value in input {
        if values.last() == Some(&value) {
            *lengths.last_mut().expect("a repeated value has a run") += 1;
        } else {
            values.push(value);
            lengths.push(1);
        }
    }
}

fn validate_gpu_output(
    context: &Context,
    values: &wgpu::Buffer,
    lengths: &wgpu::Buffer,
    count: &wgpu::Buffer,
    expected_values: &[u32],
    expected_lengths: &[u32],
) {
    let output_bytes =
        u64::try_from(size_of_val(expected_values)).expect("RLE validation output fits in u64");
    let lengths_offset = size_of::<u32>() as u64 + output_bytes;
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("RLE Benchmark Validation Readback"),
        size: lengths_offset + output_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RLE Benchmark Validation Encoder"),
        });
    encoder.copy_buffer_to_buffer(count, 0, &readback, 0, size_of::<u32>() as u64);
    encoder.copy_buffer_to_buffer(values, 0, &readback, size_of::<u32>() as u64, output_bytes);
    encoder.copy_buffer_to_buffer(lengths, 0, &readback, lengths_offset, output_bytes);
    context.queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("validation receiver dropped");
    });
    support::wait_for_gpu(&context.device);
    receiver
        .recv()
        .expect("validation callback did not run")
        .expect("validation mapping failed");
    let mapped = slice
        .get_mapped_range()
        .expect("validation readback unavailable");
    let words: &[u32] = bytemuck::cast_slice(&mapped);
    assert_eq!(words[0] as usize, expected_values.len());
    assert_eq!(&words[1..1 + expected_values.len()], expected_values);
    assert_eq!(&words[1 + expected_values.len()..], expected_lengths);
}

fn benchmark_run_length(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let context = runtime.block_on(Context::init()).unwrap();
    let mut encoder = RunLengthEncoder::from_context(&context);
    support::report_adapter(&context);
    let mut group = c.benchmark_group("run_length_encoding");

    for item_count in support::INPUT_SIZES {
        support::configure_group(&mut group, item_count);
        for average_run in RUN_LENGTHS {
            let input: Vec<_> = (0..item_count as u32)
                .map(|index| index / average_run)
                .collect();
            let mut cpu_values = Vec::with_capacity(input.len());
            let mut cpu_lengths = Vec::with_capacity(input.len());
            cpu_encode(&input, &mut cpu_values, &mut cpu_lengths);
            let gpu_input = context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("RLE Benchmark Input"),
                    contents: bytemuck::cast_slice(&input),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let values = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RLE Benchmark Values"),
                size: gpu_input.size(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let lengths = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RLE Benchmark Lengths"),
                size: gpu_input.size(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let count = context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RLE Benchmark Count"),
                size: size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let input_count =
                context
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("RLE Benchmark Input Count"),
                        contents: bytemuck::bytes_of(&(item_count as u32)),
                        usage: wgpu::BufferUsages::STORAGE,
                    });

            encoder
                .encode_gpu_to_gpu(
                    &gpu_input,
                    RunLengthOutputBuffers::new(&values, &lengths, &count),
                    item_count as u32,
                )
                .expect("fixed GPU RLE validation dispatch failed");
            validate_gpu_output(
                &context,
                &values,
                &lengths,
                &count,
                &cpu_values,
                &cpu_lengths,
            );
            encoder
                .encode_counted_gpu_to_gpu(
                    &gpu_input,
                    &input_count,
                    RunLengthOutputBuffers::new(&values, &lengths, &count),
                    item_count as u32,
                )
                .expect("counted GPU RLE validation dispatch failed");
            validate_gpu_output(
                &context,
                &values,
                &lengths,
                &count,
                &cpu_values,
                &cpu_lengths,
            );

            group.bench_with_input(
                BenchmarkId::new(format!("cpu_scalar_run_{average_run}"), item_count),
                &item_count,
                |b, &_| {
                    b.iter(|| {
                        cpu_encode(black_box(&input), &mut cpu_values, &mut cpu_lengths);
                        black_box((&cpu_values, &cpu_lengths));
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("gpu_resident_run_{average_run}"), item_count),
                &item_count,
                |b, &_| {
                    b.iter(|| {
                        encoder
                            .encode_gpu_to_gpu(
                                &gpu_input,
                                RunLengthOutputBuffers::new(&values, &lengths, &count),
                                item_count as u32,
                            )
                            .expect("GPU-resident RLE failed");
                        support::wait_for_gpu(&context.device);
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("gpu_counted_dense_run_{average_run}"), item_count),
                &item_count,
                |b, &_| {
                    b.iter(|| {
                        encoder
                            .encode_counted_gpu_to_gpu(
                                &gpu_input,
                                &input_count,
                                RunLengthOutputBuffers::new(&values, &lengths, &count),
                                item_count as u32,
                            )
                            .expect("GPU-counted RLE failed");
                        support::wait_for_gpu(&context.device);
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_run_length);
criterion_main!(benches);
