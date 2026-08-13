#[allow(dead_code)]
mod support;

use std::{borrow::Cow, sync::mpsc};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lampshade::{ArgminByKey, Context, KeyValue, KeyValueSorter};
use wgpu::util::DeviceExt;

const BLOCK_SIZE: u32 = 256;
const PAIR_SIZE: u64 = size_of::<KeyValue>() as u64;
const INPUT_SIZES: [usize; 5] = [4_096, 65_536, 131_072, 1_000_000, 10_000_000];

struct RawArgmin {
    pipeline: wgpu::ComputePipeline,
    passes: Vec<RawPass>,
    best: wgpu::Buffer,
}

struct RawPass {
    bind_group: wgpu::BindGroup,
    _output: wgpu::Buffer,
    _params: wgpu::Buffer,
    dispatch: (u32, u32),
}

impl RawArgmin {
    fn new(device: &wgpu::Device, input: &wgpu::Buffer, items: u32) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Raw Argmin Benchmark Layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Raw Argmin Benchmark Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("raw_argmin.wgsl"))),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Raw Argmin Benchmark Pipeline Layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Raw Argmin Benchmark Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("reduce_min"),
            compilation_options: Default::default(),
            cache: None,
        });
        let max_groups = device.limits().max_compute_workgroups_per_dimension;
        let mut passes = Vec::new();
        let mut current_input = input.clone();
        let mut current_items = items;
        loop {
            let output_items = current_items.div_ceil(BLOCK_SIZE);
            let output = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Raw Argmin Benchmark Level"),
                size: u64::from(output_items) * PAIR_SIZE,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Raw Argmin Benchmark Parameters"),
                contents: bytemuck::cast_slice(&[current_items, 0, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Raw Argmin Benchmark Bind Group"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: current_input.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
            passes.push(RawPass {
                bind_group,
                _output: output.clone(),
                _params: params,
                dispatch: dispatch_dimensions(output_items, max_groups),
            });
            current_input = output;
            current_items = output_items;
            if output_items == 1 {
                break;
            }
        }
        Self {
            pipeline,
            passes,
            best: current_input,
        }
    }

    fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        for pass_info in &self.passes {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Raw Argmin Benchmark"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &pass_info.bind_group, &[]);
            pass.dispatch_workgroups(pass_info.dispatch.0, pass_info.dispatch.1, 1);
        }
    }
}

fn benchmark_argmin(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let context = runtime.block_on(Context::init()).unwrap();
    support::report_adapter(&context);
    let mut group = c.benchmark_group("argmin_by_key");

    for item_count in INPUT_SIZES {
        support::configure_group(&mut group, item_count);
        let input = input_for(item_count);
        let expected = cpu_argmin(&input);
        let gpu_input = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Argmin Benchmark Input"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Argmin Benchmark Output"),
            size: PAIR_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let count = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Argmin Benchmark Count"),
                contents: bytemuck::bytes_of(&(item_count as u32)),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let sparse_items = (item_count / 10).max(1);
        let sparse_count = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sparse Argmin Benchmark Count"),
                contents: bytemuck::bytes_of(&(sparse_items as u32)),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let sort_output = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Argmin Sort Baseline Output"),
            size: item_count as u64 * PAIR_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut selector = ArgminByKey::from_context(&context);
        let mut sorter = KeyValueSorter::from_context(&context);
        let raw = RawArgmin::new(&context.device, &gpu_input, item_count as u32);

        selector
            .argmin_gpu_to_gpu(&gpu_input, &output, item_count as u32)
            .expect("Lampshade argmin validation failed");
        support::wait_for_gpu(&context.device);
        assert_eq!(read_pair(&context, &output), expected);
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        raw.record(&mut encoder);
        context.queue.submit(Some(encoder.finish()));
        support::wait_for_gpu(&context.device);
        assert_eq!(read_pair(&context, &raw.best), expected);
        selector
            .argmin_counted_gpu_to_gpu(&gpu_input, &output, &sparse_count, item_count as u32)
            .expect("Lampshade sparse counted argmin validation failed");
        support::wait_for_gpu(&context.device);
        assert_eq!(
            read_pair(&context, &output),
            cpu_argmin(&input[..sparse_items])
        );
        sorter
            .sort_gpu_to_gpu(&gpu_input, &sort_output, item_count as u32)
            .expect("sort baseline validation failed");
        support::wait_for_gpu(&context.device);
        assert_eq!(read_pair(&context, &sort_output), expected);

        group.bench_with_input(
            BenchmarkId::new("raw_resident", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    let mut encoder = context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    raw.record(&mut encoder);
                    context.queue.submit(Some(encoder.finish()));
                    support::wait_for_gpu(&context.device);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("lampshade_fixed", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    selector
                        .argmin_gpu_to_gpu(&gpu_input, &output, item_count as u32)
                        .expect("Lampshade fixed argmin failed");
                    support::wait_for_gpu(&context.device);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("lampshade_counted_dense", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    selector
                        .argmin_counted_gpu_to_gpu(&gpu_input, &output, &count, item_count as u32)
                        .expect("Lampshade counted argmin failed");
                    support::wait_for_gpu(&context.device);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("lampshade_counted_10pct", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    selector
                        .argmin_counted_gpu_to_gpu(
                            &gpu_input,
                            &output,
                            &sparse_count,
                            item_count as u32,
                        )
                        .expect("Lampshade sparse counted argmin failed");
                    support::wait_for_gpu(&context.device);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("full_width_sort_baseline", item_count),
            &item_count,
            |b, &_| {
                b.iter(|| {
                    sorter
                        .sort_gpu_to_gpu(&gpu_input, &sort_output, item_count as u32)
                        .expect("sort baseline failed");
                    support::wait_for_gpu(&context.device);
                });
            },
        );
    }
    group.finish();
}

fn input_for(size: usize) -> Vec<KeyValue> {
    (0..size)
        .map(|index| KeyValue::new((index as u32).wrapping_mul(2_654_435_761), index as u32))
        .collect()
}

fn cpu_argmin(input: &[KeyValue]) -> KeyValue {
    input
        .iter()
        .copied()
        .min_by_key(|item| (item.key, item.value))
        .unwrap()
}

fn read_pair(context: &Context, buffer: &wgpu::Buffer) -> KeyValue {
    let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Argmin Benchmark Readback"),
        size: PAIR_SIZE,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, PAIR_SIZE);
    context.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender
            .send(result)
            .expect("argmin readback receiver dropped");
    });
    support::wait_for_gpu(&context.device);
    receiver
        .recv()
        .expect("argmin readback callback did not run")
        .expect("argmin readback mapping failed");
    let result = {
        let mapped = slice.get_mapped_range();
        bytemuck::from_bytes::<KeyValue>(&mapped).to_owned()
    };
    staging.unmap();
    result
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn dispatch_dimensions(groups: u32, max: u32) -> (u32, u32) {
    (groups.min(max), groups.div_ceil(max))
}

criterion_group!(benches, benchmark_argmin);
criterion_main!(benches);
