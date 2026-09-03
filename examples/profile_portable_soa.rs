use std::mem::size_of;
use std::time::{Duration, Instant};

use futures::channel::oneshot;
use lampshade::{Context, KeyValue, KeyValueSoaSorter, KeyValueSorter};
use wgpu::util::DeviceExt;

const BLOCK_SIZE: u32 = 256;
const MAX_WORKGROUPS_X: u32 = 65_535;

fn main() {
    let mut args = std::env::args().skip(1);
    let items: usize = args
        .next()
        .unwrap_or_else(|| "1000000".to_owned())
        .parse()
        .expect("items must be an integer");
    let samples: usize = args
        .next()
        .unwrap_or_else(|| "11".to_owned())
        .parse()
        .expect("samples must be an integer");
    let warmups: usize = args
        .next()
        .unwrap_or_else(|| "4".to_owned())
        .parse()
        .expect("warmups must be an integer");
    let backend = args.next().unwrap_or_else(|| "selected".to_owned());
    assert!(
        matches!(backend.as_str(), "selected" | "portable"),
        "backend must be selected or portable"
    );
    let capacity = u32::try_from(items).expect("item count must fit in u32");

    let runtime = tokio::runtime::Runtime::new().expect("profiling runtime");
    let context = runtime
        .block_on(Context::init())
        .expect("profiling GPU context");
    let keys = seeded_input(items);
    let values: Vec<u32> = (0..capacity).collect();
    let records: Vec<KeyValue> = keys
        .iter()
        .copied()
        .zip(values.iter().copied())
        .map(|(key, value)| KeyValue::new(key, value))
        .collect();
    let mut expected = records.clone();
    expected.sort_by_key(|item| item.key);

    let count = storage_buffer(&context, "Profile SoA Count", bytemuck::bytes_of(&capacity));
    let config = uniform_buffer(
        &context,
        "Profile SoA Config",
        bytemuck::cast_slice(&[capacity, 0_u32, 0, 0]),
    );
    let clamped_count = empty_buffer(
        &context,
        "Profile SoA Clamped Count",
        size_of::<u32>() as u64,
        wgpu::BufferUsages::STORAGE,
    );

    let (pack_layout, pack_pipeline) = bridge_pipeline(
        &context.device,
        "Profile SoA Pack",
        include_str!("../src/sort/soa_pack.wgsl"),
        true,
    );
    let (unpack_layout, unpack_pipeline) = bridge_pipeline(
        &context.device,
        "Profile SoA Unpack",
        include_str!("../src/sort/soa_unpack.wgsl"),
        false,
    );

    let bridge_keys = storage_buffer(&context, "Profile Bridge Keys", bytemuck::cast_slice(&keys));
    let bridge_values = storage_buffer(
        &context,
        "Profile Bridge Values",
        bytemuck::cast_slice(&values),
    );
    let bridge_packed = empty_buffer(
        &context,
        "Profile Bridge Packed",
        size_of_val(records.as_slice()) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let pack_bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Profile SoA Pack Bind Group"),
            layout: &pack_layout,
            entries: &[
                entry(0, bridge_keys.as_entire_binding()),
                entry(1, bridge_values.as_entire_binding()),
                entry(2, bridge_packed.as_entire_binding()),
                entry(3, count.as_entire_binding()),
                entry(4, config.as_entire_binding()),
                entry(5, clamped_count.as_entire_binding()),
            ],
        });
    let unpack_bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Profile SoA Unpack Bind Group"),
            layout: &unpack_layout,
            entries: &[
                entry(0, bridge_packed.as_entire_binding()),
                entry(1, bridge_keys.as_entire_binding()),
                entry(2, bridge_values.as_entire_binding()),
                entry(3, count.as_entire_binding()),
                entry(4, config.as_entire_binding()),
            ],
        });

    submit_bridge(
        &context,
        &pack_pipeline,
        &pack_bind_group,
        capacity,
        "Validate SoA Pack",
    );
    submit_bridge(
        &context,
        &unpack_pipeline,
        &unpack_bind_group,
        capacity,
        "Validate SoA Unpack",
    );
    assert_eq!(read_pod::<u32>(&context, &bridge_keys, items), keys);
    assert_eq!(read_pod::<u32>(&context, &bridge_values, items), values);

    let packed_input = storage_buffer(
        &context,
        "Profile Packed Input",
        bytemuck::cast_slice(&records),
    );
    let packed_output = empty_buffer(
        &context,
        "Profile Packed Output",
        size_of_val(records.as_slice()) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let mut inner_sorter = if backend == "portable" {
        KeyValueSorter::new(&context.device, &context.queue)
    } else {
        KeyValueSorter::from_context(&context)
    };
    submit_inner_sort(
        &context,
        &mut inner_sorter,
        &packed_input,
        &packed_output,
        &count,
        capacity,
    );
    assert_eq!(
        read_pod::<KeyValue>(&context, &packed_output, items),
        expected
    );

    let full_keys = storage_buffer(&context, "Profile Full Keys", bytemuck::cast_slice(&keys));
    let full_values = storage_buffer(
        &context,
        "Profile Full Values",
        bytemuck::cast_slice(&values),
    );
    let mut full_sorter = if backend == "portable" {
        KeyValueSoaSorter::new_portable(&context.device, &context.queue)
    } else {
        KeyValueSoaSorter::from_context(&context)
    };
    full_sorter
        .prepare_sort(&full_keys, &full_values, capacity)
        .expect("prepare full SoA sort");
    submit_full_sort(&context, &full_sorter, &full_keys, &full_values, capacity);
    let actual: Vec<KeyValue> = read_pod::<u32>(&context, &full_keys, items)
        .into_iter()
        .zip(read_pod::<u32>(&context, &full_values, items))
        .map(|(key, value)| KeyValue::new(key, value))
        .collect();
    assert_eq!(actual, expected);

    let empty = measure(samples, warmups, || submit_empty(&context));
    let pack = measure(samples, warmups, || {
        submit_bridge(
            &context,
            &pack_pipeline,
            &pack_bind_group,
            capacity,
            "Profile SoA Pack",
        );
    });
    let inner = measure(samples, warmups, || {
        submit_inner_sort(
            &context,
            &mut inner_sorter,
            &packed_input,
            &packed_output,
            &count,
            capacity,
        );
    });
    let unpack = measure(samples, warmups, || {
        submit_bridge(
            &context,
            &unpack_pipeline,
            &unpack_bind_group,
            capacity,
            "Profile SoA Unpack",
        );
    });
    let full = measure(samples, warmups, || {
        submit_full_sort(&context, &full_sorter, &full_keys, &full_values, capacity);
    });

    let adapter = &context.adapter_info;
    println!(
        "adapter={:?} backend={:?} sorter_backend={backend} items={items} samples={samples} warmups={warmups} accelerated={} \
empty_ms={:.6} pack_ms={:.6} pack_net_ms={:.6} sort_ms={:.6} sort_net_ms={:.6} \
unpack_ms={:.6} unpack_net_ms={:.6} full_ms={:.6} full_net_ms={:.6}",
        adapter.name,
        adapter.backend,
        full_sorter.is_accelerated(),
        millis(empty),
        millis(pack),
        millis(pack.saturating_sub(empty)),
        millis(inner),
        millis(inner.saturating_sub(empty)),
        millis(unpack),
        millis(unpack.saturating_sub(empty)),
        millis(full),
        millis(full.saturating_sub(empty)),
    );
}

fn seeded_input(items: usize) -> Vec<u32> {
    let mut state = 0x6a09_e667_u32;
    (0..items)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state ^ (index as u32).wrapping_mul(0x9e37_79b9)
        })
        .collect()
}

fn measure(mut samples: usize, warmups: usize, mut operation: impl FnMut()) -> Duration {
    assert!(samples > 0, "at least one sample is required");
    for _ in 0..warmups {
        operation();
    }
    let mut durations = Vec::with_capacity(samples);
    while samples > 0 {
        let started = Instant::now();
        operation();
        durations.push(started.elapsed());
        samples -= 1;
    }
    durations.sort_unstable();
    durations[durations.len() / 2]
}

fn submit_empty(context: &Context) {
    let encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let submission = context.queue.submit(Some(encoder.finish()));
    wait(context, submission);
}

fn submit_bridge(
    context: &Context,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    capacity: u32,
    label: &'static str,
) {
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let groups = capacity.div_ceil(BLOCK_SIZE);
        let groups_x = groups.min(MAX_WORKGROUPS_X);
        pass.dispatch_workgroups(groups_x, groups.div_ceil(groups_x), 1);
    }
    let submission = context.queue.submit(Some(encoder.finish()));
    wait(context, submission);
}

fn submit_inner_sort(
    context: &Context,
    sorter: &mut KeyValueSorter,
    input: &wgpu::Buffer,
    output: &wgpu::Buffer,
    count: &wgpu::Buffer,
    capacity: u32,
) {
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    sorter
        .record_sort_counted(&mut encoder, input, output, count, capacity)
        .expect("record packed radix sort");
    let submission = context.queue.submit(Some(encoder.finish()));
    wait(context, submission);
}

fn submit_full_sort(
    context: &Context,
    sorter: &KeyValueSoaSorter,
    keys: &wgpu::Buffer,
    values: &wgpu::Buffer,
    capacity: u32,
) {
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    sorter
        .record_reserved_sort(&mut encoder, keys, values, capacity)
        .expect("record full SoA sort");
    let submission = context.queue.submit(Some(encoder.finish()));
    wait(context, submission);
}

fn wait(context: &Context, submission: wgpu::SubmissionIndex) {
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("GPU wait failed");
}

fn read_pod<T: bytemuck::Pod>(context: &Context, buffer: &wgpu::Buffer, len: usize) -> Vec<T> {
    let size = size_of::<T>() as u64 * len as u64;
    let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Profile SoA Validation Readback"),
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
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("validation GPU wait failed");
    pollster::block_on(receiver)
        .expect("validation channel closed")
        .expect("validation map failed");
    let mapped = slice
        .get_mapped_range()
        .expect("validation mapped range unavailable");
    let result = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    result
}

fn storage_buffer(context: &Context, label: &str, contents: &[u8]) -> wgpu::Buffer {
    context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
}

fn uniform_buffer(context: &Context, label: &str, contents: &[u8]) -> wgpu::Buffer {
    context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

fn empty_buffer(
    context: &Context,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn bridge_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    source: &'static str,
    pack: bool,
) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
    let mut entries = vec![
        layout_entry(0, true, false),
        layout_entry(1, pack, false),
        layout_entry(2, false, false),
        layout_entry(3, true, false),
        layout_entry(4, false, true),
    ];
    if pack {
        entries.push(layout_entry(5, false, false));
    }
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (bind_group_layout, pipeline)
}

fn layout_entry(binding: u32, read_only: bool, uniform: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: if uniform {
                wgpu::BufferBindingType::Uniform
            } else {
                wgpu::BufferBindingType::Storage { read_only }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn entry(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
