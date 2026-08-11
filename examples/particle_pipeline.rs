use futures::channel::oneshot;
use lampshade::{
    Context, KeyValue, KeyValueField, U32Predicate,
    pipeline::{GpuCount, GpuSlice, GpuSliceMut, Primitives, SortOptions, WorkspaceRequirements},
};
use wgpu::util::DeviceExt;

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let context = Context::init()
        .await
        .expect("failed to create wgpu context");

    // A renderer can encode sortable depth in `key` and an entity or particle
    // identifier in `value`. Duplicate depths make sort stability observable.
    let particles = [
        KeyValue::new(900, 101),
        KeyValue::new(250, 7),
        KeyValue::new(1_000, 5),
        KeyValue::new(400, 42),
        KeyValue::new(250, 9),
        KeyValue::new(50, 1),
        KeyValue::new(700, 88),
        KeyValue::new(400, 43),
    ];
    let capacity = particles.len() as u32;
    let input_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Records"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let mask_buffer = storage_buffer(
        &context.device,
        "Visible Particle Mask",
        u64::from(capacity) * size_of::<u32>() as u64,
        wgpu::BufferUsages::COPY_SRC,
    );
    let compacted_buffer = storage_buffer(
        &context.device,
        "Visible Particles",
        input_buffer.size(),
        wgpu::BufferUsages::empty(),
    );
    let sorted_buffer = storage_buffer(
        &context.device,
        "Depth-Sorted Particles",
        input_buffer.size(),
        wgpu::BufferUsages::COPY_SRC,
    );
    let count_buffer = storage_buffer(
        &context.device,
        "Visible Particle Count",
        size_of::<u32>() as u64,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Particle Pipeline Final Readback"),
        size: 8 + input_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let input = GpuSlice::from_range(&input_buffer, 0..capacity).expect("invalid input view");
    let mask = GpuSliceMut::from_range(&mask_buffer, 0..capacity).expect("invalid mask view");
    let compacted = GpuSliceMut::from_range(&compacted_buffer, 0..capacity)
        .expect("invalid compaction output view");
    let sorted =
        GpuSliceMut::from_range(&sorted_buffer, 0..capacity).expect("invalid sort output view");
    let visible_count = GpuCount::new(&count_buffer).expect("invalid count view");

    let mut primitives = Primitives::from_context(&context);
    primitives
        .reserve_workspace(
            WorkspaceRequirements::new(capacity)
                .predicate()
                .compact_key_values()
                .counted_key_value_sort(),
        )
        .expect("failed to reserve particle workspace");
    primitives
        .reserve_count(visible_count, capacity)
        .expect("failed to reserve count metadata");

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Cull Compact Sort Particles"),
        });
    {
        let mut recorder = primitives.record(&mut encoder);
        let mask = recorder
            .mask_key_values(
                input,
                mask,
                KeyValueField::Key,
                U32Predicate::BetweenInclusive { min: 100, max: 700 },
            )
            .expect("failed to record visibility mask");
        let visible = recorder
            .compact_key_values(input, mask, compacted, visible_count)
            .expect("failed to record particle compaction");
        recorder
            .sort_by_key(visible, sorted, SortOptions::default().key_bits(10))
            .expect("failed to record particle sort");
    }

    // No count readback separates the kernels. Count and records are copied to
    // one staging allocation after all GPU work and mapped once.
    encoder.copy_buffer_to_buffer(&count_buffer, 0, &readback, 0, size_of::<u32>() as u64);
    encoder.copy_buffer_to_buffer(&sorted_buffer, 0, &readback, 8, input_buffer.size());
    let submission = context.queue.submit([encoder.finish()]);
    let mapped = readback.slice(..);
    let (sender, receiver) = oneshot::channel();
    mapped.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("failed to wait for particle pipeline");
    receiver
        .await
        .expect("readback channel closed")
        .expect("failed to map particle readback");

    let bytes = mapped
        .get_mapped_range()
        .expect("particle readback is mapped");
    let selected = bytemuck::cast_slice::<u8, u32>(&bytes[..4])[0] as usize;
    let sorted: &[KeyValue] = bytemuck::cast_slice(&bytes[8..]);
    assert_eq!(selected, 5);
    assert_eq!(
        &sorted[..selected],
        &[
            KeyValue::new(250, 7),
            KeyValue::new(250, 9),
            KeyValue::new(400, 42),
            KeyValue::new(400, 43),
            KeyValue::new(700, 88),
        ]
    );
    println!("visible={selected} depth_sorted={:?}", &sorted[..selected]);
    drop(bytes);
    readback.unmap();
}

fn storage_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    extra_usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | extra_usage,
        mapped_at_creation: false,
    })
}
