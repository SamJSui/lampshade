use futures::channel::oneshot;
use lampshade::{
    Context, MaskGenerator, Reducer, U32Predicate, U32Reduction,
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
    let mut primitives = Primitives::from_context(&context);

    // The predicate determines the selected length at GPU execution time. The
    // CPU supplies only the allocation capacity; it never reads the count
    // between compaction, sorting, and reduction.
    let input = [15_u32, 2, 11, 4, 8, 1, 14, 6, 10, 3, 12, 0, 9, 7, 13, 5];
    let item_count = input.len() as u32;
    let input_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Pipeline Input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let mask_buffer = storage_buffer(
        &context.device,
        "Pipeline Mask",
        MaskGenerator::mask_buffer_size(item_count).expect("mask size overflow"),
        wgpu::BufferUsages::COPY_SRC,
    );
    let compacted_buffer = storage_buffer(
        &context.device,
        "Pipeline Compacted Values",
        input_buffer.size(),
        wgpu::BufferUsages::empty(),
    );
    let count_buffer = storage_buffer(
        &context.device,
        "Pipeline Selected Count",
        size_of::<u32>() as u64,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    let sorted_buffer = storage_buffer(
        &context.device,
        "Pipeline Sorted Values",
        u64::from(item_count) * size_of::<u32>() as u64,
        wgpu::BufferUsages::COPY_SRC,
    );
    let sum_buffer = storage_buffer(
        &context.device,
        "Pipeline Sum",
        Reducer::output_buffer_size(),
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Pipeline Final Readback"),
        size: u64::from(2 + item_count) * size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let input = GpuSlice::from_range(&input_buffer, 0..item_count).expect("invalid input view");
    let mask = GpuSliceMut::from_range(&mask_buffer, 0..item_count).expect("invalid mask view");
    let compacted = GpuSliceMut::from_range(&compacted_buffer, 0..item_count)
        .expect("invalid compaction output view");
    let sorted =
        GpuSliceMut::from_range(&sorted_buffer, 0..item_count).expect("invalid sort output view");
    let sum = GpuSliceMut::from_range(&sum_buffer, 0..1).expect("invalid sum output view");
    let selected_count = GpuCount::new(&count_buffer).expect("invalid count view");
    primitives
        .reserve_workspace(
            WorkspaceRequirements::new(item_count)
                .predicate()
                .compact()
                .counted_sort()
                .counted_reduce(),
        )
        .expect("failed to reserve primitive workspaces");
    primitives
        .reserve_count(selected_count, item_count)
        .expect("failed to reserve count metadata");

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Predicate Compact Sort Reduce Pipeline"),
        });
    {
        let mut recorder = primitives.record(&mut encoder);
        let mask = recorder
            .mask(input, mask, U32Predicate::GreaterThanOrEqual(8))
            .expect("failed to record predicate mask");
        let compacted = recorder
            .compact(input, mask, compacted, selected_count)
            .expect("failed to record compaction");
        let sorted = recorder
            .sort(compacted, sorted, SortOptions::default())
            .expect("failed to record sort");
        recorder
            .reduce(sorted, sum, U32Reduction::Sum)
            .expect("failed to record reduction");
    }

    // The count, reduced sum, and sorted values enter one staging allocation,
    // so the program has one submission and one final map/readback rather than
    // a synchronization point between every primitive.
    encoder.copy_buffer_to_buffer(&count_buffer, 0, &readback, 0, size_of::<u32>() as u64);
    encoder.copy_buffer_to_buffer(
        &sum_buffer,
        0,
        &readback,
        size_of::<u32>() as u64,
        size_of::<u32>() as u64,
    );
    encoder.copy_buffer_to_buffer(
        &sorted_buffer,
        0,
        &readback,
        2 * size_of::<u32>() as u64,
        u64::from(item_count) * size_of::<u32>() as u64,
    );
    let submission = context.queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (sender, receiver) = oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("failed to wait for the composed pipeline");
    receiver
        .await
        .expect("readback channel closed")
        .expect("failed to map pipeline readback");
    let result = {
        let mapped = slice.get_mapped_range().expect("readback is mapped");
        bytemuck::cast_slice::<u8, u32>(&mapped).to_vec()
    };
    readback.unmap();

    let selected_count = result[0] as usize;
    assert_eq!(selected_count, 8);
    assert_eq!(result[1], 92);
    assert_eq!(
        &result[2..2 + selected_count],
        &[8, 9, 10, 11, 12, 13, 14, 15]
    );
    println!(
        "selected={} sorted={:?} sorted_sum={}",
        result[0],
        &result[2..2 + selected_count],
        result[1]
    );
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
