use futures::channel::oneshot;
use wgpu::util::DeviceExt;
use wgpu_primitives::{
    Compactor, Context, MaskGenerator, Reducer, Sorter, U32Predicate, U32Reduction,
};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let context = Context::init()
        .await
        .expect("failed to create wgpu context");
    let generator = MaskGenerator::from_context(&context);
    let mut compactor = Compactor::from_context(&context);
    let mut sorter = Sorter::from_context(&context);
    let mut reducer = Reducer::from_context(&context);

    // This input contains every value from 0 through 15 exactly once. That
    // construction lets the CPU know that `>= 8` selects eight values without
    // reading the GPU-resident compaction count between stages.
    let input = [15_u32, 2, 11, 4, 8, 1, 14, 6, 10, 3, 12, 0, 9, 7, 13, 5];
    let item_count = input.len() as u32;
    let selected_count = 8_u32;
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
        u64::from(selected_count) * size_of::<u32>() as u64,
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
        size: u64::from(2 + selected_count) * size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Predicate Compact Sort Reduce Pipeline"),
        });
    generator
        .record_mask(
            &mut encoder,
            &input_buffer,
            &mask_buffer,
            item_count,
            U32Predicate::GreaterThanOrEqual(8),
        )
        .expect("failed to record predicate mask");
    compactor
        .record_compact(
            &mut encoder,
            &input_buffer,
            &mask_buffer,
            &compacted_buffer,
            &count_buffer,
            item_count,
        )
        .expect("failed to record compaction");
    sorter
        .record_sort(
            &mut encoder,
            &compacted_buffer,
            &sorted_buffer,
            selected_count,
        )
        .expect("failed to record sort");
    reducer
        .record_reduce(
            &mut encoder,
            &sorted_buffer,
            &sum_buffer,
            selected_count,
            U32Reduction::Sum,
        )
        .expect("failed to record reduction");

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
        u64::from(selected_count) * size_of::<u32>() as u64,
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

    assert_eq!(result, [selected_count, 92, 8, 9, 10, 11, 12, 13, 14, 15]);
    println!(
        "selected={} sorted={:?} sorted_sum={}",
        result[0],
        &result[2..],
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
