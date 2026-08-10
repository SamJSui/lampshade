mod support;

use wgpu::util::DeviceExt;
use wgpu_primitives::{
    Error, U32Reduction,
    v2::{GpuCount, GpuSlice, GpuSliceMut, Primitives, SortOptions, WorkspaceRequirements},
};

const VIEW_START: u32 = 64;
const BUFFER_ITEMS: usize = 128;

#[tokio::test]
async fn typed_views_compose_compact_sort_reduce_in_one_submission() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = [15_u32, 2, 11, 4, 8, 1, 14, 6, 10, 3, 12, 0, 9, 7, 13, 5];
    let mask = [1_u32, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
    let capacity = input.len() as u32;
    let input_buffer = initialized_at(&context.device, "v2 input", &input);
    let mask_buffer = initialized_at(&context.device, "v2 mask", &mask);
    let compacted_buffer = empty_buffer(&context.device, "v2 compacted");
    let sorted_buffer = empty_buffer(&context.device, "v2 sorted");
    let count_buffer = empty_buffer(&context.device, "v2 count");
    let sum_buffer = empty_buffer(&context.device, "v2 sum");

    let input = GpuSlice::from_range(&input_buffer, VIEW_START..VIEW_START + capacity).unwrap();
    let mask = GpuSlice::from_range(&mask_buffer, VIEW_START..VIEW_START + capacity).unwrap();
    let compacted =
        GpuSliceMut::from_range(&compacted_buffer, VIEW_START..VIEW_START + capacity).unwrap();
    let sorted =
        GpuSliceMut::from_range(&sorted_buffer, VIEW_START..VIEW_START + capacity).unwrap();
    let sum = GpuSliceMut::from_range(&sum_buffer, VIEW_START..VIEW_START + 1).unwrap();
    let count = GpuCount::at(
        &count_buffer,
        u64::from(VIEW_START) * size_of::<u32>() as u64,
    )
    .unwrap();

    let mut primitives = Primitives::from_context(&context);
    primitives
        .reserve_workspace(
            WorkspaceRequirements::new(capacity)
                .compact()
                .counted_sort()
                .counted_reduce(),
        )
        .unwrap();
    primitives.reserve_count(count, capacity).unwrap();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("v2 compact sort reduce"),
        });
    {
        let mut recorder = primitives.record(&mut encoder);
        let compacted = recorder.compact(input, mask, compacted, count).unwrap();
        let sorted = recorder
            .sort(compacted, sorted, SortOptions::default())
            .unwrap();
        recorder.reduce(sorted, sum, U32Reduction::Sum).unwrap();
    }
    context.queue.submit(Some(encoder.finish()));

    let count_result = support::read_u32(&context, &count_buffer, BUFFER_ITEMS).await;
    let sum_result = support::read_u32(&context, &sum_buffer, BUFFER_ITEMS).await;
    let sorted_result = support::read_u32(&context, &sorted_buffer, BUFFER_ITEMS).await;
    assert_eq!(count_result[VIEW_START as usize], 8);
    assert_eq!(sum_result[VIEW_START as usize], 92);
    assert_eq!(
        &sorted_result[VIEW_START as usize..VIEW_START as usize + 8],
        &[8, 9, 10, 11, 12, 13, 14, 15]
    );
}

#[tokio::test]
async fn fixed_extents_use_nonzero_offsets() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let values = [9_u32, 3, 7, 1, 5];
    let capacity = values.len() as u32;
    let input_buffer = initialized_at(&context.device, "v2 fixed input", &values);
    let sorted_buffer = empty_buffer(&context.device, "v2 fixed sorted");
    let sum_buffer = empty_buffer(&context.device, "v2 fixed sum");
    let input = GpuSlice::from_range(&input_buffer, VIEW_START..VIEW_START + capacity).unwrap();
    let sorted =
        GpuSliceMut::from_range(&sorted_buffer, VIEW_START..VIEW_START + capacity).unwrap();
    let sum = GpuSliceMut::from_range(&sum_buffer, VIEW_START..VIEW_START + 1).unwrap();
    let mut primitives = Primitives::from_context(&context);
    primitives
        .reserve_workspace(
            WorkspaceRequirements::new(capacity)
                .fixed_sort()
                .fixed_reduce(),
        )
        .unwrap();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut recorder = primitives.record(&mut encoder);
        let sorted = recorder
            .sort(input, sorted, SortOptions::default())
            .unwrap();
        recorder.reduce(sorted, sum, U32Reduction::Sum).unwrap();
    }
    context.queue.submit(Some(encoder.finish()));

    let sorted_result = support::read_u32(&context, &sorted_buffer, BUFFER_ITEMS).await;
    let sum_result = support::read_u32(&context, &sum_buffer, BUFFER_ITEMS).await;
    assert_eq!(
        &sorted_result[VIEW_START as usize..VIEW_START as usize + values.len()],
        &[1, 3, 5, 7, 9]
    );
    assert_eq!(sum_result[VIEW_START as usize], 25);
}

#[tokio::test]
async fn typed_views_reject_shared_handles_and_misaligned_bindings() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let arena = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("v2 validation arena"),
        size: 512,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut primitives = Primitives::from_context(&context);

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let input = GpuSlice::from_range(&arena, 0..16).unwrap();
    let disjoint_output = GpuSliceMut::from_range(&arena, 64..80).unwrap();
    let error = primitives
        .record(&mut encoder)
        .sort(input, disjoint_output, SortOptions::default())
        .err()
        .expect("shared buffer handles must be rejected");
    assert!(matches!(error, Error::BufferAlias { .. }));

    let input_buffer = empty_buffer(&context.device, "v2 misaligned input");
    let output_buffer = empty_buffer(&context.device, "v2 aligned output");
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let misaligned_input = GpuSlice::from_range(&input_buffer, 1..9).unwrap();
    let aligned_output = GpuSliceMut::from_range(&output_buffer, 64..72).unwrap();
    let error = primitives
        .record(&mut encoder)
        .sort(misaligned_input, aligned_output, SortOptions::default())
        .err()
        .expect("misaligned views must be rejected");
    assert!(matches!(
        error,
        Error::MisalignedBufferOffset {
            name: "sort input",
            offset: 4,
            ..
        }
    ));
}

fn initialized_at(device: &wgpu::Device, label: &'static str, values: &[u32]) -> wgpu::Buffer {
    let mut data = vec![0_u32; BUFFER_ITEMS];
    data[VIEW_START as usize..VIEW_START as usize + values.len()].copy_from_slice(values);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn empty_buffer(device: &wgpu::Device, label: &'static str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: BUFFER_ITEMS as u64 * size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
