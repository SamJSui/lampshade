mod support;

use lampshade::{
    Error, KeyValue, KeyValueField, U32Predicate, U32Reduction,
    pipeline::{GpuCount, GpuSlice, GpuSliceMut, Primitives, SortOptions, WorkspaceRequirements},
};
use wgpu::util::DeviceExt;

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
    let input_buffer = initialized_at(&context.device, "pipeline input", &input);
    let mask_buffer = initialized_at(&context.device, "pipeline mask", &mask);
    let compacted_buffer = empty_buffer(&context.device, "pipeline compacted");
    let sorted_buffer = empty_buffer(&context.device, "pipeline sorted");
    let count_buffer = empty_buffer(&context.device, "pipeline count");
    let sum_buffer = empty_buffer(&context.device, "pipeline sum");

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
            label: Some("pipeline compact sort reduce"),
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
    let input_buffer = initialized_at(&context.device, "pipeline fixed input", &values);
    let sorted_buffer = empty_buffer(&context.device, "pipeline fixed sorted");
    let sum_buffer = empty_buffer(&context.device, "pipeline fixed sum");
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
async fn fixed_key_value_sort_uses_the_adapter_path_at_nonzero_offsets() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let records = [
        KeyValue::new(7, 70),
        KeyValue::new(2, 20),
        KeyValue::new(2, 21),
        KeyValue::new(1, 10),
    ];
    let capacity = records.len() as u32;
    let input_buffer =
        initialized_key_values_at(&context.device, "pipeline fixed KV input", &records);
    let output_buffer = empty_key_value_buffer(&context.device, "pipeline fixed KV output");
    let input = GpuSlice::from_range(&input_buffer, VIEW_START..VIEW_START + capacity).unwrap();
    let output =
        GpuSliceMut::from_range(&output_buffer, VIEW_START..VIEW_START + capacity).unwrap();
    let mut primitives = Primitives::from_context(&context);
    primitives
        .reserve_workspace(WorkspaceRequirements::new(capacity).fixed_key_value_sort())
        .unwrap();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    primitives
        .record(&mut encoder)
        .sort_by_key(input, output, SortOptions::default())
        .unwrap();
    context.queue.submit([encoder.finish()]);

    let actual = support::read_pod::<KeyValue>(&context, &output_buffer, BUFFER_ITEMS).await;
    assert_eq!(
        &actual[VIEW_START as usize..VIEW_START as usize + records.len()],
        &[
            KeyValue::new(1, 10),
            KeyValue::new(2, 20),
            KeyValue::new(2, 21),
            KeyValue::new(7, 70),
        ]
    );
}

#[tokio::test]
async fn typed_views_reject_shared_handles_and_misaligned_bindings() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let arena = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pipeline validation arena"),
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

    let input_buffer = empty_buffer(&context.device, "pipeline misaligned input");
    let output_buffer = empty_buffer(&context.device, "pipeline aligned output");
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

#[tokio::test]
async fn typed_key_value_pipeline_handles_selection_edges_and_stability() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let records = [
        KeyValue::new(900, 101),
        KeyValue::new(250, 7),
        KeyValue::new(1_000, 5),
        KeyValue::new(400, 42),
        KeyValue::new(250, 9),
        KeyValue::new(50, 1),
        KeyValue::new(700, 88),
        KeyValue::new(400, 43),
    ];
    let cases = [
        (U32Predicate::GreaterThan(u32::MAX), Vec::<KeyValue>::new()),
        (U32Predicate::GreaterThanOrEqual(0), {
            let mut expected = records.to_vec();
            expected.sort_by_key(|item| item.key);
            expected
        }),
        (
            U32Predicate::BetweenInclusive { min: 100, max: 700 },
            vec![
                KeyValue::new(250, 7),
                KeyValue::new(250, 9),
                KeyValue::new(400, 42),
                KeyValue::new(400, 43),
                KeyValue::new(700, 88),
            ],
        ),
    ];

    for (predicate, expected) in cases {
        let actual = run_key_value_pipeline(&context, &records, predicate).await;
        assert_eq!(actual, expected);
    }
}

#[tokio::test]
async fn counted_key_value_sort_clamps_count_to_capacity() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let records = [
        KeyValue::new(5, 50),
        KeyValue::new(1, 10),
        KeyValue::new(3, 30),
        KeyValue::new(1, 11),
    ];
    let capacity = records.len() as u32;
    let input_buffer =
        initialized_key_values_at(&context.device, "pipeline counted KV input", &records);
    let output_buffer = empty_key_value_buffer(&context.device, "pipeline counted KV output");
    let mut counts = vec![0_u32; BUFFER_ITEMS];
    counts[VIEW_START as usize] = capacity + 100;
    let count_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pipeline oversized count"),
            contents: bytemuck::cast_slice(&counts),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
    let input = GpuSlice::counted(
        &input_buffer,
        VIEW_START..VIEW_START + capacity,
        GpuCount::at(
            &count_buffer,
            u64::from(VIEW_START) * size_of::<u32>() as u64,
        )
        .unwrap(),
    )
    .unwrap();
    let output =
        GpuSliceMut::from_range(&output_buffer, VIEW_START..VIEW_START + capacity).unwrap();
    let count = match input.extent() {
        lampshade::pipeline::Extent::Gpu(count) => count,
        lampshade::pipeline::Extent::Fixed(_) => unreachable!(),
    };
    let mut primitives = Primitives::from_context(&context);
    primitives
        .reserve_workspace(WorkspaceRequirements::new(capacity).counted_key_value_sort())
        .unwrap();
    primitives.reserve_count(count, capacity).unwrap();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    primitives
        .record(&mut encoder)
        .sort_by_key(input, output, SortOptions::default())
        .unwrap();
    context.queue.submit([encoder.finish()]);

    let count_result = support::read_u32(&context, &count_buffer, BUFFER_ITEMS).await;
    let output_result = support::read_pod::<KeyValue>(&context, &output_buffer, BUFFER_ITEMS).await;
    // The plan clamps generated dispatch metadata without mutating the
    // application-owned source count.
    assert_eq!(count_result[VIEW_START as usize], capacity + 100);
    assert_eq!(
        &output_result[VIEW_START as usize..VIEW_START as usize + records.len()],
        &[
            KeyValue::new(1, 10),
            KeyValue::new(1, 11),
            KeyValue::new(3, 30),
            KeyValue::new(5, 50),
        ]
    );
}

#[tokio::test]
async fn typed_key_value_views_validate_capacity_usage_and_aliases() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let records = [KeyValue::new(2, 20), KeyValue::new(1, 10)];
    let input_buffer =
        initialized_key_values_at(&context.device, "pipeline KV validation input", &records);
    let mask_buffer = empty_buffer(&context.device, "pipeline KV validation mask");
    let output_buffer = empty_key_value_buffer(&context.device, "pipeline KV validation output");
    let count_buffer = empty_buffer(&context.device, "pipeline KV validation count");
    let missing_copy_src = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pipeline mask missing COPY_SRC"),
        size: BUFFER_ITEMS as u64 * size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let input = GpuSlice::from_range(&input_buffer, VIEW_START..VIEW_START + 2).unwrap();
    let mask = GpuSlice::from_range(&mask_buffer, VIEW_START..VIEW_START + 2).unwrap();
    let output = GpuSliceMut::from_range(&output_buffer, VIEW_START..VIEW_START + 2).unwrap();
    let count = GpuCount::at(
        &count_buffer,
        u64::from(VIEW_START) * size_of::<u32>() as u64,
    )
    .unwrap();
    let mut primitives = Primitives::from_context(&context);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let short_output = GpuSliceMut::from_range(&output_buffer, VIEW_START..VIEW_START + 1).unwrap();
    assert!(matches!(
        primitives
            .record(&mut encoder)
            .compact_key_values(input, mask, short_output, count),
        Err(Error::BufferTooSmall { .. })
    ));

    let invalid_mask = GpuSlice::from_range(&missing_copy_src, VIEW_START..VIEW_START + 2).unwrap();
    assert!(matches!(
        primitives
            .record(&mut encoder)
            .compact_key_values(input, invalid_mask, output, count),
        Err(Error::MissingBufferUsage { .. })
    ));

    let aliased_mask_output = GpuSliceMut::from_range(&input_buffer, 0..2).unwrap();
    assert!(matches!(
        primitives.record(&mut encoder).mask_key_values(
            input,
            aliased_mask_output,
            KeyValueField::Key,
            U32Predicate::GreaterThan(0),
        ),
        Err(Error::BufferAlias { .. })
    ));

    let aliased_sort_output =
        GpuSliceMut::from_range(&input_buffer, VIEW_START + 16..VIEW_START + 18).unwrap();
    assert!(matches!(
        primitives.record(&mut encoder).sort_by_key(
            input,
            aliased_sort_output,
            SortOptions::default()
        ),
        Err(Error::BufferAlias { .. })
    ));

    let aliased_compact_output =
        GpuSliceMut::from_range(&input_buffer, VIEW_START + 16..VIEW_START + 18).unwrap();
    assert!(matches!(
        primitives.record(&mut encoder).compact_key_values(
            input,
            mask,
            aliased_compact_output,
            count
        ),
        Err(Error::BufferAlias { .. })
    ));

    let mask_as_output = GpuSliceMut::from_range(&mask_buffer, 32..34).unwrap();
    assert!(matches!(
        primitives
            .record(&mut encoder)
            .compact_key_values(input, mask, mask_as_output, count),
        Err(Error::BufferAlias { .. })
    ));

    let input_as_count = GpuCount::at(&input_buffer, 0).unwrap();
    assert!(matches!(
        primitives
            .record(&mut encoder)
            .compact_key_values(input, mask, output, input_as_count),
        Err(Error::BufferAlias { .. })
    ));

    let mask_as_count = GpuCount::at(&mask_buffer, 0).unwrap();
    assert!(matches!(
        primitives
            .record(&mut encoder)
            .compact_key_values(input, mask, output, mask_as_count),
        Err(Error::BufferAlias { .. })
    ));

    let output_as_count = GpuCount::at(&output_buffer, 0).unwrap();
    assert!(matches!(
        primitives
            .record(&mut encoder)
            .compact_key_values(input, mask, output, output_as_count),
        Err(Error::BufferAlias { .. })
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

async fn run_key_value_pipeline(
    context: &lampshade::Context,
    records: &[KeyValue],
    predicate: U32Predicate,
) -> Vec<KeyValue> {
    let capacity = records.len() as u32;
    let input_buffer =
        initialized_key_values_at(&context.device, "pipeline particle input", records);
    let mask_buffer = empty_buffer(&context.device, "pipeline particle mask");
    let compacted_buffer = empty_key_value_buffer(&context.device, "pipeline particle compacted");
    let sorted_buffer = empty_key_value_buffer(&context.device, "pipeline particle sorted");
    let count_buffer = empty_buffer(&context.device, "pipeline particle count");
    let readback_size = 8 + u64::from(capacity) * size_of::<KeyValue>() as u64;
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pipeline particle readback"),
        size: readback_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let input = GpuSlice::from_range(&input_buffer, VIEW_START..VIEW_START + capacity).unwrap();
    let mask = GpuSliceMut::from_range(&mask_buffer, VIEW_START..VIEW_START + capacity).unwrap();
    let compacted =
        GpuSliceMut::from_range(&compacted_buffer, VIEW_START..VIEW_START + capacity).unwrap();
    let sorted =
        GpuSliceMut::from_range(&sorted_buffer, VIEW_START..VIEW_START + capacity).unwrap();
    let count = GpuCount::at(
        &count_buffer,
        u64::from(VIEW_START) * size_of::<u32>() as u64,
    )
    .unwrap();
    let mut primitives = Primitives::from_context(context);
    primitives
        .reserve_workspace(
            WorkspaceRequirements::new(capacity)
                .predicate()
                .compact_key_values()
                .counted_key_value_sort(),
        )
        .unwrap();
    primitives.reserve_count(count, capacity).unwrap();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut recorder = primitives.record(&mut encoder);
        let mask = recorder
            .mask_key_values(input, mask, KeyValueField::Key, predicate)
            .unwrap();
        let selected = recorder
            .compact_key_values(input, mask, compacted, count)
            .unwrap();
        recorder
            .sort_by_key(selected, sorted, SortOptions::default().key_bits(10))
            .unwrap();
    }
    encoder.copy_buffer_to_buffer(
        &count_buffer,
        u64::from(VIEW_START) * size_of::<u32>() as u64,
        &readback,
        0,
        size_of::<u32>() as u64,
    );
    encoder.copy_buffer_to_buffer(
        &sorted_buffer,
        u64::from(VIEW_START) * size_of::<KeyValue>() as u64,
        &readback,
        8,
        u64::from(capacity) * size_of::<KeyValue>() as u64,
    );
    let submission = context.queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    let (sender, receiver) = futures::channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    receiver.await.unwrap().unwrap();
    let result = {
        let mapped = slice
            .get_mapped_range()
            .expect("pipeline readback mapped range unavailable");
        let count = bytemuck::cast_slice::<u8, u32>(&mapped[..4])[0] as usize;
        bytemuck::cast_slice::<u8, KeyValue>(&mapped[8..])[..count].to_vec()
    };
    readback.unmap();
    result
}

fn initialized_key_values_at(
    device: &wgpu::Device,
    label: &'static str,
    values: &[KeyValue],
) -> wgpu::Buffer {
    let mut data = vec![KeyValue::default(); BUFFER_ITEMS];
    data[VIEW_START as usize..VIEW_START as usize + values.len()].copy_from_slice(values);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn empty_key_value_buffer(device: &wgpu::Device, label: &'static str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: BUFFER_ITEMS as u64 * size_of::<KeyValue>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
