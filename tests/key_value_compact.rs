mod support;

use wgpu::util::DeviceExt;
use wgpu_primitives::{Error, KeyValue, KeyValueCompactor};

const COMPACTION_SIZES: [usize; 18] = [
    0, 1, 2, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513, 2_047, 2_048, 4_097,
];

fn mask_for(case: usize, size: usize) -> Vec<u32> {
    match case % 5 {
        0 => vec![0; size],
        1 => vec![1; size],
        2 => (0..size).map(|index| (index % 2) as u32).collect(),
        3 => (0..size).map(|index| u32::from(index % 11 == 0)).collect(),
        _ => support::random_u32(size, case as u64)
            .into_iter()
            .map(|value| value & 1)
            .collect(),
    }
}

fn cpu_compact(input: &[KeyValue], mask: &[u32]) -> Vec<KeyValue> {
    input
        .iter()
        .zip(mask)
        .filter_map(|(&item, &keep)| (keep == 1).then_some(item))
        .collect()
}

#[tokio::test]
async fn key_value_compact_matches_cpu_across_boundaries_and_patterns() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut compactor = KeyValueCompactor::from_context(&context);

    for (case, size) in COMPACTION_SIZES.into_iter().enumerate() {
        let input: Vec<_> = (0..size as u32)
            .map(|index| KeyValue::new(index % 7, index ^ 0xA5A5_0000))
            .collect();
        let mask = mask_for(case, size);
        let actual = compactor
            .compact(&input, &mask)
            .await
            .expect("GPU key-value compaction failed");

        assert_eq!(actual, cpu_compact(&input, &mask), "size {size}");
    }
}

#[tokio::test]
async fn key_value_compact_preserves_whole_records_and_order() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut compactor = KeyValueCompactor::from_context(&context);
    let input = [
        KeyValue::new(2, 10),
        KeyValue::new(1, 20),
        KeyValue::new(2, 30),
        KeyValue::new(1, 40),
        KeyValue::new(2, 50),
    ];
    let mask = [1_u32, 0, 1, 1, 1];

    assert_eq!(
        compactor
            .compact(&input, &mask)
            .await
            .expect("GPU key-value compaction failed"),
        [
            KeyValue::new(2, 10),
            KeyValue::new(2, 30),
            KeyValue::new(1, 40),
            KeyValue::new(2, 50),
        ]
    );
}

#[tokio::test]
async fn key_value_gpu_compaction_writes_output_and_resident_count() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = [
        KeyValue::new(9, 90),
        KeyValue::new(8, 80),
        KeyValue::new(7, 70),
        KeyValue::new(6, 60),
        KeyValue::new(5, 50),
        KeyValue::new(4, 40),
        KeyValue::new(999, 999),
    ];
    let mask = [1_u32, 0, 1, 0, 0, 1, 1];
    let input_buffer = storage_input(&context.device, "Key-Value Compaction Input", &input);
    let mask_buffer = storage_mask(&context.device, &mask);
    let output = compaction_output(&context.device, input.len());
    let count = count_buffer(&context.device, 99);
    let mut compactor = KeyValueCompactor::from_context(&context);

    compactor
        .compact_gpu_to_gpu(&input_buffer, &mask_buffer, &output, &count, 6)
        .expect("GPU-resident key-value compaction failed");

    assert_eq!(support::read_u32(&context, &count, 1).await, [3]);
    assert_eq!(
        support::read_pod::<KeyValue>(&context, &output, 3).await,
        [
            KeyValue::new(9, 90),
            KeyValue::new(7, 70),
            KeyValue::new(4, 40),
        ]
    );
}

#[tokio::test]
async fn key_value_recording_composes_multiple_invocations() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let first = [
        KeyValue::new(3, 0),
        KeyValue::new(1, 1),
        KeyValue::new(4, 2),
    ];
    let second = [
        KeyValue::new(2, 3),
        KeyValue::new(7, 4),
        KeyValue::new(1, 5),
    ];
    let first_input = storage_input(&context.device, "First Key-Value Input", &first);
    let second_input = storage_input(&context.device, "Second Key-Value Input", &second);
    let first_mask = storage_mask(&context.device, &[1, 0, 1]);
    let second_mask = storage_mask(&context.device, &[0, 1, 1]);
    let first_output = compaction_output(&context.device, first.len());
    let second_output = compaction_output(&context.device, second.len());
    let first_count = count_buffer(&context.device, 0);
    let second_count = count_buffer(&context.device, 0);
    let mut compactor = KeyValueCompactor::from_context(&context);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    compactor
        .record_compact(
            &mut encoder,
            &first_input,
            &first_mask,
            &first_output,
            &first_count,
            first.len() as u32,
        )
        .expect("first key-value compaction recording failed");
    compactor
        .record_compact(
            &mut encoder,
            &second_input,
            &second_mask,
            &second_output,
            &second_count,
            second.len() as u32,
        )
        .expect("second key-value compaction recording failed");
    context.queue.submit(Some(encoder.finish()));

    assert_eq!(support::read_u32(&context, &first_count, 1).await, [2]);
    assert_eq!(
        support::read_pod::<KeyValue>(&context, &first_output, 2).await,
        [KeyValue::new(3, 0), KeyValue::new(4, 2)]
    );
    assert_eq!(support::read_u32(&context, &second_count, 1).await, [2]);
    assert_eq!(
        support::read_pod::<KeyValue>(&context, &second_output, 2).await,
        [KeyValue::new(7, 4), KeyValue::new(1, 5)]
    );
}

#[tokio::test]
async fn key_value_compact_validates_slice_and_record_buffer_contracts() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut compactor = KeyValueCompactor::from_context(&context);

    assert!(matches!(
        compactor.compact(&[KeyValue::new(1, 2)], &[]).await,
        Err(Error::CompactionLengthMismatch { input: 1, mask: 0 })
    ));
    assert!(matches!(
        compactor.compact(&[KeyValue::new(1, 2)], &[2]).await,
        Err(Error::InvalidCompactionFlag { index: 0, value: 2 })
    ));

    let input = storage_input(
        &context.device,
        "Valid Key-Value Input",
        &[KeyValue::new(1, 10), KeyValue::new(2, 20)],
    );
    let mask = storage_mask(&context.device, &[1, 0]);
    let short_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Short Key-Value Output"),
        size: size_of::<KeyValue>() as u64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let count = count_buffer(&context.device, 0);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    assert!(matches!(
        compactor.record_compact(&mut encoder, &input, &mask, &short_output, &count, 2),
        Err(Error::BufferTooSmall {
            required: 16,
            actual: 8,
            ..
        })
    ));
    assert!(matches!(
        compactor.record_compact(&mut encoder, &input, &mask, &input, &count, 2),
        Err(Error::BufferAlias { .. })
    ));
}

#[tokio::test]
async fn key_value_profile_reports_scatter_and_valid_output() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    if !context
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
    {
        return;
    }
    let input = [
        KeyValue::new(3, 0),
        KeyValue::new(1, 1),
        KeyValue::new(4, 2),
        KeyValue::new(1, 3),
    ];
    let input_buffer = storage_input(&context.device, "Profile Key-Value Input", &input);
    let mask = storage_mask(&context.device, &[1, 0, 1, 1]);
    let output = compaction_output(&context.device, input.len());
    let count = count_buffer(&context.device, 0);
    let mut compactor = KeyValueCompactor::from_context(&context);

    let profile = compactor
        .profile_compact_gpu_to_gpu(&input_buffer, &mask, &output, &count, input.len() as u32)
        .await
        .expect("profiled key-value compaction failed");

    assert!(
        profile
            .spans
            .iter()
            .any(|span| span.label == "compact.scatter")
    );
    assert_eq!(support::read_u32(&context, &count, 1).await, [3]);
    assert_eq!(
        support::read_pod::<KeyValue>(&context, &output, 3).await,
        [input[0], input[2], input[3]]
    );
}

fn storage_input<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &'static str,
    data: &[T],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn storage_mask(device: &wgpu::Device, data: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Key-Value Compaction Mask"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn compaction_output(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Key-Value Compaction Output"),
        size: (len * size_of::<KeyValue>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn count_buffer(device: &wgpu::Device, initial: u32) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Key-Value Compaction Output Count"),
        contents: bytemuck::bytes_of(&initial),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    })
}
