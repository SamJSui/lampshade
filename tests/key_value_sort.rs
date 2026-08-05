mod support;

use wgpu::util::DeviceExt;
use wgpu_primitives::{Error, KeyValue, KeyValueSorter};

const SORT_SIZES: [usize; 18] = [
    0, 1, 2, 31, 32, 33, 127, 128, 129, 511, 512, 513, 2_047, 2_048, 2_049, 4_097, 65_537, 17,
];

fn cpu_stable_sort(input: &[KeyValue]) -> Vec<KeyValue> {
    let mut expected = input.to_vec();
    expected.sort_by_key(|item| item.key);
    expected
}

fn duplicate_keys(len: usize, seed: u64) -> Vec<KeyValue> {
    support::random_u32(len, seed)
        .into_iter()
        .enumerate()
        .map(|(index, key)| KeyValue::new(key & 0x0f, index as u32))
        .collect()
}

#[tokio::test]
async fn key_value_sort_matches_stable_cpu_sort_across_boundaries() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = KeyValueSorter::from_context(&context);

    for (case, size) in SORT_SIZES.into_iter().enumerate() {
        let input = duplicate_keys(size, case as u64);
        let actual = sorter
            .sort(&input)
            .await
            .expect("GPU key-value sort failed");
        assert_eq!(
            actual,
            cpu_stable_sort(&input),
            "key-value sort mismatch for size {size}"
        );
    }
}

#[tokio::test]
async fn key_value_sort_preserves_value_order_for_equal_keys() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = KeyValueSorter::from_context(&context);
    let input: Vec<_> = (0..4_097).map(|value| KeyValue::new(42, value)).collect();

    let actual = sorter
        .sort(&input)
        .await
        .expect("GPU key-value sort failed");
    assert_eq!(actual, input);

    let edge_keys = [
        KeyValue::new(u32::MAX, 0),
        KeyValue::new(0, 1),
        KeyValue::new(u32::MAX, 2),
        KeyValue::new(0, 3),
        KeyValue::new(1, 4),
    ];
    assert_eq!(
        sorter
            .sort(&edge_keys)
            .await
            .expect("GPU key-value sort failed"),
        cpu_stable_sort(&edge_keys)
    );
}

#[tokio::test]
async fn key_value_sort_gpu_to_gpu_writes_the_caller_output_buffer() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = KeyValueSorter::from_context(&context);
    let input = duplicate_keys(4_097, 100);
    let input_buffer = create_sort_input(&context.device, &input);
    let output = create_sort_output(&context.device, input.len());

    sorter
        .sort_gpu_to_gpu(&input_buffer, &output, input.len() as u32)
        .expect("GPU key-value sort failed");
    let actual = support::read_pod::<KeyValue>(&context, &output, input.len()).await;
    assert_eq!(actual, cpu_stable_sort(&input));
}

#[tokio::test]
async fn record_key_value_sort_composes_multiple_invocations_in_one_encoder() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = KeyValueSorter::new(&context.device, &context.queue);
    let first = [KeyValue::new(7, 11)];
    let second = duplicate_keys(4_097, 200);
    let first_input = create_sort_input(&context.device, &first);
    let second_input = create_sort_input(&context.device, &second);
    let first_output = create_sort_output(&context.device, first.len());
    let second_output = create_sort_output(&context.device, second.len());
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    sorter
        .record_sort(
            &mut encoder,
            &first_input,
            &first_output,
            first.len() as u32,
        )
        .expect("first key-value sort recording failed");
    sorter
        .record_sort(
            &mut encoder,
            &second_input,
            &second_output,
            second.len() as u32,
        )
        .expect("second key-value sort recording failed");
    context.queue.submit(Some(encoder.finish()));

    assert_eq!(
        support::read_pod::<KeyValue>(&context, &first_output, first.len()).await,
        cpu_stable_sort(&first)
    );
    assert_eq!(
        support::read_pod::<KeyValue>(&context, &second_output, second.len()).await,
        cpu_stable_sort(&second)
    );
}

#[tokio::test]
async fn record_key_value_sort_rejects_short_pair_buffers() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = KeyValueSorter::from_context(&context);
    let input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Short Key-Value Sort Input"),
        size: 24,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Key-Value Sort Output"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let error = sorter
        .record_sort(&mut encoder, &input, &output, 4)
        .expect_err("short key-value input must be rejected");
    assert!(matches!(error, Error::BufferTooSmall { .. }));
}

fn create_sort_input(device: &wgpu::Device, input: &[KeyValue]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Key-Value Sort Input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn create_sort_output(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Key-Value Sort Output"),
        size: (len * size_of::<KeyValue>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
