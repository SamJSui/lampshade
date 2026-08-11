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
async fn key_value_sort_handles_full_width_keys_across_many_tiles() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = KeyValueSorter::from_context(&context);
    let input: Vec<_> = support::random_u32(262_147, 0x00F0_1132)
        .into_iter()
        .enumerate()
        .map(|(index, key)| KeyValue::new(key, index as u32))
        .collect();

    let actual = sorter
        .sort(&input)
        .await
        .expect("full-width GPU key-value sort failed");
    assert_eq!(actual, cpu_stable_sort(&input));
}

#[tokio::test]
async fn key_value_sort_is_stable_for_large_duplicate_heavy_input() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = KeyValueSorter::from_context(&context);
    let input = duplicate_keys(1_000_003, 0xD001_1CA7);

    let actual = sorter
        .sort(&input)
        .await
        .expect("large duplicate-heavy GPU key-value sort failed");
    assert_eq!(actual, cpu_stable_sort(&input));
}

#[tokio::test]
async fn bounded_key_value_sort_is_stable_across_pass_parities() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut portable_sorter = KeyValueSorter::new(&context.device, &context.queue);
    let mut adapter_sorter = KeyValueSorter::from_context(&context);

    for key_bits in [0, 1, 8, 9, 16, 17, 24, 25, 32] {
        let mask = if key_bits == 32 {
            u32::MAX
        } else if key_bits == 0 {
            0
        } else {
            (1_u32 << key_bits) - 1
        };
        let input: Vec<_> = support::random_u32(4_097, u64::from(key_bits) + 900)
            .into_iter()
            .enumerate()
            .map(|(index, key)| KeyValue::new(key & mask, index as u32))
            .collect();
        let expected = cpu_stable_sort(&input);
        let portable_actual = portable_sorter
            .sort_with_key_bits(&input, key_bits)
            .await
            .expect("portable bounded key-value sort failed");
        assert_eq!(
            portable_actual, expected,
            "portable mismatch for {key_bits} key bits"
        );
        let adapter_actual = adapter_sorter
            .sort_with_key_bits(&input, key_bits)
            .await
            .expect("adapter bounded key-value sort failed");
        assert_eq!(
            adapter_actual, expected,
            "adapter mismatch for {key_bits} key bits"
        );
    }
}

#[tokio::test]
async fn bounded_key_value_sort_validates_host_keys() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = KeyValueSorter::new(&context.device, &context.queue);
    let error = sorter
        .sort_with_key_bits(&[KeyValue::new(256, 0)], 8)
        .await
        .expect_err("a nine-bit key must not satisfy an eight-bit bound");
    assert!(matches!(error, Error::KeyExceedsBitRange { .. }));
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
async fn counted_key_value_sort_uses_the_gpu_resident_prefix_and_stays_stable() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = [
        KeyValue::new(7, 70),
        KeyValue::new(2, 20),
        KeyValue::new(2, 21),
        KeyValue::new(1, 10),
        KeyValue::new(0, 0),
    ];
    let selected = 4_u32;
    let input_buffer = create_sort_input(&context.device, &input);
    let output = create_sort_output(&context.device, input.len());
    let count = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Key-Value Sort GPU Count"),
            contents: bytemuck::bytes_of(&selected),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let mut sorter = KeyValueSorter::from_context(&context);

    sorter
        .sort_counted_gpu_to_gpu_with_key_bits(
            &input_buffer,
            &output,
            &count,
            input.len() as u32,
            3,
        )
        .expect("counted key-value sort failed");
    let actual = support::read_pod::<KeyValue>(&context, &output, selected as usize).await;
    assert_eq!(
        actual,
        [
            KeyValue::new(1, 10),
            KeyValue::new(2, 20),
            KeyValue::new(2, 21),
            KeyValue::new(7, 70),
        ]
    );
}

#[tokio::test]
async fn bounded_key_value_gpu_sort_writes_output_after_one_and_three_byte_passes() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = KeyValueSorter::from_context(&context);

    for (key_bits, input) in [
        (
            8,
            vec![
                KeyValue::new(255, 0),
                KeyValue::new(0, 1),
                KeyValue::new(17, 2),
                KeyValue::new(3, 3),
                KeyValue::new(17, 4),
            ],
        ),
        (
            17,
            vec![
                KeyValue::new(0x1ffff, 0),
                KeyValue::new(0, 1),
                KeyValue::new(0x10001, 2),
                KeyValue::new(0xff, 3),
                KeyValue::new(0x10001, 4),
            ],
        ),
    ] {
        let input_buffer = create_sort_input(&context.device, &input);
        let output = create_sort_output(&context.device, input.len());

        sorter
            .sort_gpu_to_gpu_with_key_bits(&input_buffer, &output, input.len() as u32, key_bits)
            .expect("bounded key-value GPU sort failed");
        let actual = support::read_pod::<KeyValue>(&context, &output, input.len()).await;
        assert_eq!(
            actual,
            cpu_stable_sort(&input),
            "caller output mismatch for {key_bits} key bits"
        );
    }
}

#[tokio::test]
async fn bounded_key_value_sort_rebuilds_cached_bindings_when_pass_count_changes() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = KeyValueSorter::from_context(&context);
    let input = [
        KeyValue::new(255, 0),
        KeyValue::new(0, 1),
        KeyValue::new(17, 2),
        KeyValue::new(3, 3),
        KeyValue::new(17, 4),
    ];
    let input_buffer = create_sort_input(&context.device, &input);
    let output = create_sort_output(&context.device, input.len());
    let expected = cpu_stable_sort(&input);

    for key_bits in [17, 8, 16, 24, 32, 8] {
        sorter
            .sort_gpu_to_gpu_with_key_bits(&input_buffer, &output, input.len() as u32, key_bits)
            .expect("bounded key-value GPU sort failed");
        assert_eq!(
            support::read_pod::<KeyValue>(&context, &output, input.len()).await,
            expected,
            "cache rebuild mismatch for {key_bits} key bits"
        );
    }
}

#[tokio::test]
async fn zero_bit_gpu_sort_stably_overwrites_the_caller_output() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = KeyValueSorter::from_context(&context);
    let input = [
        KeyValue::new(0, 41),
        KeyValue::new(0, 7),
        KeyValue::new(0, 99),
        KeyValue::new(0, 3),
    ];
    let input_buffer = create_sort_input(&context.device, &input);
    let sentinels = [KeyValue::new(u32::MAX, u32::MAX); 4];
    let output = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Initialized Key-Value Sort Output"),
            contents: bytemuck::cast_slice(&sentinels),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    sorter
        .sort_gpu_to_gpu_with_key_bits(&input_buffer, &output, input.len() as u32, 0)
        .expect("zero-bit key-value GPU sort failed");
    assert_eq!(
        support::read_pod::<KeyValue>(&context, &output, input.len()).await,
        input
    );
}

#[tokio::test]
async fn key_value_sort_rejects_aliased_input_and_output() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = KeyValueSorter::from_context(&context);
    let input = create_sort_input(&context.device, &[KeyValue::new(1, 0)]);

    let error = sorter
        .sort_gpu_to_gpu_with_key_bits(&input, &input, 1, 8)
        .expect_err("in-place key-value scatter must be rejected");
    assert!(matches!(error, Error::BufferAlias { .. }));
}

#[tokio::test]
async fn record_key_value_sort_composes_multiple_invocations_in_one_encoder() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = KeyValueSorter::from_context(&context);
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
