mod support;

use lampshade::{
    Compactor, Error, KeyValue, KeyValueCompactor, KeyValueField, MaskGenerator, U32Predicate,
};
use wgpu::util::DeviceExt;

const PREDICATE_SIZES: [usize; 8] = [0, 1, 31, 255, 256, 257, 2_048, 4_097];

fn cpu_matches(value: u32, predicate: U32Predicate) -> bool {
    match predicate {
        U32Predicate::Equal(target) => value == target,
        U32Predicate::NotEqual(target) => value != target,
        U32Predicate::LessThan(target) => value < target,
        U32Predicate::LessThanOrEqual(target) => value <= target,
        U32Predicate::GreaterThan(target) => value > target,
        U32Predicate::GreaterThanOrEqual(target) => value >= target,
        U32Predicate::BetweenInclusive { min, max } => value >= min && value <= max,
    }
}

fn cpu_mask(input: &[u32], predicate: U32Predicate) -> Vec<u32> {
    input
        .iter()
        .map(|&value| u32::from(cpu_matches(value, predicate)))
        .collect()
}

#[tokio::test]
async fn predicate_masks_match_cpu_across_dispatch_boundaries() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let generator = MaskGenerator::from_context(&context);
    let predicates = [
        U32Predicate::Equal(127),
        U32Predicate::NotEqual(127),
        U32Predicate::LessThan(127),
        U32Predicate::LessThanOrEqual(127),
        U32Predicate::GreaterThan(127),
        U32Predicate::GreaterThanOrEqual(127),
        U32Predicate::BetweenInclusive { min: 63, max: 191 },
        U32Predicate::BetweenInclusive { min: 9, max: 3 },
    ];

    for size in PREDICATE_SIZES {
        let mut input: Vec<_> = (0..size as u32)
            .map(|index| index.wrapping_mul(73) & 255)
            .collect();
        if let Some(first) = input.first_mut() {
            *first = u32::MIN;
        }
        if let Some(last) = input.last_mut() {
            *last = u32::MAX;
        }

        for predicate in predicates {
            let actual = generator
                .mask(&input, predicate)
                .await
                .expect("GPU predicate mask failed");
            assert_eq!(actual, cpu_mask(&input, predicate), "size {size}");
        }
    }
}

#[tokio::test]
async fn key_value_predicates_can_test_either_field() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let generator = MaskGenerator::from_context(&context);
    let input = [
        KeyValue::new(3, 90),
        KeyValue::new(7, 20),
        KeyValue::new(7, 70),
        KeyValue::new(1, 40),
    ];

    assert_eq!(
        generator
            .mask_key_values(&input, KeyValueField::Key, U32Predicate::Equal(7))
            .await
            .expect("key predicate failed"),
        [0, 1, 1, 0]
    );
    assert_eq!(
        generator
            .mask_key_values(
                &input,
                KeyValueField::Value,
                U32Predicate::BetweenInclusive { min: 40, max: 70 },
            )
            .await
            .expect("value predicate failed"),
        [0, 0, 1, 1]
    );
}

#[tokio::test]
async fn recorded_u32_predicate_composes_with_compaction_in_one_submission() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = [4_u32, 17, 9, 22, 11, 3];
    let input_buffer = storage_input(&context.device, "Predicate U32 Input", &input);
    let mask = mask_buffer(&context.device, input.len());
    let output = value_output(&context.device, input.len());
    let count = count_buffer(&context.device);
    let generator = MaskGenerator::from_context(&context);
    let mut compactor = Compactor::from_context(&context);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    generator
        .record_mask(
            &mut encoder,
            &input_buffer,
            &mask,
            input.len() as u32,
            U32Predicate::GreaterThanOrEqual(10),
        )
        .expect("predicate recording failed");
    compactor
        .record_compact(
            &mut encoder,
            &input_buffer,
            &mask,
            &output,
            &count,
            input.len() as u32,
        )
        .expect("compaction recording failed");
    context.queue.submit(Some(encoder.finish()));

    assert_eq!(
        support::read_u32(&context, &mask, input.len()).await,
        [0, 1, 0, 1, 1, 0]
    );
    assert_eq!(support::read_u32(&context, &count, 1).await, [3]);
    assert_eq!(support::read_u32(&context, &output, 3).await, [17, 22, 11]);
}

#[tokio::test]
async fn multiple_predicates_keep_distinct_parameters_in_one_submission() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = [3_u32, 8, 13, 21];
    let input_buffer = storage_input(&context.device, "Shared Predicate Input", &input);
    let lower_mask = mask_buffer(&context.device, input.len());
    let upper_mask = mask_buffer(&context.device, input.len());
    let generator = MaskGenerator::from_context(&context);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    generator
        .record_mask(
            &mut encoder,
            &input_buffer,
            &lower_mask,
            input.len() as u32,
            U32Predicate::LessThan(10),
        )
        .expect("lower predicate recording failed");
    generator
        .record_mask(
            &mut encoder,
            &input_buffer,
            &upper_mask,
            input.len() as u32,
            U32Predicate::GreaterThanOrEqual(13),
        )
        .expect("upper predicate recording failed");
    context.queue.submit(Some(encoder.finish()));

    assert_eq!(
        support::read_u32(&context, &lower_mask, input.len()).await,
        [1, 1, 0, 0]
    );
    assert_eq!(
        support::read_u32(&context, &upper_mask, input.len()).await,
        [0, 0, 1, 1]
    );
}

#[tokio::test]
async fn recorded_key_value_predicate_composes_with_compaction() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = [
        KeyValue::new(5, 50),
        KeyValue::new(2, 20),
        KeyValue::new(5, 51),
        KeyValue::new(8, 80),
    ];
    let input_buffer = storage_input(&context.device, "Predicate Key-Value Input", &input);
    let mask = mask_buffer(&context.device, input.len());
    let output = key_value_output(&context.device, input.len());
    let count = count_buffer(&context.device);
    let generator = MaskGenerator::from_context(&context);
    let mut compactor = KeyValueCompactor::from_context(&context);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    generator
        .record_key_value_mask(
            &mut encoder,
            &input_buffer,
            &mask,
            input.len() as u32,
            KeyValueField::Key,
            U32Predicate::Equal(5),
        )
        .expect("key-value predicate recording failed");
    compactor
        .record_compact(
            &mut encoder,
            &input_buffer,
            &mask,
            &output,
            &count,
            input.len() as u32,
        )
        .expect("key-value compaction recording failed");
    context.queue.submit(Some(encoder.finish()));

    assert_eq!(support::read_u32(&context, &count, 1).await, [2]);
    assert_eq!(
        support::read_pod::<KeyValue>(&context, &output, 2).await,
        [input[0], input[2]]
    );
}

#[tokio::test]
async fn predicate_recording_validates_buffer_contracts() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = storage_input(
        &context.device,
        "Validated Predicate Input",
        &[1_u32, 2, 3, 4],
    );
    let output = mask_buffer(&context.device, 4);
    let short_output = mask_buffer(&context.device, 3);
    let storage_only_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Predicate Mask Without Copy Source"),
        size: MaskGenerator::mask_buffer_size(4).expect("mask size overflow"),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let generator = MaskGenerator::from_context(&context);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    assert!(matches!(
        generator.record_mask(
            &mut encoder,
            &input,
            &short_output,
            4,
            U32Predicate::Equal(2),
        ),
        Err(Error::BufferTooSmall { .. })
    ));
    assert!(matches!(
        generator.record_mask(
            &mut encoder,
            &input,
            &storage_only_output,
            4,
            U32Predicate::Equal(2),
        ),
        Err(Error::MissingBufferUsage { .. })
    ));
    assert!(matches!(
        generator.record_mask(&mut encoder, &input, &input, 4, U32Predicate::Equal(2),),
        Err(Error::BufferAlias { .. })
    ));

    generator
        .record_mask(&mut encoder, &input, &output, 4, U32Predicate::Equal(2))
        .expect("valid predicate buffers were rejected");
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

fn mask_buffer(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Generated Predicate Mask"),
        size: MaskGenerator::mask_buffer_size(len as u32).expect("mask size overflow"),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn value_output(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Predicate Value Output"),
        size: (len * size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn key_value_output(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Predicate Key-Value Output"),
        size: (len * size_of::<KeyValue>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn count_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Predicate Compaction Count"),
        contents: bytemuck::bytes_of(&0_u32),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    })
}
