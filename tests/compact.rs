mod support;

use lampshade::{Compactor, Error};
use wgpu::util::DeviceExt;

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

fn cpu_compact(input: &[u32], mask: &[u32]) -> Vec<u32> {
    input
        .iter()
        .zip(mask)
        .filter_map(|(&value, &keep)| (keep == 1).then_some(value))
        .collect()
}

#[tokio::test]
async fn compact_matches_cpu_across_boundaries_and_patterns() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut compactor = Compactor::from_context(&context);

    for (case, size) in COMPACTION_SIZES.into_iter().enumerate() {
        let input: Vec<_> = (0..size as u32).map(|index| index ^ 0xA5A5_0000).collect();
        let mask = mask_for(case, size);
        let actual = compactor
            .compact(&input, &mask)
            .await
            .expect("GPU compaction failed");

        assert_eq!(actual, cpu_compact(&input, &mask), "size {size}");
    }
}

#[tokio::test]
async fn portable_compaction_fuses_block_prefixes_without_subgroups() {
    let Some(context) = support::gpu_context_without_optional_features().await else {
        return;
    };
    assert!(!context.device.features().contains(wgpu::Features::SUBGROUP));
    let size = 4_097;
    let input: Vec<_> = (0..size as u32).map(|index| index ^ 0x5A5A_0000).collect();
    let mask: Vec<_> = (0..size).map(|index| u32::from(index % 3 != 1)).collect();
    let mut compactor = Compactor::from_context(&context);

    assert_eq!(
        compactor
            .compact(&input, &mask)
            .await
            .expect("portable GPU compaction failed"),
        cpu_compact(&input, &mask)
    );
}

#[tokio::test]
async fn compact_preserves_selected_input_order() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut compactor = Compactor::from_context(&context);
    let input = [40_u32, 10, 30, 20, 50, 60];
    let mask = [0_u32, 1, 1, 0, 1, 0];

    assert_eq!(
        compactor
            .compact(&input, &mask)
            .await
            .expect("GPU compaction failed"),
        [10, 30, 50]
    );
}

#[tokio::test]
async fn gpu_compaction_writes_output_and_resident_count() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = [9_u32, 8, 7, 6, 5, 4, 999];
    let mask = [1_u32, 0, 1, 0, 0, 1, 1];
    let input_buffer = storage_input(&context.device, "Compaction Input", &input);
    let mask_buffer = storage_mask(&context.device, "Compaction Mask", &mask);
    let output = compaction_output(&context.device, input.len());
    let count = count_buffer(&context.device, 99);
    let mut compactor = Compactor::from_context(&context);

    compactor
        .compact_gpu_to_gpu(&input_buffer, &mask_buffer, &output, &count, 6)
        .expect("GPU-resident compaction failed");

    assert_eq!(support::read_u32(&context, &count, 1).await, [3]);
    assert_eq!(support::read_u32(&context, &output, 3).await, [9, 7, 4]);
}

#[tokio::test]
async fn empty_gpu_compaction_clears_resident_count() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = storage_input(&context.device, "Empty Compaction Input", &[0]);
    let mask = storage_mask(&context.device, "Empty Compaction Mask", &[0]);
    let output = compaction_output(&context.device, 1);
    let count = count_buffer(&context.device, 99);
    let mut compactor = Compactor::from_context(&context);

    compactor
        .compact_gpu_to_gpu(&input, &mask, &output, &count, 0)
        .expect("empty GPU compaction failed");

    assert_eq!(support::read_u32(&context, &count, 1).await, [0]);
}

#[tokio::test]
async fn record_compact_composes_multiple_invocations_in_one_encoder() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let first_input = [3_u32, 1, 4, 1];
    let first_mask = [1_u32, 0, 1, 1];
    let second_input = [2_u32, 7, 1, 8, 2, 8];
    let second_mask = [0_u32, 1, 0, 1, 0, 1];
    let first_input_buffer = storage_input(&context.device, "First Input", &first_input);
    let first_mask_buffer = storage_mask(&context.device, "First Mask", &first_mask);
    let first_output = compaction_output(&context.device, first_input.len());
    let first_count = count_buffer(&context.device, 0);
    let second_input_buffer = storage_input(&context.device, "Second Input", &second_input);
    let second_mask_buffer = storage_mask(&context.device, "Second Mask", &second_mask);
    let second_output = compaction_output(&context.device, second_input.len());
    let second_count = count_buffer(&context.device, 0);
    let mut compactor = Compactor::from_context(&context);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    compactor
        .record_compact(
            &mut encoder,
            &first_input_buffer,
            &first_mask_buffer,
            &first_output,
            &first_count,
            first_input.len() as u32,
        )
        .expect("first compaction recording failed");
    compactor
        .record_compact(
            &mut encoder,
            &second_input_buffer,
            &second_mask_buffer,
            &second_output,
            &second_count,
            second_input.len() as u32,
        )
        .expect("second compaction recording failed");
    context.queue.submit(Some(encoder.finish()));

    assert_eq!(support::read_u32(&context, &first_count, 1).await, [3]);
    assert_eq!(
        support::read_u32(&context, &first_output, 3).await,
        [3, 4, 1]
    );
    assert_eq!(support::read_u32(&context, &second_count, 1).await, [3]);
    assert_eq!(
        support::read_u32(&context, &second_output, 3).await,
        [7, 8, 8]
    );
}

#[tokio::test]
async fn compact_rejects_invalid_slice_masks() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut compactor = Compactor::from_context(&context);

    assert!(matches!(
        compactor.compact(&[1, 2], &[1]).await,
        Err(Error::CompactionLengthMismatch { input: 2, mask: 1 })
    ));
    assert!(matches!(
        compactor.compact(&[1, 2], &[1, 2]).await,
        Err(Error::InvalidCompactionFlag { index: 1, value: 2 })
    ));
}

#[tokio::test]
async fn record_compact_rejects_invalid_buffer_contracts() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let device = &context.device;
    let input = storage_input(device, "Valid Compaction Input", &[1, 2, 3, 4]);
    let mask = storage_mask(device, "Valid Compaction Mask", &[1, 0, 1, 0]);
    let output = compaction_output(device, 4);
    let count = count_buffer(device, 0);
    let mut compactor = Compactor::from_context(&context);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let short_output = compaction_output(device, 3);
    assert!(matches!(
        compactor.record_compact(&mut encoder, &input, &mask, &short_output, &count, 4),
        Err(Error::BufferTooSmall { .. })
    ));

    let mask_without_copy_src = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Invalid Compaction Mask"),
        contents: bytemuck::cast_slice(&[1_u32, 0, 1, 0]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    assert!(matches!(
        compactor.record_compact(
            &mut encoder,
            &input,
            &mask_without_copy_src,
            &output,
            &count,
            4,
        ),
        Err(Error::MissingBufferUsage { .. })
    ));

    assert!(matches!(
        compactor.record_compact(&mut encoder, &input, &mask, &input, &count, 4),
        Err(Error::BufferAlias { .. })
    ));

    let invalid_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Invalid Compaction Count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    assert!(matches!(
        compactor.record_compact(&mut encoder, &input, &mask, &output, &invalid_count, 4),
        Err(Error::MissingBufferUsage { .. })
    ));
}

fn storage_input(device: &wgpu::Device, label: &'static str, data: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn storage_mask(device: &wgpu::Device, label: &'static str, data: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn compaction_output(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Compaction Output"),
        size: (len * size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn count_buffer(device: &wgpu::Device, initial: u32) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Compaction Output Count"),
        contents: bytemuck::bytes_of(&initial),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    })
}
