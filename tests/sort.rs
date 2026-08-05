mod support;

use wgpu::util::DeviceExt;
use wgpu_primitives::{Error, Sorter};

fn cpu_sort(input: &[u32]) -> Vec<u32> {
    let mut expected = input.to_vec();
    expected.sort_unstable();
    expected
}

#[tokio::test]
async fn radix_sort_matches_cpu_across_boundaries_and_patterns() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = Sorter::from_context(&context);
    let sizes = [
        0, 1, 2, 31, 32, 33, 127, 128, 129, 511, 512, 513, 2_047, 2_048, 2_049, 4_097, 17,
    ];

    for (case, size) in sizes.into_iter().enumerate() {
        let input = match case % 4 {
            0 => vec![42; size],
            1 => (0..size as u32).collect(),
            2 => (0..size as u32).rev().collect(),
            _ => support::random_u32(size, case as u64),
        };
        let actual = sorter.sort(&input).await.expect("GPU radix sort failed");
        assert_eq!(actual, cpu_sort(&input), "sort mismatch for size {size}");
    }

    let edge_values = [u32::MAX, 0, u32::MAX, 1, 0, 42, 42];
    assert_eq!(
        sorter
            .sort(&edge_values)
            .await
            .expect("GPU radix sort failed"),
        cpu_sort(&edge_values)
    );
}

#[tokio::test]
async fn sorter_reuses_workspace_for_growing_and_shrinking_inputs() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = Sorter::from_context(&context);

    for (case, size) in [8, 65_537, 257, 4_097, 3].into_iter().enumerate() {
        let input = support::random_u32(size, case as u64 + 100);
        let actual = sorter.sort(&input).await.expect("GPU radix sort failed");
        assert_eq!(actual, cpu_sort(&input), "sort mismatch for size {size}");
    }
}

#[tokio::test]
async fn sort_gpu_to_gpu_writes_the_caller_output_buffer() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = Sorter::from_context(&context);
    let input = [9, 1, 4, 1, u32::MAX, 0];
    let input_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sort Input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sort Output"),
        size: input_buffer.size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    sorter
        .sort_gpu_to_gpu(&input_buffer, &output, input.len() as u32)
        .expect("GPU radix sort failed");
    let actual = support::read_u32(&context, &output, input.len()).await;
    assert_eq!(actual, cpu_sort(&input));
}

#[tokio::test]
async fn record_sort_composes_multiple_invocations_in_one_encoder() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = Sorter::new(&context.device, &context.queue);
    let first = [7_u32];
    let second = support::random_u32(4_097, 300);
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
        .expect("first GPU sort recording failed");
    sorter
        .record_sort(
            &mut encoder,
            &second_input,
            &second_output,
            second.len() as u32,
        )
        .expect("second GPU sort recording failed");
    context.queue.submit(Some(encoder.finish()));

    assert_eq!(
        support::read_u32(&context, &first_output, first.len()).await,
        cpu_sort(&first)
    );
    assert_eq!(
        support::read_u32(&context, &second_output, second.len()).await,
        cpu_sort(&second)
    );
}

#[tokio::test]
async fn record_sort_rejects_invalid_buffer_contracts() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = Sorter::from_context(&context);
    let input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Short Sort Input"),
        size: 12,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sort Output"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let error = sorter
        .record_sort(&mut encoder, &input, &output, 4)
        .expect_err("short input must be rejected");
    assert!(matches!(error, Error::BufferTooSmall { .. }));
}

fn create_sort_input(device: &wgpu::Device, input: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Sort Input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn create_sort_output(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Sort Output"),
        size: (len * size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
