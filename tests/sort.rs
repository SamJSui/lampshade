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
async fn bounded_sort_handles_odd_and_even_portable_pass_counts() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = Sorter::new(&context.device, &context.queue);
    let cases = [
        (0, vec![0, 0, 0]),
        (1, vec![1, 0, 1, 0]),
        (3, vec![7, 0, 4, 1, 7]),
        (5, vec![31, 0, 17, 3, 16, 3]),
        (16, vec![u16::MAX as u32, 0, 42, 42, 1]),
        (32, vec![u32::MAX, 0, 0x8000_0000, 7]),
    ];

    for (key_bits, input) in cases {
        let actual = sorter
            .sort_with_key_bits(&input, key_bits)
            .await
            .expect("bounded GPU radix sort failed");
        assert_eq!(actual, cpu_sort(&input), "mismatch for {key_bits} key bits");
    }
}

#[tokio::test]
async fn bounded_sort_validates_host_keys_and_key_width() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = Sorter::new(&context.device, &context.queue);

    let error = sorter
        .sort_with_key_bits(&[4], 2)
        .await
        .expect_err("a three-bit key must not satisfy a two-bit bound");
    assert!(matches!(error, Error::KeyExceedsBitRange { .. }));

    let error = sorter
        .sort_with_key_bits(&[], 33)
        .await
        .expect_err("key widths above 32 must be rejected");
    assert!(matches!(error, Error::InvalidKeyBits { bits: 33 }));
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
async fn bounded_sort_gpu_to_gpu_writes_output_after_an_odd_pass_count() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = Sorter::new(&context.device, &context.queue);
    let input = [31, 0, 17, 3, 16, 3];
    let input_buffer = create_sort_input(&context.device, &input);
    let output = create_sort_output(&context.device, input.len());

    sorter
        .sort_gpu_to_gpu_with_key_bits(&input_buffer, &output, input.len() as u32, 5)
        .expect("bounded GPU radix sort failed");
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

    let error = sorter
        .record_sort(&mut encoder, &output, &output, 4)
        .expect_err("in-place radix scatter must be rejected");
    assert!(matches!(error, Error::BufferAlias { .. }));
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
