mod support;

use lampshade::{Error, Reducer, U32Reduction};
use wgpu::util::DeviceExt;

const REDUCTION_SIZES: [usize; 17] = [
    0, 1, 2, 31, 32, 33, 255, 256, 257, 2_047, 2_048, 2_049, 4_095, 4_096, 4_097, 8_193, 17,
];

fn input_for(case: usize, size: usize) -> Vec<u32> {
    match case % 3 {
        0 => vec![1; size],
        1 => (0..size).map(|index| (index % 97) as u32).collect(),
        _ => support::random_u32(size, case as u64),
    }
}

fn cpu_reduce(input: &[u32], operation: U32Reduction) -> u32 {
    input
        .iter()
        .copied()
        .fold(operation.identity(), |lhs, rhs| match operation {
            U32Reduction::Sum => lhs.wrapping_add(rhs),
            U32Reduction::Min => lhs.min(rhs),
            U32Reduction::Max => lhs.max(rhs),
        })
}

#[tokio::test]
async fn reductions_match_cpu_across_boundaries_and_patterns() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut reducer = Reducer::from_context(&context);

    for operation in [U32Reduction::Sum, U32Reduction::Min, U32Reduction::Max] {
        for (case, size) in REDUCTION_SIZES.into_iter().enumerate() {
            let input = input_for(case, size);
            let actual = reducer
                .reduce(&input, operation)
                .await
                .expect("GPU reduction failed");
            assert_eq!(
                actual,
                cpu_reduce(&input, operation),
                "{operation:?} mismatch for size {size}"
            );
        }
    }

    let overflow = [u32::MAX, 2, u32::MAX, 7];
    assert_eq!(
        reducer.sum(&overflow).await.expect("GPU sum failed"),
        overflow
            .into_iter()
            .fold(0_u32, |sum, value| sum.wrapping_add(value))
    );
}

#[tokio::test]
async fn reduction_crosses_multiple_hierarchy_levels() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut reducer = Reducer::from_context(&context);
    let input = support::random_u32(4_194_305, 0x005E_D0CE);

    for operation in [U32Reduction::Sum, U32Reduction::Min, U32Reduction::Max] {
        assert_eq!(
            reducer
                .reduce(&input, operation)
                .await
                .expect("multi-level GPU reduction failed"),
            cpu_reduce(&input, operation)
        );
    }
}

#[tokio::test]
async fn empty_gpu_reductions_write_operation_identities() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut reducer = Reducer::from_context(&context);
    let input = storage_buffer(&context.device, "Empty Reduction Input", &[123]);
    let output = output_buffer(&context.device, "Empty Reduction Output");

    for operation in [U32Reduction::Sum, U32Reduction::Min, U32Reduction::Max] {
        reducer
            .reduce_gpu_to_gpu(&input, &output, 0, operation)
            .expect("empty GPU reduction failed");
        assert_eq!(
            support::read_u32(&context, &output, 1).await,
            [operation.identity()]
        );
    }
}

#[tokio::test]
async fn reduction_uses_the_explicit_logical_length() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut reducer = Reducer::from_context(&context);
    let input = storage_buffer(&context.device, "Padded Reduction Input", &[9, 3, 1_000]);
    let output = output_buffer(&context.device, "Padded Reduction Output");

    reducer
        .reduce_gpu_to_gpu(&input, &output, 2, U32Reduction::Min)
        .expect("GPU reduction failed");
    assert_eq!(support::read_u32(&context, &output, 1).await, [3]);
}

#[tokio::test]
async fn recorded_reductions_compose_in_one_encoder() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut reducer = Reducer::from_context(&context);
    let first = input_for(1, 4_097);
    let second = input_for(2, 8_193);
    let first_input = storage_buffer(&context.device, "First Reduction Input", &first);
    let second_input = storage_buffer(&context.device, "Second Reduction Input", &second);
    let first_output = output_buffer(&context.device, "First Reduction Output");
    let second_output = output_buffer(&context.device, "Second Reduction Output");
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    reducer
        .record_reduce(
            &mut encoder,
            &first_input,
            &first_output,
            first.len() as u32,
            U32Reduction::Sum,
        )
        .expect("first reduction recording failed");
    reducer
        .record_reduce(
            &mut encoder,
            &second_input,
            &second_output,
            second.len() as u32,
            U32Reduction::Max,
        )
        .expect("second reduction recording failed");
    context.queue.submit(Some(encoder.finish()));

    assert_eq!(
        support::read_u32(&context, &first_output, 1).await,
        [cpu_reduce(&first, U32Reduction::Sum)]
    );
    assert_eq!(
        support::read_u32(&context, &second_output, 1).await,
        [cpu_reduce(&second, U32Reduction::Max)]
    );
}

#[tokio::test]
async fn reduction_rejects_invalid_buffer_contracts() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut reducer = Reducer::from_context(&context);
    let invalid_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Invalid Reduction Input"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output = output_buffer(&context.device, "Reduction Output");
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let error = reducer
        .record_reduce(&mut encoder, &invalid_input, &output, 4, U32Reduction::Sum)
        .expect_err("missing STORAGE input usage must be rejected");
    assert!(matches!(error, Error::MissingBufferUsage { .. }));

    let short_input = storage_buffer(&context.device, "Short Reduction Input", &[1, 2]);
    let error = reducer
        .record_reduce(&mut encoder, &short_input, &output, 3, U32Reduction::Sum)
        .expect_err("input shorter than the logical length must be rejected");
    assert!(matches!(error, Error::BufferTooSmall { .. }));

    let input = storage_buffer(&context.device, "Reduction Input", &[1, 2, 3, 4]);
    let invalid_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Invalid Reduction Output"),
        size: Reducer::output_buffer_size(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let error = reducer
        .record_reduce(&mut encoder, &input, &invalid_output, 4, U32Reduction::Sum)
        .expect_err("missing COPY_DST output usage must be rejected");
    assert!(matches!(error, Error::MissingBufferUsage { .. }));

    let aliased = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Aliased Reduction Buffer"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let error = reducer
        .record_reduce(&mut encoder, &aliased, &aliased, 4, U32Reduction::Sum)
        .expect_err("aliased reduction buffers must be rejected");
    assert!(matches!(error, Error::BufferAlias { .. }));
}

fn storage_buffer(device: &wgpu::Device, label: &'static str, input: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn output_buffer(device: &wgpu::Device, label: &'static str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: Reducer::output_buffer_size(),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
