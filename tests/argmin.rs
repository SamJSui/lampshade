mod support;

use lampshade::{
    ArgminByKey, Error, KeyValue,
    pipeline::{GpuCount, GpuSlice, GpuSliceMut, Primitives, WorkspaceRequirements},
};
use wgpu::util::DeviceExt;

fn cpu_argmin(input: &[KeyValue]) -> KeyValue {
    input
        .iter()
        .copied()
        .min_by_key(|item| (item.key, item.value))
        .unwrap_or(KeyValue::new(u32::MAX, u32::MAX))
}

fn input_for(size: usize) -> Vec<KeyValue> {
    let mut input = (0..size)
        .map(|index| KeyValue::new((index as u32).wrapping_mul(2_654_435_761), index as u32))
        .collect::<Vec<_>>();
    if size > 2 {
        input[size / 2] = KeyValue::new(3, 17);
        input[size - 1] = KeyValue::new(3, 9);
    }
    input
}

#[tokio::test]
async fn argmin_matches_cpu_across_hierarchy_boundaries() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut selector = ArgminByKey::from_context(&context);
    for size in [0, 1, 2, 255, 256, 257, 65_536, 65_537] {
        let input = input_for(size);
        assert_eq!(
            selector.argmin(&input).await.expect("GPU argmin failed"),
            cpu_argmin(&input),
            "argmin mismatch for {size} records"
        );
    }
}

#[tokio::test]
async fn fixed_and_counted_recording_handle_empty_and_clamped_extents() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = input_for(4_097);
    let input_buffer = storage_buffer(&context.device, "Argmin Input", &input);
    let fixed_output = output_buffer(&context.device, "Fixed Argmin Output");
    let counted_output = output_buffer(&context.device, "Counted Argmin Output");
    let count = count_buffer(&context.device, input.len() as u32 + 100);
    let mut selector = ArgminByKey::from_context(&context);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    selector
        .record_argmin(
            &mut encoder,
            &input_buffer,
            &fixed_output,
            input.len() as u32,
        )
        .expect("fixed argmin recording failed");
    selector
        .record_argmin_counted(
            &mut encoder,
            &input_buffer,
            &counted_output,
            &count,
            input.len() as u32,
        )
        .expect("counted argmin recording failed");
    context.queue.submit(Some(encoder.finish()));
    let expected = cpu_argmin(&input);
    assert_eq!(
        support::read_pod::<KeyValue>(&context, &fixed_output, 1).await,
        [expected]
    );
    assert_eq!(
        support::read_pod::<KeyValue>(&context, &counted_output, 1).await,
        [expected]
    );

    context
        .queue
        .write_buffer(&count, 0, bytemuck::bytes_of(&0_u32));
    selector
        .argmin_counted_gpu_to_gpu(&input_buffer, &counted_output, &count, input.len() as u32)
        .expect("empty counted argmin failed");
    assert_eq!(
        support::read_pod::<KeyValue>(&context, &counted_output, 1).await,
        [KeyValue::new(u32::MAX, u32::MAX)]
    );
}

#[tokio::test]
async fn typed_argmin_composes_fixed_and_gpu_counted_ranges() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    const START: u32 = 32;
    const CAPACITY: u32 = 4_097;
    let active = 2_049_u32;
    let values = input_for(CAPACITY as usize);
    let input = ranged_storage_buffer(&context.device, "Typed Argmin Input", &values, START);
    let fixed_output = ranged_output_buffer(&context.device, "Typed Fixed Argmin Output", START);
    let counted_output =
        ranged_output_buffer(&context.device, "Typed Counted Argmin Output", START);
    let count_alignment = context.device.limits().min_storage_buffer_offset_alignment;
    let count = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Typed Argmin Count"),
        size: u64::from(count_alignment) + 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    context.queue.write_buffer(
        &count,
        u64::from(count_alignment),
        bytemuck::bytes_of(&active),
    );
    let gpu_count = GpuCount::at(&count, u64::from(count_alignment)).unwrap();
    let mut primitives = Primitives::from_context(&context);
    primitives
        .reserve_workspace(WorkspaceRequirements::new(CAPACITY).argmin_by_key())
        .unwrap();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut recorder = primitives.record(&mut encoder);
        recorder
            .argmin_by_key(
                GpuSlice::from_range(&input, START..START + CAPACITY).unwrap(),
                GpuSliceMut::from_range(&fixed_output, START..START + 1).unwrap(),
            )
            .unwrap();
        recorder
            .argmin_by_key(
                GpuSlice::counted(&input, START..START + CAPACITY, gpu_count).unwrap(),
                GpuSliceMut::from_range(&counted_output, START..START + 1).unwrap(),
            )
            .unwrap();
    }
    context.queue.submit(Some(encoder.finish()));

    assert_eq!(
        read_ranged_pair(&context, &fixed_output, START).await,
        cpu_argmin(&values)
    );
    assert_eq!(
        read_ranged_pair(&context, &counted_output, START).await,
        cpu_argmin(&values[..active as usize])
    );
}

#[tokio::test]
async fn counted_argmin_consumes_a_count_written_earlier_in_the_same_encoder() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = [
        KeyValue::new(9, 0),
        KeyValue::new(4, 1),
        KeyValue::new(4, 0),
        KeyValue::new(7, 3),
        KeyValue::new(0, 4),
    ];
    let input_buffer = storage_buffer(&context.device, "Produced-count Argmin Input", &input);
    let output = output_buffer(&context.device, "Produced-count Argmin Output");
    let produced_count = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Produced Argmin Count"),
            contents: bytemuck::bytes_of(&4_u32),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
    let count = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("GPU-written Argmin Count"),
        size: size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let mut selector = ArgminByKey::from_context(&context);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(&produced_count, 0, &count, 0, size_of::<u32>() as u64);
    selector
        .record_argmin_counted(
            &mut encoder,
            &input_buffer,
            &output,
            &count,
            input.len() as u32,
        )
        .expect("same-encoder counted argmin recording failed");
    context.queue.submit(Some(encoder.finish()));

    assert_eq!(
        support::read_pod::<KeyValue>(&context, &output, 1).await,
        [KeyValue::new(4, 0)]
    );
}

#[tokio::test]
async fn argmin_rejects_aliases_and_invalid_usages() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut selector = ArgminByKey::from_context(&context);
    let aliased = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Aliased Argmin Buffer"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let count = count_buffer(&context.device, 1);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let error = selector
        .record_argmin(&mut encoder, &aliased, &aliased, 1)
        .expect_err("aliased argmin buffers must be rejected");
    assert!(matches!(
        error,
        Error::BufferAlias {
            first: "argmin input",
            second: "argmin output"
        }
    ));
    let output = output_buffer(&context.device, "Argmin Validation Output");
    let error = selector
        .record_argmin_counted(&mut encoder, &count, &output, &count, 1)
        .expect_err("count/input alias must be rejected");
    assert!(matches!(
        error,
        Error::BufferAlias {
            first: "argmin input",
            second: "argmin item count"
        }
    ));
    let invalid_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Invalid Argmin Input"),
        size: 8,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let error = selector
        .record_argmin(&mut encoder, &invalid_input, &output, 1)
        .expect_err("missing STORAGE input usage must be rejected");
    assert!(matches!(error, Error::MissingBufferUsage { .. }));

    let count_output_alias = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Aliased Argmin Output and Count"),
        size: ArgminByKey::output_buffer_size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let valid_input = storage_buffer(
        &context.device,
        "Valid Argmin Validation Input",
        &[KeyValue::new(2, 1)],
    );
    let error = selector
        .record_argmin_counted(
            &mut encoder,
            &valid_input,
            &count_output_alias,
            &count_output_alias,
            1,
        )
        .expect_err("count/output alias must be rejected");
    assert!(matches!(
        error,
        Error::BufferAlias {
            first: "argmin output",
            second: "argmin item count"
        }
    ));

    let invalid_count = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Invalid Argmin Count"),
        size: size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let error = selector
        .record_argmin_counted(&mut encoder, &valid_input, &output, &invalid_count, 1)
        .expect_err("missing STORAGE count usage must be rejected");
    assert!(matches!(
        error,
        Error::MissingBufferUsage {
            name: "argmin item count",
            ..
        }
    ));

    let limit = context
        .device
        .limits()
        .max_buffer_size
        .min(context.device.limits().max_storage_buffer_binding_size);
    let oversized_items = limit / size_of::<KeyValue>() as u64 + 1;
    if let Ok(oversized_items) = u32::try_from(oversized_items) {
        let requested = u64::from(oversized_items) * size_of::<KeyValue>() as u64;
        let error = selector
            .record_argmin(&mut encoder, &valid_input, &output, oversized_items)
            .expect_err("oversized argmin binding must be rejected before recording");
        assert!(matches!(
            error,
            Error::BufferLimitExceeded {
                requested: actual_requested,
                limit: actual_limit,
            } if actual_requested == requested && actual_limit == limit
        ));
    }
}

#[tokio::test]
#[ignore = "allocates about 128 MiB to cross the two-dimensional dispatch boundary"]
async fn argmin_validates_two_dimensional_dispatch() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    const ITEMS: usize = 65_535 * 256 + 1;
    let mut input = vec![KeyValue::new(7, 1); ITEMS];
    input[ITEMS - 1] = KeyValue::new(0, 99);
    let input_buffer = storage_buffer(&context.device, "2D Argmin Input", &input);
    let fixed_output = output_buffer(&context.device, "2D Fixed Argmin Output");
    let counted_output = output_buffer(&context.device, "2D Counted Argmin Output");
    let count = count_buffer(&context.device, ITEMS as u32);
    let mut selector = ArgminByKey::from_context(&context);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    selector
        .record_argmin(&mut encoder, &input_buffer, &fixed_output, ITEMS as u32)
        .expect("2D fixed argmin recording failed");
    selector
        .record_argmin_counted(
            &mut encoder,
            &input_buffer,
            &counted_output,
            &count,
            ITEMS as u32,
        )
        .expect("2D counted argmin recording failed");
    context.queue.submit(Some(encoder.finish()));

    let expected = [KeyValue::new(0, 99)];
    assert_eq!(
        support::read_pod::<KeyValue>(&context, &fixed_output, 1).await,
        expected
    );
    assert_eq!(
        support::read_pod::<KeyValue>(&context, &counted_output, 1).await,
        expected
    );
}

fn storage_buffer(device: &wgpu::Device, label: &'static str, input: &[KeyValue]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn output_buffer(device: &wgpu::Device, label: &'static str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: ArgminByKey::output_buffer_size(),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn count_buffer(device: &wgpu::Device, count: u32) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Argmin Count"),
        contents: bytemuck::bytes_of(&count),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn ranged_storage_buffer(
    device: &wgpu::Device,
    label: &'static str,
    input: &[KeyValue],
    start: u32,
) -> wgpu::Buffer {
    let mut data = vec![KeyValue::default(); start as usize + input.len()];
    data[start as usize..].copy_from_slice(input);
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn ranged_output_buffer(device: &wgpu::Device, label: &'static str, start: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: u64::from(start + 1) * size_of::<KeyValue>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

async fn read_ranged_pair(
    context: &lampshade::Context,
    buffer: &wgpu::Buffer,
    start: u32,
) -> KeyValue {
    let size = u64::from(start + 1) * size_of::<KeyValue>() as u64;
    let data = support::read_pod::<KeyValue>(context, buffer, size as usize / 8).await;
    data[start as usize]
}
