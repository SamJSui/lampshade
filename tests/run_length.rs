mod support;

use lampshade::{
    Error, RunLengthEncoder, RunLengthOutputBuffers, U32Reduction,
    pipeline::{
        Extent, GpuCount, GpuSlice, GpuSliceMut, Primitives, SortOptions, WorkspaceRequirements,
    },
};
use wgpu::util::DeviceExt;

const SIZES: [usize; 13] = [
    0, 1, 2, 31, 32, 255, 256, 257, 511, 2_047, 2_048, 4_097, 65_537,
];

fn cpu_encode(input: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut values = Vec::new();
    let mut lengths = Vec::new();
    for &value in input {
        if values.last() == Some(&value) {
            *lengths.last_mut().expect("a repeated value has a run") += 1;
        } else {
            values.push(value);
            lengths.push(1);
        }
    }
    (values, lengths)
}

fn patterned_input(case: usize, size: usize) -> Vec<u32> {
    match case % 4 {
        0 => vec![7; size],
        1 => (0..size as u32).collect(),
        2 => (0..size as u32).map(|index| index / 7).collect(),
        _ => (0..size as u32)
            .map(|index| match index % 9 {
                0..=2 => 4,
                3..=6 => 1,
                _ => 4,
            })
            .collect(),
    }
}

#[tokio::test]
async fn run_length_encoding_matches_cpu_across_boundaries_and_patterns() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut encoder = RunLengthEncoder::from_context(&context);

    for (case, size) in SIZES.into_iter().enumerate() {
        let input = patterned_input(case, size);
        assert_eq!(
            encoder.encode(&input).await.expect("GPU RLE failed"),
            cpu_encode(&input),
            "size {size}"
        );
    }
}

#[tokio::test]
async fn resident_fixed_and_counted_paths_write_gpu_run_counts() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = [5_u32, 5, 2, 2, 2, 9, 5, 5, 77];
    let input_buffer = storage_buffer(&context.device, "RLE Input", &input);
    let mut rle = RunLengthEncoder::from_context(&context);

    let fixed_values = output_buffer(&context.device, "Fixed RLE Values", input.len());
    let fixed_lengths = output_buffer(&context.device, "Fixed RLE Lengths", input.len());
    let fixed_count = count_buffer(&context.device, "Fixed RLE Count", 99);
    rle.encode_gpu_to_gpu(
        &input_buffer,
        RunLengthOutputBuffers::new(&fixed_values, &fixed_lengths, &fixed_count),
        8,
    )
    .expect("fixed GPU RLE failed");
    assert_eq!(support::read_u32(&context, &fixed_count, 1).await, [4]);
    assert_eq!(
        support::read_u32(&context, &fixed_values, 4).await,
        [5, 2, 9, 5]
    );
    assert_eq!(
        support::read_u32(&context, &fixed_lengths, 4).await,
        [2, 3, 1, 2]
    );

    for (resident_count, expected_values, expected_lengths) in [
        (0, vec![], vec![]),
        (1, vec![5], vec![1]),
        (6, vec![5, 2, 9], vec![2, 3, 1]),
        (99, vec![5, 2, 9, 5, 77], vec![2, 3, 1, 2, 1]),
    ] {
        let input_count = count_buffer(&context.device, "RLE Input Count", resident_count);
        let values = output_buffer(&context.device, "Counted RLE Values", input.len());
        let lengths = output_buffer(&context.device, "Counted RLE Lengths", input.len());
        let run_count = count_buffer(&context.device, "Counted RLE Count", 99);
        rle.encode_counted_gpu_to_gpu(
            &input_buffer,
            &input_count,
            RunLengthOutputBuffers::new(&values, &lengths, &run_count),
            input.len() as u32,
        )
        .expect("counted GPU RLE failed");
        assert_eq!(
            support::read_u32(&context, &run_count, 1).await,
            [expected_values.len() as u32]
        );
        assert_eq!(
            support::read_u32(&context, &values, expected_values.len()).await,
            expected_values
        );
        assert_eq!(
            support::read_u32(&context, &lengths, expected_lengths.len()).await,
            expected_lengths
        );
    }
}

#[tokio::test]
async fn empty_fixed_encoding_clears_the_resident_count() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = storage_buffer(&context.device, "Empty RLE Input", &[0]);
    let values = output_buffer(&context.device, "Empty RLE Values", 1);
    let lengths = output_buffer(&context.device, "Empty RLE Lengths", 1);
    let count = count_buffer(&context.device, "Empty RLE Count", 99);
    let mut rle = RunLengthEncoder::from_context(&context);

    rle.encode_gpu_to_gpu(
        &input,
        RunLengthOutputBuffers::new(&values, &lengths, &count),
        0,
    )
    .expect("empty GPU RLE failed");
    assert_eq!(support::read_u32(&context, &count, 1).await, [0]);
}

#[tokio::test]
async fn recorded_encodings_compose_in_one_encoder() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let first = [1_u32, 1, 1, 4, 4, 2];
    let second = [8_u32, 3, 3, 8, 8, 8, 1];
    let first_input = storage_buffer(&context.device, "First RLE Input", &first);
    let second_input = storage_buffer(&context.device, "Second RLE Input", &second);
    let first_values = output_buffer(&context.device, "First RLE Values", first.len());
    let first_lengths = output_buffer(&context.device, "First RLE Lengths", first.len());
    let first_count = count_buffer(&context.device, "First RLE Count", 0);
    let second_values = output_buffer(&context.device, "Second RLE Values", second.len());
    let second_lengths = output_buffer(&context.device, "Second RLE Lengths", second.len());
    let second_count = count_buffer(&context.device, "Second RLE Count", 0);
    let mut rle = RunLengthEncoder::from_context(&context);
    let mut commands = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    rle.record_encode(
        &mut commands,
        &first_input,
        RunLengthOutputBuffers::new(&first_values, &first_lengths, &first_count),
        first.len() as u32,
    )
    .expect("first RLE recording failed");
    rle.record_encode(
        &mut commands,
        &second_input,
        RunLengthOutputBuffers::new(&second_values, &second_lengths, &second_count),
        second.len() as u32,
    )
    .expect("second RLE recording failed");
    context.queue.submit(Some(commands.finish()));

    assert_eq!(support::read_u32(&context, &first_count, 1).await, [3]);
    assert_eq!(
        support::read_u32(&context, &first_values, 3).await,
        [1, 4, 2]
    );
    assert_eq!(
        support::read_u32(&context, &first_lengths, 3).await,
        [3, 2, 1]
    );
    assert_eq!(support::read_u32(&context, &second_count, 1).await, [4]);
    assert_eq!(
        support::read_u32(&context, &second_values, 4).await,
        [8, 3, 8, 1]
    );
    assert_eq!(
        support::read_u32(&context, &second_lengths, 4).await,
        [1, 2, 3, 1]
    );
}

#[tokio::test]
async fn encoding_rejects_size_usage_and_alias_contract_violations() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let device = &context.device;
    let input = storage_buffer(device, "RLE Contract Input", &[1, 1, 2, 2]);
    let values = output_buffer(device, "RLE Contract Values", 4);
    let lengths = output_buffer(device, "RLE Contract Lengths", 4);
    let count = count_buffer(device, "RLE Contract Count", 0);
    let mut rle = RunLengthEncoder::from_context(&context);
    let mut commands = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let short = output_buffer(device, "Short RLE Output", 3);
    assert!(matches!(
        rle.record_encode(
            &mut commands,
            &input,
            RunLengthOutputBuffers::new(&short, &lengths, &count),
            4,
        ),
        Err(Error::BufferTooSmall { .. })
    ));
    let invalid_count = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Invalid RLE Count"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    assert!(matches!(
        rle.record_encode(
            &mut commands,
            &input,
            RunLengthOutputBuffers::new(&values, &lengths, &invalid_count),
            4,
        ),
        Err(Error::MissingBufferUsage { .. })
    ));

    let all_usage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("RLE Alias Buffer"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    for result in [
        rle.record_encode(
            &mut commands,
            &all_usage,
            RunLengthOutputBuffers::new(&all_usage, &lengths, &count),
            4,
        ),
        rle.record_encode(
            &mut commands,
            &all_usage,
            RunLengthOutputBuffers::new(&values, &all_usage, &count),
            4,
        ),
        rle.record_encode(
            &mut commands,
            &all_usage,
            RunLengthOutputBuffers::new(&values, &lengths, &all_usage),
            4,
        ),
        rle.record_encode(
            &mut commands,
            &input,
            RunLengthOutputBuffers::new(&all_usage, &all_usage, &count),
            4,
        ),
        rle.record_encode(
            &mut commands,
            &input,
            RunLengthOutputBuffers::new(&all_usage, &lengths, &all_usage),
            4,
        ),
        rle.record_encode(
            &mut commands,
            &input,
            RunLengthOutputBuffers::new(&values, &all_usage, &all_usage),
            4,
        ),
    ] {
        assert!(matches!(result, Err(Error::BufferAlias { .. })));
    }

    for result in [
        rle.record_encode_counted(
            &mut commands,
            &input,
            &all_usage,
            RunLengthOutputBuffers::new(&all_usage, &lengths, &count),
            4,
        ),
        rle.record_encode_counted(
            &mut commands,
            &input,
            &all_usage,
            RunLengthOutputBuffers::new(&values, &all_usage, &count),
            4,
        ),
        rle.record_encode_counted(
            &mut commands,
            &input,
            &all_usage,
            RunLengthOutputBuffers::new(&values, &lengths, &all_usage),
            4,
        ),
    ] {
        assert!(matches!(result, Err(Error::BufferAlias { .. })));
    }
}

#[tokio::test]
async fn rejected_encoding_does_not_mutate_the_command_stream() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = storage_buffer(&context.device, "Rejected RLE Input", &[1, 1, 2, 2]);
    let short_values = output_buffer(&context.device, "Rejected Short Values", 3);
    let lengths = output_buffer(&context.device, "Rejected RLE Lengths", 4);
    let count = count_buffer(&context.device, "Rejected RLE Count", 55);
    let mut rle = RunLengthEncoder::from_context(&context);
    let mut commands = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    assert!(matches!(
        rle.record_encode(
            &mut commands,
            &input,
            RunLengthOutputBuffers::new(&short_values, &lengths, &count),
            4,
        ),
        Err(Error::BufferTooSmall { .. })
    ));
    context.queue.submit(Some(commands.finish()));
    assert_eq!(support::read_u32(&context, &count, 1).await, [55]);
}

#[tokio::test]
async fn encoding_rejects_logical_bindings_above_the_device_limit() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let limit = context
        .device
        .limits()
        .max_buffer_size
        .min(context.device.limits().max_storage_buffer_binding_size);
    let requested = limit + size_of::<u32>() as u64;
    let oversized_items = u32::try_from(requested / size_of::<u32>() as u64).unwrap();
    let input = storage_buffer(&context.device, "Tiny Oversized RLE Input", &[1]);
    let values = output_buffer(&context.device, "Tiny Oversized RLE Values", 1);
    let lengths = output_buffer(&context.device, "Tiny Oversized RLE Lengths", 1);
    let count = count_buffer(&context.device, "Tiny Oversized RLE Count", 7);
    let mut commands = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let mut rle = RunLengthEncoder::from_context(&context);

    assert!(matches!(
        rle.record_encode(
            &mut commands,
            &input,
            RunLengthOutputBuffers::new(&values, &lengths, &count),
            oversized_items,
        ),
        Err(Error::BufferLimitExceeded {
            requested: actual,
            limit: actual_limit,
        }) if actual == requested && actual_limit == limit
    ));
}

#[tokio::test]
async fn typed_sort_rle_reduce_composes_at_nonzero_offsets_in_one_submission() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    const START: u32 = 64;
    let input = [5_u32, 1, 5, 2, 2, 1, 9, 5];
    let mut padded_input = vec![0xDEAD_BEEF; START as usize];
    padded_input.extend(input);
    let input_buffer = storage_buffer(&context.device, "Typed RLE Input", &padded_input);
    let sorted_buffer = typed_output_buffer(&context.device, "Typed RLE Sorted", START + 8);
    let values_buffer = typed_output_buffer(&context.device, "Typed RLE Values", START + 8);
    let lengths_buffer = typed_output_buffer(&context.device, "Typed RLE Lengths", START + 8);
    let sum_buffer = typed_output_buffer(&context.device, "Typed RLE Length Sum", START + 1);
    let count_buffer = typed_output_buffer(&context.device, "Typed RLE Count", START + 1);
    let input_view = GpuSlice::from_range(&input_buffer, START..START + 8).unwrap();
    let sorted_view = GpuSliceMut::from_range(&sorted_buffer, START..START + 8).unwrap();
    let values_view = GpuSliceMut::from_range(&values_buffer, START..START + 8).unwrap();
    let lengths_view = GpuSliceMut::from_range(&lengths_buffer, START..START + 8).unwrap();
    let sum_view = GpuSliceMut::from_range(&sum_buffer, START..START + 1).unwrap();
    let run_count = GpuCount::at(&count_buffer, u64::from(START) * 4).unwrap();
    let mut primitives = Primitives::from_context(&context);
    primitives
        .reserve_workspace(
            WorkspaceRequirements::new(8)
                .fixed_sort()
                .run_length_encode()
                .counted_reduce(),
        )
        .unwrap();
    primitives.reserve_count(run_count, 8).unwrap();
    let mut commands = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    {
        let mut recorder = primitives.record(&mut commands);
        let sorted = recorder
            .sort(input_view, sorted_view, SortOptions::default())
            .unwrap();
        let encoded = recorder
            .run_length_encode(sorted, values_view, lengths_view, run_count)
            .unwrap();
        assert!(matches!(encoded.unique_values.extent(), Extent::Gpu(_)));
        assert!(matches!(encoded.run_lengths.extent(), Extent::Gpu(_)));
        recorder
            .reduce(encoded.run_lengths, sum_view, U32Reduction::Sum)
            .unwrap();
    }
    context.queue.submit(Some(commands.finish()));

    assert_eq!(read_range(&context, &count_buffer, START, 1).await, [4]);
    assert_eq!(
        read_range(&context, &values_buffer, START, 4).await,
        [1, 2, 5, 9]
    );
    assert_eq!(
        read_range(&context, &lengths_buffer, START, 4).await,
        [2, 2, 3, 1]
    );
    assert_eq!(read_range(&context, &sum_buffer, START, 1).await, [8]);
}

#[tokio::test]
async fn typed_counted_input_clamps_and_masks_inactive_scan_levels() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    const START: u32 = 64;
    const CAPACITY: u32 = 4_097;
    let input: Vec<_> = (0..CAPACITY).map(|index| index / 5).collect();
    let mut padded_input = vec![0xCAFE_BABE; START as usize];
    padded_input.extend(&input);
    let input_buffer = storage_buffer(&context.device, "Typed Counted RLE Input", &padded_input);
    let input_count_buffer =
        typed_output_buffer(&context.device, "Typed Counted RLE Input Count", START + 1);
    let values_buffer = typed_output_buffer(
        &context.device,
        "Typed Counted RLE Values",
        START + CAPACITY,
    );
    let lengths_buffer = typed_output_buffer(
        &context.device,
        "Typed Counted RLE Lengths",
        START + CAPACITY,
    );
    let run_count_buffer =
        typed_output_buffer(&context.device, "Typed Counted RLE Run Count", START + 1);
    let input_count = GpuCount::at(&input_count_buffer, u64::from(START) * 4).unwrap();
    let run_count = GpuCount::at(&run_count_buffer, u64::from(START) * 4).unwrap();
    let input_view =
        GpuSlice::counted(&input_buffer, START..START + CAPACITY, input_count).unwrap();
    let values_view = GpuSliceMut::from_range(&values_buffer, START..START + CAPACITY).unwrap();
    let lengths_view = GpuSliceMut::from_range(&lengths_buffer, START..START + CAPACITY).unwrap();
    let mut primitives = Primitives::from_context(&context);
    primitives
        .reserve_workspace(WorkspaceRequirements::new(CAPACITY).run_length_encode())
        .unwrap();

    for resident_count in [2_049_u32, CAPACITY + 1_000] {
        context.queue.write_buffer(
            &input_count_buffer,
            u64::from(START) * 4,
            bytemuck::bytes_of(&resident_count),
        );
        let mut commands = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let encoded = primitives
            .record(&mut commands)
            .run_length_encode(input_view, values_view, lengths_view, run_count)
            .unwrap();
        assert!(matches!(encoded.unique_values.extent(), Extent::Gpu(_)));
        context.queue.submit(Some(commands.finish()));

        let active = resident_count.min(CAPACITY) as usize;
        let (expected_values, expected_lengths) = cpu_encode(&input[..active]);
        assert_eq!(
            read_range(&context, &run_count_buffer, START, 1).await,
            [expected_values.len() as u32]
        );
        assert_eq!(
            read_range(&context, &values_buffer, START, expected_values.len()).await,
            expected_values
        );
        assert_eq!(
            read_range(&context, &lengths_buffer, START, expected_lengths.len()).await,
            expected_lengths
        );
    }
}

fn storage_buffer(device: &wgpu::Device, label: &'static str, values: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn output_buffer(device: &wgpu::Device, label: &'static str, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len * size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn count_buffer(device: &wgpu::Device, label: &'static str, value: u32) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(&value),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    })
}

fn typed_output_buffer(device: &wgpu::Device, label: &'static str, len: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: u64::from(len) * size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

async fn read_range(
    context: &lampshade::Context,
    source: &wgpu::Buffer,
    start: u32,
    len: usize,
) -> Vec<u32> {
    let bytes = (len * size_of::<u32>()) as u64;
    let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Typed RLE Range Readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut commands = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    commands.copy_buffer_to_buffer(source, u64::from(start) * 4, &staging, 0, bytes);
    let submission = context.queue.submit(Some(commands.finish()));
    let slice = staging.slice(..);
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
    let output = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    staging.unmap();
    output
}
