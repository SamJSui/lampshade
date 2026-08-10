mod support;

use wgpu::util::DeviceExt;
use wgpu_primitives::{
    Compactor, CountedSortDispatch, Error, GpuCountPlan, MaskGenerator, Reducer, Sorter,
    U32Predicate, U32Reduction,
};

#[tokio::test]
async fn gpu_count_drives_sort_and_reduction_across_boundaries() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let capacity = 4_097_u32;
    let input = support::random_u32(capacity as usize, 0x00C0_1DED);
    let input_buffer = initialized_storage(&context.device, "Counted Input", &input);
    let mut sorter = Sorter::from_context(&context);
    let mut reducer = Reducer::from_context(&context);

    for requested_count in [0, 1, 2_049, capacity, u32::MAX] {
        let count = initialized_storage(&context.device, "GPU Item Count", &[requested_count]);
        let sorted = storage_output(
            &context.device,
            "Counted Sort Output",
            input_buffer.size(),
            wgpu::BufferUsages::COPY_SRC,
        );
        let sum = reduction_output(&context.device, "Counted Sum Output");
        let minimum = reduction_output(&context.device, "Counted Min Output");
        let maximum = reduction_output(&context.device, "Counted Max Output");
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        sorter
            .record_sort_counted(&mut encoder, &input_buffer, &sorted, &count, capacity)
            .expect("counted sort recording failed");
        for (output, operation) in [
            (&sum, U32Reduction::Sum),
            (&minimum, U32Reduction::Min),
            (&maximum, U32Reduction::Max),
        ] {
            reducer
                .record_reduce_counted(&mut encoder, &sorted, output, &count, capacity, operation)
                .expect("counted reduction recording failed");
        }
        context.queue.submit(Some(encoder.finish()));

        let selected = requested_count.min(capacity) as usize;
        let mut expected = input[..selected].to_vec();
        expected.sort_unstable();
        assert_eq!(
            support::read_u32(&context, &sorted, selected).await,
            expected,
            "sort mismatch for GPU count {requested_count}"
        );
        assert_eq!(
            support::read_u32(&context, &sum, 1).await,
            [cpu_reduce(&expected, U32Reduction::Sum)]
        );
        assert_eq!(
            support::read_u32(&context, &minimum, 1).await,
            [cpu_reduce(&expected, U32Reduction::Min)]
        );
        assert_eq!(
            support::read_u32(&context, &maximum, 1).await,
            [cpu_reduce(&expected, U32Reduction::Max)]
        );
    }
}

#[tokio::test]
async fn predicate_compact_sort_reduce_stays_in_one_submission() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = support::random_u32(8_193, 0x051E_C7ED);
    let capacity = input.len() as u32;
    let predicate = U32Predicate::LessThan(1_u32 << 29);
    let expected_selected: Vec<_> = input
        .iter()
        .copied()
        .filter(|value| *value < 1_u32 << 29)
        .collect();
    let input_buffer = initialized_storage(&context.device, "Pipeline Input", &input);
    let mask = storage_output(
        &context.device,
        "Pipeline Mask",
        input_buffer.size(),
        wgpu::BufferUsages::COPY_SRC,
    );
    let compacted = storage_output(
        &context.device,
        "Pipeline Compacted",
        input_buffer.size(),
        wgpu::BufferUsages::empty(),
    );
    let count = storage_output(
        &context.device,
        "Pipeline Count",
        size_of::<u32>() as u64,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    let sorted = storage_output(
        &context.device,
        "Pipeline Sorted",
        input_buffer.size(),
        wgpu::BufferUsages::COPY_SRC,
    );
    let sum = reduction_output(&context.device, "Pipeline Sum");
    let generator = MaskGenerator::from_context(&context);
    let mut compactor = Compactor::from_context(&context);
    let mut sorter = Sorter::from_context(&context);
    let mut reducer = Reducer::from_context(&context);
    let mut expected_sorted = expected_selected;
    expected_sorted.sort_unstable();

    for strategy in [CountedSortDispatch::Indirect, CountedSortDispatch::Capacity] {
        let count_plan =
            GpuCountPlan::new_with_sort_dispatch(&context.device, &count, capacity, strategy)
                .expect("count plan creation failed");
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        generator
            .record_mask(&mut encoder, &input_buffer, &mask, capacity, predicate)
            .expect("predicate recording failed");
        compactor
            .record_compact(
                &mut encoder,
                &input_buffer,
                &mask,
                &compacted,
                &count,
                capacity,
            )
            .expect("compaction recording failed");
        count_plan.record_prepare(&mut encoder);
        sorter
            .record_sort_with_count_plan(&mut encoder, &compacted, &sorted, &count_plan)
            .expect("counted sort recording failed");
        reducer
            .record_reduce_with_count_plan(
                &mut encoder,
                &sorted,
                &sum,
                &count_plan,
                U32Reduction::Sum,
            )
            .expect("counted reduction recording failed");
        context.queue.submit(Some(encoder.finish()));

        let actual_count = support::read_u32(&context, &count, 1).await[0] as usize;
        let actual_sorted = support::read_u32(&context, &sorted, actual_count).await;
        assert_eq!(actual_count, expected_sorted.len());
        assert_eq!(actual_sorted, expected_sorted, "strategy {strategy:?}");
        assert_eq!(
            support::read_u32(&context, &sum, 1).await,
            [cpu_reduce(&expected_sorted, U32Reduction::Sum)],
            "strategy {strategy:?}"
        );
    }
}

#[tokio::test]
async fn immediate_counted_apis_submit_without_a_host_count_readback() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = [9, 4, 7, 1, 99, 98];
    let input_buffer = initialized_storage(&context.device, "Immediate Input", &input);
    // A larger count allocation is valid: the APIs bind exactly the first
    // scalar rather than accidentally exposing the whole buffer to WGSL.
    let count = initialized_storage(&context.device, "Immediate Count", &[4, u32::MAX]);
    let sorted = storage_output(
        &context.device,
        "Immediate Sorted",
        input_buffer.size(),
        wgpu::BufferUsages::COPY_SRC,
    );
    let sum = reduction_output(&context.device, "Immediate Sum");
    let mut sorter = Sorter::from_context(&context);
    let mut reducer = Reducer::from_context(&context);

    sorter
        .sort_counted_gpu_to_gpu(&input_buffer, &sorted, &count, input.len() as u32)
        .expect("immediate counted sort failed");
    reducer
        .reduce_counted_gpu_to_gpu(&sorted, &sum, &count, input.len() as u32, U32Reduction::Sum)
        .expect("immediate counted reduction failed");

    assert_eq!(support::read_u32(&context, &sorted, 4).await, [1, 4, 7, 9]);
    assert_eq!(support::read_u32(&context, &sum, 1).await, [21]);
}

#[tokio::test]
async fn counted_paths_reject_invalid_buffer_contracts() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let input = initialized_storage(&context.device, "Validation Input", &[3, 2, 1, 0]);
    let output = storage_output(
        &context.device,
        "Validation Output",
        input.size(),
        wgpu::BufferUsages::COPY_SRC,
    );
    let count = initialized_storage(&context.device, "Validation Count", &[4]);
    let invalid_count = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Count Missing Storage"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut sorter = Sorter::from_context(&context);
    let mut reducer = Reducer::from_context(&context);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let error = match GpuCountPlan::new(&context.device, &invalid_count, 4) {
        Ok(_) => panic!("a plan count without STORAGE must fail"),
        Err(error) => error,
    };
    assert!(matches!(error, Error::MissingBufferUsage { .. }));

    let error = sorter
        .record_sort_counted(&mut encoder, &input, &output, &invalid_count, 4)
        .expect_err("count without STORAGE must fail");
    assert!(matches!(error, Error::MissingBufferUsage { .. }));

    let error = sorter
        .record_sort_counted(&mut encoder, &output, &output, &count, 4)
        .expect_err("sort buffers must be distinct");
    assert!(matches!(error, Error::BufferAlias { .. }));

    let error = sorter
        .record_sort_counted(&mut encoder, &input, &output, &count, 5)
        .expect_err("capacity beyond the input must fail");
    assert!(matches!(error, Error::BufferTooSmall { .. }));

    let error = sorter
        .record_sort_counted_with_key_bits(&mut encoder, &input, &output, &count, 4, 33)
        .expect_err("key widths above 32 must fail");
    assert!(matches!(error, Error::InvalidKeyBits { bits: 33 }));

    let invalid_reduction_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Reduction Output Missing Copy Destination"),
        size: Reducer::output_buffer_size(),
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let error = reducer
        .record_reduce_counted(
            &mut encoder,
            &input,
            &invalid_reduction_output,
            &count,
            4,
            U32Reduction::Sum,
        )
        .expect_err("reduction output without COPY_DST must fail");
    assert!(matches!(error, Error::MissingBufferUsage { .. }));

    let reduction_output = reduction_output(&context.device, "Validation Reduction Output");
    let error = reducer
        .record_reduce_counted(
            &mut encoder,
            &input,
            &reduction_output,
            &reduction_output,
            4,
            U32Reduction::Sum,
        )
        .expect_err("the count and output must be distinct");
    assert!(matches!(error, Error::BufferAlias { .. }));

    let aliased_plan = GpuCountPlan::new(&context.device, &reduction_output, 4)
        .expect("storage output should be a valid count-plan binding");
    let error = reducer
        .record_reduce_with_count_plan(
            &mut encoder,
            &input,
            &reduction_output,
            &aliased_plan,
            U32Reduction::Sum,
        )
        .expect_err("a plan count must not alias the reduction output");
    assert!(matches!(error, Error::BufferAlias { .. }));

    let sort_aliased_plan = GpuCountPlan::new(&context.device, &output, 4)
        .expect("storage output should be a valid count-plan binding");
    let error = sorter
        .record_sort_with_count_plan(&mut encoder, &input, &output, &sort_aliased_plan)
        .expect_err("a plan count must not alias the sort output");
    assert!(matches!(error, Error::BufferAlias { .. }));
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

fn initialized_storage(device: &wgpu::Device, label: &'static str, values: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn storage_output(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    extra_usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | extra_usage,
        mapped_at_creation: false,
    })
}

fn reduction_output(device: &wgpu::Device, label: &'static str) -> wgpu::Buffer {
    storage_output(
        device,
        label,
        Reducer::output_buffer_size(),
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    )
}
