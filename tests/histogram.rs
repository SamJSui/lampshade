mod support;

use wgpu::util::DeviceExt;
use wgpu_primitives::{Error, Histogram};

const HISTOGRAM_SIZES: [usize; 13] = [
    0, 1, 2, 255, 256, 257, 2_047, 2_048, 2_049, 4_095, 4_096, 4_097, 8_193,
];

fn input_for(size: usize) -> Vec<u32> {
    let mut state = 0xA11C_E5ED_u32 ^ size as u32;
    (0..size)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state % 320
        })
        .collect()
}

fn cpu_histogram(input: &[u32], bin_count: u32) -> Vec<u32> {
    let mut bins = vec![0_u32; bin_count as usize];
    for &value in input {
        if let Some(bin) = bins.get_mut(value as usize) {
            *bin += 1;
        }
    }
    bins
}

#[tokio::test]
async fn histograms_match_cpu_across_boundaries_and_bin_counts() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let histogram = Histogram::from_context(&context);

    for bin_count in [1, 7, 256] {
        for size in HISTOGRAM_SIZES {
            let input = input_for(size);
            assert_eq!(
                histogram
                    .histogram(&input, bin_count)
                    .await
                    .expect("GPU histogram failed"),
                cpu_histogram(&input, bin_count),
                "histogram mismatch for {size} items and {bin_count} bins"
            );
        }
    }
}

#[tokio::test]
async fn histogram_uses_the_explicit_logical_length_and_clears_output() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let histogram = Histogram::from_context(&context);
    let input = storage_buffer(&context.device, "Padded Histogram Input", &[0, 1, 1, 7]);
    let output = output_buffer(&context.device, "Histogram Output", 4);

    histogram
        .histogram_gpu_to_gpu(&input, &output, 3, 4)
        .expect("GPU histogram failed");
    assert_eq!(support::read_u32(&context, &output, 4).await, [1, 2, 0, 0]);

    histogram
        .histogram_gpu_to_gpu(&input, &output, 0, 4)
        .expect("empty GPU histogram failed");
    assert_eq!(support::read_u32(&context, &output, 4).await, [0, 0, 0, 0]);
}

#[tokio::test]
async fn recorded_histograms_compose_in_one_encoder() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let histogram = Histogram::from_context(&context);
    let first = [0_u32, 0, 3, 5, 8];
    let second = [1_u32, 2, 2, 2, 3, 3];
    let first_input = storage_buffer(&context.device, "First Histogram Input", &first);
    let second_input = storage_buffer(&context.device, "Second Histogram Input", &second);
    let first_output = output_buffer(&context.device, "First Histogram Output", 6);
    let second_output = output_buffer(&context.device, "Second Histogram Output", 4);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    histogram
        .record_histogram(
            &mut encoder,
            &first_input,
            &first_output,
            first.len() as u32,
            6,
        )
        .expect("first histogram recording failed");
    histogram
        .record_histogram(
            &mut encoder,
            &second_input,
            &second_output,
            second.len() as u32,
            4,
        )
        .expect("second histogram recording failed");
    context.queue.submit(Some(encoder.finish()));

    assert_eq!(
        support::read_u32(&context, &first_output, 6).await,
        cpu_histogram(&first, 6)
    );
    assert_eq!(
        support::read_u32(&context, &second_output, 4).await,
        cpu_histogram(&second, 4)
    );
}

#[tokio::test]
async fn histogram_rejects_invalid_buffer_contracts() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let histogram = Histogram::from_context(&context);
    assert!(matches!(
        Histogram::output_buffer_size(0),
        Err(Error::InvalidHistogramBinCount { .. })
    ));
    assert!(matches!(
        Histogram::output_buffer_size(Histogram::MAX_BINS + 1),
        Err(Error::InvalidHistogramBinCount { .. })
    ));

    let invalid_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Invalid Histogram Input"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let output = output_buffer(&context.device, "Histogram Output", 4);
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let error = histogram
        .record_histogram(&mut encoder, &invalid_input, &output, 4, 4)
        .expect_err("missing STORAGE input usage must be rejected");
    assert!(matches!(error, Error::MissingBufferUsage { .. }));

    let input = storage_buffer(&context.device, "Histogram Input", &[0, 1, 2, 3]);
    let short_output = output_buffer(&context.device, "Short Histogram Output", 3);
    let error = histogram
        .record_histogram(&mut encoder, &input, &short_output, 4, 4)
        .expect_err("short output must be rejected");
    assert!(matches!(error, Error::BufferTooSmall { .. }));

    let no_copy_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Histogram Output Missing Copy Destination"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let error = histogram
        .record_histogram(&mut encoder, &input, &no_copy_output, 4, 4)
        .expect_err("missing COPY_DST output usage must be rejected");
    assert!(matches!(error, Error::MissingBufferUsage { .. }));

    let aliased = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Aliased Histogram Buffer"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let error = histogram
        .record_histogram(&mut encoder, &aliased, &aliased, 4, 4)
        .expect_err("aliased histogram buffers must be rejected");
    assert!(matches!(error, Error::BufferAlias { .. }));
}

fn storage_buffer(device: &wgpu::Device, label: &'static str, input: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn output_buffer(device: &wgpu::Device, label: &'static str, bins: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: Histogram::output_buffer_size(bins).expect("histogram output size overflow"),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
