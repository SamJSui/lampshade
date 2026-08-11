mod support;

use lampshade::{Error, Scanner};
use wgpu::util::DeviceExt;

const SCAN_SIZES: [usize; 20] = [
    0, 1, 2, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1_023, 1_024, 1_025, 2_047, 2_048, 2_049,
    4_097, 17,
];

fn cpu_inclusive_scan(input: &[u32]) -> Vec<u32> {
    input
        .iter()
        .scan(0_u32, |sum, value| {
            *sum = sum.wrapping_add(*value);
            Some(*sum)
        })
        .collect()
}

fn cpu_exclusive_scan(input: &[u32]) -> Vec<u32> {
    input
        .iter()
        .scan(0_u32, |sum, value| {
            let prefix = *sum;
            *sum = sum.wrapping_add(*value);
            Some(prefix)
        })
        .collect()
}

fn scan_input(case: usize, size: usize) -> Vec<u32> {
    match case % 3 {
        0 => vec![1; size],
        1 => (0..size).map(|index| (index % 7) as u32).collect(),
        _ => support::random_u32(size, case as u64)
            .into_iter()
            .map(|value| value & 0x0f)
            .collect(),
    }
}

#[tokio::test]
async fn scan_matches_cpu_across_boundaries_and_patterns() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut scanner = Scanner::from_context(&context);
    for (case, size) in SCAN_SIZES.into_iter().enumerate() {
        let input = scan_input(case, size);
        let actual = scanner.scan(&input).await.expect("GPU scan failed");
        assert_eq!(
            actual,
            cpu_inclusive_scan(&input),
            "scan mismatch for size {size}"
        );
    }

    let overflow = [u32::MAX, 1, u32::MAX, 2];
    assert_eq!(
        scanner.scan(&overflow).await.expect("GPU scan failed"),
        cpu_inclusive_scan(&overflow)
    );
}

#[tokio::test]
async fn exclusive_scan_matches_cpu_across_boundaries_and_patterns() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut scanner = Scanner::from_context(&context);

    for (case, size) in SCAN_SIZES.into_iter().enumerate() {
        let input = scan_input(case, size);
        let actual = scanner
            .scan_exclusive(&input)
            .await
            .expect("GPU exclusive scan failed");
        assert_eq!(
            actual,
            cpu_exclusive_scan(&input),
            "exclusive scan mismatch for size {size}"
        );
    }

    let overflow = [u32::MAX, 1, u32::MAX, 2];
    assert_eq!(
        scanner
            .scan_exclusive(&overflow)
            .await
            .expect("GPU exclusive scan failed"),
        cpu_exclusive_scan(&overflow)
    );
}

#[tokio::test]
async fn exclusive_scan_matches_cpu_across_multiple_hierarchy_levels() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut scanner = Scanner::from_context(&context);

    // This crosses at least three hierarchy levels on both the portable
    // 2,048-item blocks and the subgroup path's 256-item blocks.
    let input = scan_input(2, 4_194_305);
    let expected = cpu_exclusive_scan(&input);
    let actual = scanner
        .scan_exclusive(&input)
        .await
        .expect("GPU exclusive scan failed");

    if let Some(index) = actual
        .iter()
        .zip(&expected)
        .position(|(actual, expected)| actual != expected)
    {
        panic!(
            "exclusive scan mismatch at index {index}: expected {}, got {}",
            expected[index], actual[index]
        );
    }
}

#[tokio::test]
async fn portable_scan_fallback_matches_cpu_without_subgroups() {
    let Some(context) = support::gpu_context_without_optional_features().await else {
        return;
    };
    assert!(!context.device.features().contains(wgpu::Features::SUBGROUP));
    let mut scanner = Scanner::from_context(&context);
    let input = scan_input(1, 4_097);

    assert_eq!(
        scanner
            .scan_exclusive(&input)
            .await
            .expect("portable GPU exclusive scan failed"),
        cpu_exclusive_scan(&input)
    );
}

#[tokio::test]
async fn scan_uses_the_explicit_logical_length() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut scanner = Scanner::from_context(&context);
    let input = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Padded Scan Input"),
            contents: bytemuck::cast_slice(&[1_u32, 2, 3, 999]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Padded Scan Output"),
        size: input.size(),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    scanner
        .scan_gpu_to_gpu(&input, &output, 3)
        .expect("GPU scan failed");
    assert_eq!(support::read_u32(&context, &output, 3).await, [1, 3, 6]);

    scanner
        .scan_exclusive_gpu_to_gpu(&input, &output, 3)
        .expect("GPU exclusive scan failed");
    assert_eq!(support::read_u32(&context, &output, 3).await, [0, 1, 3]);
}

#[tokio::test]
async fn record_exclusive_scan_composes_multiple_invocations_in_one_encoder() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut scanner = Scanner::from_context(&context);
    let first = [3_u32, 1, 4, 1];
    let second = [2_u32, 7, 1];
    let first_input = create_scan_input(&context.device, &first);
    let second_input = create_scan_input(&context.device, &second);
    let first_output = create_scan_output(&context.device, first.len());
    let second_output = create_scan_output(&context.device, second.len());
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    scanner
        .record_exclusive_scan(
            &mut encoder,
            &first_input,
            &first_output,
            first.len() as u32,
        )
        .expect("first exclusive scan recording failed");
    scanner
        .record_exclusive_scan(
            &mut encoder,
            &second_input,
            &second_output,
            second.len() as u32,
        )
        .expect("second exclusive scan recording failed");
    context.queue.submit(Some(encoder.finish()));

    assert_eq!(
        support::read_u32(&context, &first_output, first.len()).await,
        cpu_exclusive_scan(&first)
    );
    assert_eq!(
        support::read_u32(&context, &second_output, second.len()).await,
        cpu_exclusive_scan(&second)
    );
}

#[tokio::test]
async fn scan_rejects_invalid_buffer_contracts() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut scanner = Scanner::from_context(&context);
    let missing_copy_source = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Invalid Scan Input"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scan Output"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let error = scanner
        .record_scan(&mut encoder, &missing_copy_source, &output, 4)
        .expect_err("missing COPY_SRC must be rejected");
    assert!(matches!(error, Error::MissingBufferUsage { .. }));

    let valid_input = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scan Input"),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let short_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Short Scan Output"),
        size: 12,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let error = scanner
        .record_scan(&mut encoder, &valid_input, &short_output, 4)
        .expect_err("short output must be rejected");
    assert!(matches!(error, Error::BufferTooSmall { .. }));
}

fn create_scan_input(device: &wgpu::Device, input: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scan Test Input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn create_scan_output(device: &wgpu::Device, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Scan Test Output"),
        size: (len * size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
