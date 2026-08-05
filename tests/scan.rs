mod support;

use wgpu::util::DeviceExt;
use wgpu_primitives::{Error, Scanner};

fn cpu_scan(input: &[u32]) -> Vec<u32> {
    input
        .iter()
        .scan(0_u32, |sum, value| {
            *sum = sum.wrapping_add(*value);
            Some(*sum)
        })
        .collect()
}

#[tokio::test]
async fn scan_matches_cpu_across_boundaries_and_patterns() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut scanner = Scanner::from_context(&context);
    let sizes = [
        0, 1, 2, 31, 32, 33, 127, 128, 129, 511, 512, 513, 1_023, 1_024, 1_025, 2_047, 2_048,
        2_049, 4_097, 17,
    ];

    for (case, size) in sizes.into_iter().enumerate() {
        let input = match case % 3 {
            0 => vec![1; size],
            1 => (0..size).map(|index| (index % 7) as u32).collect(),
            _ => support::random_u32(size, case as u64)
                .into_iter()
                .map(|value| value & 0x0f)
                .collect(),
        };
        let actual = scanner.scan(&input).await.expect("GPU scan failed");
        assert_eq!(actual, cpu_scan(&input), "scan mismatch for size {size}");
    }

    let overflow = [u32::MAX, 1, u32::MAX, 2];
    assert_eq!(
        scanner.scan(&overflow).await.expect("GPU scan failed"),
        cpu_scan(&overflow)
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
