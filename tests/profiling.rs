mod support;

use wgpu::util::DeviceExt;
use wgpu_primitives::{Error, KeyValue, KeyValueSorter, Scanner, Sorter};

fn timestamp_queries_available(context: &wgpu_primitives::Context) -> bool {
    context
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
}

fn storage_buffer(
    device: &wgpu::Device,
    label: &'static str,
    data: &[impl bytemuck::Pod],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn output_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

#[tokio::test]
async fn profiles_prefix_scan_dispatches() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    if !timestamp_queries_available(&context) {
        eprintln!("skipping timestamp profile test because the adapter lacks timestamp queries");
        return;
    }

    let input = support::random_u32(8_193, 0x710F);
    let gpu_input = storage_buffer(&context.device, "Profile Scan Input", &input);
    let gpu_output = output_buffer(&context.device, "Profile Scan Output", gpu_input.size());
    let mut scanner = Scanner::from_context(&context);

    let profile = scanner
        .profile_scan_gpu_to_gpu(&gpu_input, &gpu_output, input.len() as u32)
        .await
        .expect("profiled scan failed");
    let actual = support::read_u32(&context, &gpu_output, input.len()).await;
    let mut running = 0_u32;
    let expected: Vec<_> = input
        .iter()
        .map(|value| {
            running = running.wrapping_add(*value);
            running
        })
        .collect();

    assert_eq!(actual, expected);
    assert!(!profile.spans.is_empty());
    assert!(
        profile
            .spans
            .iter()
            .any(|span| span.label == "scan.level.0")
    );
    assert!(profile.spans.iter().any(|span| span.label == "scan.add.0"));
    assert!(profile.gpu_elapsed >= profile.dispatch_time);
}

#[tokio::test]
async fn profiles_key_and_key_value_radix_stages() {
    const PORTABLE_RADIX_PASS_COUNT: usize = 16;

    let Some(context) = support::gpu_context().await else {
        return;
    };
    if !timestamp_queries_available(&context) {
        eprintln!("skipping timestamp profile test because the adapter lacks timestamp queries");
        return;
    }

    let input = support::random_u32(8_193, 0x50A7);
    let gpu_input = storage_buffer(&context.device, "Profile Sort Input", &input);
    let gpu_output = output_buffer(&context.device, "Profile Sort Output", gpu_input.size());
    let mut sorter = Sorter::from_context(&context);
    let profile = sorter
        .profile_sort_gpu_to_gpu(&gpu_input, &gpu_output, input.len() as u32)
        .await
        .expect("profiled key sort failed");
    let actual = support::read_u32(&context, &gpu_output, input.len()).await;
    let mut expected = input.clone();
    expected.sort_unstable();

    assert_eq!(actual, expected);
    assert_eq!(
        profile
            .spans
            .iter()
            .filter(|span| span.label.ends_with(".reduce"))
            .count(),
        PORTABLE_RADIX_PASS_COUNT
    );
    assert_eq!(
        profile
            .spans
            .iter()
            .filter(|span| span.label.ends_with(".scatter"))
            .count(),
        PORTABLE_RADIX_PASS_COUNT
    );
    assert!(
        profile
            .spans
            .iter()
            .any(|span| span.label == "radix.00.scan.level.0")
    );

    let pairs: Vec<_> = input
        .iter()
        .enumerate()
        .map(|(index, key)| KeyValue::new(key & 0xff, index as u32))
        .collect();
    let pair_input = storage_buffer(&context.device, "Profile Pair Input", &pairs);
    let pair_output = output_buffer(&context.device, "Profile Pair Output", pair_input.size());
    let mut pair_sorter = KeyValueSorter::from_context(&context);
    let pair_profile = pair_sorter
        .profile_sort_gpu_to_gpu(&pair_input, &pair_output, pairs.len() as u32)
        .await
        .expect("profiled key-value sort failed");
    let actual: Vec<KeyValue> = support::read_pod(&context, &pair_output, pairs.len()).await;
    let mut expected = pairs.clone();
    expected.sort_by_key(|item| item.key);

    assert_eq!(actual, expected);
    let pair_reduce_passes = pair_profile
        .spans
        .iter()
        .filter(|span| span.label.ends_with(".reduce"))
        .count();
    assert!(matches!(pair_reduce_passes, 8 | PORTABLE_RADIX_PASS_COUNT));
    assert_eq!(
        pair_profile
            .spans
            .iter()
            .filter(|span| span.label.ends_with(".scatter"))
            .count(),
        pair_reduce_passes
    );
}

#[tokio::test]
async fn trivial_profiles_complete_without_timestamp_dispatches() -> Result<(), Error> {
    let Some(context) = support::gpu_context().await else {
        return Ok(());
    };
    let buffer = output_buffer(&context.device, "Empty Profile Buffer", 4);
    let mut scanner = Scanner::from_context(&context);
    let profile = scanner.profile_scan_gpu_to_gpu(&buffer, &buffer, 0).await?;

    assert!(profile.spans.is_empty());
    assert!(profile.gpu_elapsed.is_zero());

    let input = storage_buffer(&context.device, "Single Scan Input", &[42_u32]);
    let output = output_buffer(&context.device, "Single Scan Output", 4);
    let profile = scanner.profile_scan_gpu_to_gpu(&input, &output, 1).await?;
    assert_eq!(support::read_u32(&context, &output, 1).await, [42]);
    assert!(profile.spans.is_empty());
    Ok(())
}
