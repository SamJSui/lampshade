mod support;

use wgpu::util::DeviceExt;
use wgpu_primitives::{Compactor, Error, KeyValue, KeyValueSorter, Scanner, Sorter};

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
async fn profiles_stream_compaction_scan_and_scatter() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    if !timestamp_queries_available(&context) {
        eprintln!("skipping timestamp profile test because the adapter lacks timestamp queries");
        return;
    }

    let input = support::random_u32(8_193, 0x0C0A_0AC7);
    let mask: Vec<_> = input.iter().map(|value| value & 1).collect();
    let gpu_input = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Profile Compaction Input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let gpu_mask = storage_buffer(&context.device, "Profile Compaction Mask", &mask);
    let gpu_output = output_buffer(
        &context.device,
        "Profile Compaction Output",
        gpu_input.size(),
    );
    let gpu_count = output_buffer(&context.device, "Profile Compaction Count", 4);
    let mut compactor = Compactor::from_context(&context);

    let profile = compactor
        .profile_compact_gpu_to_gpu(
            &gpu_input,
            &gpu_mask,
            &gpu_output,
            &gpu_count,
            input.len() as u32,
        )
        .await
        .expect("profiled compaction failed");
    let expected = cpu_compact(&input, &mask);

    assert_eq!(
        support::read_u32(&context, &gpu_count, 1).await,
        [expected.len() as u32]
    );
    assert_eq!(
        support::read_u32(&context, &gpu_output, expected.len()).await,
        expected
    );
    assert!(
        profile
            .spans
            .iter()
            .any(|span| span.label == "compact.scan.level.0")
    );
    assert!(
        profile
            .spans
            .iter()
            .any(|span| span.label == "compact.scatter")
    );
    assert!(profile.gpu_elapsed >= profile.dispatch_time);
}

fn cpu_compact(input: &[u32], mask: &[u32]) -> Vec<u32> {
    input
        .iter()
        .zip(mask)
        .filter_map(|(&value, &keep)| (keep == 1).then_some(value))
        .collect()
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

    let bounded_input: Vec<_> = input.iter().map(|key| key & 0x1f).collect();
    let bounded_gpu_input = storage_buffer(
        &context.device,
        "Bounded Profile Sort Input",
        &bounded_input,
    );
    let bounded_gpu_output = output_buffer(
        &context.device,
        "Bounded Profile Sort Output",
        bounded_gpu_input.size(),
    );
    let bounded_profile = sorter
        .profile_sort_gpu_to_gpu_with_key_bits(
            &bounded_gpu_input,
            &bounded_gpu_output,
            bounded_input.len() as u32,
            5,
        )
        .await
        .expect("profiled bounded key sort failed");
    let mut bounded_expected = bounded_input.clone();
    bounded_expected.sort_unstable();

    assert_eq!(
        support::read_u32(&context, &bounded_gpu_output, bounded_input.len()).await,
        bounded_expected
    );
    assert_eq!(
        bounded_profile
            .spans
            .iter()
            .filter(|span| span.label.ends_with(".reduce"))
            .count(),
        3
    );
    assert_eq!(
        bounded_profile
            .spans
            .iter()
            .filter(|span| span.label.ends_with(".scatter"))
            .count(),
        3
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
    let pair_scatter_passes = pair_profile
        .spans
        .iter()
        .filter(|span| span.label.ends_with(".scatter"))
        .count();
    let pair_histogram_passes = pair_profile
        .spans
        .iter()
        .filter(|span| span.label.ends_with(".histogram"))
        .count();
    let pair_prefix_passes = pair_profile
        .spans
        .iter()
        .filter(|span| span.label.ends_with(".prefix"))
        .count();
    if pair_histogram_passes == 1 {
        assert_eq!(pair_prefix_passes, 1);
        assert_eq!(pair_reduce_passes, 0);
        assert_eq!(pair_scatter_passes, 4);

        for (key_bits, expected_scatter_passes) in [(8, 1), (16, 2), (24, 3), (32, 4)] {
            let bounded_pair_profile = pair_sorter
                .profile_sort_gpu_to_gpu_with_key_bits(
                    &pair_input,
                    &pair_output,
                    pairs.len() as u32,
                    key_bits,
                )
                .await
                .expect("profiled bounded key-value sort failed");
            assert_eq!(
                bounded_pair_profile
                    .spans
                    .iter()
                    .filter(|span| span.label.ends_with(".histogram"))
                    .count(),
                1
            );
            assert_eq!(
                bounded_pair_profile
                    .spans
                    .iter()
                    .filter(|span| span.label.ends_with(".prefix"))
                    .count(),
                1
            );
            assert_eq!(
                bounded_pair_profile
                    .spans
                    .iter()
                    .filter(|span| span.label.ends_with(".scatter"))
                    .count(),
                expected_scatter_passes
            );
        }
    } else {
        assert_eq!(pair_prefix_passes, 0);
        assert!(matches!(pair_reduce_passes, 8 | PORTABLE_RADIX_PASS_COUNT));
        assert_eq!(pair_scatter_passes, pair_reduce_passes);
    }
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

    let mask = storage_buffer(&context.device, "Empty Compaction Mask", &[0_u32]);
    let compacted = output_buffer(&context.device, "Empty Compaction Output", 4);
    let count = output_buffer(&context.device, "Empty Compaction Count", 4);
    let mut compactor = Compactor::from_context(&context);
    let profile = compactor
        .profile_compact_gpu_to_gpu(&input, &mask, &compacted, &count, 0)
        .await?;
    assert!(profile.spans.is_empty());
    assert!(profile.gpu_elapsed.is_zero());
    assert_eq!(support::read_u32(&context, &count, 1).await, [0]);
    Ok(())
}
