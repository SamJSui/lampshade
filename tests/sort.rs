mod support;

use wgpu_algorithms::Sorter;

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
    let mut sorter = Sorter::new(&context);
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
        let actual = sorter
            .sort_radix(&input)
            .await
            .expect("GPU radix sort failed");
        assert_eq!(actual, cpu_sort(&input), "sort mismatch for size {size}");
    }

    let edge_values = [u32::MAX, 0, u32::MAX, 1, 0, 42, 42];
    assert_eq!(
        sorter
            .sort_radix(&edge_values)
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
    let mut sorter = Sorter::new(&context);

    for (case, size) in [8, 65_537, 257, 4_097, 3].into_iter().enumerate() {
        let input = support::random_u32(size, case as u64 + 100);
        let actual = sorter
            .sort_radix(&input)
            .await
            .expect("GPU radix sort failed");
        assert_eq!(actual, cpu_sort(&input), "sort mismatch for size {size}");
    }
}

#[tokio::test]
async fn adaptive_sort_is_correct_on_both_sides_of_the_threshold() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = Sorter::new(&context);

    for (case, size) in [999_999, 1_000_000].into_iter().enumerate() {
        let input = support::random_u32(size, case as u64 + 200);
        let actual = sorter.sort(&input).await.expect("adaptive sort failed");
        assert_eq!(actual, cpu_sort(&input), "sort mismatch for size {size}");
    }
}

#[tokio::test]
async fn sort_without_readback_leaves_the_result_on_the_gpu() {
    let Some(context) = support::gpu_context().await else {
        return;
    };
    let mut sorter = Sorter::new(&context);
    let input = [9, 1, 4, 1, u32::MAX, 0];
    let output = sorter.sort_resident(&input).expect("GPU radix sort failed");
    let actual = support::read_u32(&context, output, input.len()).await;
    assert_eq!(actual, cpu_sort(&input));
}
