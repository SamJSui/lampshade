#[cfg(test)]
mod tests {
    use crate::context::Context;
    use crate::sort::sorter::Sorter;

    #[tokio::test]
    async fn test_sort() {
        let ctx = Context::init().await.unwrap();
        let mut sorter = Sorter::new(&ctx);

        let n = 1_234_567;
        let input: Vec<u32> = (0..n).map(|_| rand::random::<u32>()).collect();

        let mut cpu_sorted = input.clone();
        cpu_sorted.sort_unstable();

        let gpu_sorted = sorter.sort_radix(&input).await;

        assert_eq!(cpu_sorted, gpu_sorted, "GPU Sort result matches CPU");
    }
}
