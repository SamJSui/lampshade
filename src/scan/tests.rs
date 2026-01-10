#[cfg(test)]
mod tests {
    use crate::common;
    use crate::context::Context;
    use crate::scan::Scanner;

    #[tokio::test]
    async fn test_scan() {
        let ctx = Context::init().await.unwrap();
        let mut scanner = Scanner::new(&ctx);

        let n = 1_000_000;
        let input: Vec<u32> = (0..n).map(|_| rand::random::<u32>() % 100).collect();

        let cpu_result: Vec<u32> = input
            .iter()
            .scan(0, |state, &x| {
                *state += x;
                Some(*state)
            })
            .collect();

        let buf_src = common::buffers::create_storage_buffer(&ctx.device, &input);
        let buf_dst = common::buffers::create_empty_storage_buffer(&ctx.device, (n * 4) as u64);

        scanner.scan_gpu_to_gpu(&buf_src, &buf_dst).await;

        let gpu_result =
            common::buffers::download_buffer(&ctx.device, &ctx.queue, &buf_dst, (n * 4) as u64)
                .await;

        assert_eq!(cpu_result, gpu_result, "GPU Scan result matches CPU");
    }
}
