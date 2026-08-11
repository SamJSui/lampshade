use lampshade::{context::Context, scan::Scanner};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    println!("Initializing WGPU Context...");
    let ctx = Context::init()
        .await
        .expect("Failed to create WGPU context");

    let mut scanner = Scanner::from_context(&ctx);

    let input = vec![1, 1, 1, 1, 1, 1, 1, 1];
    println!("Input: {:?}", input);

    let inclusive = scanner.scan(&input).await.expect("GPU scan failed");
    let exclusive = scanner
        .scan_exclusive(&input)
        .await
        .expect("GPU exclusive scan failed");

    println!("Inclusive: {:?}", inclusive);
    println!("Exclusive: {:?}", exclusive);

    assert_eq!(inclusive, [1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(exclusive, [0, 1, 2, 3, 4, 5, 6, 7]);
    println!("Scans verified successfully!");
}
