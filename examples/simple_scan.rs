use wgpu_primitives::{context::Context, scan::Scanner};

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

    let result = scanner.scan(&input).await.expect("GPU scan failed");

    println!("Output: {:?}", result);

    let expected = vec![1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(result, expected, "GPU scan returned an incorrect result");
    println!("Scan Verified Successfully!");
}
