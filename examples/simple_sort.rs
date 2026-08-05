use wgpu_algorithms::{context::Context, sort::Sorter};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    println!("Initializing Context...");
    let ctx = Context::init().await.expect("Failed");
    let mut sorter = Sorter::from_context(&ctx);

    let input = vec![10, 5, 8, 1, 2, 9, 3, 4, 7, 6, 0, 11];
    println!("Input:  {:?}", input);

    let result = sorter.sort(&input).await.expect("GPU sort failed");
    println!("Output: {:?}", result);

    let mut expected = input.clone();
    expected.sort();

    let result_slice = &result[0..input.len()];
    assert_eq!(result_slice, expected.as_slice());
    println!("Radix Sort Verified!");
}
