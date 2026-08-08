use wgpu_primitives::{Compactor, Context};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    println!("Initializing WGPU Context...");
    let context = Context::init()
        .await
        .expect("failed to create WGPU context");
    let mut compactor = Compactor::from_context(&context);

    let input = [40_u32, 10, 30, 20];
    let mask = [0_u32, 1, 1, 0];
    let output = compactor
        .compact(&input, &mask)
        .await
        .expect("GPU compaction failed");

    println!("Input:  {input:?}");
    println!("Mask:   {mask:?}");
    println!("Output: {output:?}");
    assert_eq!(output, [10, 30]);
    println!("Stable stream compaction verified successfully!");
}
