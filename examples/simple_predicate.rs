use wgpu_primitives::{Compactor, Context, MaskGenerator, U32Predicate};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    println!("Initializing WGPU Context...");
    let context = Context::init()
        .await
        .expect("failed to create WGPU context");
    let generator = MaskGenerator::from_context(&context);
    let mut compactor = Compactor::from_context(&context);

    let input = [4_u32, 17, 9, 22, 11, 3];
    let predicate = U32Predicate::GreaterThanOrEqual(10);
    let mask = generator
        .mask(&input, predicate)
        .await
        .expect("GPU predicate mask failed");
    let output = compactor
        .compact(&input, &mask)
        .await
        .expect("GPU compaction failed");

    println!("Input:  {input:?}");
    println!("Mask:   {mask:?}");
    println!("Output: {output:?}");
    assert_eq!(mask, [0, 1, 0, 1, 1, 0]);
    assert_eq!(output, [17, 22, 11]);
    println!("Predicate mask and stable compaction verified successfully!");
}
