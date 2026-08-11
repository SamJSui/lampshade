use lampshade::{Context, Histogram};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let context = Context::init()
        .await
        .expect("failed to create wgpu context");
    let histogram = Histogram::from_context(&context);
    let input = [0_u32, 3, 1, 3, 7, 3, 9];

    let bins = histogram
        .histogram(&input, 8)
        .await
        .expect("GPU histogram failed");

    // Values 0 through 7 are counted. The value 9 is outside the requested
    // range and is intentionally ignored.
    assert_eq!(bins, [1, 1, 0, 3, 0, 0, 0, 1]);
    println!("input={input:?} bins={bins:?}");
}
