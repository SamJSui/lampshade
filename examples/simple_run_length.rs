use lampshade::{Context, RunLengthEncoder};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let context = Context::init()
        .await
        .expect("failed to create wgpu context");
    let mut encoder = RunLengthEncoder::from_context(&context);
    let input = [4_u32, 4, 4, 1, 1, 7, 4, 4];
    let (values, lengths) = encoder.encode(&input).await.expect("GPU RLE failed");

    assert_eq!(values, [4, 1, 7, 4]);
    assert_eq!(lengths, [3, 2, 1, 2]);
    println!("input={input:?} values={values:?} lengths={lengths:?}");
}
