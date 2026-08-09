use wgpu_primitives::{Context, Reducer};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let context = Context::init()
        .await
        .expect("failed to create wgpu context");
    let mut reducer = Reducer::from_context(&context);
    let input = [3_u32, 1, 4, 2];

    let sum = reducer.sum(&input).await.expect("GPU sum failed");
    let min = reducer.min(&input).await.expect("GPU minimum failed");
    let max = reducer.max(&input).await.expect("GPU maximum failed");

    assert_eq!(sum, 10);
    assert_eq!(min, 1);
    assert_eq!(max, 4);
    println!("input={input:?} sum={sum} min={min} max={max}");
}
