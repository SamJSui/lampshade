use lampshade::{ArgminByKey, Context, KeyValue};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let context = Context::init()
        .await
        .expect("failed to create wgpu context");
    let mut selector = ArgminByKey::from_context(&context);
    let candidates = [
        KeyValue::new(42, 0),
        KeyValue::new(7, 3),
        KeyValue::new(7, 1),
        KeyValue::new(19, 2),
    ];

    let best = selector.argmin(&candidates).await.expect("argmin failed");
    assert_eq!(best, KeyValue::new(7, 1));
    println!("best key={} value={}", best.key, best.value);
}
