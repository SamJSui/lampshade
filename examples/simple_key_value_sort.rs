use wgpu_primitives::{Context, KeyValue, KeyValueSorter};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    let context = Context::init()
        .await
        .expect("failed to create wgpu context");
    let mut sorter = KeyValueSorter::from_context(&context);
    let input = [
        KeyValue::new(2, 10),
        KeyValue::new(1, 20),
        KeyValue::new(2, 30),
        KeyValue::new(1, 40),
    ];

    let sorted = sorter
        .sort(&input)
        .await
        .expect("GPU key-value sort failed");

    assert_eq!(
        sorted,
        [
            KeyValue::new(1, 20),
            KeyValue::new(1, 40),
            KeyValue::new(2, 10),
            KeyValue::new(2, 30),
        ]
    );
    println!("{sorted:?}");
}
