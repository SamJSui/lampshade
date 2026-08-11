use lampshade::{Context, KeyValue, KeyValueCompactor};

fn main() {
    env_logger::init();
    pollster::block_on(run());
}

async fn run() {
    println!("Initializing WGPU Context...");
    let context = Context::init()
        .await
        .expect("failed to create WGPU context");
    let mut compactor = KeyValueCompactor::from_context(&context);

    let input = [
        KeyValue::new(2, 10),
        KeyValue::new(1, 20),
        KeyValue::new(2, 30),
        KeyValue::new(1, 40),
    ];
    let mask = [1_u32, 0, 1, 1];
    let output = compactor
        .compact(&input, &mask)
        .await
        .expect("GPU key-value compaction failed");

    println!("Input:  {input:?}");
    println!("Mask:   {mask:?}");
    println!("Output: {output:?}");
    assert_eq!(
        output,
        [
            KeyValue::new(2, 10),
            KeyValue::new(2, 30),
            KeyValue::new(1, 40),
        ]
    );
    println!("Stable key-value stream compaction verified successfully!");
}
