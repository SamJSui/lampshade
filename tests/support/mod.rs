#![allow(dead_code)]

use futures::channel::oneshot;
use rand::{Rng, SeedableRng, rngs::StdRng};
use wgpu_primitives::{Context, Error};

pub async fn gpu_context() -> Option<Context> {
    match Context::init().await {
        Ok(context) => Some(context),
        Err(Error::RequestAdapter(error)) => {
            eprintln!("skipping GPU test because no adapter is available: {error}");
            None
        }
        Err(error) => panic!("failed to initialize the GPU test context: {error}"),
    }
}

pub fn random_u32(len: usize, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random()).collect()
}

pub async fn read_u32(context: &Context, buffer: &wgpu::Buffer, len: usize) -> Vec<u32> {
    if len == 0 {
        return Vec::new();
    }

    let size = (len * size_of::<u32>()) as u64;
    let staging = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    let submission = context.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .expect("GPU test poll failed");
    receiver
        .await
        .expect("GPU test readback channel closed")
        .expect("GPU test readback map failed");

    let result = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    staging.unmap();
    result
}
