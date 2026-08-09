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

pub async fn gpu_context_without_optional_features() -> Option<Context> {
    let descriptor = wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    }
    .with_env();
    let instance = wgpu::Instance::new(descriptor);
    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
    {
        Ok(adapter) => adapter,
        Err(error) => {
            eprintln!("skipping portable GPU test because no adapter is available: {error}");
            return None;
        }
    };
    let adapter_info = adapter.get_info();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Portable Test Device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .expect("failed to request portable GPU test device");

    Some(Context {
        adapter_info,
        device,
        queue,
    })
}

pub fn random_u32(len: usize, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random()).collect()
}

pub async fn read_u32(context: &Context, buffer: &wgpu::Buffer, len: usize) -> Vec<u32> {
    read_pod(context, buffer, len).await
}

pub async fn read_pod<T: bytemuck::Pod>(
    context: &Context,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Vec<T> {
    if len == 0 {
        return Vec::new();
    }

    let size = (len * size_of::<T>()) as u64;
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

    let result = {
        let mapped = slice
            .get_mapped_range()
            .expect("test readback buffer is mapped");
        bytemuck::cast_slice(&mapped).to_vec()
    };
    staging.unmap();
    result
}
