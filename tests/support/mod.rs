#![allow(dead_code)]

use std::sync::OnceLock;

use futures::channel::oneshot;
use lampshade::{Context, Error};
use rand::{Rng, SeedableRng, rngs::StdRng};

enum SharedContext {
    Ready(Box<Context>),
    Unavailable(String),
    Failed(String),
}

static GPU_CONTEXT: OnceLock<SharedContext> = OnceLock::new();
static PORTABLE_GPU_CONTEXT: OnceLock<SharedContext> = OnceLock::new();

pub async fn gpu_context() -> Option<Context> {
    shared_context(
        GPU_CONTEXT.get_or_init(|| match pollster::block_on(Context::init()) {
            Ok(context) => SharedContext::Ready(Box::new(context)),
            Err(Error::RequestAdapter(error)) => SharedContext::Unavailable(error.to_string()),
            Err(error) => SharedContext::Failed(error.to_string()),
        }),
        "GPU",
    )
}

fn shared_context(context: &SharedContext, kind: &str) -> Option<Context> {
    match context {
        SharedContext::Ready(context) => Some(Context {
            adapter_info: context.adapter_info.clone(),
            device: context.device.clone(),
            queue: context.queue.clone(),
        }),
        SharedContext::Unavailable(error) if gpu_tests_required() => {
            panic!("{kind} test adapter is required but unavailable: {error}");
        }
        SharedContext::Unavailable(error) => {
            eprintln!("skipping {kind} test because no adapter is available: {error}");
            None
        }
        SharedContext::Failed(error) => {
            panic!("failed to initialize the {kind} test context: {error}")
        }
    }
}

fn gpu_tests_required() -> bool {
    std::env::var("LAMPSHADE_REQUIRE_GPU_TESTS")
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

pub async fn gpu_context_without_optional_features() -> Option<Context> {
    shared_context(
        PORTABLE_GPU_CONTEXT.get_or_init(|| {
            pollster::block_on(async {
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
                    Err(error) => return SharedContext::Unavailable(error.to_string()),
                };
                let adapter_info = adapter.get_info();
                match adapter
                    .request_device(&wgpu::DeviceDescriptor {
                        label: Some("Portable Test Device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: adapter.limits(),
                        memory_hints: wgpu::MemoryHints::Performance,
                        ..Default::default()
                    })
                    .await
                {
                    Ok((device, queue)) => SharedContext::Ready(Box::new(Context {
                        adapter_info,
                        device,
                        queue,
                    })),
                    Err(error) => SharedContext::Failed(error.to_string()),
                }
            })
        }),
        "portable GPU",
    )
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

    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission.clone()),
            timeout: None,
        })
        .expect("GPU test copy poll failed");

    let slice = staging.slice(..);
    let (sender, receiver) = oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
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
