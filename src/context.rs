use crate::Error;
use wgpu::{AdapterInfo, Backends, Device, Instance, MemoryHints, Queue, RequestAdapterOptions};

pub struct Context {
    pub adapter_info: AdapterInfo,
    pub device: Device,
    pub queue: Queue,
}

impl Context {
    pub async fn init() -> Result<Self, Error> {
        let descriptor = wgpu::InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        }
        .with_env();
        let instance = Instance::new(&descriptor);

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Context Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: MemoryHints::Performance,
                ..Default::default()
            })
            .await?;

        Ok(Self {
            adapter_info,
            device,
            queue,
        })
    }
}
