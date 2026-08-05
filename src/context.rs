use crate::Error;
use wgpu::{Backends, Device, Instance, MemoryHints, Queue, RequestAdapterOptions};

pub struct Context {
    pub device: Device,
    pub queue: Queue,
}

impl Context {
    pub async fn init() -> Result<Self, Error> {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Context Device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: MemoryHints::Performance,
                ..Default::default()
            })
            .await?;

        Ok(Self { device, queue })
    }
}
