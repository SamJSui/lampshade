use futures::channel::oneshot;
use wgpu::util::DeviceExt;

use crate::Error;

// --- Allocation ---

pub fn create_storage_buffer<T: bytemuck::Pod>(device: &wgpu::Device, data: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

pub fn create_empty_storage_buffer(device: &wgpu::Device, size_bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Empty Storage"),
        size: size_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

// --- Bindings ---

pub fn bind_entry(binding: u32, read_only: bool, uniform: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: if uniform {
                wgpu::BufferBindingType::Uniform
            } else {
                wgpu::BufferBindingType::Storage { read_only }
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// --- IO (Download/Read) ---

pub async fn download_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    size_bytes: u64,
) -> Result<Vec<u32>, Error> {
    if size_bytes == 0 {
        return Ok(Vec::new());
    }

    validate_buffer(
        buffer,
        "readback source",
        size_bytes,
        wgpu::BufferUsages::COPY_SRC,
    )?;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: size_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size_bytes);

    let index = queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    device.poll(wgpu::PollType::Wait {
        submission_index: Some(index),
        timeout: None,
    })?;

    receiver.await.map_err(|_| Error::ReadbackChannelClosed)??;

    let result = {
        let data = buffer_slice.get_mapped_range();
        bytemuck::cast_slice(&data).to_vec()
    };
    staging_buffer.unmap();
    Ok(result)
}

pub fn validate_buffer(
    buffer: &wgpu::Buffer,
    name: &'static str,
    required_bytes: u64,
    required_usage: wgpu::BufferUsages,
) -> Result<(), Error> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(Error::BufferTooSmall {
            name,
            required: required_bytes,
            actual: actual_bytes,
        });
    }

    let actual_usage = buffer.usage();
    if !actual_usage.contains(required_usage) {
        return Err(Error::MissingBufferUsage {
            name,
            required: required_usage,
            actual: actual_usage,
        });
    }

    Ok(())
}
