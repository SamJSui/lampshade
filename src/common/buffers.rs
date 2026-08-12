use futures::channel::oneshot;
use wgpu::util::DeviceExt;

use crate::Error;

#[derive(Clone, Copy)]
pub(crate) struct BufferRange<'a> {
    pub(crate) buffer: &'a wgpu::Buffer,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

impl<'a> BufferRange<'a> {
    pub(crate) fn whole(buffer: &'a wgpu::Buffer) -> Self {
        Self {
            buffer,
            offset: 0,
            size: buffer.size(),
        }
    }

    pub(crate) fn new(
        buffer: &'a wgpu::Buffer,
        offset: u64,
        size: u64,
        name: &'static str,
    ) -> Result<Self, Error> {
        let end = offset
            .checked_add(size)
            .ok_or(Error::BufferRangeOutOfBounds {
                name,
                offset,
                size,
                buffer_size: buffer.size(),
            })?;
        if end > buffer.size() {
            return Err(Error::BufferRangeOutOfBounds {
                name,
                offset,
                size,
                buffer_size: buffer.size(),
            });
        }
        Ok(Self {
            buffer,
            offset,
            size,
        })
    }

    pub(crate) fn validate(
        self,
        name: &'static str,
        required_bytes: u64,
        required_usage: wgpu::BufferUsages,
    ) -> Result<(), Error> {
        if self.size < required_bytes {
            return Err(Error::BufferTooSmall {
                name,
                required: required_bytes,
                actual: self.size,
            });
        }
        let actual_usage = self.buffer.usage();
        if !actual_usage.contains(required_usage) {
            return Err(Error::MissingBufferUsage {
                name,
                required: required_usage,
                actual: actual_usage,
            });
        }
        Ok(())
    }

    pub(crate) fn validate_storage_offset(
        self,
        device: &wgpu::Device,
        name: &'static str,
    ) -> Result<(), Error> {
        let alignment = u64::from(device.limits().min_storage_buffer_offset_alignment);
        if !self.offset.is_multiple_of(alignment) {
            return Err(Error::MisalignedBufferOffset {
                name,
                offset: self.offset,
                alignment,
            });
        }
        Ok(())
    }

    pub(crate) fn validate_storage_binding_size(
        self,
        device: &wgpu::Device,
        required_bytes: u64,
    ) -> Result<(), Error> {
        validate_storage_binding_size(device, required_bytes)
    }

    pub(crate) fn binding(self, size: u64) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.buffer,
            offset: self.offset,
            size: wgpu::BufferSize::new(size),
        })
    }
}

pub(crate) fn validate_storage_binding_size(
    device: &wgpu::Device,
    requested: u64,
) -> Result<(), Error> {
    let limits = device.limits();
    validate_storage_binding_size_against_limits(
        requested,
        limits.max_buffer_size,
        limits.max_storage_buffer_binding_size,
    )
}

fn validate_storage_binding_size_against_limits(
    requested: u64,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
) -> Result<(), Error> {
    let limit = max_buffer_size.min(max_storage_buffer_binding_size);
    if requested > limit {
        Err(Error::BufferLimitExceeded { requested, limit })
    } else {
        Ok(())
    }
}

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

pub async fn download_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    element_count: usize,
) -> Result<Vec<T>, Error> {
    if element_count == 0 {
        return Ok(Vec::new());
    }

    let size_bytes =
        crate::common::math::checked_byte_size(element_count as u64, size_of::<T>() as u64)?;

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

    // Finish the device-local copy before asking the backend to map the
    // staging allocation. Some integrated Vulkan drivers otherwise expose a
    // previously recycled staging allocation through the completed callback.
    device.poll(wgpu::PollType::Wait {
        submission_index: Some(index.clone()),
        timeout: None,
    })?;

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    device.poll(wgpu::PollType::Wait {
        submission_index: None,
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

#[cfg(test)]
mod tests {
    use super::validate_storage_binding_size_against_limits;
    use crate::Error;

    #[test]
    fn storage_binding_size_uses_the_stricter_device_limit() {
        assert!(validate_storage_binding_size_against_limits(512, 1_024, 512).is_ok());
        assert!(matches!(
            validate_storage_binding_size_against_limits(513, 1_024, 512),
            Err(Error::BufferLimitExceeded {
                requested: 513,
                limit: 512
            })
        ));
        assert!(matches!(
            validate_storage_binding_size_against_limits(257, 256, 512),
            Err(Error::BufferLimitExceeded {
                requested: 257,
                limit: 256
            })
        ));
    }
}
