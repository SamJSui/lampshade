/// A lazily allocated GPU buffer that grows to the exact requested capacity.
///
/// Primitive owners keep this state across calls, avoiding allocation on a
/// steady-size workload without exposing scratch buffers in the public API.
#[derive(Default)]
pub(crate) struct ReusableBuffer {
    buffer: Option<wgpu::Buffer>,
    capacity_bytes: u64,
}

impl ReusableBuffer {
    pub(crate) fn ensure(
        &mut self,
        device: &wgpu::Device,
        required_bytes: u64,
        label: &'static str,
        usage: wgpu::BufferUsages,
    ) {
        if self.buffer.is_none() || required_bytes > self.capacity_bytes {
            self.buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: required_bytes,
                usage,
                mapped_at_creation: false,
            }));
            self.capacity_bytes = required_bytes;
        }
    }

    pub(crate) fn get(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }
}
