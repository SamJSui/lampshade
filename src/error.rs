use std::fmt;

/// Errors returned while preparing or executing a GPU algorithm.
#[derive(Debug)]
pub enum Error {
    RequestAdapter(wgpu::RequestAdapterError),
    RequestDevice(wgpu::RequestDeviceError),
    BufferMap(wgpu::BufferAsyncError),
    DevicePoll(wgpu::PollError),
    ReadbackChannelClosed,
    ElementCountTooLarge {
        count: u64,
    },
    SizeOverflow,
    BufferLimitExceeded {
        requested: u64,
        limit: u64,
    },
    BufferTooSmall {
        name: &'static str,
        required: u64,
        actual: u64,
    },
    MissingBufferUsage {
        name: &'static str,
        required: wgpu::BufferUsages,
        actual: wgpu::BufferUsages,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestAdapter(error) => write!(f, "failed to request a GPU adapter: {error}"),
            Self::RequestDevice(error) => write!(f, "failed to request a GPU device: {error}"),
            Self::BufferMap(error) => write!(f, "failed to map a GPU readback buffer: {error}"),
            Self::DevicePoll(error) => write!(f, "failed while waiting for GPU work: {error}"),
            Self::ReadbackChannelClosed => {
                f.write_str("GPU readback completed without reporting its mapping result")
            }
            Self::ElementCountTooLarge { count } => {
                write!(f, "element count {count} exceeds the supported u32 range")
            }
            Self::SizeOverflow => f.write_str("buffer size calculation overflowed"),
            Self::BufferLimitExceeded { requested, limit } => write!(
                f,
                "requested GPU buffer size {requested} exceeds the device limit {limit}"
            ),
            Self::BufferTooSmall {
                name,
                required,
                actual,
            } => write!(
                f,
                "{name} buffer is too small: requires {required} bytes, has {actual} bytes"
            ),
            Self::MissingBufferUsage {
                name,
                required,
                actual,
            } => write!(
                f,
                "{name} buffer is missing usage {required:?}; actual usage is {actual:?}"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<wgpu::RequestAdapterError> for Error {
    fn from(error: wgpu::RequestAdapterError) -> Self {
        Self::RequestAdapter(error)
    }
}

impl From<wgpu::RequestDeviceError> for Error {
    fn from(error: wgpu::RequestDeviceError) -> Self {
        Self::RequestDevice(error)
    }
}

impl From<wgpu::BufferAsyncError> for Error {
    fn from(error: wgpu::BufferAsyncError) -> Self {
        Self::BufferMap(error)
    }
}

impl From<wgpu::PollError> for Error {
    fn from(error: wgpu::PollError) -> Self {
        Self::DevicePoll(error)
    }
}
