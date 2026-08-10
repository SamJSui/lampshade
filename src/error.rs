use std::fmt;

/// Errors returned while preparing or executing a GPU algorithm.
#[derive(Debug)]
pub enum Error {
    RequestAdapter(wgpu::RequestAdapterError),
    RequestDevice(wgpu::RequestDeviceError),
    BufferMap(wgpu::BufferAsyncError),
    MapRange(wgpu::MapRangeError),
    DevicePoll(wgpu::PollError),
    ReadbackChannelClosed,
    TimestampQueriesUnsupported,
    ElementCountTooLarge {
        count: u64,
    },
    RadixElementCountLimitExceeded {
        count: u32,
        limit: u32,
    },
    InvalidKeyBits {
        bits: u32,
    },
    KeyExceedsBitRange {
        key: u32,
        bits: u32,
    },
    CompactionLengthMismatch {
        input: usize,
        mask: usize,
    },
    InvalidCompactionFlag {
        index: usize,
        value: u32,
    },
    InvalidHistogramBinCount {
        bins: u32,
        max: u32,
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
    BufferRangeOutOfBounds {
        name: &'static str,
        offset: u64,
        size: u64,
        buffer_size: u64,
    },
    MisalignedBufferOffset {
        name: &'static str,
        offset: u64,
        alignment: u64,
    },
    UnsupportedDynamicExtent {
        operation: &'static str,
    },
    MissingBufferUsage {
        name: &'static str,
        required: wgpu::BufferUsages,
        actual: wgpu::BufferUsages,
    },
    BufferAlias {
        first: &'static str,
        second: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestAdapter(error) => write!(f, "failed to request a GPU adapter: {error}"),
            Self::RequestDevice(error) => write!(f, "failed to request a GPU device: {error}"),
            Self::BufferMap(error) => write!(f, "failed to map a GPU readback buffer: {error}"),
            Self::MapRange(error) => write!(f, "failed to access a mapped GPU buffer: {error}"),
            Self::DevicePoll(error) => write!(f, "failed while waiting for GPU work: {error}"),
            Self::ReadbackChannelClosed => {
                f.write_str("GPU readback completed without reporting its mapping result")
            }
            Self::TimestampQueriesUnsupported => {
                f.write_str("the selected GPU adapter does not support timestamp queries")
            }
            Self::ElementCountTooLarge { count } => {
                write!(f, "element count {count} exceeds the supported u32 range")
            }
            Self::RadixElementCountLimitExceeded { count, limit } => write!(
                f,
                "element count {count} exceeds the optimized radix-sort limit of {limit}"
            ),
            Self::InvalidKeyBits { bits } => {
                write!(
                    f,
                    "radix-sort key width must be at most 32 bits, got {bits}"
                )
            }
            Self::KeyExceedsBitRange { key, bits } => write!(
                f,
                "key {key} does not fit in the declared {bits}-bit radix-sort range"
            ),
            Self::CompactionLengthMismatch { input, mask } => write!(
                f,
                "compaction input and mask lengths differ: input has {input} items, mask has {mask}"
            ),
            Self::InvalidCompactionFlag { index, value } => write!(
                f,
                "compaction mask value at index {index} must be 0 or 1, got {value}"
            ),
            Self::InvalidHistogramBinCount { bins, max } => write!(
                f,
                "histogram bin count must be between 1 and {max}, got {bins}"
            ),
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
            Self::BufferRangeOutOfBounds {
                name,
                offset,
                size,
                buffer_size,
            } => write!(
                f,
                "{name} range at byte offset {offset} with size {size} exceeds buffer size {buffer_size}"
            ),
            Self::MisalignedBufferOffset {
                name,
                offset,
                alignment,
            } => write!(
                f,
                "{name} byte offset {offset} is not aligned to {alignment} bytes"
            ),
            Self::UnsupportedDynamicExtent { operation } => {
                write!(f, "{operation} currently requires a CPU-known input extent")
            }
            Self::MissingBufferUsage {
                name,
                required,
                actual,
            } => write!(
                f,
                "{name} buffer is missing usage {required:?}; actual usage is {actual:?}"
            ),
            Self::BufferAlias { first, second } => {
                write!(f, "{first} and {second} must be distinct buffers")
            }
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

impl From<wgpu::MapRangeError> for Error {
    fn from(error: wgpu::MapRangeError) -> Self {
        Self::MapRange(error)
    }
}

impl From<wgpu::PollError> for Error {
    fn from(error: wgpu::PollError) -> Self {
        Self::DevicePoll(error)
    }
}
