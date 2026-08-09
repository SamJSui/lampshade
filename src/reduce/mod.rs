mod pipeline;
mod reducer;

pub use reducer::Reducer;

/// An associative reduction over unsigned 32-bit values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum U32Reduction {
    /// Wrapping unsigned addition.
    Sum,
    /// The smallest input value.
    Min,
    /// The largest input value.
    Max,
}

impl U32Reduction {
    /// Returns the result of reducing an empty input.
    pub const fn identity(self) -> u32 {
        match self {
            Self::Sum | Self::Max => 0,
            Self::Min => u32::MAX,
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Min => "min",
            Self::Max => "max",
        }
    }

    pub(crate) const fn pass_label(self) -> &'static str {
        match self {
            Self::Sum => "Sum Reduction",
            Self::Min => "Minimum Reduction",
            Self::Max => "Maximum Reduction",
        }
    }

    pub(crate) const fn identity_offset(self) -> u64 {
        match self {
            Self::Sum => 0,
            Self::Min => 4,
            Self::Max => 8,
        }
    }
}
