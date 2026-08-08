mod generator;
mod pipeline;

pub use generator::MaskGenerator;

/// A comparison that maps one `u32` value to a compaction flag.
///
/// A matching value produces `1`; a non-matching value produces `0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum U32Predicate {
    /// Matches values equal to the operand.
    Equal(u32),
    /// Matches values unequal to the operand.
    NotEqual(u32),
    /// Matches values strictly less than the operand.
    LessThan(u32),
    /// Matches values less than or equal to the operand.
    LessThanOrEqual(u32),
    /// Matches values strictly greater than the operand.
    GreaterThan(u32),
    /// Matches values greater than or equal to the operand.
    GreaterThanOrEqual(u32),
    /// Matches values in the closed interval `min..=max`.
    ///
    /// A `min` greater than `max` matches no values.
    BetweenInclusive { min: u32, max: u32 },
}

impl U32Predicate {
    pub(crate) const fn encode(self) -> (u32, u32, u32) {
        match self {
            Self::Equal(value) => (0, value, 0),
            Self::NotEqual(value) => (1, value, 0),
            Self::LessThan(value) => (2, value, 0),
            Self::LessThanOrEqual(value) => (3, value, 0),
            Self::GreaterThan(value) => (4, value, 0),
            Self::GreaterThanOrEqual(value) => (5, value, 0),
            Self::BetweenInclusive { min, max } => (6, min, max),
        }
    }
}

/// The field tested when generating a mask for [`crate::KeyValue`] records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyValueField {
    /// Tests [`crate::KeyValue::key`].
    Key,
    /// Tests [`crate::KeyValue::value`].
    Value,
}

impl KeyValueField {
    pub(crate) const fn encode(self) -> u32 {
        match self {
            Self::Key => 0,
            Self::Value => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{KeyValueField, U32Predicate};

    #[test]
    fn encodes_predicates_for_the_shader() {
        assert_eq!(U32Predicate::Equal(7).encode(), (0, 7, 0));
        assert_eq!(U32Predicate::NotEqual(7).encode(), (1, 7, 0));
        assert_eq!(U32Predicate::LessThan(7).encode(), (2, 7, 0));
        assert_eq!(U32Predicate::LessThanOrEqual(7).encode(), (3, 7, 0));
        assert_eq!(U32Predicate::GreaterThan(7).encode(), (4, 7, 0));
        assert_eq!(U32Predicate::GreaterThanOrEqual(7).encode(), (5, 7, 0));
        assert_eq!(
            U32Predicate::BetweenInclusive { min: 3, max: 9 }.encode(),
            (6, 3, 9)
        );
    }

    #[test]
    fn encodes_key_value_fields_for_the_shader() {
        assert_eq!(KeyValueField::Key.encode(), 0);
        assert_eq!(KeyValueField::Value.encode(), 1);
    }
}
