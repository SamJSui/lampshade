/// Aligns a size to the WebGPU requirement (256 bytes for Uniforms, 4 for storage)
pub const fn align_to(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

pub fn checked_align_to(value: u64, alignment: u64) -> Result<u64, crate::Error> {
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded & !(alignment - 1))
        .ok_or(crate::Error::SizeOverflow)
}

/// Calculates how many workgroups to dispatch
pub fn calc_groups(total_items: u32, block_size: u32) -> u32 {
    total_items.div_ceil(block_size)
}

pub fn checked_u32(value: u64) -> Result<u32, crate::Error> {
    u32::try_from(value).map_err(|_| crate::Error::ElementCountTooLarge { count: value })
}

pub fn checked_byte_size(elements: u64, element_size: u64) -> Result<u64, crate::Error> {
    elements
        .checked_mul(element_size)
        .ok_or(crate::Error::SizeOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculates_dispatch_groups() {
        assert_eq!(calc_groups(0, 256), 0);
        assert_eq!(calc_groups(1, 256), 1);
        assert_eq!(calc_groups(256, 256), 1);
        assert_eq!(calc_groups(257, 256), 2);
    }

    #[test]
    fn checked_alignment_rejects_overflow() {
        assert!(matches!(
            checked_align_to(u64::MAX, 256),
            Err(crate::Error::SizeOverflow)
        ));
    }

    #[test]
    fn checked_element_count_rejects_values_above_u32() {
        let count = u64::from(u32::MAX) + 1;
        assert!(matches!(
            checked_u32(count),
            Err(crate::Error::ElementCountTooLarge { count: actual }) if actual == count
        ));
    }

    #[test]
    fn checked_byte_size_rejects_overflow() {
        assert!(matches!(
            checked_byte_size(u64::MAX, 4),
            Err(crate::Error::SizeOverflow)
        ));
    }
}
