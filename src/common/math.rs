/// Aligns a size to the WebGPU requirement (256 bytes for Uniforms, 4 for storage)
pub const fn align_to(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) & !(alignment - 1)
}

/// Calculates how many workgroups to dispatch
pub const fn calc_groups(total_items: u32, block_size: u32) -> u32 {
    (total_items + block_size - 1) / block_size
}
