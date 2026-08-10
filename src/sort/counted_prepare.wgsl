struct DispatchConfig {
    capacity_items: u32,
    items_per_block: u32,
    max_workgroups_x: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> item_count: array<u32>;
@group(0) @binding(1) var<storage, read_write> dispatch_args: array<u32>;
@group(0) @binding(2) var<uniform> config: DispatchConfig;

fn divide_round_up(value: u32, divisor: u32) -> u32 {
    return value / divisor + select(0u, 1u, value % divisor != 0u);
}

@compute @workgroup_size(1)
fn main() {
    let items = min(item_count[0], config.capacity_items);
    let workgroups = divide_round_up(items, config.items_per_block);
    dispatch_args[0] = min(workgroups, config.max_workgroups_x);
    dispatch_args[1] = divide_round_up(workgroups, config.max_workgroups_x);
    dispatch_args[2] = 1u;
}
