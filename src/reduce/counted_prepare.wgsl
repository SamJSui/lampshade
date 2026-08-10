struct Config {
    capacity_items: u32,
    pass_count: u32,
    items_per_block: u32,
    plan_stride_words: u32,
    max_workgroups_x: u32,
    _padding_0: u32,
    _padding_1: u32,
    _padding_2: u32,
}

@group(0) @binding(0) var<storage, read> item_count: array<u32>;
@group(0) @binding(1) var<storage, read_write> plans: array<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch_args: array<u32>;
@group(0) @binding(3) var<uniform> config: Config;

fn divide_round_up(value: u32, divisor: u32) -> u32 {
    return value / divisor + select(0u, 1u, value % divisor != 0u);
}

@compute @workgroup_size(1)
fn main() {
    var input_items = min(item_count[0], config.capacity_items);

    for (var level = 0u; level < config.pass_count; level++) {
        let output_items = divide_round_up(input_items, config.items_per_block);
        let plan_offset = level * config.plan_stride_words;
        plans[plan_offset] = input_items;
        plans[plan_offset + 1u] = output_items;

        let args_offset = level * 3u;
        dispatch_args[args_offset] = min(output_items, config.max_workgroups_x);
        dispatch_args[args_offset + 1u] = divide_round_up(
            output_items,
            config.max_workgroups_x
        );
        dispatch_args[args_offset + 2u] = 1u;
        input_items = output_items;
    }
}
