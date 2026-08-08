struct Params {
    num_items: u32,
    groups_x: u32,
    operation: u32,
    field: u32,
    lower: u32,
    upper: u32,
    _padding_0: u32,
    _padding_1: u32,
}

struct KeyValue {
    key: u32,
    value: u32,
}

@group(0) @binding(0) var<storage, read> input: array<{{ITEM_TYPE}}>;
@group(0) @binding(1) var<storage, read_write> mask: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 256u;

fn matches(value: u32) -> bool {
    switch params.operation {
        case 0u: { return value == params.lower; }
        case 1u: { return value != params.lower; }
        case 2u: { return value < params.lower; }
        case 3u: { return value <= params.lower; }
        case 4u: { return value > params.lower; }
        case 5u: { return value >= params.lower; }
        case 6u: { return value >= params.lower && value <= params.upper; }
        default: { return false; }
    }
}

@compute @workgroup_size(BLOCK_SIZE)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let flat_group_id = group_id.y * params.groups_x + group_id.x;
    let index = flat_group_id * BLOCK_SIZE + local_id.x;
    if (index >= params.num_items) {
        return;
    }

    let item = input[index];
    let value = {{VALUE_EXPRESSION}};
    mask[index] = select(0u, 1u, matches(value));
}
