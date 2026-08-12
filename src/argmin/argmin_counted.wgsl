struct KeyValue {
    key: u32,
    value: u32,
}

struct Params {
    capacity: u32,
    level: u32,
    output_capacity: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<KeyValue>;
@group(0) @binding(1) var<storage, read_write> output: array<KeyValue>;
@group(0) @binding(2) var<storage, read> item_count: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

const BLOCK_SIZE: u32 = {{BLOCK_SIZE}}u;
const MAX_WORKGROUPS_X: u32 = {{MAX_WORKGROUPS_X}}u;
const IDENTITY: KeyValue = KeyValue(0xffffffffu, 0xffffffffu);

var<workgroup> partials: array<KeyValue, {{BLOCK_SIZE}}>;
var<workgroup> active_items: u32;

fn better(lhs: KeyValue, rhs: KeyValue) -> KeyValue {
    if (lhs.key < rhs.key || (lhs.key == rhs.key && lhs.value <= rhs.value)) {
        return lhs;
    }
    return rhs;
}

fn level_input_items() -> u32 {
    var items = min(item_count[0], params.capacity);
    for (var level = 0u; level < params.level; level++) {
        items = items / BLOCK_SIZE + select(0u, 1u, items % BLOCK_SIZE != 0u);
    }
    return items;
}

@compute @workgroup_size(BLOCK_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let flat_group = group_id.y * MAX_WORKGROUPS_X + group_id.x;
    if (flat_group >= params.output_capacity) {
        return;
    }
    if (local_id.x == 0u) {
        active_items = level_input_items();
    }
    workgroupBarrier();
    let index = flat_group * BLOCK_SIZE + local_id.x;
    var candidate = IDENTITY;
    if (index < active_items) {
        candidate = input[index];
    }
    partials[local_id.x] = candidate;
    workgroupBarrier();

    for (var stride = BLOCK_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (local_id.x < stride) {
            partials[local_id.x] = better(partials[local_id.x], partials[local_id.x + stride]);
        }
        workgroupBarrier();
    }

    if (local_id.x == 0u) {
        output[flat_group] = partials[0];
    }
}
