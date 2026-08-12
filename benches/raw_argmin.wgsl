struct KeyValue {
    key: u32,
    value: u32,
}

struct Params {
    input_items: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<KeyValue>;
@group(0) @binding(1) var<storage, read_write> output: array<KeyValue>;
@group(0) @binding(2) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 256u;
var<workgroup> partials: array<KeyValue, BLOCK_SIZE>;

fn better(lhs: KeyValue, rhs: KeyValue) -> KeyValue {
    if (lhs.key < rhs.key || (lhs.key == rhs.key && lhs.value <= rhs.value)) {
        return lhs;
    }
    return rhs;
}

@compute @workgroup_size(BLOCK_SIZE)
fn reduce_min(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let group = group_id.y * num_workgroups.x + group_id.x;
    let index = group * BLOCK_SIZE + local_id.x;
    var candidate = KeyValue(0xffffffffu, 0xffffffffu);
    if (index < params.input_items) {
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
        output[group] = partials[0];
    }
}
