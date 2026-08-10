@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read> plan: array<u32>;

const VT: u32 = {{VT}}u;
const BLOCK_SIZE: u32 = {{BLOCK_SIZE}}u;
const ITEMS_PER_BLOCK: u32 = VT * BLOCK_SIZE;
const IDENTITY: u32 = {{IDENTITY}};

var<workgroup> partials: array<u32, {{BLOCK_SIZE}}>;

fn combine(lhs: u32, rhs: u32) -> u32 {
    return {{COMBINE}};
}

@compute @workgroup_size(BLOCK_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let thread = local_id.x;
    let flat_group = group_id.y * {{MAX_WORKGROUPS_X}}u + group_id.x;
    let input_items = plan[0];
    let output_items = plan[1];
    if (flat_group >= output_items) {
        return;
    }
    let group_base = flat_group * ITEMS_PER_BLOCK;

    var value = IDENTITY;
    for (var item = 0u; item < VT; item++) {
        let index = group_base + thread + item * BLOCK_SIZE;
        if (index < input_items) {
            value = combine(value, input[index]);
        }
    }

    partials[thread] = value;
    workgroupBarrier();

    for (var stride = BLOCK_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (thread < stride) {
            partials[thread] = combine(partials[thread], partials[thread + stride]);
        }
        workgroupBarrier();
    }

    if (thread == 0u) {
        output[flat_group] = partials[0];
    }
}
