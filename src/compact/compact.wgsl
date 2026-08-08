struct Params {
    num_items: u32,
    groups_x: u32,
    _padding_0: u32,
    _padding_1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> mask: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_count: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 256u;

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

    let keep = mask[index];
    if (keep == 1u) {
        output[offsets[index]] = input[index];
    }

    if (index + 1u == params.num_items) {
        output_count[0] = offsets[index] + keep;
    }
}
