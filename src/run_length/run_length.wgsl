struct Params {
    capacity_items: u32,
    fixed_items: u32,
    groups_x: u32,
    counted: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> heads: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> unique_values: array<u32>;
@group(0) @binding(4) var<storage, read_write> run_lengths: array<u32>;
@group(0) @binding(5) var<storage, read> input_count: array<u32>;
@group(0) @binding(6) var<storage, read_write> run_count: array<u32>;
@group(0) @binding(7) var<uniform> params: Params;

const BLOCK_SIZE: u32 = 256u;

fn active_items() -> u32 {
    if (params.counted != 0u) {
        return min(input_count[0], params.capacity_items);
    }
    return params.fixed_items;
}

fn item_index(group_id: vec3<u32>, local_id: vec3<u32>) -> u32 {
    let flat_group_id = group_id.y * params.groups_x + group_id.x;
    let total_groups = params.capacity_items / BLOCK_SIZE
        + select(0u, 1u, params.capacity_items % BLOCK_SIZE != 0u);
    if (flat_group_id >= total_groups) {
        return params.capacity_items;
    }
    return flat_group_id * BLOCK_SIZE + local_id.x;
}

@compute @workgroup_size(BLOCK_SIZE)
fn mark_heads(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let index = item_index(group_id, local_id);
    if (index >= params.capacity_items) {
        return;
    }

    let count = active_items();
    var is_head = 0u;
    if (index < count) {
        if (index == 0u) {
            is_head = 1u;
        } else if (input[index] != input[index - 1u]) {
            is_head = 1u;
        }
    }
    heads[index] = is_head;
}

@compute @workgroup_size(BLOCK_SIZE)
fn scatter_starts(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let index = item_index(group_id, local_id);
    let count = active_items();
    if (index >= count) {
        return;
    }

    let is_head = heads[index];
    if (is_head != 0u) {
        let run_index = offsets[index] + is_head - 1u;
        unique_values[run_index] = input[index];
        run_lengths[run_index] = index;
    }
}

@compute @workgroup_size(BLOCK_SIZE)
fn finalize_lengths(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let index = item_index(group_id, local_id);
    let count = active_items();
    if (index >= count) {
        return;
    }

    let is_last = index == count - 1u;
    var is_tail = is_last;
    if (!is_last) {
        is_tail = heads[index + 1u] != 0u;
    }
    if (is_tail) {
        let run_index = offsets[index] + heads[index] - 1u;
        let start = run_lengths[run_index];
        var end = count;
        if (!is_last) {
            end = index + 1u;
        }
        run_lengths[run_index] = end - start;
        if (is_last) {
            run_count[0] = run_index + 1u;
        }
    }
}
