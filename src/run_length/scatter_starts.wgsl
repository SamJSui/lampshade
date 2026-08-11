@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> run_starts: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<u32>;
@group(0) @binding(5) var<storage, read> input_count: array<u32>;

const BLOCK_SIZE: u32 = 256u;

fn active_items() -> u32 {
    return min(input_count[0], arrayLength(&input));
}

fn item_index(group_id: vec3<u32>, local_id: vec3<u32>, groups_x: u32) -> u32 {
    let capacity_items = arrayLength(&input);
    let flat_group_id = group_id.y * groups_x + group_id.x;
    let total_groups = capacity_items / BLOCK_SIZE
        + select(0u, 1u, capacity_items % BLOCK_SIZE != 0u);
    if (flat_group_id >= total_groups) {
        return capacity_items;
    }
    return flat_group_id * BLOCK_SIZE + local_id.x;
}

@compute @workgroup_size(BLOCK_SIZE)
fn scatter_starts(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let index = item_index(group_id, local_id, num_workgroups.x);
    let count = active_items();
    if (index >= count) {
        return;
    }

    let is_head = index == 0u || input[index] != input[index - 1u];
    if (is_head) {
        let run_index = offsets[index];
        run_starts[run_index] = index;
    }
}
