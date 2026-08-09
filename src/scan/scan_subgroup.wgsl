@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

@group(0) @binding(1)
var<storage, read_write> aux: array<u32>;

var<workgroup> subgroup_prefixes: array<u32, {{BLOCK_SIZE}}>;

const VT: u32 = {{VT}}u;
const BLOCK_SIZE: u32 = {{BLOCK_SIZE}}u;
const ITEMS_PER_BLOCK: u32 = VT * BLOCK_SIZE;
override EXCLUSIVE: bool = false;

@compute @workgroup_size(BLOCK_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) lane: u32,
    @builtin(subgroup_size) subgroup_size: u32
) {
    let tid = local_id.x;
    let flat_group_id = group_id.y * 65535u + group_id.x;
    let group_base = flat_group_id * ITEMS_PER_BLOCK;
    let thread_base = group_base + tid * VT;

    var my_vals: array<u32, {{VT}}>;
    var my_sum = 0u;
    for (var i = 0u; i < VT; i++) {
        let idx = thread_base + i;
        if (idx < arrayLength(&data)) {
            my_vals[i] = data[idx];
            my_sum += my_vals[i];
        } else {
            my_vals[i] = 0u;
        }
    }

    let lane_prefix = subgroupExclusiveAdd(my_sum);
    if (lane == subgroup_size - 1u) {
        subgroup_prefixes[subgroup_id] = lane_prefix + my_sum;
    }
    workgroupBarrier();

    let subgroup_count = (BLOCK_SIZE + subgroup_size - 1u) / subgroup_size;
    if (tid == 0u) {
        var prefix = 0u;
        for (var group = 0u; group < subgroup_count; group++) {
            let total = subgroup_prefixes[group];
            subgroup_prefixes[group] = prefix;
            prefix += total;
        }
    }
    workgroupBarrier();

    var running_prefix = subgroup_prefixes[subgroup_id] + lane_prefix;
    for (var i = 0u; i < VT; i++) {
        let idx = thread_base + i;
        if (idx < arrayLength(&data)) {
            if (EXCLUSIVE) {
                data[idx] = running_prefix;
                running_prefix += my_vals[i];
            } else {
                running_prefix += my_vals[i];
                data[idx] = running_prefix;
            }
        }
    }

    if (tid == BLOCK_SIZE - 1u && flat_group_id < arrayLength(&aux)) {
        aux[flat_group_id] = subgroup_prefixes[subgroup_id] + lane_prefix + my_sum;
    }
}
