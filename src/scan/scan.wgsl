@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

@group(0) @binding(1)
var<storage, read_write> aux: array<u32>;

var<workgroup> temp: array<u32, {{BLOCK_SIZE}}>;

const VT: u32 = {{VT}}u;
const BLOCK_SIZE: u32 = {{BLOCK_SIZE}}u;
const ITEMS_PER_BLOCK: u32 = VT * BLOCK_SIZE;

@compute @workgroup_size(BLOCK_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let tid = local_id.x;
    let flat_group_id = group_id.y * 65535u + group_id.x;
    let group_base = flat_group_id * ITEMS_PER_BLOCK;
    let thread_base = group_base + (tid * VT);

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

    // Hillis-Steele Scan (Shared Mem)
    temp[tid] = my_sum;
    workgroupBarrier();

    for (var offset = 1u; offset < BLOCK_SIZE; offset <<= 1u) {
        var val = 0u;
        if (tid >= offset) { val = temp[tid - offset]; }
        workgroupBarrier();
        if (tid >= offset) { temp[tid] += val; }
        workgroupBarrier();
    }

    let group_prefix = temp[tid];
    let thread_start_prefix = group_prefix - my_sum;

    var running_prefix = thread_start_prefix;
    for (var i = 0u; i < VT; i++) {
        let idx = thread_base + i;
        if (idx < arrayLength(&data)) {
            running_prefix += my_vals[i];
            data[idx] = running_prefix;
        }
    }

    if (tid == (BLOCK_SIZE - 1u)) {
        if (flat_group_id < arrayLength(&aux)) {
            aux[flat_group_id] = temp[BLOCK_SIZE - 1u];
        }
    }
}