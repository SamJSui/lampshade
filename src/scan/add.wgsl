@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> aux: array<u32>;

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
    
    if (flat_group_id == 0u) { return; }

    let group_base = flat_group_id * ITEMS_PER_BLOCK;
    let thread_base = group_base + (tid * VT);
    let add_val = aux[flat_group_id - 1u];

    for (var i = 0u; i < VT; i++) {
        let idx = thread_base + i;
        if (idx < arrayLength(&data)) {
            data[idx] += add_val;
        }
    }
}