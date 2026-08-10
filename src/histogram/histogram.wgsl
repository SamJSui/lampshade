struct Params {
    num_items: u32,
    bin_count: u32,
    groups_x: u32,
    workgroups: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

const BLOCK_SIZE: u32 = {{BLOCK_SIZE}}u;
const ITEMS_PER_THREAD: u32 = {{VT}}u;
const ITEMS_PER_WORKGROUP: u32 = BLOCK_SIZE * ITEMS_PER_THREAD;

var<workgroup> local_bins: array<atomic<u32>, {{MAX_BINS}}>;

@compute @workgroup_size(BLOCK_SIZE)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let thread = local_id.x;
    let flat_group = group_id.y * params.groups_x + group_id.x;
    if (flat_group >= params.workgroups) {
        return;
    }

    atomicStore(&local_bins[thread], 0u);
    workgroupBarrier();

    let group_base = flat_group * ITEMS_PER_WORKGROUP;
    for (var item = 0u; item < ITEMS_PER_THREAD; item++) {
        let index = group_base + thread + item * BLOCK_SIZE;
        if (index < params.num_items) {
            let bin = input[index];
            if (bin < params.bin_count) {
                atomicAdd(&local_bins[bin], 1u);
            }
        }
    }
    workgroupBarrier();

    if (thread < params.bin_count) {
        let count = atomicLoad(&local_bins[thread]);
        if (count != 0u) {
            atomicAdd(&output[thread], count);
        }
    }
}
