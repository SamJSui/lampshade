struct KeyValue {
    key: u32,
    value: u32,
}

struct Uniforms {
    num_items: u32,
    num_tiles: u32,
    generation: u32,
    bit_index: u32,
    pass_count: u32,
    _padding_0: u32,
    _padding_1: u32,
    _padding_2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<KeyValue>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

const BLOCK_SIZE: u32 = 256u;
const MAX_PASS_COUNT: u32 = 4u;
const BUCKET_COUNT: u32 = 256u;

var<workgroup> local_histogram: array<atomic<u32>, MAX_PASS_COUNT * BUCKET_COUNT>;

@compute @workgroup_size(BLOCK_SIZE)
fn main_histogram(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) workgroup_count: vec3<u32>,
) {
    let tid = local_id.x;
    let histogram_entries = uniforms.pass_count * BUCKET_COUNT;
    for (var slot = tid; slot < histogram_entries; slot += BLOCK_SIZE) {
        atomicStore(&local_histogram[slot], 0u);
    }
    workgroupBarrier();

    let stride = workgroup_count.x * BLOCK_SIZE;
    var index = group_id.x * BLOCK_SIZE + tid;
    while (index < uniforms.num_items) {
        let key = input[index].key;
        for (var digit_index = 0u; digit_index < uniforms.pass_count; digit_index++) {
            let digit = (key >> (digit_index * 8u)) & 0xffu;
            atomicAdd(&local_histogram[digit_index * BUCKET_COUNT + digit], 1u);
        }
        index += stride;
    }
    workgroupBarrier();

    for (var slot = tid; slot < histogram_entries; slot += BLOCK_SIZE) {
        let count = atomicLoad(&local_histogram[slot]);
        if (count != 0u) {
            atomicAdd(&histogram[slot], count);
        }
    }
}
