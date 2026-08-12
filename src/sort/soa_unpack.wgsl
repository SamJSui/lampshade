struct KeyValue {
    key: u32,
    value: u32,
}

struct Config {
    capacity_items: u32,
    count_word: u32,
    _padding_0: u32,
    _padding_1: u32,
}

@group(0) @binding(0) var<storage, read> packed_input: array<KeyValue>;
@group(0) @binding(1) var<storage, read_write> keys_output: array<u32>;
@group(0) @binding(2) var<storage, read_write> values_output: array<u32>;
@group(0) @binding(3) var<storage, read> count_words: array<u32>;
@group(0) @binding(4) var<uniform> config: Config;

const BLOCK_SIZE: u32 = 256u;

fn item_index(
    group_id: vec3<u32>,
    workgroup_count: vec3<u32>,
    local_id: vec3<u32>,
) -> u32 {
    let groups = config.capacity_items / BLOCK_SIZE
        + select(0u, 1u, config.capacity_items % BLOCK_SIZE != 0u);
    let flat_group = group_id.y * workgroup_count.x + group_id.x;
    if (flat_group >= groups) {
        return config.capacity_items;
    }
    return flat_group * BLOCK_SIZE + local_id.x;
}

@compute @workgroup_size(BLOCK_SIZE)
fn main(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(num_workgroups) workgroup_count: vec3<u32>,
) {
    let index = item_index(group_id, workgroup_count, local_id);
    let count = min(count_words[config.count_word], config.capacity_items);
    if (index < count) {
        let item = packed_input[index];
        keys_output[index] = item.key;
        values_output[index] = item.value;
    }
}
