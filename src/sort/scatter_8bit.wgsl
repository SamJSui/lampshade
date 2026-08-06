struct KeyValue {
    key: u32,
    value: u32,
}

struct Uniforms {
    num_items: u32,
    num_tiles: u32,
    generation: u32,
    bit_index: u32,
}

@group(0) @binding(0) var<storage, read> input: array<KeyValue>;
@group(0) @binding(1) var<storage, read_write> output: array<KeyValue>;
@group(0) @binding(2) var<storage, read> digit_offsets: array<u32>;
@group(0) @binding(3) var<storage, read_write> partition_state: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

const BLOCK_SIZE: u32 = 256u;
const SUBGROUP_SIZE: u32 = 32u;
const SUBGROUP_COUNT: u32 = BLOCK_SIZE / SUBGROUP_SIZE;
const ITEMS_PER_THREAD: u32 = 7u;
const ITEMS_PER_SUBGROUP: u32 = SUBGROUP_SIZE * ITEMS_PER_THREAD;
const ITEMS_PER_TILE: u32 = BLOCK_SIZE * ITEMS_PER_THREAD;
const BUCKET_COUNT: u32 = 256u;
const TILE_COUNTER_COUNT: u32 = 4u;
const COUNT_MASK: u32 = 0x0fffffffu;
const PREFIX_BIT: u32 = 0x10000000u;
const GENERATION_SHIFT: u32 = 29u;

var<workgroup> sorted_items: array<KeyValue, 1792>;
var<workgroup> bucket_counts: array<atomic<u32>, 256>;
var<workgroup> tile_offsets: array<u32, 256>;
var<workgroup> assigned_tile: u32;

fn state_value(generation: u32, prefix: bool, count: u32) -> u32 {
    var state = generation << GENERATION_SHIFT;
    if (prefix) {
        state |= PREFIX_BIT;
    }
    return state | count;
}

fn wait_for_state(index: u32, generation: u32) -> u32 {
    loop {
        let state = atomicLoad(&partition_state[index]);
        if ((state >> GENERATION_SHIFT) == generation) {
            return state;
        }
    }
    return 0u;
}

fn tile_prefix(tile: u32, bucket: u32, local_count: u32) -> u32 {
    let state_index = TILE_COUNTER_COUNT + tile * BUCKET_COUNT + bucket;
    atomicStore(
        &partition_state[state_index],
        state_value(uniforms.generation, false, local_count),
    );

    var prefix = 0u;
    if (tile > 0u) {
        var cursor = tile;
        loop {
            cursor--;
            let state = wait_for_state(
                TILE_COUNTER_COUNT + cursor * BUCKET_COUNT + bucket,
                uniforms.generation,
            );
            prefix += state & COUNT_MASK;
            if ((state & PREFIX_BIT) != 0u || cursor == 0u) {
                break;
            }
        }
    }

    atomicStore(
        &partition_state[state_index],
        state_value(uniforms.generation, true, prefix + local_count),
    );
    return prefix;
}

fn subgroup_rank(is_valid: bool, digit: u32, lane: u32) -> u32 {
    var matching_lanes = subgroupBallot(is_valid).x;
    for (var bit = 0u; bit < 8u; bit++) {
        let bit_mask = 1u << bit;
        let lanes_with_bit = subgroupBallot(is_valid && (digit & bit_mask) != 0u).x;
        if ((digit & bit_mask) != 0u) {
            matching_lanes &= lanes_with_bit;
        } else {
            matching_lanes &= ~lanes_with_bit;
        }
    }

    if (!is_valid) {
        return 0u;
    }
    let lower_lanes = select(0u, (1u << lane) - 1u, lane > 0u);
    let rank = countOneBits(matching_lanes & lower_lanes) + 1u;
    let count = countOneBits(matching_lanes);
    return (count << 16u) | rank;
}

fn exclusive_scan_bucket_counts(tid: u32, subgroup_id: u32, lane: u32) {
    let count = atomicLoad(&bucket_counts[tid]);
    let lane_prefix = subgroupExclusiveAdd(count);
    if (lane == SUBGROUP_SIZE - 1u) {
        sorted_items[subgroup_id].key = lane_prefix + count;
    }
    workgroupBarrier();

    var subgroup_prefix = 0u;
    for (var group = 0u; group < subgroup_id; group++) {
        subgroup_prefix += sorted_items[group].key;
    }
    atomicStore(&bucket_counts[tid], subgroup_prefix + lane_prefix);
    workgroupBarrier();
}

@compute @workgroup_size(BLOCK_SIZE)
fn main_scatter(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let tid = local_id.x;
    var items: array<KeyValue, 7>;
    var ranks: array<u32, 7>;

    if (tid == 0u) {
        assigned_tile = atomicAdd(&partition_state[uniforms.generation - 1u], 1u);
    }
    atomicStore(&bucket_counts[tid], 0u);
    workgroupBarrier();

    let tile = assigned_tile;
    let tile_base = tile * ITEMS_PER_TILE;

    for (var item_index = 0u; item_index < ITEMS_PER_THREAD; item_index++) {
        let source_index = tile_base
            + subgroup_id * ITEMS_PER_SUBGROUP
            + item_index * SUBGROUP_SIZE
            + lane;
        let is_valid = source_index < uniforms.num_items;
        var digit = 0u;
        if (is_valid) {
            items[item_index] = input[source_index];
            digit = (items[item_index].key >> uniforms.bit_index) & 0xffu;
        }
        ranks[item_index] = subgroup_rank(is_valid, digit, lane);
    }

    for (var group = 0u; group < SUBGROUP_COUNT; group++) {
        if (subgroup_id == group) {
            for (var item_index = 0u; item_index < ITEMS_PER_THREAD; item_index++) {
                let source_index = tile_base
                    + subgroup_id * ITEMS_PER_SUBGROUP
                    + item_index * SUBGROUP_SIZE
                    + lane;
                if (source_index < uniforms.num_items) {
                    let digit = (items[item_index].key >> uniforms.bit_index) & 0xffu;
                    let rank = ranks[item_index] & 0xffffu;
                    let count = ranks[item_index] >> 16u;
                    let preceding = atomicLoad(&bucket_counts[digit]);
                    ranks[item_index] = preceding + rank - 1u;
                    if (rank == count) {
                        atomicStore(&bucket_counts[digit], preceding + count);
                    }
                }
            }
        }
        workgroupBarrier();
    }

    let local_count = atomicLoad(&bucket_counts[tid]);
    let prefix = tile_prefix(tile, tid, local_count);
    tile_offsets[tid] =
        digit_offsets[(uniforms.generation - 1u) * BUCKET_COUNT + tid] + prefix;

    exclusive_scan_bucket_counts(tid, subgroup_id, lane);

    for (var item_index = 0u; item_index < ITEMS_PER_THREAD; item_index++) {
        let source_index = tile_base
            + subgroup_id * ITEMS_PER_SUBGROUP
            + item_index * SUBGROUP_SIZE
            + lane;
        if (source_index < uniforms.num_items) {
            let item = items[item_index];
            let digit = (item.key >> uniforms.bit_index) & 0xffu;
            let local_index = atomicLoad(&bucket_counts[digit]) + ranks[item_index];
            sorted_items[local_index] = item;
        }
    }
    workgroupBarrier();

    let tile_items = min(ITEMS_PER_TILE, uniforms.num_items - tile_base);
    for (var item_index = 0u; item_index < ITEMS_PER_THREAD; item_index++) {
        let local_index = item_index * BLOCK_SIZE + tid;
        if (local_index < tile_items) {
            let item = sorted_items[local_index];
            let digit = (item.key >> uniforms.bit_index) & 0xffu;
            let bucket_start = atomicLoad(&bucket_counts[digit]);
            output[tile_offsets[digit] + local_index - bucket_start] = item;
        }
    }
}
