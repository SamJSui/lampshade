struct KeyValue {
    key: u32,
    value: u32,
}

@group(0) @binding(0) var<storage, read> input: array<{{ITEM_TYPE}}>;
@group(0) @binding(1) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<{{ITEM_TYPE}}>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<storage, read> item_count: array<u32>;

struct Uniforms {
    bit_index: u32,
    capacity_items: u32,
    capacity_blocks: u32,
    _padding: u32,
}

const VT: u32 = {{VT}}u;
const BLOCK_SIZE: u32 = {{BLOCK_SIZE}}u;
const ITEMS_PER_BLOCK: u32 = VT * BLOCK_SIZE;
const RADIX_BUCKETS: u32 = {{RADIX_BUCKETS}}u;
const RADIX_BUCKET_GROUPS: u32 = {{RADIX_BUCKET_GROUPS}}u;
const RADIX_MASK: u32 = RADIX_BUCKETS - 1u;
const MAX_WORKGROUPS_X: u32 = {{MAX_WORKGROUPS_X}}u;

var<workgroup> local_histogram: array<vec4<u32>, {{LOCAL_HISTOGRAM_SIZE}}>;

fn actual_items() -> u32 {
    return min(item_count[0], uniforms.capacity_items);
}

fn flat_group_id(group_id: vec3<u32>) -> u32 {
    return group_id.y * MAX_WORKGROUPS_X + group_id.x;
}

fn active_blocks(items: u32) -> u32 {
    return items / ITEMS_PER_BLOCK + select(0u, 1u, items % ITEMS_PER_BLOCK != 0u);
}

fn local_hist_index(thread_index: u32, bucket_group: u32) -> u32 {
    return thread_index * RADIX_BUCKET_GROUPS + bucket_group;
}

fn store_local_counts(thread: u32, counts: array<u32, {{RADIX_BUCKETS}}>) {
    for (var bucket_group = 0u; bucket_group < RADIX_BUCKET_GROUPS; bucket_group++) {
        let bucket = bucket_group * 4u;
        local_histogram[local_hist_index(thread, bucket_group)] = vec4<u32>(
            counts[bucket],
            counts[bucket + 1u],
            counts[bucket + 2u],
            counts[bucket + 3u],
        );
    }
}

fn exclusive_scan_local_counts(thread: u32) {
    for (var stride = 1u; stride < BLOCK_SIZE; stride <<= 1u) {
        let right_thread = ((thread + 1u) * stride * 2u) - 1u;
        if (right_thread < BLOCK_SIZE) {
            let left_thread = right_thread - stride;
            for (var bucket_group = 0u; bucket_group < RADIX_BUCKET_GROUPS; bucket_group++) {
                let right_index = local_hist_index(right_thread, bucket_group);
                let left_index = local_hist_index(left_thread, bucket_group);
                local_histogram[right_index] += local_histogram[left_index];
            }
        }
        workgroupBarrier();
    }

    if (thread == 0u) {
        for (var bucket_group = 0u; bucket_group < RADIX_BUCKET_GROUPS; bucket_group++) {
            local_histogram[local_hist_index(BLOCK_SIZE - 1u, bucket_group)] = vec4<u32>(0u);
        }
    }
    workgroupBarrier();

    for (var stride = BLOCK_SIZE >> 1u; stride > 0u; stride >>= 1u) {
        let right_thread = ((thread + 1u) * stride * 2u) - 1u;
        if (right_thread < BLOCK_SIZE) {
            let left_thread = right_thread - stride;
            for (var bucket_group = 0u; bucket_group < RADIX_BUCKET_GROUPS; bucket_group++) {
                let right_index = local_hist_index(right_thread, bucket_group);
                let left_index = local_hist_index(left_thread, bucket_group);
                let left = local_histogram[left_index];
                let right = local_histogram[right_index];
                local_histogram[left_index] = right;
                local_histogram[right_index] = right + left;
            }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(BLOCK_SIZE)
fn reduce(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let thread = local_id.x;
    let block = flat_group_id(group_id);
    let items = actual_items();
    if (block >= active_blocks(items)) {
        return;
    }
    let thread_base = block * ITEMS_PER_BLOCK + thread * VT;
    var counts: array<u32, {{RADIX_BUCKETS}}>;

    for (var i = 0u; i < VT; i++) {
        let index = thread_base + i;
        if (index < items) {
            let item = input[index];
            let key = {{KEY_ACCESS}};
            let digit = (key >> uniforms.bit_index) & RADIX_MASK;
            counts[digit]++;
        }
    }

    store_local_counts(thread, counts);
    workgroupBarrier();

    for (var stride = BLOCK_SIZE >> 1u; stride > 0u; stride >>= 1u) {
        if (thread < stride) {
            for (var bucket_group = 0u; bucket_group < RADIX_BUCKET_GROUPS; bucket_group++) {
                let left_index = local_hist_index(thread, bucket_group);
                let right_index = local_hist_index(thread + stride, bucket_group);
                local_histogram[left_index] += local_histogram[right_index];
            }
        }
        workgroupBarrier();
    }

    if (thread == 0u) {
        for (var bucket = 0u; bucket < RADIX_BUCKETS; bucket++) {
            let bucket_group = bucket >> 2u;
            let lane = bucket & 3u;
            histograms[bucket * uniforms.capacity_blocks + block] =
                local_histogram[local_hist_index(0u, bucket_group)][lane];
        }
    }
}

@compute @workgroup_size(BLOCK_SIZE)
fn scatter(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let thread = local_id.x;
    let block = flat_group_id(group_id);
    let items = actual_items();
    if (block >= active_blocks(items)) {
        return;
    }
    let thread_base = block * ITEMS_PER_BLOCK + thread * VT;
    var thread_items: array<{{ITEM_TYPE}}, {{VT}}>;
    var digits: array<u32, {{VT}}>;
    var counts: array<u32, {{RADIX_BUCKETS}}>;

    for (var i = 0u; i < VT; i++) {
        let index = thread_base + i;
        if (index < items) {
            let item = input[index];
            let key = {{KEY_ACCESS}};
            let digit = (key >> uniforms.bit_index) & RADIX_MASK;
            thread_items[i] = item;
            digits[i] = digit;
            counts[digit]++;
        }
    }

    store_local_counts(thread, counts);
    workgroupBarrier();
    exclusive_scan_local_counts(thread);

    var local_counts: array<u32, {{RADIX_BUCKETS}}>;
    for (var i = 0u; i < VT; i++) {
        let index = thread_base + i;
        if (index < items) {
            let digit = digits[i];
            let bucket_group = digit >> 2u;
            let lane = digit & 3u;
            let thread_offset = local_histogram[local_hist_index(thread, bucket_group)][lane];
            var block_offset = 0u;
            if (block > 0u) {
                block_offset = histograms[digit * uniforms.capacity_blocks + block - 1u];
            } else if (digit > 0u) {
                block_offset = histograms[digit * uniforms.capacity_blocks - 1u];
            }
            let destination = block_offset + thread_offset + local_counts[digit];
            local_counts[digit]++;
            output[destination] = thread_items[i];
        }
    }
}
