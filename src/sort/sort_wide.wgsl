struct KeyValue {
    key: u32,
    value: u32,
}

@group(0) @binding(0) var<storage, read> input: array<{{ITEM_TYPE}}>;
@group(0) @binding(1) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<{{ITEM_TYPE}}>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

struct Uniforms {
    bit_index: u32,
    num_items: u32,
    num_blocks: u32,
}

const VT: u32 = {{VT}}u;
const BLOCK_SIZE: u32 = {{BLOCK_SIZE}}u;
const ITEMS_PER_BLOCK: u32 = VT * BLOCK_SIZE;
const RADIX_BUCKETS: u32 = {{RADIX_BUCKETS}}u;
const RADIX_BUCKET_GROUPS: u32 = {{RADIX_BUCKET_GROUPS}}u;
const RADIX_MASK: u32 = RADIX_BUCKETS - 1u;

var<workgroup> s_local_hist: array<vec4<u32>, {{LOCAL_HISTOGRAM_SIZE}}>;

fn get_flat_group_id(group_id: vec3<u32>) -> u32 {
    return group_id.y * 65535u + group_id.x;
}

fn local_hist_index(thread_index: u32, bucket_group: u32) -> u32 {
    return thread_index * RADIX_BUCKET_GROUPS + bucket_group;
}

fn store_local_counts(tid: u32, counts: array<u32, {{RADIX_BUCKETS}}>) {
    for (var bucket_group = 0u; bucket_group < RADIX_BUCKET_GROUPS; bucket_group++) {
        let bucket = bucket_group * 4u;
        s_local_hist[local_hist_index(tid, bucket_group)] = vec4<u32>(
            counts[bucket],
            counts[bucket + 1u],
            counts[bucket + 2u],
            counts[bucket + 3u],
        );
    }
}

fn exclusive_scan_local_counts(tid: u32) {
    for (var stride = 1u; stride < BLOCK_SIZE; stride <<= 1u) {
        let right_thread = ((tid + 1u) * stride * 2u) - 1u;
        if (right_thread < BLOCK_SIZE) {
            let left_thread = right_thread - stride;
            for (var bucket_group = 0u; bucket_group < RADIX_BUCKET_GROUPS; bucket_group++) {
                let right_index = local_hist_index(right_thread, bucket_group);
                let left_index = local_hist_index(left_thread, bucket_group);
                s_local_hist[right_index] += s_local_hist[left_index];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        for (var bucket_group = 0u; bucket_group < RADIX_BUCKET_GROUPS; bucket_group++) {
            s_local_hist[local_hist_index(BLOCK_SIZE - 1u, bucket_group)] = vec4<u32>(0u);
        }
    }
    workgroupBarrier();

    for (var stride = (BLOCK_SIZE >> 1u); stride > 0u; stride >>= 1u) {
        let right_thread = ((tid + 1u) * stride * 2u) - 1u;
        if (right_thread < BLOCK_SIZE) {
            let left_thread = right_thread - stride;
            for (var bucket_group = 0u; bucket_group < RADIX_BUCKET_GROUPS; bucket_group++) {
                let right_index = local_hist_index(right_thread, bucket_group);
                let left_index = local_hist_index(left_thread, bucket_group);
                let left = s_local_hist[left_index];
                let right = s_local_hist[right_index];
                s_local_hist[left_index] = right;
                s_local_hist[right_index] = right + left;
            }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(BLOCK_SIZE)
fn main_reduce(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let block_idx = get_flat_group_id(group_id);
    let thread_base_idx = block_idx * ITEMS_PER_BLOCK + tid * VT;
    var my_counts: array<u32, {{RADIX_BUCKETS}}>;

    for (var i = 0u; i < VT; i++) {
        let idx = thread_base_idx + i;
        if (idx < uniforms.num_items) {
            let item = input[idx];
            let val = {{KEY_ACCESS}};
            let digit = (val >> uniforms.bit_index) & RADIX_MASK;
            my_counts[digit]++;
        }
    }

    store_local_counts(tid, my_counts);
    workgroupBarrier();

    for (var stride = (BLOCK_SIZE >> 1u); stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            for (var bucket_group = 0u; bucket_group < RADIX_BUCKET_GROUPS; bucket_group++) {
                let left_index = local_hist_index(tid, bucket_group);
                let right_index = local_hist_index(tid + stride, bucket_group);
                s_local_hist[left_index] += s_local_hist[right_index];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        for (var bucket = 0u; bucket < RADIX_BUCKETS; bucket++) {
            let bucket_group = bucket >> 2u;
            let lane = bucket & 3u;
            histograms[bucket * uniforms.num_blocks + block_idx] =
                s_local_hist[local_hist_index(0u, bucket_group)][lane];
        }
    }
}

@compute @workgroup_size(BLOCK_SIZE)
fn main_scatter(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let block_idx = get_flat_group_id(group_id);
    let thread_base_idx = block_idx * ITEMS_PER_BLOCK + tid * VT;
    var my_items: array<{{ITEM_TYPE}}, {{VT}}>;
    var my_digits: array<u32, {{VT}}>;
    var my_counts: array<u32, {{RADIX_BUCKETS}}>;

    for (var i = 0u; i < VT; i++) {
        let idx = thread_base_idx + i;
        if (idx < uniforms.num_items) {
            let item = input[idx];
            let val = {{KEY_ACCESS}};
            let digit = (val >> uniforms.bit_index) & RADIX_MASK;
            my_items[i] = item;
            my_digits[i] = digit;
            my_counts[digit]++;
        }
    }

    store_local_counts(tid, my_counts);
    workgroupBarrier();
    exclusive_scan_local_counts(tid);

    var local_running_counts: array<u32, {{RADIX_BUCKETS}}>;

    for (var i = 0u; i < VT; i++) {
        let idx = thread_base_idx + i;
        if (idx < uniforms.num_items) {
            let digit = my_digits[i];
            let bucket_group = digit >> 2u;
            let lane = digit & 3u;
            let thread_offset = s_local_hist[local_hist_index(tid, bucket_group)][lane];
            var block_offset = 0u;

            if (block_idx > 0u) {
                block_offset = histograms[digit * uniforms.num_blocks + block_idx - 1u];
            } else if (digit > 0u) {
                block_offset = histograms[digit * uniforms.num_blocks - 1u];
            }

            let destination = block_offset + thread_offset + local_running_counts[digit];
            local_running_counts[digit]++;
            output[destination] = my_items[i];
        }
    }
}
