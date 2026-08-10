@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> histograms: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
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
const MAX_WORKGROUPS_X: u32 = {{MAX_WORKGROUPS_X}}u;

var<workgroup> local_histogram: array<vec4<u32>, {{BLOCK_SIZE}}>;

fn actual_items() -> u32 {
    return min(item_count[0], uniforms.capacity_items);
}

fn flat_group_id(group_id: vec3<u32>) -> u32 {
    return group_id.y * MAX_WORKGROUPS_X + group_id.x;
}

fn active_blocks(items: u32) -> u32 {
    return items / ITEMS_PER_BLOCK + select(0u, 1u, items % ITEMS_PER_BLOCK != 0u);
}

@compute @workgroup_size(BLOCK_SIZE)
fn reduce(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread = local_id.x;
    let block = flat_group_id(group_id);
    let block_base = block * ITEMS_PER_BLOCK;
    let thread_base = block_base + thread * VT;
    let items = actual_items();
    if (block >= active_blocks(items)) {
        return;
    }

    var counts = vec4<u32>(0u);
    for (var i = 0u; i < VT; i++) {
        let index = thread_base + i;
        if (index < items) {
            let digit = (input[index] >> uniforms.bit_index) & 3u;
            if (digit == 0u) { counts.x++; }
            else if (digit == 1u) { counts.y++; }
            else if (digit == 2u) { counts.z++; }
            else { counts.w++; }
        }
    }

    local_histogram[thread] = counts;
    workgroupBarrier();

    for (var stride = BLOCK_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (thread < stride) {
            local_histogram[thread] += local_histogram[thread + stride];
        }
        workgroupBarrier();
    }

    if (thread == 0u) {
        let stride = uniforms.capacity_blocks;
        histograms[block] = local_histogram[0].x;
        histograms[stride + block] = local_histogram[0].y;
        histograms[2u * stride + block] = local_histogram[0].z;
        histograms[3u * stride + block] = local_histogram[0].w;
    }
}

@compute @workgroup_size(BLOCK_SIZE)
fn scatter(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let thread = local_id.x;
    let block = flat_group_id(group_id);
    let block_base = block * ITEMS_PER_BLOCK;
    let thread_base = block_base + thread * VT;
    let items = actual_items();
    if (block >= active_blocks(items)) {
        return;
    }
    let stride = uniforms.capacity_blocks;

    var block_offset = vec4<u32>(0u);
    if (block > 0u) {
        block_offset.x = histograms[block - 1u];
        block_offset.y = histograms[stride + block - 1u];
        block_offset.z = histograms[2u * stride + block - 1u];
        block_offset.w = histograms[3u * stride + block - 1u];
    } else {
        block_offset.x = 0u;
        block_offset.y = histograms[stride - 1u];
        block_offset.z = histograms[2u * stride - 1u];
        block_offset.w = histograms[3u * stride - 1u];
    }

    var values: array<u32, {{VT}}>;
    var digits: array<u32, {{VT}}>;
    var counts = vec4<u32>(0u);

    for (var i = 0u; i < VT; i++) {
        let index = thread_base + i;
        if (index < items) {
            let value = input[index];
            let digit = (value >> uniforms.bit_index) & 3u;
            values[i] = value;
            digits[i] = digit;
            if (digit == 0u) { counts.x++; }
            else if (digit == 1u) { counts.y++; }
            else if (digit == 2u) { counts.z++; }
            else { counts.w++; }
        }
    }

    local_histogram[thread] = counts;
    workgroupBarrier();

    for (var offset = 1u; offset < BLOCK_SIZE; offset <<= 1u) {
        var prior = vec4<u32>(0u);
        if (thread >= offset) {
            prior = local_histogram[thread - offset];
        }
        workgroupBarrier();
        if (thread >= offset) {
            local_histogram[thread] += prior;
        }
        workgroupBarrier();
    }

    let thread_start = local_histogram[thread] - counts;
    var local_counts = vec4<u32>(0u);
    for (var i = 0u; i < VT; i++) {
        let index = thread_base + i;
        if (index < items) {
            let digit = digits[i];
            var destination = 0u;
            if (digit == 0u) {
                destination = block_offset.x + thread_start.x + local_counts.x;
                local_counts.x++;
            } else if (digit == 1u) {
                destination = block_offset.y + thread_start.y + local_counts.y;
                local_counts.y++;
            } else if (digit == 2u) {
                destination = block_offset.z + thread_start.z + local_counts.z;
                local_counts.z++;
            } else {
                destination = block_offset.w + thread_start.w + local_counts.w;
                local_counts.w++;
            }
            output[destination] = values[i];
        }
    }
}
