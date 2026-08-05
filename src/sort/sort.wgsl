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

var<workgroup> s_local_hist: array<vec4<u32>, {{BLOCK_SIZE}}>;

fn get_flat_group_id(group_id: vec3<u32>) -> u32 {
    return group_id.y * 65535u + group_id.x;
}

@compute @workgroup_size(BLOCK_SIZE)
fn main_reduce(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let flat_group_id = get_flat_group_id(group_id);
    let group_base_idx = flat_group_id * ITEMS_PER_BLOCK;
    let thread_base_idx = group_base_idx + (tid * VT);

    var my_counts = vec4<u32>(0u);

    for (var i = 0u; i < VT; i++) {
        let idx = thread_base_idx + i;
        if (idx < uniforms.num_items) {
            let item = input[idx];
            let val = {{KEY_ACCESS}};
            let digit = (val >> uniforms.bit_index) & 3u;
            if (digit == 0u) { my_counts.x++; }
            else if (digit == 1u) { my_counts.y++; }
            else if (digit == 2u) { my_counts.z++; }
            else { my_counts.w++; }
        }
    }

    s_local_hist[tid] = my_counts;
    workgroupBarrier();

    for (var s = (BLOCK_SIZE >> 1u); s > 0u; s >>= 1u) {
        if (tid < s) {
            s_local_hist[tid] += s_local_hist[tid + s];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let total_blocks = uniforms.num_blocks;
        
        // Region 0: All D0 counts
        histograms[flat_group_id] = s_local_hist[0].x;
        // Region 1: All D1 counts
        histograms[total_blocks + flat_group_id] = s_local_hist[0].y;
        // Region 2: All D2 counts
        histograms[(2u * total_blocks) + flat_group_id] = s_local_hist[0].z;
        // Region 3: All D3 counts
        histograms[(3u * total_blocks) + flat_group_id] = s_local_hist[0].w;
    }
}

@compute @workgroup_size(BLOCK_SIZE)
fn main_scatter(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let block_idx = get_flat_group_id(group_id);
    let group_base_idx = block_idx * ITEMS_PER_BLOCK;
    let thread_base_idx = group_base_idx + (tid * VT);

    let total_blocks = uniforms.num_blocks;
    var block_offset = vec4<u32>(0u);

    if (block_idx > 0u) {
        block_offset.x = histograms[block_idx - 1u];
        block_offset.y = histograms[total_blocks + block_idx - 1u];
        block_offset.z = histograms[(2u * total_blocks) + block_idx - 1u];
        block_offset.w = histograms[(3u * total_blocks) + block_idx - 1u];
    } else {
        if (block_idx == 0u) {
             // Digit 0 always starts at 0
             block_offset.x = 0u; 
             // Digit 1 starts where Digit 0 ended (last element of Region 0)
             block_offset.y = histograms[total_blocks - 1u];
             // Digit 2 starts where Digit 1 ended
             block_offset.z = histograms[(2u * total_blocks) - 1u];
             // Digit 3 starts where Digit 2 ended
             block_offset.w = histograms[(3u * total_blocks) - 1u];
        }
    }

    var my_items: array<{{ITEM_TYPE}}, {{VT}}>;
    var my_digits: array<u32, {{VT}}>;
    var my_counts = vec4<u32>(0u);

    for (var i = 0u; i < VT; i++) {
        let idx = thread_base_idx + i;
        if (idx < uniforms.num_items) {
            let item = input[idx];
            let val = {{KEY_ACCESS}};
            let digit = (val >> uniforms.bit_index) & 3u;
            
            my_items[i] = item;
            my_digits[i] = digit;
            
            if (digit == 0u) { my_counts.x++; }
            else if (digit == 1u) { my_counts.y++; }
            else if (digit == 2u) { my_counts.z++; }
            else { my_counts.w++; }
        }
    }
    workgroupBarrier();

    s_local_hist[tid] = my_counts;
    workgroupBarrier();

    for (var offset = 1u; offset < BLOCK_SIZE; offset <<= 1u) {
        var temp = vec4<u32>(0u);
        if (tid >= offset) { temp = s_local_hist[tid - offset]; }
        workgroupBarrier();

        if (tid >= offset) { s_local_hist[tid] += temp; }
        workgroupBarrier();
    }

    let thread_inclusive_prefix = s_local_hist[tid];
    let thread_exclusive_start = thread_inclusive_prefix - my_counts;
    
    var local_running_counts = vec4<u32>(0u);
    
    for (var i = 0u; i < VT; i++) {
        let idx = thread_base_idx + i;
        if (idx < uniforms.num_items) {
            let digit = my_digits[i];
            var dest = 0u;
            
            if (digit == 0u) {
                dest = block_offset.x + thread_exclusive_start.x + local_running_counts.x;
                local_running_counts.x++;
            } else if (digit == 1u) {
                dest = block_offset.y + thread_exclusive_start.y + local_running_counts.y;
                local_running_counts.y++;
            } else if (digit == 2u) {
                dest = block_offset.z + thread_exclusive_start.z + local_running_counts.z;
                local_running_counts.z++;
            } else {
                dest = block_offset.w + thread_exclusive_start.w + local_running_counts.w;
                local_running_counts.w++;
            }

            output[dest] = my_items[i];
        }
    }
}
