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

@group(0) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> dispatch_args: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

const BUCKET_COUNT: u32 = 256u;
const PASS_COUNT: u32 = 4u;
var<workgroup> scan_values: array<u32, 256>;
var<workgroup> upper_nonempty: array<atomic<u32>, 2>;

@compute @workgroup_size(BUCKET_COUNT)
fn main_prefix(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    if (tid < 2u) {
        atomicStore(&upper_nonempty[tid], 0u);
    }
    workgroupBarrier();

    for (var radix_pass = 0u; radix_pass < uniforms.pass_count; radix_pass++) {
        let base = radix_pass * BUCKET_COUNT;
        let value = atomicLoad(&histogram[base + tid]);
        scan_values[tid] = value;
        if (uniforms.pass_count == PASS_COUNT && radix_pass >= 2u && value != 0u) {
            atomicAdd(&upper_nonempty[radix_pass - 2u], 1u);
        }
        workgroupBarrier();

        for (var stride = 1u; stride < BUCKET_COUNT; stride <<= 1u) {
            let right = ((tid + 1u) * stride * 2u) - 1u;
            if (right < BUCKET_COUNT) {
                scan_values[right] += scan_values[right - stride];
            }
            workgroupBarrier();
        }

        if (tid == 0u) {
            scan_values[BUCKET_COUNT - 1u] = 0u;
        }
        workgroupBarrier();

        for (var stride = BUCKET_COUNT >> 1u; stride > 0u; stride >>= 1u) {
            let right = ((tid + 1u) * stride * 2u) - 1u;
            if (right < BUCKET_COUNT) {
                let left = right - stride;
                let left_value = scan_values[left];
                scan_values[left] = scan_values[right];
                scan_values[right] += left_value;
            }
            workgroupBarrier();
        }

        offsets[base + tid] = scan_values[tid];
        workgroupBarrier();
    }

    if (tid < uniforms.pass_count) {
        var groups = uniforms.num_tiles;
        if (uniforms.pass_count == PASS_COUNT
            && tid >= 2u
            && atomicLoad(&upper_nonempty[0]) == 1u
            && atomicLoad(&upper_nonempty[1]) == 1u) {
            groups = 0u;
        }
        let offset = tid * 3u;
        dispatch_args[offset] = groups;
        dispatch_args[offset + 1u] = 1u;
        dispatch_args[offset + 2u] = 1u;
    }
}
