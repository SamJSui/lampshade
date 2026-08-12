struct Params {
    capacity_items: u32,
    pass_count: u32,
    count_word: u32,
    _padding: u32,
}

struct UniformRecord {
    words: array<u32, {{UNIFORM_STRIDE_WORDS}}>,
}

@group(0) @binding(0) var<storage, read> count_words: array<u32>;
@group(0) @binding(1) var<storage, read_write> records: array<UniformRecord>;
@group(0) @binding(2) var<uniform> params: Params;

const ITEMS_PER_TILE: u32 = 1792u;

@compute @workgroup_size(1)
fn main() {
    let num_items = min(count_words[params.count_word], params.capacity_items);
    let num_tiles = num_items / ITEMS_PER_TILE
        + select(0u, 1u, num_items % ITEMS_PER_TILE != 0u);

    for (var radix_pass = 0u; radix_pass < params.pass_count; radix_pass++) {
        records[radix_pass].words[0] = num_items;
        records[radix_pass].words[1] = num_tiles;
        records[radix_pass].words[2] = radix_pass + 1u;
        records[radix_pass].words[3] = radix_pass * 8u;
        records[radix_pass].words[4] = params.pass_count;
    }
}
