# GPU-counted portable radix tuning

## Outcome

GPU-counted key/value sorting now reuses the measured 4-bit radix
variant on capable Intel Vulkan adapters. The old counted implementation always
processed two key bits per pass, so a full-width key required 16 passes. The
selected Intel path processes four bits per pass and requires eight.

The separate-buffer SoA bridge was not the bottleneck. At one million items on
Intel, pack and unpack each cost about 1 ms while the old counted radix stage
cost about 35 ms. The optimization therefore targets the inner radix sort and
does not add another bridge or copy.

Apple Metal remains on the 2-bit path. Both 3-bit and 4-bit experiments were
correct, but neither improved the stable 10-million-item result by 5%. Shipping
either would add backend policy without a defensible gain.

## Results

The table reports the full pack -> counted stable sort -> unpack pipeline after
subtracting the same-process empty-submission median. Each cell is the median
of three independent process medians; every process used four warmups and 11
samples.

| Adapter | Items | 2-bit control | Selected | Change |
| --- | ---: | ---: | ---: | ---: |
| Intel ADL-N, Vulkan | 1M | 36.606 ms | 30.377 ms | **-17.02%** |
| Intel ADL-N, Vulkan | 10M | 372.767 ms | 278.484 ms | **-25.29%** |
| Apple M3 Pro, Metal | 10M | 32.979 ms | 32.952 ms | -0.08% |

The Apple labels select the same 2-bit implementation in the final source; the
10M row is a parity control. Apple 1M completion times were bimodal even with
identical routing, so they are preserved in the JSON but excluded from the
claim. A five-process RTX 4070 Ti SUPER diagnostic remained healthy, and the
native NVIDIA SoA source and routing are untouched by this change.

The formal same-runtime regression harness also compared the complete checkout
with crates.io 0.11.0 on RTX at 1M, 10M, and 100M. All 18 rows passed the 2%
budget. The largest regression was +1.81%; counted full-width sort changed by
-0.29%, -0.39%, and +0.02% across the three sizes.

## Method and correctness

`examples/profile_portable_soa.rs` validates full key order, key/value
association, duplicate stability, the GPU-selected prefix, and a count written
earlier in the same command encoder before timing. Encoding is excluded; queue
submission through completion is included. Because Apple timestamps remain
disabled, stage estimates subtract an empty submission measured in the same
process.

The physical `key_value_sort` integration suite passed 21/21 on both Apple
Metal and Intel Vulkan. Local unit tests passed 32/32. The forced `portable`
backend uses the existing 2-bit shader, while `selected` exercises adapter
routing from the same candidate source.

Reproduction:

```text
cargo run --release --example profile_portable_soa -- 1000000 11 4 portable
cargo run --release --example profile_portable_soa -- 1000000 11 4 selected
cargo run --release --example profile_portable_soa -- 10000000 11 4 portable
cargo run --release --example profile_portable_soa -- 10000000 11 4 selected
```

The machine-readable artifact contains every process median and the exact
claim boundaries: [2026-08-13-portable-soa-counted-radix.json](2026-08-13-portable-soa-counted-radix.json).
The measurements used branch `perf/portable-soa` at parent
`b95cba617b61ef91e37f0155854f70d293f63150`. The dirty candidate is pinned by a
30-file LF-normalized source manifest with SHA-256
`7a90f56574f2b82e15f1d70f2914180340521243c7ce78b17d11e6d5c18c1cb4`.
