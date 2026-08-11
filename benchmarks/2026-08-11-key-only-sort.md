# Fixed-length key-only radix routing

Date: 2026-08-11

The v0.8 audit found that the `Sorter` key-only `u32` path always used sixteen
portable 2-bit passes, while the adapter-selected key/value sorter used four
8-bit passes.
This change templates the established NVIDIA subgroup kernel over either a
`u32` key or a `KeyValue` record. It does not change GPU-counted sorting or
routing on unmeasured adapters.

## Method

- Baseline: clean `063e5452c8dcd50b3a04022bb51fab365316466d` (`v0.8.0`)
- Candidate: working tree based on `063e545`; the exact measured sort and
  profiler sources are pinned by normalized SHA-256 hashes in the JSON artifact
- Adapter: NVIDIA GeForce RTX 4070 Ti SUPER, Vulkan, driver 591.86
- Five independent processes per source state
- Each process: 250 ms warm-up and five measured samples
- Timing: resident command recording, submission, execution, and completion;
  upload and result readback excluded
- Control: unchanged full-width `KeyValueSorter` workload
- Correctness: CPU parity through 1,000,003 random keys, explicit 8/16/24/32-bit
  paths, and an ignored 100M descending-input validator

The baseline processes ran before the candidate processes. The unchanged
key/value control bounds same-session drift; this is a targeted routing result,
not the cross-machine release matrix.

## Resident wall-time result

Values are medians of five independent process medians.

| Workload | Items | v0.8 baseline | Candidate | Change | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Key-only sort | 1M | 1.386 ms | 0.351 ms | -74.68% | 3.95x |
| Key-only sort | 10M | 4.273 ms | 1.443 ms | -66.23% | 2.96x |
| Key-only sort | 100M | 35.153 ms | 12.309 ms | -64.98% | 2.86x |
| Key/value control | 1M | 0.294 ms | 0.287 ms | -2.38% | 1.02x |
| Key/value control | 10M | 1.616 ms | 1.611 ms | -0.31% | 1.00x |
| Key/value control | 100M | 14.487 ms | 14.475 ms | -0.08% | 1.00x |

The candidate reduces fixed full-width key sorting from sixteen scatter passes
to four. The existing key/value path remains within the 2% regression budget at
10M and 100M; its 1M result improved.

One candidate 100M key/value control process reported a 27.819 ms wall-time
median; the other four were 14.415-14.606 ms. The outlier is retained in the
JSON and does not change the five-process median.

Machine-readable process medians are in
[`2026-08-11-key-only-sort.json`](2026-08-11-key-only-sort.json).

## Commands

```powershell
$env:WGPU_BACKEND='vulkan'
$env:WGPU_PRIMITIVES_PROFILE_CASES='key_sort,key_value_full_width'
$env:WGPU_PRIMITIVES_PROFILE_ITEMS='1000000,10000000,100000000'
$env:WGPU_PRIMITIVES_PROFILE_SAMPLES='5'
$env:WGPU_PRIMITIVES_PROFILE_WARMUP_MS='250'
cargo run --release --example profile_primitives
```

```powershell
$env:LAMPSHADE_REQUIRE_GPU_TESTS='1'
cargo test --release --test sort adapter_selected_sort_validates_100m_items -- --ignored --exact --nocapture
```
