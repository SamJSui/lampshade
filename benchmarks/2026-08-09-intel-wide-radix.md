# Intel Vulkan 4-bit radix experiment

This experiment routes capable Intel Vulkan adapters to the existing 4-bit
stable key-value radix kernels. It halves the number of full-array passes
relative to the portable 2-bit path while preserving that path as the fallback.

## System and method

- Beelink Mini S13, Intel N150, integrated Intel Alder Lake-N graphics
  (`8086:46d4`).
- Ubuntu 24.04, x86-64, Vulkan, Mesa 25.2.8.
- `wgpu-primitives` 0.5.0 based on merged revision
  `9bc5da70f3dcd470cf374502a9f2fe4cce51b99e` plus the
  `src/sort/pipeline.rs` routing change in this report.
- Massively 0.96.0 at
  `ef9de55190529be98203aca207edab9d560d312e`.

The comparison harness times the resident public API through GPU completion.
It excludes upload, validation readback, and result download. Each 1M and 10M
case warms for at least two seconds and four calls, retains 11 samples, and
reports the median of three independent process medians. The 100M cases use two
warmups and seven retained samples per process. Every process validates its
output before timing.

## Sort results

`Change` compares the new 4-bit path with the previously published portable
2-bit result. `Speedup` is `Massively time / new time`.

| Workload | Items | Portable | 4-bit | Change | Massively | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bounded 16-bit stable sort | 1M | 20.338 ms | 15.364 ms | -24.46% | 47.945 ms | 3.12x |
| Full-width stable sort | 1M | 40.645 ms | 28.495 ms | -29.89% | 48.751 ms | 1.71x |
| Bounded 16-bit stable sort | 10M | 174.815 ms | 130.349 ms | -25.44% | 580.788 ms | 4.46x |
| Full-width stable sort | 10M | 352.525 ms | 261.765 ms | -25.75% | 587.289 ms | 2.24x |
| Bounded 16-bit stable sort | 100M | 1769.833 ms | 1308.816 ms | -26.05% | 12801.820 ms | 9.78x |
| Full-width stable sort | 100M | 3532.261 ms | 2617.170 ms | -25.91% | 12547.157 ms | 4.79x |

The 1M and 10M Massively values were measured in the same comparison run. The
unchanged 100M comparator values come from the preceding Intel validation; the
new `wgpu-primitives` 100M values are fresh three-process medians.

## Why it is faster

GPU timestamps show that the win comes from fewer complete passes, not a
cheaper individual pass:

| Workload | Items | Path | Passes | GPU elapsed | Reduce | Histogram scan | Scatter |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Bounded | 10M | 2-bit | 8 | 145.919 ms | 51.230 ms | 0.275 ms | 93.921 ms |
| Bounded | 10M | 4-bit | 4 | 106.866 ms | 38.288 ms | 0.403 ms | 67.866 ms |
| Full width | 10M | 2-bit | 16 | 301.489 ms | 102.322 ms | 0.557 ms | 197.546 ms |
| Full width | 10M | 4-bit | 8 | 230.158 ms | 80.205 ms | 0.809 ms | 148.483 ms |
| Bounded | 100M | 2-bit | 8 | 1604.758 ms | 569.318 ms | 1.764 ms | 1032.868 ms |
| Bounded | 100M | 4-bit | 4 | 1108.825 ms | 414.297 ms | 3.279 ms | 690.807 ms |
| Full width | 100M | 2-bit | 16 | 3344.316 ms | 1146.926 ms | 3.536 ms | 2193.741 ms |
| Full width | 100M | 4-bit | 8 | 2364.696 ms | 868.582 ms | 6.571 ms | 1488.656 ms |

Four-bit passes use 16 buckets instead of four, so their local bookkeeping is
more expensive. Halving the reduce and scatter traversals still wins by roughly
25%-30%. Scatter remains the largest stage at 62%-65% of GPU time.

## Gates and decision

- All 64 release tests pass on Intel, including stable duplicate ordering,
  odd/even pass parity, command composition, and the portable fallback.
- Intel 10M scan changes from 13.084 to 12.820 ms and compaction from 17.179 to
  15.609 ms; neither unrelated primitive regresses.
- RTX 4070 Ti SUPER 10M controls change by -0.60% to -0.19% across both sorts,
  scan, and compaction. NVIDIA routing is unchanged.
- Direct tests keep key-only sort, non-Vulkan Intel, and adapters below the
  required workgroup limits on the portable path.

The experiment clears the 5% improvement target at every tested size, stays
inside the 2% regression budget, and is accepted. Structured values are in
[`2026-08-09-intel-wide-radix.json`](2026-08-09-intel-wide-radix.json).
