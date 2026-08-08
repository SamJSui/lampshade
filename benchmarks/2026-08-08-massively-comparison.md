# Massively 0.96 Comparison

This report compares the overlapping public GPU APIs in `wgpu-primitives` and
[Massively 0.96.0](https://crates.io/crates/massively). It answers two separate
questions: whether the specialized wgpu primitives are competitive with a broader
Thrust-style library, and which remaining kernels have the strongest optimization evidence.

## Source and method

- Measured `wgpu-primitives` base revision:
  `2ada503c600ce064c0995d507284a7bdaa4d0dc9`
- Pinned Massively version: `0.96.0`, tag revision
  `ef9de55190529be98203aca207edab9d560d312e`
- Runtime isolation: separate processes because `wgpu-primitives` uses wgpu 28 and
  Massively/CubeCL uses wgpu 30.
- Workloads: stable 16-bit-bounded and full-width `u32` key/value radix sort,
  wrapping `u32` exclusive scan, and stable copy-if with a precomputed 50% mask.
- Input: identical deterministic `xorshift32-v1` data in both runners.
- Correctness: readback before timing validates sort order, stability, permutation,
  key/value association, every scan prefix, or every selected compaction value.
- Timing: resident public API call through confirmed GPU completion. Upload,
  readback, and validation are excluded.
- Sampling: at least two seconds and four warmup calls followed by 11 samples;
  100M cases use two warmups and seven samples. Tables report the median of three
  independent process medians.

The public APIs have a real allocation difference. `wgpu-primitives` reuses a
caller-owned output and primitive workspace. Massively returns a fresh owned output
per call, although CubeCL may recycle an allocation after warmup. The benchmark keeps
that cost because it is part of the API applications actually call; this is not a
kernel-only comparison.

There is also a conservative sort-output asymmetry. Massively's public
`radix_sort_by_key` returns the values permuted by their keys, while
`KeyValueSorter` writes both sorted keys and values. Validation reconstructs each
Massively output key from its original value index, so both paths must preserve the
same stable ordering and permutation even though `wgpu-primitives` materializes the
larger public output.

The working directories were dirty only because they contained this uncommitted
comparison harness. The measured production source remained at the revision above.

Test systems:

| System | GPU | TPCs | Backend | Driver |
| --- | --- | ---: | --- | --- |
| RTX | NVIDIA GeForce RTX 4070 Ti SUPER | — | Vulkan | NVIDIA 591.86 |
| dopey | NVIDIA Jetson Orin Nano Super | 8 | Vulkan | NVIDIA 595.78 |
| grumpy | NVIDIA Jetson Orin Nano Super | 4 | Vulkan | NVIDIA 595.78 |

Both Jetsons used `MAXN_SUPER` with CPU, GPU, and EMC clocks pinned during
measurement. Their saved dynamic CPU governors, GPU frequency range, EMC range, and
`nvfancontrol` state were verified after the run. JetPack's restore script printed
internal persistence/path errors, so restoration was confirmed from live settings
rather than inferred from its exit code.

## RTX results

`Speedup` is `Massively time / wgpu-primitives time`; values above `1.0x` favor
`wgpu-primitives`.

| Workload | Items | `wgpu-primitives` | Massively | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Stable sort, bounded 16-bit keys | 1M | 0.177 ms | 1.111 ms | 6.30x |
| Stable sort, full-width keys | 1M | 0.281 ms | 1.108 ms | 3.94x |
| Exclusive scan | 1M | 0.150 ms | 0.373 ms | 2.49x |
| Compact, 50% selected | 1M | 0.180 ms | 0.372 ms | 2.07x |
| Stable sort, bounded 16-bit keys | 10M | 0.914 ms | 7.231 ms | 7.91x |
| Stable sort, full-width keys | 10M | 1.623 ms | 7.222 ms | 4.45x |
| Exclusive scan | 10M | 0.720 ms | 0.940 ms | 1.31x |
| Compact, 50% selected | 10M | 0.943 ms | 0.989 ms | 1.05x |
| Stable sort, bounded 16-bit keys | 100M | 8.371 ms | 165.336 ms | 19.75x |
| Stable sort, full-width keys | 100M | 14.952 ms | 165.799 ms | 11.09x |

The sort advantage grows with size. Massively's radix API does not accept a known
key-width bound, so the bounded row measures a useful capability exposed only by
`wgpu-primitives`; the full-width row is the direct general-case comparison. Scan is
also faster on the RTX, while 10M compaction is approximately tied.

## Jetson results

| Host | Workload | Items | `wgpu-primitives` | Massively | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| dopey, 8 TPC | Bounded 16-bit sort | 1M | 1.376 ms | 9.909 ms | 7.20x |
| dopey, 8 TPC | Full-width sort | 1M | 2.592 ms | 9.904 ms | 3.82x |
| dopey, 8 TPC | Exclusive scan | 1M | 0.760 ms | 1.642 ms | 2.16x |
| dopey, 8 TPC | Compact 50% | 1M | 1.013 ms | 1.648 ms | 1.63x |
| dopey, 8 TPC | Bounded 16-bit sort | 10M | 11.255 ms | 96.007 ms | 8.53x |
| dopey, 8 TPC | Full-width sort | 10M | 21.752 ms | 98.809 ms | 4.54x |
| dopey, 8 TPC | Exclusive scan | 10M | 6.223 ms | 5.415 ms | 0.87x |
| dopey, 8 TPC | Compact 50% | 10M | 7.726 ms | 6.996 ms | 0.91x |
| grumpy, 4 TPC | Bounded 16-bit sort | 1M | 1.400 ms | 10.212 ms | 7.29x |
| grumpy, 4 TPC | Full-width sort | 1M | 2.660 ms | 10.220 ms | 3.84x |
| grumpy, 4 TPC | Exclusive scan | 1M | 1.062 ms | 1.552 ms | 1.46x |
| grumpy, 4 TPC | Compact 50% | 1M | 1.314 ms | 1.621 ms | 1.23x |
| grumpy, 4 TPC | Bounded 16-bit sort | 10M | 11.443 ms | 97.401 ms | 8.51x |
| grumpy, 4 TPC | Full-width sort | 10M | 22.098 ms | 100.381 ms | 4.54x |
| grumpy, 4 TPC | Exclusive scan | 10M | 9.196 ms | 5.452 ms | 0.59x |
| grumpy, 4 TPC | Compact 50% | 10M | 10.558 ms | 6.849 ms | 0.65x |

At 10M, Massively is 1.15x faster than `wgpu-primitives` scan and 1.10x faster
than compaction on dopey. On grumpy it is 1.69x and 1.54x faster. This crossover
is the clearest evidence for the next performance work: our hierarchical scan and
scan-derived compaction do not scale as well on the 4-TPC integrated adapter.

Both 8 GB systems completed and validated the 100M sorts:

| Host | Workload | `wgpu-primitives` | Massively | Speedup |
| --- | --- | ---: | ---: | ---: |
| dopey, 8 TPC | Bounded 16-bit sort | 112.557 ms | 8,290.482 ms | 73.66x |
| dopey, 8 TPC | Full-width sort | 215.014 ms | 8,455.212 ms | 39.32x |
| grumpy, 4 TPC | Bounded 16-bit sort | 113.847 ms | 8,294.100 ms | 72.85x |
| grumpy, 4 TPC | Full-width sort | 217.199 ms | 8,460.408 ms | 38.95x |

Massively remained correct instead of failing allocation, but its natural public API
put the shared-memory systems under much greater allocation pressure and scaled far
worse than linearly from 10M. The 100M result therefore demonstrates support and API
cost; it should not be interpreted as a kernel-only radix comparison.

## 100M scan correctness finding

> **Resolved:** exact scratch binding ranges and a multi-level regression landed in
> fix revision `f9b4982`. All 10M and 100M scan/compaction validations now pass on
> RTX, dopey, and grumpy. The previously withheld results are published in the
> [fix follow-up](2026-08-08-scan-scratch-binding-fix.md). The text below is retained
> as the original discovery record.

The RTX comparison deliberately does not publish a 100M scan or compaction speedup.
All three process-isolated `wgpu-primitives` scan validations first diverged at item
96,471,040. All three compaction validations returned too few selected items because
compaction uses the same exclusive scan to generate offsets. Massively validated at
100M, with medians of 3.100 ms for scan and 5.626 ms for compaction.

The failure is scheduling-sensitive: an uncaptured runner can pass, while the normal
captured benchmark process reproduces it. Inspection identifies the unsafe boundary.
Hierarchical scan levels place `data` and `auxiliary` at different offsets in one
scratch buffer, but `ScanPipeline::dispatch` binds both writable views with
`size: None`. The data view therefore extends through the auxiliary level. Rounded-up
threads use `arrayLength(&data)` and may write into the next level during the add pass,
creating an overlapping writable-storage race. Compaction inherits the bad offsets.

This comparison branch does not change the production kernel. The fix should give
each binding its exact logical byte range, add a multi-level regression above the
`2,048 × 2,048` hierarchy boundary, and rerun the 100M matrix before any new scan or
compaction claim is published.

## Decision

The specialized sort is the strongest differentiator: it wins the direct full-width
case on every device and exposes a bounded-key optimization that Massively lacks.
Massively does not invalidate the crate's direction; it provides evidence for a
narrower next target. After the correctness fix, profile and optimize the portable
integrated scan path, then let compaction inherit that improvement.

The reproduction harness is in
[`massively-comparison/`](massively-comparison/README.md), and the compact process-
median snapshot is
[`2026-08-08-massively-comparison.json`](2026-08-08-massively-comparison.json).
