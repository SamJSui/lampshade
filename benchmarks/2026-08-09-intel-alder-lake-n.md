# Intel Alder Lake-N Vulkan validation

This run adds Intel integrated graphics to the physical-hardware matrix and
checks the internal runtime/workspace migration against the exact pre-change
revision. The Intel path uses the portable radix implementation, so it
exercises a different backend from the NVIDIA subgroup-specialized path.

## System and method

- Beelink Mini S13 with an Intel N150 and integrated Intel Alder Lake-N
  graphics (`8086:46d4`).
- Ubuntu 24.04, x86-64, Vulkan through the Intel open-source Mesa driver
  25.2.8.
- Balanced power profile with dynamic clocks; CPU governor reported
  `powersave`. Clocks were not pinned.
- `wgpu-primitives` 0.5.0 / wgpu 28, based on revision
  `cc9fe339ac9def0839d16188cb680b34a373197c`.
- The uncommitted benchmarked production sources are identified by the ordered
  SHA-256 manifest digest
  `46d7cb6438901d12d59923de92d490cc6e22ec388f8b3201608e471dc7ec3c13`;
  per-file hashes are recorded in the JSON artifact.
- Massively 0.96.0 at
  `ef9de55190529be98203aca207edab9d560d312e`, using CubeCL and wgpu 30.

The timing boundary is the resident public GPU API call through confirmed GPU
completion. Upload, validation readback, and result download are excluded.
Each normal case warms for at least two seconds and four calls, retains 11
samples, and reports the median of three independent process medians. Every
process validates deterministic output before timing. The 100M cases use two
warmups and seven retained samples per process.

## Intel results

`Speedup` is `Massively time / wgpu-primitives time`.

| Workload | Items | `wgpu-primitives` | Massively | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Bounded 16-bit stable sort | 1M | 20.338 ms | 47.563 ms | 2.34x |
| Full-width stable sort | 1M | 40.645 ms | 48.212 ms | 1.19x |
| Exclusive scan | 1M | 2.543 ms | 6.049 ms | 2.38x |
| Stable compaction, 50% | 1M | 2.583 ms | 5.248 ms | 2.03x |
| Bounded 16-bit stable sort | 10M | 174.815 ms | 563.336 ms | 3.22x |
| Full-width stable sort | 10M | 352.525 ms | 588.937 ms | 1.67x |
| Exclusive scan | 10M | 13.084 ms | 33.872 ms | 2.59x |
| Stable compaction, 50% | 10M | 17.179 ms | 41.848 ms | 2.44x |
| Bounded 16-bit stable sort | 100M | 1769.833 ms | 12801.820 ms | 7.23x |
| Full-width stable sort | 100M | 3532.261 ms | 12547.157 ms | 3.55x |
| Exclusive scan | 100M | 129.370 ms | 315.466 ms | 2.44x |
| Stable compaction, 50% | 100M | 160.307 ms | 404.164 ms | 2.52x |

All 64 release tests passed on the Intel adapter, including the forced
no-subgroup fallback, multi-level scan hierarchy, composed command recording,
profiling, stable sort, and stable compaction tests. The adapter reports
subgroup sizes from 8 through 32, but Intel does not satisfy the NVIDIA Vulkan
sort policy and correctly selects the portable implementation.

## Migration regression gate

The migration changes Rust ownership boundaries around command sessions,
profiling, capability snapshots, and reusable buffers. It does not change WGSL,
dispatch dimensions, bind groups, or pass order.

| System | Workload | Items | Before | After | Change |
| --- | --- | ---: | ---: | ---: | ---: |
| Intel | Bounded sort | 1M | 20.673 ms | 20.338 ms | -1.62% |
| Intel | Full-width sort | 1M | 41.269 ms | 40.645 ms | -1.51% |
| Intel | Exclusive scan | 1M | 2.842 ms | 2.543 ms | -10.51% |
| Intel | Compaction | 1M | 3.127 ms | 2.583 ms | -17.41% |
| Intel | Bounded sort | 10M | 177.956 ms | 174.815 ms | -1.76% |
| Intel | Full-width sort | 10M | 353.554 ms | 352.525 ms | -0.29% |
| Intel | Exclusive scan | 10M | 13.462 ms | 13.084 ms | -2.81% |
| RTX 4070 Ti SUPER | Bounded sort | 10M | 0.911 ms | 0.916 ms | +0.57% |
| RTX 4070 Ti SUPER | Full-width sort | 10M | 1.610 ms | 1.606 ms | -0.29% |
| RTX 4070 Ti SUPER | Exclusive scan | 10M | 0.339 ms | 0.340 ms | +0.32% |
| RTX 4070 Ti SUPER | Compaction | 10M | 0.483 ms | 0.483 ms | -0.12% |

The first three-process Intel compaction comparison moved from 16.827 ms to
17.179 ms (`+2.09%`), narrowly outside the 2% acceptance gate while its 1M
control improved sharply. A targeted five-process rerun from a separate clean
baseline worktree measured 16.099 ms before and 16.096 ms after (`-0.02%`). The
isolated control therefore identifies the first difference as run variance,
not a reproducible regression.

## Decision

Intel Vulkan is validated for the current primitive set through 100M items.
`wgpu-primitives` beats pinned Massively 0.96 in every overlapping Intel row,
and the internal migration passes its 2% Intel and RTX regression controls.
Exact aggregate values and source-state notes are in
[`2026-08-09-intel-alder-lake-n.json`](2026-08-09-intel-alder-lake-n.json).
