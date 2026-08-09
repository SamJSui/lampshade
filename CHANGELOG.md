# Changelog

All notable changes to `wgpu-primitives` are documented here.

## Unreleased

### Added

- Physical Intel Alder Lake-N Vulkan validation, including all 64 release
  tests and reproducible 1M/10M comparisons against pinned Massively 0.96.
- An architecture guide defining the slice convenience, resident composition,
  and private kernel/runtime layers and the evidence required before a crate
  split.

### Changed

- Immediate submission, timestamp profiling, reusable buffer ownership, and
  adapter capability capture now use shared private engine components. Public
  APIs, WGSL kernels, dispatch sizes, and command order are unchanged.
- Capable Intel Vulkan adapters now select the existing 4-bit stable key-value
  radix kernels; key-only sort, other backends, and devices below the required
  workgroup limits retain the portable 2-bit path.

### Performance

- On Intel Alder Lake-N, the 4-bit radix route reduces stable key-value sort
  latency by 24.46%-29.89% across 1M-100M items relative to the portable path.
- At 10M items, `wgpu-primitives` leads Massively by 4.46x for bounded stable
  sort, 2.24x for full-width stable sort, 2.62x for exclusive scan, and 2.59x
  for 50%-selective stable compaction. At 100M, the corresponding speedups are
  9.78x, 4.79x, 2.44x, and 2.52x.
- Identical pre/post migration controls pass a 2% regression gate on Intel and
  RTX 4070 Ti SUPER. The largest RTX increase is 0.57%; a targeted five-process
  Intel compaction control changes by -0.02%.

## 0.5.0 - 2026-08-09

### Added

- Stable `u32` stream compaction from caller-provided 0/1 masks, including
  slice, immediate GPU submission, command recording, resident output count,
  and GPU timestamp profiling APIs.
- Stable `KeyValue` stream compaction with matching slice, resident-buffer,
  command-recording, resident-count, and GPU timestamp profiling APIs.
- Reusable GPU predicate masks for `u32` values and either field of `KeyValue`
  records, with slice, immediate GPU submission, command-recording, and GPU
  timestamp profiling APIs.
- Opt-in `*_with_key_bits` sort APIs for host slices, immediate GPU submission,
  command recording, and GPU timestamp profiling.
- Host-slice validation for declared key widths, including explicit errors for
  invalid widths and keys outside the declared range.

### Changed

- Portable and wide radix paths now record only the passes required by a
  declared key width and route both odd and even pass counts to the caller's
  output buffer. Existing APIs remain full-width by default.
- Hierarchical scan dispatches bind exact logical data and auxiliary ranges,
  preventing deeper scratch levels from overlapping. A regression now covers
  the first three-level high-end hierarchy size.
- The top scan pass reads caller input and writes caller output directly instead
  of copying the full buffer first. Devices with enabled subgroups use a
  coalesced one-item-per-lane scan; other devices retain the portable fallback.
- Stable compaction keeps level-zero offsets block-local and combines them with
  scanned block totals during scatter, removing the full-size prefix-add pass.
- The POSIX Massively comparison now records individual runner failures and
  continues the matrix, matching the PowerShell harness. Failed implementations
  remain explicit and are excluded from timing comparisons.

### Performance

- Against pinned Massively 0.96, exclusive scan is 2.81x faster at 10M and
  1.09x faster at 100M on an RTX 4070 Ti SUPER. On 8-TPC and 4-TPC Jetson Orin
  Nano configurations it is 1.49x-1.53x faster at 10M and 1.18x-1.19x at
  100M. Scan-derived 50% compaction leads by 1.09x-1.65x across the same cases.
- Fusing compaction's block prefix improves the PR #17 compaction baseline by
  19.9%-27.3% across RTX and both Jetsons. Against pinned Massively 0.96,
  50%-selective compaction is 2.06x/1.52x faster at 10M/100M on RTX,
  1.66x/1.52x on the 8-TPC Jetson, and 1.63x/1.49x on the 4-TPC Jetson.
- On an Apple M3 Pro through Metal, all 64 release tests and 100M benchmark
  validators pass. At 100M, exclusive scan measures 13.736 ms and 50%-selective
  compaction measures 18.302 ms. Pinned Massively 0.96 cannot initialize the
  compared Metal pipelines because generated layouts require 42 or 47 storage
  buffers against the adapter limit of 29, so no speedup is reported.

- On an RTX 4070 Ti SUPER, resident predicate-mask wall time measured 0.218 ms
  for 10 million values and 1.379 ms for 100 million values, 4.80x and 17.28x
  faster than the scalar CPU mask. Isolated GPU timestamps measured 0.127 ms
  and 1.268 ms. CPU-reference validation passed at both sizes on 8-TPC and
  4-TPC Jetson Orin Nano systems; their predicate times were effectively tied
  at 100 million items.
- At 10 million items and 50% selectivity, resident compaction measured
  0.892 ms on an RTX 4070 Ti SUPER, 7.601 ms on an 8-TPC Jetson Orin Nano,
  and 10.428 ms on a 4-TPC Jetson Orin Nano.
- At 10 million `KeyValue` records and 50% selectivity, resident compaction
  measured 0.973 ms on the RTX, 7.873 ms on the 8-TPC Jetson, and 10.903 ms
  on the 4-TPC Jetson. Same-source `u32` controls remained within 0.5% of the
  published baseline on all three systems.
- On physical 8-TPC and 4-TPC Jetson Orin Nano configurations, the explicit
  16-bit path reduced 10-million-pair portable Vulkan latency by 52.2% and
  51.3% respectively. Full-width changes were +0.01% and +0.004%.

### Compatibility

- The deprecated `wgpu-algorithms` forwarding package is updated to 0.5.0.

## 0.4.0 - 2026-08-07

### Added

- Capability-gated GPU timestamp profiling for scan, key-only radix sort, and
  stable key-value radix sort, including labeled dispatch spans.
- An adapter-selected 8-bit stable key-value radix path for discrete NVIDIA
  Vulkan GPUs with 32-wide subgroups.
- A reproducible comparison harness pinned to `wgpu_sort` commit
  `4cb640e8cae28eba0149d470c5168cc2853466dd`, with deterministic correctness
  validation and machine-readable results.

### Changed

- The NVIDIA Vulkan key-value path computes all four byte histograms in one
  input read and skips identity scatter passes for constant upper bytes.
- Profiling is opt-in per call; normal execution does not allocate timestamp
  queries or readback buffers.

### Performance

- On the documented RTX 4070 Ti SUPER Vulkan system, 100 million stable pairs
  with 16-bit keys measured 8.605 ms versus 14.884 ms for the pinned
  `wgpu_sort` revision: 42.2% lower latency, or a 1.73x speedup.
- Random full-width keys measured 15.457 ms versus 15.907 ms: a modest 2.8%
  latency reduction. These results apply to the documented GPU, driver,
  backend, input distributions, and resident timing boundary.

### Compatibility

- The portable 2-bit path remains available on other adapters and backends.
- The deprecated `wgpu-algorithms` forwarding package is updated to 0.4.0.
