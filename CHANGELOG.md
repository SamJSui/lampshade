# Changelog

All notable changes to Lampshade are documented here. Releases before 0.8 used
the `wgpu-primitives` package name.

## Unreleased

### Changed

- Move the public GPU runtime from wgpu 29.0.4 to wgpu 30.0.1. Lampshade API
  names, buffer layouts, and kernels are unchanged, but application-owned wgpu
  types must come from wgpu 30. Lampshade 0.12 remains available for wgpu 29
  consumers; see the [0.13 migration note](docs/migration-0.13.md).
- Compare the 0.13 candidate with published Lampshade 0.12.1 using the release
  harness's cross-runtime characterization mode instead of applying the
  same-runtime 2% regression gate.

### Added

- Restore `Error::MapRange` so wgpu 30 mapped-range access failures propagate
  through Lampshade's host-returning and profiling APIs.

## 0.12.1 - 2026-08-13

### Changed

- Route GPU-counted key/value sorts through the existing 4-bit radix
  variant on compatible Intel Vulkan adapters instead of always using the
  2-bit portable kernel. Apple Metal retains the measured 2-bit baseline.
- Move the published-release performance baseline to Lampshade 0.12.0 so
  subsequent WGPU 29 changes compare against this patch line.

### Performance

- Reduce the full portable SoA pack-sort-unpack pipeline by 17.02% at 1M and
  25.29% at 10M items on Intel Alder Lake-N Vulkan. The
  [counted-radix report](benchmarks/2026-08-13-portable-soa-counted-radix.md)
  records process medians, physical correctness, the unchanged RTX native
  path, and the rejected Metal experiments.

## 0.12.0 - 2026-08-13

### Added

- Lexicographic `ArgminByKey` selection for fixed and GPU-counted `KeyValue`
  buffers, including slice, immediate, recording, timestamp-profiling, typed
  pipeline, validation, example, and raw-kernel comparison coverage.
- A public API contract that classifies convenience, immediate-submit,
  recording, prepared-recording, and profiling behavior, including their
  synchronization and resource-creation boundaries.
- CI gates for Rust 1.87 MSRV compatibility, warning-denied rustdoc,
  all-feature WebAssembly compilation, extracted-package consumer compilation,
  and semver compatibility. The 0.12 check records its intentional breaking
  pre-1.0 cleanup against 0.11; versions after 0.12 are checked as
  patch-compatible against the 0.12 baseline.

### Fixed

- Restore `wasm32-unknown-unknown` library compilation by retaining transient
  WebGPU resources in thread-local storage until submitted work completes.

### Changed

- Move the published-release performance baseline to Lampshade 0.11.0 so
  future WGPU 29 changes are compared against the current release.
- Remove the deprecated `lampshade::v2` alias. Import the unchanged typed API
  from `lampshade::pipeline`.
- Make `Error` non-exhaustive so future validation improvements do not force a
  major release. Downstream matches now require a fallback arm.
- Explicitly scope `pipeline` to GPU-counted composition. Scan, histogram, and
  separate-buffer SoA sorting remain supported direct primitive APIs.

## 0.11.0 - 2026-08-12

### Added

- Add a fixed-count `KeyValueSoaSorter` API that owns the count metadata needed
  by the native GPU-counted kernel, removing that workaround from downstream
  renderers.
- Add `KeyValueSoaSorter::requirements` to merge the native SoA feature and
  compute-limit contract into an application's existing wgpu device request.
- Add a transparent portable SoA backend that packs separate key/value buffers,
  invokes the existing stable key/value sorter, and unpacks the selected prefix
  without CPU readback.

### Changed

- Keep `new_for_adapter` as a compatibility alias while making
  `KeyValueSoaSorter::new` the portable, adapter-selected constructor.

### Performance

- Record the clean 0.10.1-to-0.11 RTX regression gate, targeted noisy-row
  recheck, and physical SoA correctness validation on NVIDIA Vulkan, Apple
  Metal, and Intel Vulkan.

## 0.10.1 - 2026-08-12

### Changed

- Replace the pre-release path dependency in the README with the published
  Lampshade 0.10 and wgpu 29 dependencies.
- Move the published-release regression baseline to Lampshade 0.10 so future
  gates compare wgpu 29 against the same wgpu 29 runtime stack.

## 0.10.0 - 2026-08-12

### Added

- Add a native stable structure-of-arrays radix sorter for separate `u32` key
  and value buffers whose exact item count remains GPU-resident.
- Cache counted-sort bindings so prepared per-frame recording performs no
  buffer, bind-group, or pipeline creation.

### Changed

- Move the public GPU runtime from wgpu 30 to wgpu 29 for compatibility with
  the strongest identified graphics and simulation adoption targets. Because
  wgpu types appear in Lampshade's public API, this is a breaking 0.10 change;
  the 0.9 release and `release/wgpu30` branch preserve the wgpu 30 line.
- Route counted `KeyValue` sorting through the four-pass 8-bit backend on
  compatible discrete NVIDIA Vulkan adapters; portable adapters retain the
  existing counted sorter.
- Remove `Error::MapRange`, which has no wgpu 29 equivalent because mapped
  range access is infallible in that release. This is a public enum change.

### Performance

- Add source-pinned RTX regression evidence against Lampshade 0.9/wgpu 30 and
  downstream sorter-stage evidence for `wgpu-3dgs-viewer`; reports distinguish
  algorithmic gains from the WGPU runtime version.
- Record the M3 Pro migration result: correctness passes, but the WGPU 29
  completion boundary makes host-synchronized Metal workloads slower than the
  0.9/wgpu 30 migration control. This is accepted as the 0.10 wgpu 29 baseline;
  future release gates compare against the previous wgpu 29 Lampshade release.

## 0.9.0 - 2026-08-11

### Added

- Adjacent `u32` run-length encoding with slice, immediate GPU-buffer,
  recording, timestamp-profiling, and typed-pipeline APIs.
- Capacity-bounded RLE driven by a GPU-resident input count, producing unique
  values and lengths that share a new GPU-resident run count without readback.
- Deterministic boundary, alias, binding-limit, nonzero-offset, counted-clamp,
  one-submission composition, example, and Criterion coverage for RLE.

### Fixed

- Keep transient bind groups, uniform buffers, and counted-reduction metadata
  alive through submission instead of relying on backend handle retention.
- Pool predicate parameters and counted-reduction metadata so multiple recorded
  operations cannot overwrite state that an earlier submission still uses.
- Complete staging-buffer copies before starting host mapping, fixing recycled
  readback data observed across repeated Jetson Orin Vulkan calls.
- Disable optional subgroup and timestamp-query features in the convenience
  context on integrated NVIDIA Vulkan, where those configurations made
  repeated compute/readback dispatches unreliable. Portable kernels remain
  enabled.
- Initialize zero reduction identities with buffer clears and copy the minimum
  identity from offset zero, avoiding incorrect packed-source offsets on Orin.
- Reject aliased scan input/output handles before command recording, including
  one-item copies and ranged internal scans.
- Validate scan and predicate logical bindings against the device's effective
  storage-binding limit before creating bind groups.
- Disable unreliable Apple Metal timestamp profiling in the convenience
  context, and return a timestamp-result error instead of reporting a false
  zero-duration span when a custom device leaves a query pair unwritten.
- Require an actual Vulkan adapter for CI GPU tests instead of silently passing
  skipped integration coverage.

### Changed

- Move the published-release regression baseline from the pre-rename
  `wgpu-primitives` 0.7 package to the published `lampshade` 0.8 package.
- Lazily initialize typed-facade primitives so construction and reservation
  prepare only the operations callers request.
- Route fixed-length `u32` sorting on compatible discrete NVIDIA Vulkan
  adapters through the generalized 8-bit radix kernel; GPU-counted sorting and
  other adapters retain the portable path.
- Document the 1.87 MSRV required by wgpu 30 and clarify benchmark workload and
  historical hardware-test scope.

### Performance

- The adapter-selected key-only path reduces full-width fixed-sort passes from
  sixteen 2-bit passes to four 8-bit passes. Exact RTX before/after evidence is
  recorded in `benchmarks/2026-08-11-key-only-sort.md`.
- RLE now uses one dense bind group and separate mark, scatter, and finalize
  stages. On the RTX release candidate, fixed 100M RLE measured 7.225 ms versus
  8.036 ms in the original implementation report.
- The complete crates.io 0.8 regression gate and four-adapter correctness
  matrix are recorded in `benchmarks/2026-08-11-lampshade-0.9-release.md`.

## 0.8.0 - 2026-08-11

### Added

- Capacity-bounded `u32` sort and reduction APIs driven by a GPU-resident item
  count, including immediate submission, command recording, and timestamp
  profiling. Preparation kernels clamp the count and build indirect dispatch
  arguments without a host readback.
- End-to-end GPU coverage for zero counts, hierarchy boundaries, oversized
  count clamping, invalid buffer contracts, and one-submission predicate,
  compaction, sort, and reduction composition.
- Criterion cases comparing fixed-length and GPU-counted resident sort and
  reduction at the same capacity.
- A reusable `GpuCountPlan` that binds one resident count, prepares sort and
  reduction metadata once, and exposes explicit indirect or capacity-based
  counted-sort scheduling.
- An end-to-end Criterion comparison of one-submission GPU-counted composition
  against CPU count readback followed by fixed-length sort and reduction.
- A stable `pipeline` recording API with typed `u32` and `KeyValue` buffer
  ranges, CPU-fixed or GPU-resident extents, automatic shared-count
  preparation, and operation-specific workspace and count-metadata
  reservation.
- Nonzero buffer-offset coverage for compaction, fixed and GPU-counted sort,
  and fixed and GPU-counted reduction, including alignment and alias errors.
- Typed stable key/value predicate, compaction, and GPU-counted radix-sort
  composition, including a particle-shaped example and repository-owned
  crate-boundary validation application.
- A typed-pipeline guide that records buffer ownership, layout scope, and the
  correctness and performance evidence used for stabilization.

### Changed

- The project is renamed from `wgpu-primitives` to `lampshade`. Package imports,
  repository metadata, examples, validation, and active benchmark harnesses use
  the new name; kernels and public behavior are unchanged.
- The resident pipeline example now uses the compactor's actual GPU-produced
  count through the typed recorder for downstream sort and reduction instead
  of manual count-plan ordering or a CPU-known selected length.
- The former `v2` namespace remains as a deprecated compatibility alias for
  `pipeline`; existing raw-buffer and explicit-plan APIs remain unchanged.
- The obsolete `wgpu-algorithms` forwarding package is discontinued. A final,
  frozen `wgpu-primitives` 0.8 compatibility package reexports Lampshade and
  directs consumers to the new crate.
- The published-release regression baseline now targets crates.io 0.7.0.

### Performance

- The typed-pipeline 0.7-versus-0.8 fixed-path gate passed all 15 workload/size rows.
  Eleven passed the initial three-process matrix; four noisy rows passed
  nine-process targeted rechecks between -0.96% and +1.03%.
- The final Lampshade source passed 14 of 15 rows initially; the sole 10M
  reduction miss passed a nine-process recheck at +1.62%. All 100M changes
  remained between -0.56% and +0.38%.

## 0.7.0 - 2026-08-09

### Added

- Portable 1-256-bin `u32` histogram counting with slice, immediate
  resident-buffer, command-recording, and timestamp-profiling APIs. Values
  outside the requested bin range are ignored.
- A composed `predicate -> compact -> sort -> reduce` example using one command
  encoder, one submission, and one final readback allocation.
- A cross-platform published-release regression harness that compares identical
  checkout and crates.io workloads, emits machine-readable evidence, and fails
  above a configurable 2% budget.

### Changed

- Installation documentation now points to 0.7 and documents histogram as a
  published API.

## 0.6.0 - 2026-08-09

### Added

- Portable hierarchical `u32` sum, minimum, and maximum reduction, with slice,
  immediate resident-buffer, command-recording, and timestamp-profiling APIs.
- Deterministic reduction correctness coverage across workgroup and hierarchy
  boundaries, including empty-input identities and wrapping sum overflow.
- Physical Intel Alder Lake-N Vulkan validation, including all 64 release
  tests and reproducible 1M/10M comparisons against pinned Massively 0.96.
- An architecture guide defining the slice convenience, resident composition,
  and private kernel/runtime layers and the evidence required before a crate
  split.

### Changed

- Upgrade the public GPU runtime from wgpu 28 to wgpu 30, requiring downstream
  users of the GPU-buffer APIs to upgrade their wgpu dependency. This is a
  breaking pre-1.0 dependency change and advances the crate to 0.6. The
  migration includes explicit mapped-range error handling and the new instance,
  adapter, and pipeline-layout descriptor fields.
- Disable redundant backend workgroup-memory zero fills. Every crate shader
  initializes each shared value before its first read, and cross-adapter GPU
  correctness suites cover the affected kernels.
- Immediate submission, timestamp profiling, reusable buffer ownership, and
  adapter capability capture now use shared private engine components. Apart
  from the wgpu type-version change above, public method names, WGSL kernels,
  dispatch sizes, and command order are unchanged.
- Capable Intel Vulkan adapters now select the existing 4-bit stable key-value
  radix kernels; key-only sort, other backends, and devices below the required
  workgroup limits retain the portable 2-bit path.

### Performance

- The selected 128-thread, 32-item-per-thread portable reduction shape cuts
  100M sum dispatch time by 37.5% on Intel Alder Lake-N and 17.2% on Apple M3
  Pro relative to the initial 256-thread, 8-item implementation, while RTX
  dispatch remains within 1%.
- At 100M values, end-to-host wrapping sum is 1.94x faster than Massively on RTX,
  1.04x faster on Intel, and 1.11x faster on Apple. The wgpu 30 migration reverses
  the previous Apple host-boundary loss without adding vendor-specific code.
- On Intel Alder Lake-N, the 4-bit radix route reduces stable key-value sort
  latency by 24.46%-29.89% across 1M-100M items relative to the portable path.
- At 10M items on Intel, `wgpu-primitives` leads Massively by 4.33x for bounded
  stable sort, 2.24x for full-width stable sort, 2.75x for exclusive scan, and
  2.67x for 50%-selective stable compaction. The earlier 100M Intel sweep
  measured corresponding speedups of 9.78x, 4.79x, 2.44x, and 2.52x.
- Identical wgpu 28/30 controls pass a 2% regression gate on Apple M3 Pro, Intel,
  and RTX 4070 Ti SUPER. The largest measured increase is 1.55% for RTX bounded
  sort; stabilized Apple bounded sort improves by 1.74%, and Intel scan and
  compaction improve by 5.79% and 5.26%.

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
