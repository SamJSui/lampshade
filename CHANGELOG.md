# Changelog

All notable changes to `wgpu-primitives` are documented here.

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
