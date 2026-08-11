# Typed pipeline stabilization

Date: 2026-08-10

This change promotes the recording-first API from the experimental `v2`
namespace to the stable `pipeline` namespace. The old path remains a deprecated
compatibility alias. The implementation moved without changing kernels,
dispatch order, buffer bindings, or workspace behavior.

## Acceptance evidence

- Candidate: `wgpu-primitives` 0.8.0 release candidate on branch
  `feat/stabilize-typed-pipeline`
- Parent revision: `58a3fba57b9c5dcb0a7f9d0cd7d980e9f41b1068`
- Dirty-source manifest: `c7bc8d743773f0f3ce2beae507b88364fabf36223946cfabcf72885363631224`
- Adapter: NVIDIA GeForce RTX 4070 Ti SUPER, Vulkan, driver 591.86
- Baseline: crates.io `wgpu-primitives` 0.7.0
- Gate: candidate median may be at most 2% slower than the published median
- Aggregation: median of independent process medians

The complete three-process matrix initially passed 11 of 15 rows:

| Workload | Items | Published | Candidate | Change | Initial gate |
| --- | ---: | ---: | ---: | ---: | :---: |
| Sum reduction | 1M | 0.0861 ms | 0.0895 ms | +3.95% | Recheck |
| Bounded-16 sort | 1M | 0.1811 ms | 0.1795 ms | -0.88% | Pass |
| Full-width sort | 1M | 0.2828 ms | 0.2930 ms | +3.61% | Recheck |
| Exclusive scan | 1M | 0.1164 ms | 0.1174 ms | +0.86% | Pass |
| 50% compaction | 1M | 0.1322 ms | 0.1321 ms | -0.08% | Pass |
| Sum reduction | 10M | 0.0999 ms | 0.0994 ms | -0.50% | Pass |
| Bounded-16 sort | 10M | 0.9164 ms | 0.9143 ms | -0.23% | Pass |
| Full-width sort | 10M | 1.6228 ms | 1.6104 ms | -0.76% | Pass |
| Exclusive scan | 10M | 0.3373 ms | 0.3449 ms | +2.25% | Recheck |
| 50% compaction | 10M | 0.4859 ms | 0.4859 ms | 0.00% | Pass |
| Sum reduction | 100M | 0.7108 ms | 0.7309 ms | +2.83% | Recheck |
| Bounded-16 sort | 100M | 7.9261 ms | 7.9308 ms | +0.06% | Pass |
| Full-width sort | 100M | 14.5089 ms | 14.4909 ms | -0.12% | Pass |
| Exclusive scan | 100M | 2.8319 ms | 2.8422 ms | +0.36% | Pass |
| 50% compaction | 100M | 3.7204 ms | 3.7281 ms | +0.21% | Pass |

The four failures were repeated with nine alternating processes under the
repository's documented targeted-recheck protocol:

| Rechecked row | Published | Candidate | Change | 2% gate |
| --- | ---: | ---: | ---: | :---: |
| Sum reduction, 1M | 0.0833 ms | 0.0825 ms | -0.96% | Pass |
| Full-width sort, 1M | 0.2840 ms | 0.2842 ms | +0.07% | Pass |
| Exclusive scan, 10M | 0.3387 ms | 0.3422 ms | +1.03% | Pass |
| Sum reduction, 100M | 0.7246 ms | 0.7197 ms | -0.68% | Pass |

This resolves every initial failure. The adjacent JSON preserves all initial
and recheck process medians, timestamps, adapter identity, commands, and source
manifest.

The affected counted shader had already passed direct parent comparisons at
1M, 10M, and 100M (+0.66%, +0.07%, and +0.03%), and the repository-owned
consumer passed typed-versus-raw total-time comparisons on discrete RTX and
integrated Intel. Those results remain applicable because this promotion only
moves the Rust module and changes first-party import paths.

## Correctness and release checks

- `cargo fmt --all --check`
- strict Clippy for all targets, the standalone consumer, and the deprecated
  forwarding package
- rustdoc with warnings denied
- `cargo test --release --all-targets`, including every 100M validator
- stable `pipeline` physical-GPU tests and deprecated `v2` compile coverage
- offline `cargo package --allow-dirty` verification
- release-regression Python unit tests under WSL

All checks pass. Apple and Jetson remain useful additional coverage, but no
result is inferred for adapters absent from this stabilization run.
