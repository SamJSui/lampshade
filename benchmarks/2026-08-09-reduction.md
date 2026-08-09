# Portable `u32` reduction

Date: 2026-08-09

This change adds wrapping sum, minimum, and maximum reduction over `u32` values.
It provides slice convenience, immediate GPU-to-GPU, command-recording, and GPU
timestamp APIs. Empty reductions return the operation identity: `0` for sum and
maximum, and `u32::MAX` for minimum.

## Kernel and ownership model

Each 128-thread workgroup reads 32 coalesced values per thread and reduces its
4,096-value range in shared workgroup memory. One partial is written per
workgroup; later passes repeat until one scalar remains. Two grow-only scratch
buffers ping-pong between hierarchy levels.

`Reducer` therefore takes `&mut self`: a call may grow those private scratch
buffers. `record_reduce` instead takes `&mut CommandEncoder` because it appends
GPU commands to caller-owned state; it does not submit. The input and output are
borrowed buffer handles, while slice methods return a copied `u32` after `.await`.

## Method

- Input: deterministic xorshift32 values, validated against wrapping CPU sum.
- Sizes: 1M, 10M, and 100M values.
- Comparison: pinned Massively 0.96.0 at revision
  `ef9de55190529be98203aca207edab9d560d312e`.
- Boundary: resident GPU input through returned host scalar for both libraries;
  upload and validation are excluded, while the four-byte readback is included.
- Sampling: at least two seconds of warmup per process; 11 samples at 1M/10M
  and seven at 100M. RTX and Apple use three process medians; Intel uses five.

## Massively comparison

`Speedup` is Massively time divided by `wgpu-primitives` time.

| Adapter | Items | `wgpu-primitives` | Massively | Speedup |
| --- | ---: | ---: | ---: | ---: |
| RTX 4070 Ti SUPER, Vulkan | 1M | 0.082 ms | 0.290 ms | 3.55x |
| RTX 4070 Ti SUPER, Vulkan | 10M | 0.100 ms | 0.294 ms | 2.94x |
| RTX 4070 Ti SUPER, Vulkan | 100M | 0.747 ms | 1.038 ms | 1.39x |
| Intel Alder Lake-N, Vulkan | 1M | 1.485 ms | 2.458 ms | 1.66x |
| Intel Alder Lake-N, Vulkan | 10M | 3.676 ms | 4.471 ms | 1.22x |
| Intel Alder Lake-N, Vulkan | 100M | 22.326 ms | 22.724 ms | 1.02x |
| Apple M3 Pro, Metal | 1M | 1.729 ms | 0.524 ms | 0.30x |
| Apple M3 Pro, Metal | 10M | 1.716 ms | 0.761 ms | 0.44x |
| Apple M3 Pro, Metal | 100M | 4.643 ms | 3.493 ms | 0.75x |

The Apple deficit is primarily the synchronous host boundary. A resident 100M
profile measured 3.065 ms wall time and 2.838 ms across reduction dispatches;
roughly 1.6 ms is added when the scalar is made visible to the host. Applications
that consume the scalar on the GPU should use `record_reduce` or
`reduce_gpu_to_gpu` and avoid that synchronization.

## Tuning evidence

The initial portable shape used 256 threads and eight values per thread. Testing
one parameter pair at a time selected 128 threads and 32 values per thread:

| Adapter | Initial 100M dispatch | Selected dispatch | Change |
| --- | ---: | ---: | ---: |
| RTX 4070 Ti SUPER | 0.629 ms | 0.634 ms | +0.8% |
| Intel Alder Lake-N | 32.965 ms | 20.592 ms | -37.5% |
| Apple M3 Pro | 3.426 ms | 2.838 ms | -17.2% |

The selected shape won on Intel and Apple while staying within 1% of the initial
RTX result, so no vendor-specific route was added. Intel's 24-, 40-, and 64-item
variants and smaller/larger workgroups were slower.

## Validation

- All 74 release tests pass on Intel Vulkan and Apple Metal.
- Boundary, overflow, empty-identity, explicit-length, multi-level hierarchy,
  invalid-buffer, and same-encoder composition tests pass.
- 1M, 10M, and 100M CPU-reference validation passes on all three adapters.
- Same-session `main` versus branch controls stayed within the 2% regression
  budget; the largest increase was 0.20%.

Jetson coverage was unavailable: both previously used Jetson hosts were offline
during this run.

### Existing-primitive regression controls

These 10M controls compare merged `main` at
`bcb8b7b9aaa064802f715b2608827c5da0256ae2` with this branch in isolated
worktrees. Negative change is faster.

| Adapter | Workload | `main` | Branch | Change |
| --- | --- | ---: | ---: | ---: |
| RTX 4070 Ti SUPER | Bounded stable sort | 0.9323 ms | 0.9342 ms | +0.20% |
| RTX 4070 Ti SUPER | Full-width stable sort | 1.6243 ms | 1.6252 ms | +0.06% |
| RTX 4070 Ti SUPER | Exclusive scan | 0.3627 ms | 0.3519 ms | -2.98% |
| RTX 4070 Ti SUPER | 50% stable compaction | 0.5151 ms | 0.4964 ms | -3.63% |
| Intel Alder Lake-N | Bounded stable sort | 131.851 ms | 126.282 ms | -4.22% |
| Intel Alder Lake-N | Full-width stable sort | 263.336 ms | 251.962 ms | -4.32% |
| Intel Alder Lake-N | Exclusive scan | 13.463 ms | 13.114 ms | -2.59% |
| Intel Alder Lake-N | 50% stable compaction | 16.830 ms | 16.117 ms | -4.24% |

The benchmarked candidate was the dirty `feat/reduction` working tree with
`bcb8b7b9aaa064802f715b2608827c5da0256ae2` as its parent. The PR commit provides
the immutable candidate revision; raw runner records retain `dirty: true`.
