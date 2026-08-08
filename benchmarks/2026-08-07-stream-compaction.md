# Stable Stream Compaction Baseline

This report validates the first stable `u32` stream-compaction primitive in
`wgpu-primitives`. A caller-provided 0/1 mask is exclusively scanned into
destination indices, selected values are scattered in input order, and the
selected count remains in a caller-owned GPU buffer.

## Source and method

- Source base: `1c19619bb83939a9fcdf9008f3cc7400289a3818`
- Measured source package SHA-256:
  `C1FA5E57E14826C1762DCC30A40B01319A5C971E81442BD17D331F4AB1C57B39`
- Workload: 10 million resident `u32` values with periodic 0%, 10%, 50%, 90%,
  and 100% selectivity masks
- Backend: Vulkan
- Timing: 11 samples after a two-second warmup; tables report medians
- Validation: each profiled workload was compared with a stable CPU reference
  after timing

Resident wall time includes command encoding, submission, GPU execution, and
the GPU wait. GPU elapsed spans the recursive exclusive scan and stable scatter
dispatches. Allocation, upload, CPU-reference generation, and readback are
outside the timed region.

After the physical runs, only profiler/test presentation coverage and the
result documentation were changed. The measured library implementation and
shader remained unchanged.

## Systems

| Host | GPU | Driver | Active GPU TPCs | Controlled clocks |
| --- | --- | --- | ---: | --- |
| RTX desktop | NVIDIA GeForce RTX 4070 Ti SUPER | 591.86 | - | normal desktop state |
| `dopey` | NVIDIA Tegra Orin | 595.78 | 8 | CPU 1.728 GHz, GPU 1.020 GHz, EMC 3.199 GHz |
| `grumpy` | NVIDIA Tegra Orin | 595.78 | 4 | CPU 1.728 GHz, GPU 1.020 GHz, EMC 3.199 GHz |

Both Jetsons used `MAXN_SUPER` and had 5.8 GiB available before testing. Their
saved dynamic CPU, GPU, and EMC ranges and WFI/c7 idle states were restored and
verified afterward.

## Resident results

Times are milliseconds. `Scan` and `scatter` are GPU dispatch time.

| Host | Kept | Wall | GPU elapsed | Dispatch | Scan | Scatter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 4070 Ti SUPER | 0% | 0.791 | 0.601 | 0.575 | 0.499 | 0.075 |
| RTX 4070 Ti SUPER | 10% | 0.829 | 0.653 | 0.627 | 0.500 | 0.127 |
| RTX 4070 Ti SUPER | 50% | 0.892 | 0.714 | 0.689 | 0.500 | 0.188 |
| RTX 4070 Ti SUPER | 90% | 0.965 | 0.778 | 0.752 | 0.500 | 0.253 |
| RTX 4070 Ti SUPER | 100% | 0.992 | 0.783 | 0.757 | 0.500 | 0.257 |
| `dopey` (8 TPCs) | 0% | 6.937 | 5.023 | 4.986 | 4.059 | 0.926 |
| `dopey` (8 TPCs) | 10% | 7.347 | 5.540 | 5.503 | 4.057 | 1.446 |
| `dopey` (8 TPCs) | 50% | 7.601 | 5.789 | 5.751 | 4.057 | 1.694 |
| `dopey` (8 TPCs) | 90% | 7.794 | 5.997 | 5.960 | 4.059 | 1.902 |
| `dopey` (8 TPCs) | 100% | 7.823 | 6.015 | 5.977 | 4.058 | 1.920 |
| `grumpy` (4 TPCs) | 0% | 9.650 | 7.846 | 7.809 | 6.885 | 0.924 |
| `grumpy` (4 TPCs) | 10% | 10.165 | 8.371 | 8.334 | 6.884 | 1.452 |
| `grumpy` (4 TPCs) | 50% | 10.428 | 8.627 | 8.590 | 6.882 | 1.709 |
| `grumpy` (4 TPCs) | 90% | 10.684 | 8.872 | 8.834 | 6.884 | 1.949 |
| `grumpy` (4 TPCs) | 100% | 10.685 | 8.881 | 8.844 | 6.884 | 1.960 |

At 50% selectivity, resident throughput is 11.21 billion input items/s on the
RTX, 1.32 billion/s on `dopey`, and 0.96 billion/s on `grumpy`. The scan is the
fixed cost and accounts for 72.7%, 70.6%, and 80.1% of dispatch time
respectively. Scatter grows with the amount of retained output.

The two Jetsons have nearly equal scatter time at each selectivity despite the
different active TPC counts. Their main difference is the existing scan:
6.882 ms on the 4-TPC system versus 4.057 ms on the 8-TPC system at 50% kept.
This makes scan optimization and reuse more consequential than tuning the new
scatter kernel first.

## Correctness and portability

- The complete suite passed on all three hosts: 9 library unit tests and 37 GPU
  integration tests, including 7 compaction-specific tests.
- Every host validated 10 million items at all five selectivities against the
  stable CPU reference, including the all-discarded and all-retained cases.
- A 20-million-item RTX validation crossed the 65,535-workgroup boundary and
  passed through the two-dimensional dispatch path.
- Tests cover empty and singleton inputs, scan/workgroup boundaries, random and
  duplicate values, stable ordering, explicit logical lengths, multiple
  invocations in one encoder, resident counts, invalid masks, buffer usages,
  capacities, and forbidden aliases.

## Evidence

The compact machine-readable aggregate is
[`2026-08-07-stream-compaction.json`](2026-08-07-stream-compaction.json).
Raw outputs were copied from the measured hosts before their disposable test
directories were removed:

| Host | Raw output SHA-256 |
| --- | --- |
| RTX desktop | `C92610C3585FF2868A23A71D30BFB537322A97CBFA76C3BF16C7230508736C68` |
| `dopey` | `89F0D9A75E126A71850C837F7B1C545EFC3CEE48E2C609B76CC8D5A137788E0D` |
| `grumpy` | `C4CF35A9109E16DC57B4A6BD86326DF5F45A9FE34F26C3A9177DAEB8646B0E64` |

## Decision

The first compaction milestone is suitable to publish: it is stable,
GPU-resident, composable, profiled, and validated across discrete and
integrated NVIDIA Vulkan devices. The next extension should add structured
payloads, followed by reusable selection predicates that generate masks for
this same compaction path.
