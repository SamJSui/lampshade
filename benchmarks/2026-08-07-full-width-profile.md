# Full-Width Key-Value Sort Profile

This experiment profiles the version 0.4 release candidate's 8-bit NVIDIA
Vulkan path and repeats the pinned `wgpu_sort` comparison with a warmup and
resident-input restoration method that remains stable across fresh processes.

## System and Method

- Source revision: `7fa08b7bfbc7cf844a3dfd3cdd70d31976b6f123`
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- Backend: Vulkan, NVIDIA driver 591.86
- Subgroup: exactly 32 lanes
- Workload: 100 million resident stable `KeyValue` pairs
- Inputs: deterministic bounded 16-bit and random full-width `u32` keys
- Timing: median of seven samples after at least two seconds of warmup
- Comparison aggregate: median of three independent process medians

Timing includes command encoding, submission, GPU execution, and waiting. It
excludes initial allocation, host upload, output readback, and `wgpu_sort`
input restoration. Because `wgpu_sort` mutates its primary buffers, its input
is restored from immutable GPU-resident backup buffers and awaited before each
timer starts.

The complete raw comparison data, including every sample and adapter metadata,
is in
[`2026-08-07-rtx4070ti-super-wgpu-sort.json`](2026-08-07-rtx4070ti-super-wgpu-sort.json).

## Pinned Comparison

| Key workload | `wgpu-primitives` | `wgpu_sort` | Latency change | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Bounded 16-bit | 7.972 ms | 13.924 ms | -42.7% | 1.75x |
| Full-width `u32` | 14.484 ms | 14.910 ms | -2.9% | 1.03x |

The three `wgpu-primitives` process medians span 7.959-7.984 ms for bounded
keys and 14.454-14.526 ms for full-width keys. The three `wgpu_sort` medians
span 13.838-14.057 ms and 14.871-14.944 ms respectively.

An earlier trial used repeated 800 MB host `queue.write_buffer` restorations
and produced comparator process medians from about 15 ms to 25 ms. Those data
were discarded. The GPU-resident restore removes that staging-allocation
confounder and matches the declared resident timing boundary.

## Timestamp Breakdown

| Workload | Resident wall | GPU elapsed | Dispatch | Histogram | Prefix | Scatter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bounded 16-bit | 7.957 ms | 7.762 ms | 7.738 ms | 1.331 ms | 0.009 ms | 6.402 ms |
| Full-width `u32` | 14.517 ms | 14.508 ms | 14.486 ms | 1.321 ms | 0.009 ms | 13.150 ms |

For bounded keys, the two required scatter passes take 3.184 and 3.198 ms;
the skipped upper-byte passes each take 0.006 ms. For full-width keys, all four
passes are active and tightly grouped at 3.228-3.252 ms. No digit is an
outlier.

Full-width scatter consumes 90.8% of measured dispatch time. Histogram and
prefix work together consume only 9.2%, and the inter-pass gap is 0.022 ms.
Pass scheduling and prefix setup therefore cannot deliver a material
full-width improvement.

## Next Optimization Target

A 10% resident-wall improvement requires reducing 14.517 ms to at most
13.065 ms. If histogram, prefix, and scheduling remain fixed, scatter must fall
from 13.150 ms to at most about 11.70 ms: an 11.0% scatter reduction.

The next experiment should use NVIDIA shader counters to distinguish these
scatter costs before changing the algorithm:

1. Eight per-bit subgroup ballots for each of seven items per thread in
   `subgroup_rank`.
2. The eight serialized subgroup accumulation phases and workgroup barriers.
3. Partition-lookback atomics versus the shared-memory reorder and final global
   writes.

The acceptance gate is full-width resident wall at or below 13.065 ms on this
system, bounded-key wall no slower than 8.20 ms, stable CPU-reference
correctness, and no regression above 5% on portable backends.

## Hardware Validation Status

This report validates one discrete NVIDIA Vulkan system. The configured Jetson
Orin Nano hosts `dopey` and `grumpy` were unreachable over SSH during this run,
so no Jetson result is claimed. When available, they should run 1M and 10M
bounded/full-width cases first; their integrated GPUs exercise the portable
fallback rather than the discrete-NVIDIA fast path.
