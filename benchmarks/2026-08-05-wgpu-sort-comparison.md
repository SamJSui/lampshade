# Stable Key-Value Sort Comparison

This experiment compares the unreleased `wgpu-primitives` NVIDIA Vulkan
key-value path with [`wgpu_sort`](https://github.com/KeKsBoTer/wgpu_sort) at
commit `4cb640e8cae28eba0149d470c5168cc2853466dd`.

## System and Method

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- OS: Windows 11 Education 10.0.26200, 64-bit
- Backend: Vulkan, NVIDIA driver 591.86
- `wgpu-primitives`: wgpu 28.0.0, interleaved `KeyValue` records
- `wgpu_sort`: wgpu 0.20.1, separate key and value buffers
- Workloads: duplicate-heavy 16-bit keys and random full-width `u32` keys
- Sizes: 1 million, 10 million, and 100 million pairs
- Stability: checked before timing with original positions as values

Both harnesses use the same xorshift-generated logical input. Timing includes
command encoding, queue submission, execution, and the GPU wait. It excludes
initial allocation, upload, and readback. `wgpu_sort` input restoration happens
before the timer because its API sorts its working buffers in place;
`wgpu-primitives` leaves its input buffer unchanged.

Each process performs four warmups and 11 measured samples at 1M and 10M. At
100M it performs two warmups and seven samples. Each result below is the median
of three independent process medians.

The table values are preserved from the original controlled run. That run
predated the committed harness, so its individual process samples are not
available in the repository; the published aggregates are also preserved as a
[machine-readable snapshot](2026-08-05-wgpu-sort-comparison.json).

## Results

### Duplicate-Heavy 16-Bit Keys

| Pairs | `wgpu-primitives` | `wgpu_sort` | Time change | Throughput |
| ---: | ---: | ---: | ---: | ---: |
| 1M | 0.224 ms | 0.318 ms | -29.6% | 4.464 billion pairs/s |
| 10M | 0.989 ms | 1.718 ms | -42.4% | 10.112 billion pairs/s |
| 100M | 8.605 ms | 14.884 ms | -42.2% | 11.621 billion pairs/s |

### Random Full-Width Keys

| Pairs | `wgpu-primitives` | `wgpu_sort` | Time change | Throughput |
| ---: | ---: | ---: | ---: | ---: |
| 1M | 0.296 ms | 0.326 ms | -9.2% | 3.384 billion pairs/s |
| 10M | 1.720 ms | 1.735 ms | -0.9% | 5.812 billion pairs/s |
| 100M | 15.457 ms | 15.907 ms | -2.8% | 6.470 billion pairs/s |

The 1M and 10M full-width differences are small relative to observed process
variance and should be treated as parity. The 100M full-width result shows a
modest repeatable advantage. The much larger bounded-key result comes from
detecting that both upper bytes are constant and skipping two stable scatters
that would be identity transformations.

## Reproduction

The [committed comparison harness](wgpu-sort-comparison/README.md) builds the
current checkout and the pinned `wgpu_sort` revision as independent processes,
validates both against the same stable CPU reference, and records raw samples,
adapter metadata, exact revisions, and aggregate medians in JSON.

From the repository root on Windows PowerShell:

```powershell
& .\benchmarks\wgpu-sort-comparison\run.ps1
```

Use `-Quick` for a 1M-pair correctness and resident-path smoke test. A new full
run is a new measurement; it should not silently replace this historical
snapshot because driver, hardware, thermal state, or source revision may have
changed.

## Source-Derived Memory Model

The committed harness also reports known GPU buffer allocations calculated
from each pinned implementation's source. On this adapter, the formulas give:

| Pairs | `wgpu-primitives` known buffers | `wgpu_sort` known buffers | Reduction |
| ---: | ---: | ---: | ---: |
| 1M | 23.84 MiB | 38.47 MiB | 38.0% |
| 10M | 238.31 MiB | 384.12 MiB | 38.0% |
| 100M | 2,348.75 MiB | 3,840.17 MiB | 38.8% |

These figures are an allocation model, not measured peak VRAM. They include
the public input/output buffers and reusable algorithm workspace. They exclude
pipelines, bind groups, transient upload/readback staging, and driver-managed
allocations. The implementations also expose different physical layouts, so
the totals describe their current public paths rather than a forced common
layout.

## Implementation

The adapter-selected path uses four 8-bit digits. It computes all four 256-bin
histograms in one input read, fuses their prefix setup into one workgroup, and
uses 32-wide subgroup operations for stable rank calculation. A packed
partition state supplies cross-workgroup prefixes, while workgroup memory
reorders records before coalesced output writes. Indirect dispatch suppresses
both upper-byte passes when each contains exactly one nonempty bucket.

The shaders were implemented in this repository; no `wgpu_sort` source was
copied or incorporated. The single-read histogram and partition-lookback architecture uses established
GPU radix-sort techniques described by the
[`Onesweep` paper](https://research.nvidia.com/publication/2022-06_onesweep-faster-least-significant-digit-radix-sort-gpus).

## Boundaries

- The fast path requires a discrete NVIDIA Vulkan adapter with an enabled,
  exactly 32-wide subgroup. Other adapters retain existing fallbacks.
- The result compares current public API layouts rather than forcing a common
  physical layout: interleaved records for `wgpu-primitives`, separate arrays
  for `wgpu_sort`.
- The packed lookback state supports at most 268,435,455 pairs, subject to
  stricter device storage-buffer limits.
- These results establish a win on one GPU and driver, not across all WebGPU
  implementations.
