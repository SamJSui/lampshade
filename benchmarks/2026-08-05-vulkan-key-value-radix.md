# Vulkan Key-Value Radix Optimization

This experiment used the hardware timestamp profile to optimize the stable
`KeyValue` scatter path without changing its public ordering contract.

## System and Method

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- CPU: AMD Ryzen 7 7800X3D
- OS: Windows 11 Education 10.0.26200, 64-bit
- wgpu: 28.0.0
- Vulkan driver: NVIDIA 591.86
- Workload: 100 million resident `KeyValue` pairs
- Timestamp method: one-second warmup, median of five samples

The controlled A/B restored the original shader, measured it, enabled one
candidate, and repeated the same process in the same session. Criterion results
use `cargo bench --bench key_value_sort -- --noplot`.

## Profile-Guided Change

The baseline spent 75.8% of Vulkan key-value dispatch time in scatter. A
work-efficient replacement for the workgroup prefix produced no measurable
improvement, so it was rejected.

The retained Vulkan kernel processes four radix bits per pass. It uses 16
buckets and eight full reduce/scan/scatter passes instead of four buckets and
16 passes. This increases per-pass local histogram work but halves reads and
writes of the complete key-value array.

The generalized wide shader compiled pathologically through DX12 during the
experiment. Adapter-aware construction therefore selects it only for discrete
NVIDIA Vulkan `KeyValue` sorting, the hardware class measured here. Key-only
sorting and all other adapters retain the original portable 2-bit kernel.

## Controlled Timestamp Results

| Measurement | Baseline | Optimized | Change |
| --- | ---: | ---: | ---: |
| Key-value resident wall | 90.432 ms | 52.493 ms | -41.9% |
| Key-value dispatch total | 89.292 ms | 51.983 ms | -41.8% |
| Reduce dispatches | 21.185 ms | 11.151 ms | -47.4% |
| Histogram scan dispatches | 0.450 ms | 0.452 ms | +0.4% |
| Scatter dispatches | 67.684 ms | 40.336 ms | -40.4% |
| Key-only resident wall | 36.377 ms | 36.585 ms | +0.6% |

The unchanged DX12 path measured 68.812 ms before and 69.597 ms after for
100 million pairs, a 1.1% difference treated as run-to-run variance.

## Criterion Results

| Pairs | Stable Rayon | GPU resident | Resident speedup | GPU round trip |
| ---: | ---: | ---: | ---: | ---: |
| 10 million | 59.421 ms | 7.427 ms | 8.00x | 59.726 ms |
| 100 million | 759.420 ms | 61.429 ms | 12.36x | 477.790 ms |

At 100 million pairs, resident throughput is 1.628 billion pairs per second.
Criterion measured a 44.1% resident-time reduction relative to its saved
110.270 ms baseline. Round-trip performance improved by 17.1%, showing that
host transfer now dominates more of that path.

## Boundaries

- The fast path is specific to discrete NVIDIA Vulkan key-value sorting; it is
  not a general cross-backend speedup.
- Stability is covered by equal-key ordering tests and CPU-reference tests.
- NVIDIA Nsight Graphics was not installed, so no occupancy, register, or cache
  counter claims are made.
