# GPU Timestamp Profile

This report separates steady-state resident wall time from hardware GPU timestamps for the scan and radix-sort dispatches. It profiles the current development branch after version 0.3.0; the profiling APIs are not part of the crates.io 0.3.0 release.

## System

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- CPU: AMD Ryzen 7 7800X3D
- Memory: 32 GB
- OS: Windows 11 Education 10.0.26200, 64-bit
- Rust: 1.96.0
- wgpu: 28.0.0
- Vulkan driver: NVIDIA 591.86
- DX12 driver: 32.0.15.9186

## Method

The `profile_primitives` release example uses caller-owned GPU buffers and deterministic input. Each case warms the normal resident path for one second, reports the median of five normal submissions, discards one profiled invocation, and reports medians from five timestamped submissions.

- `resident_wall` includes command encoding, submission, and waiting, but excludes host transfer and initial workspace allocation.
- `gpu_elapsed` spans the beginning of the first compute pass through the end of the last compute pass.
- `dispatch` sums the timestamped compute-pass durations.
- `inter_pass_gap` is `gpu_elapsed - dispatch`; it includes time between the timestamped passes.
- Stage percentages use the median reduce, histogram-scan, and scatter totals. Normal wall and timestamp values come from separate steady-state samples, so their difference is diagnostic rather than an additive accounting identity.

Run the matrix from a source checkout:

```powershell
$env:WGPU_BACKEND = 'vulkan' # or 'dx12'
$env:WGPU_PRIMITIVES_PROFILE_ITEMS = '1000000,10000000,100000000'
$env:WGPU_PRIMITIVES_PROFILE_SAMPLES = '5'
cargo run --release --example profile_primitives
```

The warmup defaults to 1,000 ms and can be changed with `WGPU_PRIMITIVES_PROFILE_WARMUP_MS`.

## Timing Matrix

| Primitive | Items | Backend | Resident wall, ms | GPU elapsed, ms | Dispatch, ms | Inter-pass gap, ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Scan | 1M | Vulkan | 0.161 | 0.080 | 0.067 | 0.013 |
| Scan | 1M | DX12 | 0.270 | 0.059 | 0.055 | 0.004 |
| Scan | 10M | Vulkan | 0.752 | 0.540 | 0.515 | 0.024 |
| Scan | 10M | DX12 | 0.723 | 0.430 | 0.424 | 0.006 |
| Scan | 100M | Vulkan | 7.153 | 5.554 | 5.531 | 0.023 |
| Scan | 100M | DX12 | 6.301 | 4.489 | 4.478 | 0.011 |
| Key sort | 1M | Vulkan | 1.201 | 0.952 | 0.618 | 0.334 |
| Key sort | 1M | DX12 | 1.409 | 0.709 | 0.537 | 0.172 |
| Key sort | 10M | Vulkan | 5.117 | 4.844 | 4.364 | 0.481 |
| Key sort | 10M | DX12 | 5.454 | 4.358 | 4.099 | 0.259 |
| Key sort | 100M | Vulkan | 41.574 | 40.849 | 40.306 | 0.543 |
| Key sort | 100M | DX12 | 41.726 | 40.248 | 39.998 | 0.250 |
| Key-value sort | 1M | Vulkan | 1.692 | 1.435 | 1.107 | 0.328 |
| Key-value sort | 1M | DX12 | 1.762 | 0.988 | 0.833 | 0.156 |
| Key-value sort | 10M | Vulkan | 11.922 | 11.690 | 11.189 | 0.501 |
| Key-value sort | 10M | DX12 | 10.082 | 8.936 | 8.477 | 0.460 |
| Key-value sort | 100M | Vulkan | 102.191 | 101.976 | 101.326 | 0.650 |
| Key-value sort | 100M | DX12 | 78.739 | 76.838 | 76.393 | 0.444 |

## Sort Stage Breakdown

| Primitive | Items | Backend | Reduce | Histogram scan | Scatter |
| --- | ---: | --- | ---: | ---: | ---: |
| Key sort | 1M | Vulkan | 24.1% | 19.7% | 56.1% |
| Key sort | 1M | DX12 | 23.6% | 18.8% | 57.6% |
| Key sort | 10M | Vulkan | 21.8% | 6.4% | 71.8% |
| Key sort | 10M | DX12 | 21.7% | 6.0% | 72.3% |
| Key sort | 100M | Vulkan | 30.7% | 1.0% | 68.2% |
| Key sort | 100M | DX12 | 32.4% | 0.8% | 66.8% |
| Key-value sort | 1M | Vulkan | 20.5% | 11.2% | 68.2% |
| Key-value sort | 1M | DX12 | 30.6% | 12.0% | 57.4% |
| Key-value sort | 10M | Vulkan | 24.8% | 2.5% | 72.7% |
| Key-value sort | 10M | DX12 | 37.1% | 2.9% | 59.9% |
| Key-value sort | 100M | Vulkan | 24.8% | 0.4% | 74.8% |
| Key-value sort | 100M | DX12 | 32.2% | 0.4% | 67.3% |

## Findings

1. Stable scatter is the measured bottleneck. At 100M items it consumes 68.2% of Vulkan key-sort dispatch time and 74.8% of Vulkan key-value dispatch time.
2. Histogram scan is not the large-input bottleneck. It accounts for at most 1.0% of dispatch time in the 100M sort cases.
3. Inter-pass scheduling is already small at scale. The 100M gap is 0.25-0.65 ms across the sort cases, so eliminating host-side uniform allocation cannot produce a large kernel-time gain by itself.
4. DX12 key-value scatter is materially better on this GPU. At 100M pairs, DX12 resident wall time is 22.9% lower than Vulkan, driven primarily by a 32.2% lower measured scatter time. Key-only sort is effectively tied across the two backends.

The next optimization should focus on the scatter shader's memory traffic, branching, register pressure, and workgroup prefix implementation. An NVIDIA Nsight Graphics capture is appropriate for that analysis because the workload runs as Vulkan or DX12 compute. Nsight was not installed on this machine for this report, so no occupancy, cache, or shader-counter claims are made here.
