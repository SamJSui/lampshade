# Apple M3 Pro Metal validation

This validation adds the first Apple GPU to the physical-hardware matrix. It
checks the merged `wgpu-primitives` revision `1ff738e` on an M3 Pro through
Metal, measures the same resident workloads used by the pinned Massively
comparison, and records a Massively 0.96 Metal pipeline-layout defect instead
of converting failed runs into artificial timings.

## System and method

- MacBook Pro `Mac15,6`, Apple M3 Pro with a 14-core integrated GPU and 18 GB
  unified memory.
- macOS 15.6.1 (`24G90`), Metal 3, arm64, Rust 1.97.1.
- `wgpu-primitives` 0.4.0 at
  `1ff738ec573e9c2def8f1cf786476a0136ce19fb`, using wgpu 28.
- Massively 0.96.0 at
  `ef9de55190529be98203aca207edab9d560d312e`, using CubeCL and wgpu 30.
- AC power, Low Power Mode disabled.

The timing boundary is the resident public GPU API call through confirmed GPU
completion. Upload, readback, and correctness validation are excluded. Normal
cases warm for at least two seconds and four calls and retain 11 samples; 100M
cases use two warmups and seven samples. Each published value is the median of
three independent process medians.

## Correctness and adapter routing

`cargo test --release --all-targets` passed all 64 tests. The benchmark runners
also validated deterministic outputs through 100M items.

The adapter reports subgroup sizes from 4 through 64. Scan therefore uses the
dynamic subgroup implementation. Stable key/value radix sort does not satisfy
the fixed-32-wide NVIDIA Vulkan fast-path gate and correctly uses the portable
4-bit implementation.

## `wgpu-primitives` resident results

| Workload | 1M | 10M | 100M |
|---|---:|---:|---:|
| Bounded 16-bit stable sort | 3.949 ms | 19.508 ms | 173.919 ms |
| Full-width stable sort | 5.915 ms | 38.374 ms | 344.635 ms |
| Exclusive scan | 1.592 ms | 3.004 ms | 15.253 ms |
| 50%-selective stable compaction | 1.955 ms | 2.697 ms | 20.384 ms |

At 100M items, scan reaches 6.56 billion items/s and compaction reaches 4.91
billion items/s. These are absolute Apple measurements; no Massively speedup is
reported because its corresponding pipelines do not initialize.

## Massively 0.96 Metal defect

Massively's published package describes itself as
[multi-platform GPU parallel algorithms](https://github.com/akiradeveloper/massively/blob/ef9de55190529be98203aca207edab9d560d312e/Cargo.toml)
and its
[setup documentation](https://github.com/akiradeveloper/massively/blob/ef9de55190529be98203aca207edab9d560d312e/README.md)
says the same API runs through CubeCL's WGPU runtime. CubeCL's official
[supported-platform table](https://github.com/tracel-ai/cubecl/blob/11aec302f1b008f8540938a26f26cc46298d1fab/README.md#supported-platforms)
lists Metal through wgpu on Apple GPUs. If Massively 0.96 intentionally excludes
Metal, that restriction is not stated in its package setup.

On the M3 Pro, sort, exclusive scan, and compaction all reach wgpu pipeline
validation failures. Generated compute layouts request 42 or 47 storage-buffer
bindings while the adapter exposes `max_storage_buffers_per_shader_stage = 29`:

```text
In Device::create_bind_group_layout
Too many bindings of type StorageBuffers in Stage ShaderStages(COMPUTE),
limit is 29, count was 47.
```

The runner then fails output validation because no valid result was produced.
This is an upstream compatibility defect observed through Massively's public
WGPU API; the responsible fix may be in Massively's kernel specialization,
CubeCL lowering, or both.

The POSIX comparison harness now matches the PowerShell runner: it retains each
successful run, records each failed implementation/workload/process with its
complete error, and continues the matrix. Aggregates and speedups use successful
runs only, so unsupported cells remain explicit rather than becoming invented
durations.

Exact process medians and the compatibility evidence are in
[`2026-08-08-apple-metal-validation.json`](2026-08-08-apple-metal-validation.json).

## Decision

Apple Metal joins NVIDIA Vulkan as a validated physical backend for all current
primitives through 100M items. The next cross-vendor evidence targets are AMD
and Intel. Apple sort optimization should begin with the portable scatter path,
but only after profiling establishes where the 100M full-width 344.635 ms is
spent.
