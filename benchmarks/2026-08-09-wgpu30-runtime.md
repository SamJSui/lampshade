# wgpu 30 runtime migration

Date: 2026-08-09

The candidate upgrades `wgpu-primitives` from wgpu 28 to wgpu 30 and disables
automatic workgroup-memory zeroing. Every crate shader explicitly initializes
its workgroup values before their first read, so backend-generated zero fills
are redundant. The method set and command order are unchanged, but public
GPU-buffer types now come from wgpu 30, so this is a pre-1.0 minor-version
migration to `wgpu-primitives` 0.6.

The benchmarked candidate is the dirty `perf/apple-reduction-readback` working
tree with merged `main` commit `869e64da3292e4e69a4b7cba02fe5bf8e5c17dcc`
as its parent. Massively 0.96.0 remains pinned at
`ef9de55190529be98203aca207edab9d560d312e`.
The exact LF-normalized production and harness sources used by the final formal
comparisons have ordered SHA-256 manifest digest
`40ab3b4d8f865d2e2c71a7456629448c51cbfab0615ebc133301eddfe584f3d2`;
all 57 per-file hashes are recorded in the JSON artifact.

## Why the runtime changed

The original Apple reduction result looked like a scalar-readback problem.
Phase timing on the warmed wgpu 28 path instead showed that allocation, command
encoding, mapping, and scalar access were small:

| Phase, 100M Apple M3 Pro | Per-call staging | Reused staging |
| --- | ---: | ---: |
| Total | 4.931 ms | 4.889 ms |
| Staging allocation | 0.049 ms | 0.000 ms |
| Encoding | 0.028 ms | 0.029 ms |
| Submission | 0.203 ms | 0.208 ms |
| Mapping request | 0.003 ms | 0.004 ms |
| Completion wait | 4.632 ms | 4.619 ms |
| Scalar access | 0.002 ms | 0.002 ms |

Reusing the staging buffer saved less than 1%, below the 5% acceptance gate.
Changing only the runtime to wgpu 30 reduced the same warmed end-to-host path
to 3.125 ms in the first controlled run. This identified the old Metal runtime,
not the crate's readback allocation, as the dominant deficit.

## Massively comparison

The timing boundary is unchanged: reduction ends at a returned host scalar;
sort, scan, and compaction end at confirmed GPU completion. Upload and
validation readback are excluded. Values are medians of independent process
medians and every output is validated.

### RTX 4070 Ti SUPER, Vulkan, 100M items

| Workload | `wgpu-primitives` | Massively | Speedup |
| --- | ---: | ---: | ---: |
| Wrapping sum reduction | 0.714 ms | 1.388 ms | 1.94x |
| Stable sort, 16-bit keys | 7.961 ms | 167.915 ms | 21.09x |
| Stable sort, full-width keys | 14.559 ms | 168.132 ms | 11.55x |
| Exclusive scan | 2.837 ms | 3.550 ms | 1.25x |
| Stable compaction, 50% selected | 3.717 ms | 5.662 ms | 1.52x |

### Intel Alder Lake-N, Vulkan, 10M items

| Workload | `wgpu-primitives` | Massively | Speedup |
| --- | ---: | ---: | ---: |
| Wrapping sum reduction | 3.776 ms | 4.585 ms | 1.21x |
| Stable sort, 16-bit keys | 129.879 ms | 562.704 ms | 4.33x |
| Stable sort, full-width keys | 262.103 ms | 587.898 ms | 2.24x |
| Exclusive scan | 12.450 ms | 34.210 ms | 2.75x |
| Stable compaction, 50% selected | 15.900 ms | 42.429 ms | 2.67x |

At 100M, Intel reduction measured 21.983 ms against Massively's 22.836 ms,
a 1.04x lead.

### Apple M3 Pro, Metal reduction

The final wgpu 30 candidate reverses the previous loss:

| Items | `wgpu-primitives` | Massively | Speedup |
| ---: | ---: | ---: | ---: |
| 1M | 0.171 ms | 0.749 ms | 4.37x |
| 10M | 0.479 ms | 0.844 ms | 1.76x |
| 100M | 3.260 ms | 3.602 ms | 1.11x |

## Same-session regression controls

The controls alternate clean wgpu 28 and candidate processes at 10M items.
Negative values are faster.

| Adapter | Workload | wgpu 28 | Candidate | Change |
| --- | --- | ---: | ---: | ---: |
| RTX | Reduction | 0.1007 ms | 0.0993 ms | -1.39% |
| RTX | Bounded stable sort | 0.9184 ms | 0.9326 ms | +1.55% |
| RTX | Full-width stable sort | 1.6420 ms | 1.6428 ms | +0.05% |
| RTX | Exclusive scan | 0.3508 ms | 0.3388 ms | -3.42% |
| RTX | 50% stable compaction | 0.4993 ms | 0.4829 ms | -3.28% |
| Intel | Reduction | 2.9562 ms | 2.8976 ms | -1.98% |
| Intel | Bounded stable sort | 130.253 ms | 129.970 ms | -0.22% |
| Intel | Full-width stable sort | 260.225 ms | 259.609 ms | -0.24% |
| Intel | Exclusive scan | 12.814 ms | 12.072 ms | -5.79% |
| Intel | 50% stable compaction | 16.810 ms | 15.927 ms | -5.26% |
| Apple | Reduction | 1.4344 ms | 0.4999 ms | -65.15% |
| Apple | Bounded stable sort | 16.987 ms | 16.691 ms | -1.74% |
| Apple | Full-width stable sort | 34.450 ms | 32.724 ms | -5.01% |
| Apple | Exclusive scan | 4.188 ms | 1.916 ms | -54.25% |
| Apple | 50% stable compaction | 2.676 ms | 2.443 ms | -8.71% |

The largest measured regression is 1.55%, inside the 2% gate. Intel reduction
improves 2.90% at 100M; Apple reduction improves 18.40% at 100M. Apple bounded
sort used a targeted seven-process control with five seconds of warmup because
the normal two-second process medians were bimodal.

## Validation

- All 74 release GPU tests pass on RTX Vulkan, Intel Vulkan, and Apple Metal.
- All release benchmark smoke targets validate through 100M on RTX and Apple.
- Root and nested comparison runners pass Clippy with warnings denied.
- Automatic workgroup zeroing stays disabled only because every kernel writes
  each shared value before any read or barrier-dependent consumption.
