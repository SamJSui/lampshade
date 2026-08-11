# Standalone particle consumer validation

Date: 2026-08-10

This validation moves the particle workload across a real crate boundary. The
standalone application creates and owns its wgpu instance, adapter, device,
queue, buffers, command encoders, submissions, and final readback. It depends
on `wgpu-primitives` through the public API only.

The application records a 16-bit depth predicate, stable key/value compaction,
and GPU-counted stable key/value sort. The selected count stays GPU-resident
between operations. Every process validates the count, predicate, ascending
keys, and stable payload order.

## Interface result

The application-owned device path exposed one API defect: `Primitives::new`
could not accept adapter metadata, so it silently lost adapter-selected paths.
`Primitives::new_for_adapter` now accepts the `AdapterInfo` that normal wgpu
applications already own. `Primitives::new` remains the portable fallback.

## Method

- Source: clean `db9aa3dc41748e926ad28381ab2f525f97b42c20`
- Three independent processes per mode and size
- Raw and typed ordering alternates by process
- Three warm-ups and ten measured iterations per process
- Aggregation: median of process medians
- Timing separates CPU command recording from submission through device
  completion; total includes both
- Acceptance: typed total time no more than 2% slower than the equivalent raw
  public-API sequence

## Results

### NVIDIA RTX 4070 Ti SUPER, Vulkan, driver 591.86

| Items | Boundary | Raw | Typed | Change |
| ---: | --- | ---: | ---: | ---: |
| 1M | CPU recording | 0.30885 ms | 0.30865 ms | -0.06% |
| 1M | Submit through completion | 0.73675 ms | 0.74060 ms | +0.52% |
| 1M | Total | 1.04750 ms | 1.05045 ms | +0.28% |
| 10M | CPU recording | 0.32945 ms | 0.31845 ms | -3.34% |
| 10M | Submit through completion | 3.88175 ms | 3.87365 ms | -0.21% |
| 10M | Total | 4.20260 ms | 4.20040 ms | -0.05% |

### Intel Alder Lake-N integrated graphics, Vulkan, Mesa 25.2.8

| Items | Boundary | Raw | Typed | Change |
| ---: | --- | ---: | ---: | ---: |
| 1M | CPU recording | 3.41748 ms | 3.40152 ms | -0.47% |
| 1M | Submit through completion | 13.41243 ms | 12.94626 ms | -3.48% |
| 1M | Total | 17.02982 ms | 16.34266 ms | -4.03% |
| 10M | CPU recording | 3.58465 ms | 3.63064 ms | +1.28% |
| 10M | Submit through completion | 107.18517 ms | 107.13282 ms | -0.05% |
| 10M | Total | 110.92284 ms | 110.76459 ms | -0.14% |

All four total-time rows pass the 2% gate. The largest typed CPU-recording
increase is 1.28% on Intel at 10M.

Typed reservation is a one-time setup cost: approximately 6-7 ms on RTX and
8 ms on Intel in these runs. Raw setup reports zero reservation because the
lower-level public APIs grow private workspace during warm-up instead of
offering an explicit reservation call.

## Memory boundary

The application owns exactly 28,000,004 resident buffer bytes at 1M records
and 280,000,004 bytes at 10M. Final validation adds 8,000,008 and 80,000,008
readback bytes respectively. Wgpu does not expose portable physical-allocation
or peak-driver-memory telemetry, so internal primitive workspace remains
explicitly unobserved rather than estimated.

## Promotion decision

This clears the public crate-boundary, multi-adapter overhead, correctness, and
2% performance gates. It does not constitute independent adoption because the
consumer remains maintained in this repository. The typed API should remain
experimental until an application outside this repository uses it. Apple and
Jetson measurements were not available in this pass; no result is inferred for
those adapters.

Exact process medians, source state, adapter identity, and artifact timestamps
are preserved in the adjacent JSON file.
