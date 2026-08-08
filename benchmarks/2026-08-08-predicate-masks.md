# GPU Predicate Masks

This report measures the first primitive derived from stream compaction:
GPU-resident comparison predicates that generate one `0` or `1` flag per
`u32` value or selected `KeyValue` field.

## Source and method

- Source base: `92479c2dda8e07ca5c4e703a30292e05ec9749f1`
- Measured implementation package SHA-256:
  `07CD015B4A9B880017C5655EFE28E4F1F88874E32B39CA562F9EF2D20EB36769`
- Later changes are limited to this report, its JSON snapshot, README,
  changelog, and Rustdoc text; the measured Rust implementation and shader are
  unchanged.
- Workload: deterministic pseudorandom resident `u32` values tested with
  `U32Predicate::LessThan(1 << 31)`, yielding approximately 50% selectivity.
- Timestamp profile: five medians after a two-second warmup, with hardware GPU
  timestamps and a CPU-reference validation after measurement.
- Criterion profile: ten samples after a three-second warmup and a 15-second
  measurement window.
- Resident wall timing includes command encoding, submission, execution, and
  an explicit device wait. Allocation, upload, and readback are excluded.
- Jetsons ran in `MAXN_SUPER` with CPU, GPU, and EMC clocks pinned by
  `jetson_clocks` during measurement and restored to dynamic governors after.

Test systems:

| System | GPU | TPCs | Backend | Driver |
| --- | --- | ---: | --- | --- |
| RTX | NVIDIA GeForce RTX 4070 Ti SUPER | — | Vulkan | NVIDIA 591.86 |
| dopey | NVIDIA Jetson Orin Nano Super | 8 | Vulkan | NVIDIA 595.78 |
| grumpy | NVIDIA Jetson Orin Nano Super | 4 | Vulkan | NVIDIA 595.78 |

## Predicate kernel results

Hardware timestamps isolate the predicate dispatch from CPU submission and
waiting overhead:

| System | Items | Resident wall | GPU timestamp | GPU throughput | Selected |
| --- | ---: | ---: | ---: | ---: | ---: |
| RTX | 10M | 0.220 ms | 0.127 ms | 78.74 billion items/s | 5,001,643 |
| RTX | 100M | 1.368 ms | 1.268 ms | 78.86 billion items/s | 49,992,112 |
| dopey, 8 TPC | 10M | 1.286 ms | 1.180 ms | 8.47 billion items/s | 5,001,643 |
| dopey, 8 TPC | 100M | 16.522 ms | 16.422 ms | 6.09 billion items/s | 49,992,112 |
| grumpy, 4 TPC | 10M | 1.331 ms | 1.220 ms | 8.20 billion items/s | 5,001,643 |
| grumpy, 4 TPC | 100M | 16.489 ms | 16.373 ms | 6.11 billion items/s | 49,992,112 |

The 4-TPC and 8-TPC predicate times differ by only 3.4% at 10 million items
and are effectively tied at 100 million. That is evidence that this one-read,
one-write kernel is limited primarily by shared memory bandwidth, not shader
core count. At 100 million items, the RTX moves 800 MB of logical input and
output traffic in 1.268 ms, approximately 631 GB/s.

The scalar CPU comparison and Criterion resident timings on the RTX are:

| Items | CPU mask | GPU resident wall | GPU speedup | GPU mask + compact |
| ---: | ---: | ---: | ---: | ---: |
| 10M | 1.045 ms | 0.218 ms | 4.80x | 1.124 ms |
| 100M | 23.833 ms | 1.379 ms | 17.28x | 9.683 ms |

## Composition cost

The combined case records predicate generation followed by the existing
exclusive-scan and stable-scatter compaction path into one encoder and submits
once. Separate 50%-selective compaction controls from the same measurement
sessions provide the comparison:

| System | 10M compaction control | Predicate + compaction | Difference |
| --- | ---: | ---: | ---: |
| RTX | 0.937 ms | 1.124 ms | 0.186 ms |
| dopey, 8 TPC | 7.333 ms | 8.900 ms | 1.567 ms |
| grumpy, 4 TPC | 10.320 ms | 11.805 ms | 1.485 ms |

The control subtraction is a wall-time estimate from separate Criterion
processes, so hardware timestamps are the authoritative isolated kernel
measurement. The composed path has no CPU synchronization or readback between
predicate, scan, and scatter. Unlike the predicate alone, the full composition
benefits materially from the extra TPCs: dopey is 1.33x faster than grumpy.

## Correctness and portability

- GPU output matched a scalar CPU reference for all predicate variants,
  including equality, ordering, inclusive ranges, `u32::MIN`, `u32::MAX`, and
  an empty range.
- Deterministic tests cover empty and singleton inputs plus 255/256/257-thread
  workgroup boundaries.
- `KeyValue` tests independently select the key and value fields.
- Composition tests verify masks, counts, values, and stable order in one
  encoder. A regression test records two predicates with distinct parameters
  into one submission.
- Invalid capacity, usage, and aliasing contracts return explicit errors.
- CPU-reference validation passed at both 10M and 100M on the RTX and both
  physical Jetsons.

Repeated calls initially exposed stale predicate bindings on the 4-TPC Jetson
when temporary bind groups were dropped immediately after command recording.
The production implementation now attaches a completion callback to the
`CommandEncoder`. A `move` closure owns each bind group and parameter buffer
until that command buffer finishes on the GPU, then releases both. Five
consecutive serialized grumpy runs passed all six predicate tests after this
change, without leaking resources or expanding the public mask buffer.

## Decision

Predicate masks clear the derived-primitive gate on all three tested systems:
they are reusable, composable, CPU-reference validated, and limited by their
expected memory traffic. The similar 4-TPC and 8-TPC kernel results also show
that adding a Jetson-specific predicate kernel is not currently justified.

The compact machine-readable snapshot is
[`2026-08-08-predicate-masks.json`](2026-08-08-predicate-masks.json).
