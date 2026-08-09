# Direct Subgroup Scan

This follow-up optimizes the corrected hierarchical scan from the
[scratch-binding fix](2026-08-08-scan-scratch-binding-fix.md) and reruns the pinned
[Massively 0.96 comparison](2026-08-08-massively-comparison.md). The result reverses
the remaining scan and compaction deficit on the RTX and both Jetson Orin Nano
configurations.

## Diagnosis

The corrected 100M RTX profile separated two costs:

- The public GPU call copied 400 MB from input to output before scanning in place.
  The untimestamped portion grew to 1.88 ms at 100M versus 0.18 ms at 10M.
- The two full-size compute passes took 4.72 ms: 2.33 ms for local scan and
  2.37 ms for prefix add.

The compute path processed eight adjacent values per thread. For any fixed loop
iteration, neighboring lanes were therefore eight `u32` values apart instead of
contiguous. Its shared-memory Hillis-Steele scan also executed eight steps with two
workgroup barriers per step. Massively reads separate input and output storage,
uses one contiguous item per lane, and uses subgroup shuffles with two workgroup
barriers.

## Implementation

Optimization revision `b3f53211b600e0f700dfb4bb9e04f69f33cba13b` makes two
adapter-selected changes:

1. The top hierarchy pass reads the caller's input binding and writes prefixes
   directly to the caller's output while producing block totals. The former
   input-to-output copy is gone.
2. When the wgpu device has the `SUBGROUP` feature enabled, 256-thread workgroups
   process one coalesced item per lane. `subgroupExclusiveAdd` scans each subgroup;
   one lane combines the subgroup totals; two workgroup barriers publish the
   result. Devices without the feature retain the existing VT=8/VT=4 portable
   Hillis-Steele scan, now with the same direct top pass.

The `Scanner` still owns its pipelines and reusable hierarchy scratch allocation.
Recording only borrows the caller's input, output, and command encoder; the new
read-only input binding is a GPU view, not a Rust clone or a new full-size buffer.

## Method

The deterministic inputs, correctness readback, resident public-API timing boundary,
warmups, samples, process isolation, and median-of-three-process-medians aggregation
are unchanged from the original Massively report. RTX measured clean test revision
`46b6e7226a0835f71fba335a5b0c9c9964ed842f`. Each Jetson applied the exact production
optimization commit to the prior fix-only tree; the copied harness remained the only
untracked content.

All measured adapters exposed enabled, fixed 32-wide subgroups. A separate RTX test
requested a device with no optional features and validated the portable fallback.
Both Jetsons ran in `MAXN_SUPER` with 1.02 GHz GPU and 3.199 GHz EMC clocks, then were
verified back on `schedutil`, dynamic GPU/EMC ranges, and `nvfancontrol`. Post-run GPU
temperatures were 54.0 C on dopey and 53.3 C on grumpy.

## Results against Massively

`Speedup` is `Massively time / wgpu-primitives time`; every value is now above
`1.0x`.

| System | Workload | Items | `wgpu-primitives` | Massively | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 4070 Ti SUPER | Exclusive scan | 10M | 0.339 ms | 0.953 ms | 2.81x |
| RTX 4070 Ti SUPER | Compact 50% | 10M | 0.600 ms | 0.992 ms | 1.65x |
| RTX 4070 Ti SUPER | Exclusive scan | 100M | 2.828 ms | 3.089 ms | 1.09x |
| RTX 4070 Ti SUPER | Compact 50% | 100M | 5.082 ms | 5.634 ms | 1.11x |
| dopey, 8 TPC | Exclusive scan | 10M | 3.533 ms | 5.392 ms | 1.53x |
| dopey, 8 TPC | Compact 50% | 10M | 5.348 ms | 6.842 ms | 1.28x |
| dopey, 8 TPC | Exclusive scan | 100M | 38.449 ms | 45.782 ms | 1.19x |
| dopey, 8 TPC | Compact 50% | 100M | 62.068 ms | 68.406 ms | 1.10x |
| grumpy, 4 TPC | Exclusive scan | 10M | 3.634 ms | 5.405 ms | 1.49x |
| grumpy, 4 TPC | Compact 50% | 10M | 5.469 ms | 6.899 ms | 1.26x |
| grumpy, 4 TPC | Exclusive scan | 100M | 38.983 ms | 45.906 ms | 1.18x |
| grumpy, 4 TPC | Compact 50% | 100M | 62.946 ms | 68.515 ms | 1.09x |

Against the corrected pre-optimization baseline, scan latency fell 52.6%-53.9% on
RTX, 41.0%-42.4% on dopey, and 58.2%-60.2% on grumpy. Compaction fell 36.2%-39.6%,
29.9%-30.3%, and 46.3%-48.1% respectively.

## Regression controls

All process-isolated scan and compaction validations passed at 10M and 100M. The
4,194,305-item hierarchy regression passes on the subgroup path, and a dedicated
4,097-item test passes with `SUBGROUP` disabled. The full unit/GPU suite, Clippy,
formatting, documentation, and packaging checks pass.

The scan is also used for small radix histograms on the portable Jetson sorter.
Formal 10M controls stayed within 0.12% of the published baseline:

| System | Sort workload | Before | After | Change |
| --- | --- | ---: | ---: | ---: |
| dopey | Bounded 16-bit | 11.255 ms | 11.245 ms | -0.09% |
| dopey | Full width | 21.752 ms | 21.778 ms | +0.12% |
| grumpy | Bounded 16-bit | 11.443 ms | 11.436 ms | -0.06% |
| grumpy | Full width | 22.098 ms | 22.085 ms | -0.06% |

## Decision

The scan-performance item is complete for the measured NVIDIA Vulkan adapters:
wgpu-primitives now beats Massively on stable sort, exclusive scan, and stable
compaction at both 10M and 100M. The subgroup path is feature-gated rather than
vendor-gated, while the direct portable top pass also benefits devices without
subgroups. The next evidence step is broader AMD, Intel, and Apple validation rather
than another NVIDIA-specific kernel change.

Compact process medians are in
[`2026-08-08-direct-subgroup-scan.json`](2026-08-08-direct-subgroup-scan.json).
