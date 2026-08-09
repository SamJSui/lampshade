# Hierarchical Scan Scratch-Binding Fix

The pinned [Massively comparison](2026-08-08-massively-comparison.md) exposed a
scheduling-sensitive correctness failure in 100-million-item exclusive scan and
scan-derived compaction. This follow-up records the production fix, its regression
test, and the previously withheld comparisons.

## Fix

`ScanPipeline::dispatch` previously created two writable storage views with
`size: None`. At deeper hierarchy levels, `data` and `auxiliary` are offsets into
the same scratch buffer. The unbounded `data` view therefore included the next
auxiliary level, so rounded-up shader threads could use the larger
`arrayLength(&data)` and overwrite that level.

Fix revision `f9b4982e4afadced2d773e1c696f2763757a7fe8` binds:

- `data` to exactly `num_items * 4` bytes;
- `auxiliary` to exactly `ceil(num_items / items_per_block) * 4` bytes.

These are bounded views of the existing buffers; the change does not allocate or
copy GPU memory. A regression at 4,194,305 items (`2,048 * 2,048 + 1`) forces a
third hierarchy level on the high-end path and validates every exclusive prefix.

## Method

The workload, deterministic input, correctness readback, resident timing boundary,
warmups, samples, and median-of-three-process-medians aggregation are unchanged
from the original report. RTX used the clean fix commit. Each Jetson applied the
same two-file fix patch to the original measured production base so the run remained
a direct fix-only comparison; the copied benchmark harness was the only untracked
content.

Both Jetsons used `MAXN_SUPER`, 1.02 GHz pinned GPU clocks, and 3.199 GHz pinned EMC
during measurement. dopey exposed 8 active GPU TPCs and grumpy exposed 4. After the
run, both reported `schedutil` CPU governors, the dynamic 306 MHz-1.02 GHz GPU range,
the dynamic 204 MHz-3.199 GHz EMC range, and active `nvfancontrol`. Post-run GPU
temperatures were 53.1 C and 52.7 C. JetPack's restore script emitted its known
persistence/path errors, so restoration was verified from these live settings.

## Correctness

All three process-isolated validations passed for exclusive scan and 50%-selective
stable compaction at both 10 million and 100 million items on RTX, dopey, and grumpy.
The new 4,194,305-item integration regression also passes in the normal GPU test
suite. The original 100M failure reproduced in all three processes, so this closes
the measured correctness gap rather than merely changing an unreproduced code path.

## Results

`Speedup` is `Massively time / wgpu-primitives time`; values above `1.0x` favor
`wgpu-primitives`.

| System | Workload | Items | `wgpu-primitives` | Massively | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 4070 Ti SUPER | Exclusive scan | 10M | 0.715 ms | 0.945 ms | 1.32x |
| RTX 4070 Ti SUPER | Compact 50% | 10M | 0.941 ms | 0.986 ms | 1.05x |
| RTX 4070 Ti SUPER | Exclusive scan | 100M | 6.134 ms | 3.154 ms | 0.51x |
| RTX 4070 Ti SUPER | Compact 50% | 100M | 8.416 ms | 5.663 ms | 0.67x |
| dopey, 8 TPC | Exclusive scan | 10M | 6.136 ms | 5.428 ms | 0.88x |
| dopey, 8 TPC | Compact 50% | 10M | 7.677 ms | 6.892 ms | 0.90x |
| dopey, 8 TPC | Exclusive scan | 100M | 65.152 ms | 45.778 ms | 0.70x |
| dopey, 8 TPC | Compact 50% | 100M | 88.595 ms | 68.307 ms | 0.77x |
| grumpy, 4 TPC | Exclusive scan | 10M | 9.120 ms | 5.393 ms | 0.59x |
| grumpy, 4 TPC | Compact 50% | 10M | 10.532 ms | 6.929 ms | 0.66x |
| grumpy, 4 TPC | Exclusive scan | 100M | 93.279 ms | 45.846 ms | 0.49x |
| grumpy, 4 TPC | Compact 50% | 100M | 117.174 ms | 68.510 ms | 0.58x |

Relative to the original 10M `wgpu-primitives` controls, scan latency changed by
-0.58% on RTX, -1.39% on dopey, and -0.83% on grumpy. Compaction changed by -0.28%,
-0.63%, and -0.25%. The exact-range fix therefore introduced no measured control
regression.

## Decision

> **Optimized:** the subsequent
> [direct subgroup scan](2026-08-08-direct-subgroup-scan.md) removes the top-level
> copy and adds a feature-gated coalesced subgroup path. It now beats Massively in
> every measured 10M and 100M scan/compaction row on RTX, dopey, and grumpy. The
> text below is retained as the decision at the correctness-fix checkpoint.

The correctness blocker is closed, but the valid 100M data sharpens the next target.
Massively is 1.95x faster for scan and 1.49x for compaction on RTX; on the integrated
Jetsons it is 1.42x/1.30x faster on dopey and 2.03x/1.71x faster on grumpy. The next
optimization work should profile the corrected hierarchy on the 4-TPC Jetson and
separate scan-level traffic, synchronization, and dispatch overhead before changing
the kernel.

Compact process medians are in
[`2026-08-08-scan-scratch-binding-fix.json`](2026-08-08-scan-scratch-binding-fix.json).
