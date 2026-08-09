# Fused Compaction Prefix

This follow-up strengthens the narrow compaction margin left after the
[direct subgroup scan](2026-08-08-direct-subgroup-scan.md). It removes one
full-size offsets pass from stable value and `KeyValue` compaction while leaving
the standalone scan API unchanged.

## Diagnosis

The merged RTX profile split 100M value compaction into 2.68 ms of scan and
1.70 ms of scatter. The scan still finalized every offset with a 1.35 ms
full-array prefix-add pass before scatter read the result.

Compaction does not need that materialized global-offset array. Its scatter can
combine each block-local offset with the already scanned total of the preceding
block. At 100M, doing so removes one traversal of a 400 MB offsets buffer, or
800 MB of logical read-plus-write traffic.

## Implementation

Production revision `f2c9b5df5ffb688c02b86f649451cfc7605a2f50` adds an internal
block-local exclusive-scan recording mode. Higher hierarchy levels are still
fully propagated, but level zero skips its final add. The compaction shader
receives the scanned block totals and computes:

```text
destination = local_offset + preceding_block_total
```

The same path serves four-byte values and eight-byte `KeyValue` records. Public
`Scanner` calls continue to request complete global prefixes, so standalone scan
and radix-sort histogram scans retain their existing behavior.

In Rust terms, `CompactCore` mutably borrows its `Scanner` to record the local
scan, then immutably borrows the scanner-owned scratch-buffer handle for scatter.
No GPU data is cloned or read by the CPU. Command-encoder ordering makes the
earlier prefix writes visible to the later scatter dispatch.

## Method

The pinned Massively 0.96 harness retains deterministic inputs, correctness
readback before timing, resident public-API wall time through GPU completion,
2-second warmups, 7 or 11 samples, three isolated processes, and the median of
process medians. RTX ran the clean production revision. Each Jetson applied that
production commit plus the portable-test commit to its prior qualified tree; its
copied harness was the only untracked content.

Both Jetsons used `MAXN_SUPER`, 1.02 GHz GPU, and 3.199 GHz EMC clocks during
measurement. Afterward both were verified on `schedutil`, 115.2 MHz-1.728 GHz
CPU clusters, 306 MHz-1.02 GHz GPU, 204 MHz-3.199 GHz EMC, and active
`nvfancontrol`. Post-run GPU temperatures were 53.7 C on dopey and 54.9 C on
grumpy.

## Fresh all-fronts results

`Speedup` is `Massively time / wgpu-primitives time`; all fresh rows are above
`1.0x`.

### RTX 4070 Ti SUPER

| Workload | Items | `wgpu-primitives` | Massively | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Bounded 16-bit stable sort | 1M | 0.178 ms | 1.115 ms | 6.28x |
| Full-width stable sort | 1M | 0.280 ms | 1.113 ms | 3.98x |
| Exclusive scan | 1M | 0.114 ms | 0.371 ms | 3.25x |
| Compact 50% | 1M | 0.129 ms | 0.372 ms | 2.89x |
| Bounded 16-bit stable sort | 10M | 0.909 ms | 7.252 ms | 7.98x |
| Full-width stable sort | 10M | 1.607 ms | 7.242 ms | 4.51x |
| Exclusive scan | 10M | 0.339 ms | 0.952 ms | 2.81x |
| Compact 50% | 10M | 0.481 ms | 0.992 ms | 2.06x |
| Bounded 16-bit stable sort | 100M | 8.395 ms | 165.442 ms | 19.71x |
| Full-width stable sort | 100M | 14.990 ms | 165.862 ms | 11.07x |
| Exclusive scan | 100M | 2.836 ms | 3.174 ms | 1.12x |
| Compact 50% | 100M | 3.736 ms | 5.695 ms | 1.52x |

### Jetson Orin Nano, 10M

| System | Workload | `wgpu-primitives` | Massively | Speedup |
| --- | --- | ---: | ---: | ---: |
| dopey, 8 TPC | Bounded 16-bit stable sort | 11.275 ms | 95.669 ms | 8.48x |
| dopey, 8 TPC | Full-width stable sort | 21.774 ms | 98.587 ms | 4.53x |
| dopey, 8 TPC | Exclusive scan | 3.541 ms | 5.422 ms | 1.53x |
| dopey, 8 TPC | Compact 50% | 4.157 ms | 6.887 ms | 1.66x |
| grumpy, 4 TPC | Bounded 16-bit stable sort | 11.393 ms | 97.388 ms | 8.55x |
| grumpy, 4 TPC | Full-width stable sort | 22.046 ms | 100.321 ms | 4.55x |
| grumpy, 4 TPC | Exclusive scan | 3.640 ms | 5.381 ms | 1.48x |
| grumpy, 4 TPC | Compact 50% | 4.218 ms | 6.882 ms | 1.63x |

Fresh 100M compaction measured 45.101 ms versus 68.373 ms on dopey (`1.52x`)
and 45.834 ms versus 68.362 ms on grumpy (`1.49x`). The unchanged, previously
published 100M Jetson paths still lead Massively by `1.18x-1.19x` for scan,
`72.85x-73.66x` for bounded sort, and `38.95x-39.32x` for full-width sort. These
unchanged rows were not rerun in this follow-up.

## Improvement and controls

| System | Items | PR #17 compaction | Fused prefix | Change |
| --- | ---: | ---: | ---: | ---: |
| RTX | 10M | 0.600 ms | 0.481 ms | -19.9% |
| RTX | 100M | 5.082 ms | 3.736 ms | -26.5% |
| dopey | 10M | 5.348 ms | 4.157 ms | -22.3% |
| dopey | 100M | 62.068 ms | 45.101 ms | -27.3% |
| grumpy | 10M | 5.469 ms | 4.218 ms | -22.9% |
| grumpy | 100M | 62.946 ms | 45.834 ms | -27.2% |

RTX profiling shows why: the 100M full-size prefix-add span disappears, reducing
timestamped compaction from 4.37 ms to 3.30 ms even though fused scatter grows
from 1.70 ms to 1.97 ms. At 10M, `KeyValue` compaction wall time also fell from
0.610 ms to 0.526 ms; at 100M it fell from 5.177 ms to 4.295 ms.

Fresh Jetson standalone-scan controls changed by at most 0.21% and sort controls
by at most 0.38% from PR #17. The full RTX suite passed 64 tests. Both Jetsons
passed 26 scoped release tests covering scan, value/key-value compaction,
profiling, multi-level hierarchy, multiple recordings, and an explicitly
no-subgroup portable device. Every benchmark process passed 10M/100M correctness
readback.

## Decision

The fused-prefix path is accepted. `wgpu-primitives` is faster than pinned
Massively 0.96 on every overlapping stable-sort, exclusive-scan, and stable-
compaction row measured on RTX, dopey, and grumpy. The narrowest fresh row is
100M RTX scan at `1.12x`; the next performance work should target a safe
single-pass scan design or broader hardware validation, not add more work to the
now stronger compaction path.

Compact process medians and exact revisions are in
[`2026-08-08-fused-compaction-prefix.json`](2026-08-08-fused-compaction-prefix.json).
