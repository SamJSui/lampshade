# WGPU 29 release-candidate regression gate

Date: 2026-08-11 (America/Chicago)

## Result

The 0.10 candidate showed no slowdown above the 2% release budget in the six
RTX/Vulkan controls after one targeted recheck. This is evidence for those
workloads on this adapter, not proof of no regression across every primitive or
backend.

**Release verdict: accepted compatibility baseline.** The same source passed
the full M3 Pro correctness suite but made Metal completion-boundary timings
substantially slower than 0.9/wgpu 30. That cross-runtime cost is accepted for
0.10 because current downstream integrations require wgpu 29. It is not
described as a shader regression or a performance-neutral migration.

The changed GPU-counted full-width key/value path was 5.5x, 2.3x, and 6.8x
faster at 1M, 10M, and 100M items in the final three-process matrix. Absolute
sub-millisecond and 10M timings moved between GPU performance states, so the
large unrelated scan/compaction improvements are not attributed to WGPU 29.

## Formal comparison

Baseline: crates.io `lampshade 0.9.0` with wgpu 30. Candidate: working-tree
`lampshade 0.10.0` with wgpu 29. Both runners used identical deterministic
inputs, public resident APIs, correctness validation, and submit-to-completion
wall timing. Each matrix cell is the median of three independent process
medians. The 1M and 10M processes used 11 samples after four warmups; 100M used
seven samples after two warmups. Every process also warmed for at least two
seconds.

Adapter: NVIDIA GeForce RTX 4070 Ti SUPER, Vulkan, driver 591.86.

| Workload | Items | 0.9 / wgpu 30 (ms) | 0.10 / wgpu 29 (ms) | Change |
|---|---:|---:|---:|---:|
| Reduce sum | 1M | 0.1013 | 0.1004 | -0.89% |
| Fixed KV sort, 16-bit | 1M | 0.2802 | 0.2774 | -1.00% |
| Fixed KV sort, 32-bit | 1M | 0.4604 | 0.4645 | +0.89% |
| GPU-counted KV sort, 32-bit | 1M | 2.8345 | 0.5139 | -81.87% |
| Exclusive scan | 1M | 0.1542 | 0.1518 | -1.56% |
| Compact, 50% selected | 1M | 0.1745 | 0.1717 | -1.60% |
| Reduce sum | 10M | 0.1371 | 0.1348 | -1.68% |
| Fixed KV sort, 16-bit | 10M | 1.3014 | 1.2840 | -1.34% |
| Fixed KV sort, 32-bit | 10M | 2.2355 | 2.2031 | -1.45% |
| GPU-counted KV sort, 32-bit | 10M | 18.2569 | 7.8302 | -57.11% |
| Exclusive scan | 10M | 3.0070 | 2.0614 | -31.45% |
| Compact, 50% selected | 10M | 3.8012 | 2.9731 | -21.79% |
| Reduce sum | 100M | 0.7521 | 0.7315 | -2.74% |
| Fixed KV sort, 16-bit | 100M | 8.6411 | 9.9936 | +15.65% (rechecked) |
| Fixed KV sort, 32-bit | 100M | 31.1996 | 25.7866 | -17.35% |
| GPU-counted KV sort, 32-bit | 100M | 107.4778 | 15.8596 | -85.24% |
| Exclusive scan | 100M | 3.3395 | 3.1344 | -6.14% |
| Compact, 50% selected | 100M | 3.9752 | 3.9885 | +0.33% |

The only initial failure, 100M fixed 16-bit key/value sort, passed a dedicated
nine-process recheck: 8.6674 ms for 0.9 versus 8.6276 ms for 0.10, or -0.46%.
The per-process medians and the complete normalized 80-file source manifest are
preserved in the adjacent JSON artifacts.

## Commands and provenance

```powershell
benchmarks/release-regression/run.ps1 -Backend vulkan `
  -OutputPath target/release-regression/wgpu29-formal-final.json

benchmarks/release-regression/run.ps1 -Backend vulkan `
  -Items 100000000 -Workloads sort_bounded16 -Processes 9 `
  -OutputPath target/release-regression/wgpu29-bounded100m-recheck.json
```

Both artifacts pin source-manifest SHA-256
`76aef90abd951b73e52b52a11e701e992c5e42e7826990cfb47924f8d7d1835d`.
The candidate is intentionally recorded as dirty at base revision
`a0c91ac12d6a5800844a1378aed64169e879e953`; the manifest, rather than that
revision alone, identifies the measured source.

- `2026-08-11-wgpu29-formal.json` preserves all 18 aggregates and the source
  manifest. Its gate is false because it faithfully retains the initial
  outlier.
- `2026-08-11-wgpu29-bounded100m-recheck.json` preserves the resolving nine
  process medians and the same source manifest.

## Additional checks

- Required-GPU release suite: 27 library tests and every integration suite
  passed on the RTX adapter, including all three native SoA tests and a
  ten-process repeated stability check.
- Strict all-target/all-feature Clippy, formatting, Rustdoc warnings, locked
  standalone consumers, and package checks passed.
- An identical `profile_primitives` diagnostic at 1M and 10M found GPU-stage
  changes between -2.87% and +1.47% for predicate, histogram, reduction, scan,
  compaction, key/value compaction, key-only sort, and fixed key/value sort.
  This diagnostic is broader than the formal gate but is not process-isolated
  evidence.

## Apple M3 Pro, Metal

All release integration tests passed on the M3 Pro, including sort, scan,
reduction, compaction, histogram, predicates, RLE, counted composition, and the
typed facade. Performance did not pass.

| Workload | Items | 0.9 / wgpu 30 (ms) | 0.10 / wgpu 29 (ms) | Change |
|---|---:|---:|---:|---:|
| Reduce sum | 1M | 0.163 | 1.461 | +799.03% |
| Fixed KV sort, 16-bit | 1M | 2.277 | 3.299 | +44.89% |
| Fixed KV sort, 32-bit | 1M | 4.512 | 4.836 | +7.18% |
| GPU-counted KV sort, 32-bit | 1M | 4.372 | 4.627 | +5.84% |
| Exclusive scan | 1M | 0.404 | 1.604 | +296.87% |
| Compact, 50% selected | 1M | 0.444 | 1.786 | +301.94% |
| Reduce sum | 10M | 0.499 | 1.490 | +198.83% |
| Fixed KV sort, 16-bit | 10M | 16.497 | 16.934 | +2.65% |
| Fixed KV sort, 32-bit | 10M | 32.673 | 33.792 | +3.42% |
| GPU-counted KV sort, 32-bit | 10M | 33.227 | 33.556 | +0.99% |
| Exclusive scan | 10M | 1.861 | 4.064 | +118.41% |
| Compact, 50% selected | 10M | 2.429 | 2.656 | +9.34% |

This five-process exact-source rerun passed only one of 12 cells, so the Metal
failure was not a one-off. A separate WGPU 29 diagnostic measured an empty
submit plus `Device::poll(Wait)` at 1.53 ms and one 1M reduction plus the same
wait at 1.58 ms. Recording 32 reductions before one wait reduced the amortized
time to 0.12-0.22 ms per reduction. Busy-polling a queue completion callback
measured 0.28 ms but consumes a CPU core and is not an appropriate library
default.

The practical fix is therefore workflow composition: use the resident
`record_*` methods to put multiple primitives into an application-owned command
encoder, submit once, and wait only at the real host-readback boundary. A
host-returning convenience call necessarily pays the WGPU 29 Metal completion
cost. This diagnosis does not support changing shader kernels or claiming that
WGPU 29 Metal GPU execution itself is slower.

The exact rerun is `2026-08-12-wgpu29-m3-exact-rerun-5p.json`; it uses the same
source-manifest SHA-256 as the RTX run. The earlier three-process artifacts are
retained as exploratory history.

The M3 artifact originally labeled the candidate runtime using the manifest's
`29.0.1` lower bound. Its committed runner lock actually resolved `wgpu 29.0.3`
with core, HAL, and types `29.0.4`; the `runtime_stack` metadata was corrected
without changing samples, aggregates, or its source manifest. The final 0.10
candidate aligns the full stack on `29.0.4` and requires clean-commit reruns.

## Baseline policy

Starting with 0.10, wgpu 29 is Lampshade's supported compatibility baseline.
The cross-version 0.9/wgpu 30 comparison above characterizes migration cost; it
does not remain the pass/fail gate for later 0.x releases. After 0.10 is
published, future performance gates compare identical wgpu 29 stacks against
the previous Lampshade release. The 0.9 release and `release/wgpu30` preserve
the wgpu 30 line. A 1.0 version should wait for an intentionally stabilized
Lampshade API; it should not merely encode the WGPU dependency major. Intel
Vulkan performance remains unmeasured because the Beelink host was not
reachable in this run.
