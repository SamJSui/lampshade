# Run-length encoding

Date: 2026-08-11

This benchmark characterizes Lampshade's new adjacent `u32` run-length
encoding primitive. It compares fixed-length input with the dense
GPU-counted path; it is not a comparison against an external RLE library.

## Method

- Candidate: working tree based on `063e545`; normalized SHA-256 hashes in the
  JSON artifact pin the exact measured source.
- Adapter: NVIDIA GeForce RTX 4070 Ti SUPER, Vulkan, driver 591.86.
- Input: sorted values with an average run length of eight.
- Three independent processes per case and Criterion's default three-second
  warm-up.
- At 1M: 30 samples over five seconds per process. At 10M and 100M: ten samples
  over 15 seconds per process.
- Each process contributes Criterion's slope point estimate; the table reports
  the median of those three estimates.
- Timing includes command recording, submission, GPU execution, and completion.
  Input upload, output readback, and CPU-reference validation are outside the
  timed region.
- Before timing, the complete unique-value and run-length outputs are checked
  against a scalar CPU reference.

The fixed and counted cases execute the same four-stage GPU algorithm: mark run
heads, exclusive scan, scatter run starts and values, then finalize lengths.
The counted case keeps its input count on the GPU. It clamps that count to
capacity and intentionally scans and dispatches capacity-wide, so its throughput
is reported against capacity rather than the active count.

## Resident wall-time result

| Items | Fixed | GPU-counted dense | Counted change | Fixed throughput |
| ---: | ---: | ---: | ---: | ---: |
| 1M | 0.216 ms | 0.227 ms | +5.24% | 4.63 Gelem/s |
| 10M | 0.966 ms | 0.962 ms | -0.39% | 10.35 Gelem/s |
| 100M | 8.036 ms | 8.034 ms | -0.02% | 12.44 Gelem/s |

The dense GPU-counted path is effectively tied with the fixed path at 10M and
100M. At 1M it costs 5.24% more, about 11 microseconds, for resident-count
handling while the overall dispatch is still short. Every raw estimate is
retained in the JSON. Earlier quick, pre-validation, and pre-final-lint
diagnostics are excluded.

The Criterion bench also covers average run lengths 1, 8, and 256 and includes
a scalar CPU reference. This report uses run length 8 as the representative GPU
case. A cross-library baseline remains future work because the repository does
not yet have an equivalent external RLE harness.

Machine-readable values and the exact source manifest are in
[`2026-08-11-run-length-encoding.json`](2026-08-11-run-length-encoding.json).

## Existing-path regression gate

The final source also ran against the published 0.7 release across reduction,
bounded and full-width sort, exclusive scan, and 50% compaction at 1M, 10M, and
100M. Fourteen of 15 initial rows passed the 2% budget. The only miss was 1M
reduction at +2.17%, a 1.8-microsecond delta; the established nine-process
targeted recheck passed at +0.48%. All 10M and 100M rows passed in the initial
matrix, so the resolved gate shows no measured existing-path regression.

## Command

```powershell
$env:WGPU_BACKEND='vulkan'
cargo bench --locked --bench run_length -- 'gpu_.*run_8/(1000000|10000000|100000000)$'
```
