# GPU-resident dynamic counts

## Purpose

Compaction produces a data-dependent output length. Reading that four-byte
count on the CPU before sorting or reducing adds a submission, mapping wait,
and synchronization point. The counted APIs instead accept a CPU-known
capacity and a GPU-resident count. `GpuCountPlan` binds those facts once and
prepares bounded sort and reduction metadata on the GPU.

These APIs are a composition feature, not a replacement for fixed-length calls.
When the CPU already knows the exact length, the fixed path avoids preparation
and indirect-dispatch overhead.

## Candidate and method

- Candidate branch: `feat/gpu-resident-counts`
- Parent: `7c2df76a851fef5c77b5e5206f4b10133221d6af` (`0.7.0`)
- Source state: dirty benchmark candidate; the final PR commit should replace
  this line with its immutable revision.
- Release-gate LF-normalized source manifest (70 files):
  `391e369ecb8c315c7010d3c11232c35cc851de1c151b8bf7b3e52613462c850f`
- Adapter for isolated measurements: NVIDIA GeForce RTX 4070 Ti SUPER, Vulkan,
  driver 591.86
- Isolated Criterion: 10 samples and a 15-second measurement window for 10M
  and 100M
- Boundary: public resident API call through confirmed GPU completion; upload,
  validation readback, and primitive construction are excluded
- Count: equal to capacity, which isolates scheduling cost without changing the
  amount of useful sort or reduction work

## Isolated scheduling cost

| Primitive | Items | Fixed length | GPU count | Counted overhead |
| --- | ---: | ---: | ---: | ---: |
| Full-width `u32` sort | 10M | 5.379 ms | 5.738 ms | 6.68% |
| Full-width `u32` sort | 100M | 43.420 ms | 44.052 ms | 1.46% |
| Wrapping sum reduction | 10M | 100.23 us | 166.43 us | 66.05% |
| Wrapping sum reduction | 100M | 722.35 us | 775.70 us | 7.39% |

The reduction percentage is large at 10M because the fixed kernel is only about
0.1 ms; the counted path adds roughly 50-66 us in these measurements. The
absolute preparation cost amortizes at larger sizes.

## End-to-end composition

`counted_pipeline` compares the complete alternatives:

- GPU-only: compact, prepare one shared count plan, sort, and reduce in one
  encoder/submission, then wait for completion.
- Host-synchronized: compact and copy the count to a four-byte staging buffer,
  submit, map and wait, then record fixed-length sort/reduction in a second
  submission and wait again.

Inputs are deterministic. Before timing each case, the benchmark validates the
selected count, sorted prefix, and wrapping sum against CPU results. Primitive
construction and final result validation are outside the timed boundary.

The table reports Criterion point estimates at 10M source values. Negative
change means the GPU-only path is faster than host synchronization.

| Adapter | Selected | Host readback | GPU indirect | Change | GPU capacity | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 4070 Ti SUPER, Vulkan 591.86 | 10% | 2.0735 ms | 2.2727 ms | +9.61% | 2.2244 ms | +7.28% |
| RTX 4070 Ti SUPER, Vulkan 591.86 | 50% | 3.7869 ms | 3.9672 ms | +4.76% | 3.8420 ms | +1.46% |
| RTX 4070 Ti SUPER, Vulkan 591.86 | 90% | 5.6979 ms | 5.8186 ms | +2.12% | 5.6588 ms | -0.69% |
| Intel Alder Lake-N, Mesa 25.2.8 | 10% | 43.748 ms | 44.867 ms | +2.56% | 53.758 ms | +22.88% |
| Intel Alder Lake-N, Mesa 25.2.8 | 50% | 126.29 ms | 124.00 ms | -1.81% | 127.77 ms | +1.17% |
| Intel Alder Lake-N, Mesa 25.2.8 | 90% | 210.97 ms | 204.04 ms | -3.29% | 204.45 ms | -3.09% |
| Jetson `grumpy`, Vulkan 595.78 | 10% | 12.405 ms | 13.126 ms | +5.81% | 14.045 ms | +13.22% |
| Jetson `grumpy`, Vulkan 595.78 | 50% | 34.981 ms | 35.648 ms | +1.91% | 35.682 ms | +2.00% |
| Jetson `grumpy`, Vulkan 595.78 | 90% | 57.261 ms | 57.921 ms | +1.15% | 57.669 ms | +0.71% |
| Jetson `dopey`, Vulkan 595.78 | 10% | 10.559 ms | 11.303 ms | +7.05% | 12.119 ms | +14.78% |
| Jetson `dopey`, Vulkan 595.78 | 50% | 26.176 ms | 26.980 ms | +3.07% | 27.106 ms | +3.55% |
| Jetson `dopey`, Vulkan 595.78 | 90% | 41.952 ms | 42.572 ms | +1.48% | 42.021 ms | +0.16% |

RTX and Intel used 10 samples, a five-second target window, and two-second
warmup; Criterion extended slower Intel collections to obtain the requested
samples. Jetson strategy runs used 10 samples, a two-second target, and
one-second warmup. Apple M3 validation was unavailable because the machine was
offline during this run.

`CountedSortDispatch::Indirect` remains the default: it scales radix
reduce/scatter workgroups with the resident count and wins the 50%- and
90%-selected Intel cases.
`Capacity` is an explicit tuning option for dense workloads on adapters where
indirect dispatch is costly; it is not selected from vendor IDs. At zero or
near-zero selection, the fixed path can skip the radix hierarchy after reading
the count, while a pre-recorded GPU-only command graph still pays its scan/pass
overhead.

## Fixed-path regression control

The unchanged crates.io 0.7.0 runner and checkout runner used identical inputs,
public APIs, and adapter identity. Every 10M and 100M row passed the 2% gate;
the largest increases were 0.81% and 0.54%, respectively. At 1M, unchanged
reduction and scan crossed the percentage gate by only 4.7 us and 2.5 us in the
initial three-process matrix. Higher-process targeted rechecks passed:

| Recheck | Process medians | Change |
| --- | ---: | ---: |
| 1M wrapping sum | 5 | +0.12% |
| 1M exclusive scan | 9 | 0.00% |

Raw artifacts are intentionally kept under ignored `target/release-regression/`
for the working candidate.

## Interpretation

Use the counted path when it removes a CPU count readback or keeps a larger
command graph asynchronous. Use the fixed path when the CPU already owns the
exact length. The end-to-end result is adapter- and density-dependent: the
one-submission path wins dense Intel cases, approximately ties dense RTX and
Jetson cases, and trails when little work survives compaction.
