# ArgMin-by-key benchmark

Date: 2026-08-12

Status: the new fixed and GPU-counted primitive passes correctness, package,
WebAssembly compile, and existing-primitive regression gates. Results below are
specific to one RTX 4070 Ti SUPER using Vulkan and NVIDIA driver 591.86.

## Contract

`ArgminByKey` selects the lexicographically smallest `KeyValue { key, value }`
record. Equal keys choose the smaller value. Fixed and GPU-counted inputs are
supported; the resident count is clamped to capacity, and an empty input writes
`(u32::MAX, u32::MAX)`.

The implementation reduces 256 records per workgroup, then recursively reduces
the resulting candidates. The typed API composes a GPU-produced count and the
selection in one command encoder without a host readback.

## RTX result

Criterion measured command recording, submission, GPU execution, and completion
wait. Upload and validation readback were outside timing. Each entry is the
median of three independent process slope estimates. Runs used 30 samples and a
five-second window below 10M records, and 10 samples with a 15-second window at
10M. Complete process estimates are preserved in the adjacent JSON artifact.

| Records | Raw kernel | Fixed ArgMin | Dense counted | 10%-active counted | Full-width sort | ArgMin vs sort |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 85.965 us | 86.782 us | 88.619 us | 88.926 us | 276.947 us | 3.19x |
| 65,536 | 87.989 us | 88.608 us | 90.769 us | 90.073 us | 288.759 us | 3.26x |
| 131,072 | 104.936 us | 105.713 us | 108.410 us | 106.941 us | 304.297 us | 2.88x |
| 1,000,000 | 131.491 us | 131.282 us | 134.897 us | 132.877 us | 465.016 us | 3.54x |
| 10,000,000 | 322.738 us | 322.338 us | 350.569 us | 322.904 us | 2,138.674 us | 6.63x |

Fixed ArgMin stayed within 0.95% of the prebuilt raw kernel at every size. The
dense counted path cost 2.12-8.76% over fixed because each capacity-wide
workgroup reads and derives its active extent. At 10M capacity and 10% active,
skipping inactive input reads reduced that overhead to 0.18%. A future counted
optimization should make dispatch count-proportional; this result does not claim
that dense counted recording is free.

Sorting is the useful application baseline: if an application needs only one
best record, selecting it directly was 2.88-6.63x faster than fully sorting all
records. This does not replace sorting when ordered output is required.

## Correctness and regression gates

Physical RTX tests cover empty inputs, duplicate-key tie-breaking, count clamp,
same-encoder count production, aligned nonzero ranges, alias/usage/limit errors,
hierarchy boundaries, and the 16,777,217-record two-dimensional dispatch tail
for both fixed and counted paths. The complete release suite passed with GPU
tests required. `wasm32-unknown-unknown` library compilation also passed.

The initial three-process published-0.11 regression matrix passed 16 of 18
existing rows. The two marginal failures were in unchanged kernels and cleared
under the established nine-process recheck protocol:

| Existing control | Initial | Nine-process recheck |
| --- | ---: | ---: |
| 1M 50% compaction | +4.39% | 0.00% |
| 100M 16-bit key/value sort | +2.16% | -0.62% |

All other 1M, 10M, and 100M reduction, sort, scan, and compaction rows were
within the 2% budget in the initial matrix. No existing WGSL or sorter source
changed in this feature.

## Provenance and limitations

Measurements used clean commit `98fd1c4c1fa8ff350726917692f393f7b134c170`.
The release harness recorded normalized source-manifest digest
`31c2a832afc59c88052c9de6c5cd9e31518d8c8b93e1299b66b3aa3a664272c9`.
The adjacent JSON preserves independent ArgMin estimates, regression
comparisons, and recheck process medians.

Apple Metal, Intel Vulkan, Jetson, and browser runtime execution were not
measured for this feature. The portable WGSL path compiles for WebAssembly, but
compilation is not browser runtime validation.
