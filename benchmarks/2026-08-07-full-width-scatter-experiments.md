# Full-Width Scatter Experiments

This bounded optimization sprint follows the merged NVIDIA Vulkan radix work
from pull request 10. It profiles the 10-million-pair full-width scatter on an
RTX 4070 Ti SUPER and two physical Jetson Orin Nano Super systems, then tests
isolated changes to partition lookback, workgroup accumulation, shared-memory
reorder, and address calculation.

No kernel experiment met the acceptance gate. The production scatter shader is
therefore unchanged. The only retained code change corrects the profiling
example so its bounded-16 case calls the explicit 16-bit API rather than
recording a nominal 32-bit sort and relying on zero-work upper passes.

## Source and method

- Merged source revision: `5a655f035f50da6cb39b7de5f493d71d7867505f`
- Baseline profiling archive SHA-256:
  `15B62A80DD240219807FBE55BD138F5D0F43575FD01A8EA1694115E32DB787F7`
- Restored production scatter shader SHA-256:
  `7767C223A558755DFFE83FEB8AC012E3C896950D67BC65C46BA7E5B878DA2C9D`
- RTX: NVIDIA GeForce RTX 4070 Ti SUPER, Vulkan, driver 591.86
- Jetsons: NVIDIA Tegra Orin, Vulkan, driver 595.78, 32-wide subgroups
- Workload: 10 million resident stable key-value pairs
- Timing: 11 samples after a two-second warmup in each of three independent
  processes; tables report the median of process medians
- Profile: hardware GPU timestamps around histogram, prefix, and every scatter
  dispatch

Both Jetsons used MAXN_SUPER with CPU, GPU, and EMC fixed at 1.728 GHz,
1.020 GHz, and 3.199 GHz during each run. Their saved dynamic ranges and CPU
idle states were restored afterward.

The compact [machine-readable snapshot](2026-08-07-full-width-scatter-experiments.json)
contains every process median used below.

## Baseline decomposition

Times are milliseconds.

| Host | Workload | Wall | Dispatch | Histogram | Prefix | Scatter | Scatter share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 4070 Ti SUPER | bounded 16-bit | 0.929 | 0.835 | 0.170 | 0.006 | 0.659 | 78.9% |
| RTX 4070 Ti SUPER | full width | 1.626 | 1.506 | 0.173 | 0.009 | 1.324 | 87.9% |
| `dopey` (8 TPCs) | bounded 16-bit | 11.248 | 11.057 | 0.980 | 0.011 | 10.067 | 91.0% |
| `dopey` (8 TPCs) | full width | 21.752 | 21.517 | 1.315 | 0.018 | 20.184 | 93.8% |
| `grumpy` (4 TPCs) | bounded 16-bit | 11.330 | 11.127 | 0.980 | 0.011 | 10.137 | 91.1% |
| `grumpy` (4 TPCs) | full width | 21.925 | 21.666 | 1.304 | 0.018 | 20.343 | 93.9% |

Every active byte pass has essentially the same cost. Full-width latency is not
caused by one pathological digit; it is the fourfold repetition of the same
scatter bottleneck.

## Lookback upper bound

A deliberately incorrect diagnostic retained ballot ranking, serialized
workgroup accumulation, shared reorder, and coalesced global key-value writes,
but wrote each reordered tile contiguously instead of performing partition
lookback and global bucket placement. It is not a candidate implementation; it
measures an upper bound on the removable cost of lookback plus bucket address
calculation.

| Host | Baseline full wall | No-lookback wall | Wall change | Baseline scatter | No-lookback scatter | Scatter change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 4070 Ti SUPER | 1.626 | 1.459 | -10.3% | 1.324 | 1.163 | -12.2% |
| `dopey` | 21.752 | 21.863 | +0.5% | 20.184 | 20.302 | +0.6% |
| `grumpy` | 21.925 | 21.808 | -0.5% | 20.343 | 20.239 | -0.5% |

Lookback and global bucket placement are a material RTX cost but fall within
noise on Orin. Jetson scatter is dominated by ranking, serialized workgroup
accumulation, shared reorder, and memory movement.

## Targeted experiments

The acceptance gate was at least 5% lower RTX full-width wall time, no more than
2% bounded regression, and the complete parity/stability suite passing.

| Experiment | Correct | RTX bounded | Change | RTX full width | Change | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline | yes | 0.929 | - | 1.626 | - | control |
| Direct global writes; remove shared reorder | yes | 2.622 | +182.2% | 4.909 | +201.9% | reject |
| Parallel per-subgroup bucket tables | yes | 0.992 | +6.8% | 1.758 | +8.1% | reject |
| Fold bucket starts into tile offsets | yes | 0.920-0.924 | -1.0% to -0.5% | 1.627-1.636 | +0.1% to +0.6% | neutral; reject |
| Ordered ticket and global bucket counters | yes | 5.709 | +514.5% | 10.958 | +573.9% | reject |
| Pack digit into unused rank bits | yes | 0.925 | -0.4% | 1.630 | +0.2% | neutral; reject |

The direct-write result proves the shared-memory reorder is essential: its
coalesced output writes are worth much more than the local round trip costs.
The parallel subgroup table trades eight serialized phases for more workgroup
storage and seven synchronization rounds, which is slower. The ordered ticket
proves decoupled lookback concurrency is essential even though it uses more
atomics. Address folding and digit packing are compiler-level noise.

## Decision

Keep seven items per thread, the existing 16-bit path, and the current
decoupled-lookback scatter unchanged. None of the isolated modifications met
the 5% gate. A future full-width attempt should begin with hardware shader
counters or a genuinely different stable-scatter algorithm, not more local
instruction rearrangement.

The next crate-level milestone is stream compaction: flags, exclusive scan,
stable scatter, and a GPU-resident output count. That advances the crate's
primitive coverage instead of spending another sprint below the measured
optimization threshold.
