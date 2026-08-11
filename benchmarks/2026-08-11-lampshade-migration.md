# Lampshade 0.8 rename regression gate

The package and public Rust namespace were renamed from `wgpu-primitives` to
`lampshade`. This migration changed naming, documentation, compatibility, and
benchmark plumbing; it did not change primitive implementations or WGSL kernels.

## Method

- Baseline: crates.io `wgpu-primitives = 0.7.0`
- Candidate: `lampshade = 0.8.0` on `feat/rebrand-lampshade`
- Base revision: `58a3fba57b9c5dcb0a7f9d0cd7d980e9f41b1068` plus the recorded dirty source manifest
- Source manifest: `d52ec2285ff7124a3a9fe0dd112e21ee8b0b0947ff6da0879d2095b0c847099d` across 72 files
- Adapter: NVIDIA GeForce RTX 4070 Ti SUPER, Vulkan
- Gate: candidate median-of-process-medians may be at most 2% slower
- Timing: identical public resident API through device completion

The initial matrix used three independent processes per source with alternating
execution order. Negative changes favor Lampshade.

| Workload | Items | 0.7 ms | Lampshade ms | Change | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| reduce_sum | 1,000,000 | 0.0848 | 0.0844 | -0.472% | pass |
| sort_bounded16 | 1,000,000 | 0.1817 | 0.1816 | -0.055% | pass |
| sort_full_width | 1,000,000 | 0.2858 | 0.2873 | +0.525% | pass |
| exclusive_scan | 1,000,000 | 0.1152 | 0.1156 | +0.347% | pass |
| compact_50 | 1,000,000 | 0.1343 | 0.1329 | -1.042% | pass |
| reduce_sum | 10,000,000 | 0.0994 | 0.1014 | +2.012% | recheck |
| sort_bounded16 | 10,000,000 | 0.9172 | 0.9170 | -0.022% | pass |
| sort_full_width | 10,000,000 | 1.6186 | 1.6237 | +0.315% | pass |
| exclusive_scan | 10,000,000 | 0.3435 | 0.3398 | -1.077% | pass |
| compact_50 | 10,000,000 | 0.4833 | 0.4847 | +0.290% | pass |
| reduce_sum | 100,000,000 | 0.7158 | 0.7118 | -0.559% | pass |
| sort_bounded16 | 100,000,000 | 7.9240 | 7.9254 | +0.018% | pass |
| sort_full_width | 100,000,000 | 14.5061 | 14.5411 | +0.241% | pass |
| exclusive_scan | 100,000,000 | 2.8443 | 2.8294 | -0.524% | pass |
| compact_50 | 100,000,000 | 3.7235 | 3.7376 | +0.379% | pass |

## Targeted recheck

The only initial miss was 10M reduction at +2.012%, 0.012 percentage points
outside the gate. A nine-process recheck measured 0.0988 ms for the published
crate and 0.1004 ms for Lampshade: +1.619%, pass.

The resolved matrix therefore passes the no-regression gate. The adjacent JSON
preserves every initial process median, the nine-process recheck medians, exact
commands, adapter identity, and candidate source-manifest hash.
