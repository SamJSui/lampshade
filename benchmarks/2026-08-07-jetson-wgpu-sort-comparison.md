# Jetson Orin Nano `wgpu_sort` Comparison

This report measures the portable Vulkan stable key-value radix path against
`wgpu_sort` on two physical Jetson Orin Nano Super systems. It also validates
the opt-in significant-key-bit path added after version 0.4.

## Revisions and method

- Baseline: `15a2670028a106c3e6260d6f4eb5ad24e8676c9f`
- Bounded-key implementation: `cc460bddef06b3691569ea1afcb8d951e26b4f56`
- Pinned `wgpu_sort`: `4cb640e8cae28eba0149d470c5168cc2853466dd`
- Rust: 1.97.1 on `aarch64`
- Driver: NVIDIA 595.78 through Vulkan
- Timing: resident buffers, four warmups plus a two-second minimum warmup,
  eleven samples per process, and the median of three process medians

The baseline runner used the normal full-width API for both input
distributions. The optimized runner passed a trusted 16-bit bound for
`bounded16` and 32 bits for `full_width`. Both revisions validated the complete
stable output against the same CPU reference before timing.

CPU, GPU, and EMC clocks were fixed at 1.728 GHz, 1.020 GHz, and 3.199 GHz for
each timed run. The original dynamic ranges were captured first and restored
afterward. Jetson Linux R39.2 still emits irrelevant `nvidia-smi` persistence
errors during `jetson_clocks --restore`; the reported CPU, GPU, EMC, idle-state,
and TPC values were checked after each restore.

The hosts are distinct GPU configurations:

| Host | Active GPU TPCs | JetsonFabric during tests |
| --- | ---: | --- |
| `dopey` | 8 | stopped |
| `grumpy` | 4 | stopped |

JetsonFabric remains discontinued; no runtime or node process was restarted.

## Correctness

At the optimized revision, both hosts passed 32 tests: 6 library unit tests and
26 GPU integration tests. The new coverage includes stable 0-, 1-, odd-, even-,
16-, and 32-bit sorts, incorrect host-bound rejection, caller-output routing,
and reduced-pass GPU profiling.

## Results

Times are milliseconds. `Change` compares `wgpu-primitives` at the optimized
revision with its own baseline. `vs sort` is optimized `wgpu-primitives` time
divided by the pinned `wgpu_sort` time, so a value above 1 means `wgpu_sort` is
still faster.

| Host | Workload | Pairs | Baseline | Optimized | Change | `wgpu_sort` | vs sort |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dopey` (8 TPCs) | bounded 16-bit | 1M | 10.935 | 5.297 | -51.6% | 2.796 | 1.89x |
| `dopey` (8 TPCs) | full-width | 1M | 10.520 | 10.510 | -0.1% | 2.903 | 3.62x |
| `dopey` (8 TPCs) | bounded 16-bit | 10M | 87.633 | 41.851 | -52.2% | 25.125 | 1.67x |
| `dopey` (8 TPCs) | full-width | 10M | 83.662 | 83.672 | +0.01% | 26.378 | 3.17x |
| `grumpy` (4 TPCs) | bounded 16-bit | 1M | 15.230 | 7.453 | -51.1% | 2.825 | 2.64x |
| `grumpy` (4 TPCs) | full-width | 1M | 14.797 | 14.825 | +0.2% | 2.973 | 4.99x |
| `grumpy` (4 TPCs) | bounded 16-bit | 10M | 133.422 | 64.998 | -51.3% | 25.026 | 2.60x |
| `grumpy` (4 TPCs) | full-width | 10M | 129.909 | 129.914 | +0.004% | 26.260 | 4.95x |

The 16-bit hint reduces the portable kernel from 16 two-bit passes to 8 and
halves latency on both systems. Full-width changes remain within 0.2%, well
inside the 5% regression gate. The optimized bounded path is also faster than
the old full-width control on both systems, but it does not yet match
`wgpu_sort`: the remaining 10M gap is 1.67x on the 8-TPC host and 2.60x on the
4-TPC host.

The second host adds useful architectural evidence. `wgpu-primitives` improves
substantially with 8 rather than 4 active TPCs, while the pinned competitor is
about 25 ms at 10M on both configurations. The remaining portable scatter path
is therefore both the main performance gap and more sensitive to available GPU
parallelism.

## Artifact provenance

The committed aggregate snapshot is
[`2026-08-07-jetson-wgpu-sort-comparison.json`](2026-08-07-jetson-wgpu-sort-comparison.json).
The raw harness outputs were copied off each host and verified against these
on-host SHA-256 values:

| Host | Revision | SHA-256 |
| --- | --- | --- |
| `dopey` | baseline | `4c7438e3e07ace844dc9bebb1ef3a35d1c0add65aa4a438d64ef8432221096a5` |
| `dopey` | optimized | `b64efdadfd16ab0ca49a4c7b7d68db562d3a410f0c974549ac4ee2afbe8a6074` |
| `grumpy` | baseline | `f169b814e153112f3822f3b7ab620ecdb2f4c56147426e50bd48c3319aa503c5` |
| `grumpy` | optimized | `5511484b28923c75278198d8b701c3d9cd89031dfb890a8a7a7d2d6ccca50d51` |

## Reproduction

From a clean checkout on either Jetson:

```sh
sh benchmarks/wgpu-sort-comparison/run.sh \
  --items 1000000,10000000 \
  --workloads bounded16,full_width \
  --modes resident \
  --processes 3 \
  --backend vulkan
```

Record free memory, the active TPC count, power mode, and clock state before
timing. Save the dynamic clock state to an explicit file before running
`jetson_clocks`, and verify restoration from the reported ranges rather than
the restore command's noisy output.
