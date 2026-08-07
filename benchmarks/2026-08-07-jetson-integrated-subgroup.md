# Jetson Integrated NVIDIA Subgroup Fast Path

This follow-up qualifies the existing NVIDIA Vulkan 8-bit stable key-value
radix path on two physical Jetson Orin Nano Super systems. Selection now follows
the shader's actual requirements rather than requiring the adapter to report
`DiscreteGpu`: NVIDIA Vulkan, key-value items, the enabled wgpu `SUBGROUP`
feature, and a fixed subgroup width of 32.

## Revisions and method

- Source base revision: `be74b7fdf9262f22aa5704240bfe415054d46da3`
- Benchmarked source archive SHA-256:
  `55889069DD19687772B3D1036E168CDCCE74BFA1B8667819AEB643E1FD65CCBE`
- Pinned `wgpu_sort`: `4cb640e8cae28eba0149d470c5168cc2853466dd`
- Rust: 1.97.1 on `aarch64`
- Adapter: `NVIDIA Tegra Orin (nvgpu)`, Vulkan, `IntegratedGpu`
- Driver: NVIDIA 595.78
- Reported subgroup range: 32-32
- Timing: resident buffers, four warmups plus a two-second minimum warmup,
  eleven samples per process, and the median of three process medians

CPU, GPU, and EMC clocks were fixed at 1.728 GHz, 1.020 GHz, and 3.199 GHz for
each controlled run. The saved dynamic frequency ranges and CPU idle states
were restored afterward and verified with `jetson_clocks --show`. Jetson Linux
R39.2 still emitted the previously observed irrelevant persistence-mode errors
after restoring the effective state.

The benchmark used isolated temporary source directories. The compact
[machine-readable snapshot](2026-08-07-jetson-integrated-subgroup.json) records
the aggregate values. The complete runner outputs used to produce it had these
SHA-256 hashes:

- `dopey`: `BBFF4AA5DBD8E7B49EF15787C3A5760B84A868779EBA05D534A57A7B205A3C93`
- `grumpy`: `60612B94037D93F07CE414DB6F559CA37245EFD80CD7F58A895CE0CE28B2C436`

## Correctness

Both hosts passed all 32 library and GPU integration tests while selecting the
8-bit path. Coverage includes stable duplicate ordering, full-width keys,
bounded-key dispatch, boundary and odd sizes, caller-owned output buffers,
composed command recording, and GPU timestamp profiling.

## Results

Times are milliseconds. `Previous` is the bounded-key revision's portable path
from the immediately preceding controlled Jetson comparison. `Change` compares
the new capability-selected path with that result. `vs sort` is the speedup over
the pinned `wgpu_sort`, so a value above 1 means `wgpu-primitives` is faster.

| Host | Workload | Pairs | Previous | New | Change | `wgpu_sort` | vs sort |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dopey` (8 TPCs) | bounded 16-bit | 1M | 5.297 | 1.450 | -72.6% | 2.800 | 1.93x |
| `dopey` (8 TPCs) | full-width | 1M | 10.510 | 2.575 | -75.5% | 2.905 | 1.13x |
| `dopey` (8 TPCs) | bounded 16-bit | 10M | 41.851 | 11.398 | -72.8% | 25.111 | 2.20x |
| `dopey` (8 TPCs) | full-width | 10M | 83.672 | 21.505 | -74.3% | 26.423 | 1.23x |
| `grumpy` (4 TPCs) | bounded 16-bit | 1M | 7.453 | 1.478 | -80.2% | 2.831 | 1.92x |
| `grumpy` (4 TPCs) | full-width | 1M | 14.825 | 2.617 | -82.3% | 2.974 | 1.14x |
| `grumpy` (4 TPCs) | bounded 16-bit | 10M | 64.998 | 11.524 | -82.3% | 25.045 | 2.17x |
| `grumpy` (4 TPCs) | full-width | 10M | 129.914 | 21.832 | -83.2% | 26.253 | 1.20x |

The fast path reverses the earlier result on every measured workload. The
10-million-pair bounded workload is 2.17-2.20x faster than `wgpu_sort`, while
the full-width workload is 1.20-1.23x faster. Results also differ by only 1-2%
between the 4-TPC and 8-TPC hosts, removing the portable scatter path's strong
sensitivity to active TPC count.

No Jetson-specific shader or tuning variant was needed. The tested hardware
satisfies the same subgroup contract as the discrete NVIDIA path. Other
integrated NVIDIA devices remain capability-gated and should be measured before
making device-wide performance claims.

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

Record the adapter subgroup range, enabled wgpu features, free memory, active
TPC count, power mode, and clock state before timing. Save the dynamic clock
state to an explicit file before fixing clocks and verify the restored ranges
afterward.
