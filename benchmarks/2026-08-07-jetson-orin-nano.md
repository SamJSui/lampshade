# Jetson Orin Nano Super Vulkan Validation

This report validates the version 0.4 portable Vulkan path on two physical
Jetson Orin Nano Super developer kits. Both machines ran commit
`f41ee5dbe255dc7ee78677b6bf30aa38b7994a9e` from clean, isolated checkouts.

## Systems and method

Both systems reported:

- Jetson Linux R39.2 on `aarch64`
- NVIDIA Tegra Orin integrated GPU
- Vulkan 1.4.329 with NVIDIA driver 595.78
- subgroup size 32
- Rust 1.97.1
- `MAXN_SUPER` power mode

The timed runs locked CPU, GPU, and EMC clocks at 1.728 GHz, 1.020 GHz, and
3.199 GHz respectively. Each result is the median of five steady-state samples
after a two-second warmup. Input, output, and workspace buffers remain resident
on the GPU; wall time excludes initial allocation, upload, and readback.

The hosts were not equivalent GPU configurations despite reporting the same
board and driver:

| Host | Active GPU TPCs | Available RAM before validation |
| --- | ---: | ---: |
| `dopey` | 8 | 6.8 GiB |
| `grumpy` | 4 | 5.9 GiB |

Results must therefore be compared as 8-TPC and 4-TPC configurations, not as
repeat runs on interchangeable machines.

## Correctness

Both hosts passed all 26 unit and GPU integration tests with
`WGPU_BACKEND=vulkan`.

An initial `grumpy` run had four GPU integration tests fail with NVIDIA
`NvMapMemAllocInternalTagged` out-of-memory errors while an unrelated resident
JetsonFabric model held about 5.7 GiB of the shared 8 GiB system memory. After
that workload was paused, the same commit and binaries passed all 26 tests.
This is an important integrated-GPU precondition: free system memory must be
recorded because it is also GPU allocation headroom.

## Resident key-value results

| Host | Key workload | Pairs | Wall | GPU elapsed | Dispatch |
| --- | --- | ---: | ---: | ---: | ---: |
| `dopey` (8 TPCs) | bounded 16-bit | 1M | 9.812 ms | 8.954 ms | 8.497 ms |
| `dopey` (8 TPCs) | full-width | 1M | 9.355 ms | 8.451 ms | 7.993 ms |
| `dopey` (8 TPCs) | bounded 16-bit | 10M | 77.483 ms | 76.229 ms | 75.542 ms |
| `dopey` (8 TPCs) | full-width | 10M | 71.006 ms | 69.758 ms | 69.070 ms |
| `grumpy` (4 TPCs) | bounded 16-bit | 1M | 12.088 ms | 11.242 ms | 10.786 ms |
| `grumpy` (4 TPCs) | full-width | 1M | 11.511 ms | 10.659 ms | 10.202 ms |
| `grumpy` (4 TPCs) | bounded 16-bit | 10M | 101.583 ms | 100.365 ms | 99.684 ms |
| `grumpy` (4 TPCs) | full-width | 10M | 90.254 ms | 89.124 ms | 88.437 ms |

The 8-TPC configuration was 18.7-23.7% lower latency than the 4-TPC
configuration. Scatter accounted for most of the difference:

| Workload | `dopey` scatter | `grumpy` scatter | Reduction |
| --- | ---: | ---: | ---: |
| bounded 16-bit, 1M | 6.004 ms | 8.329 ms | 27.9% |
| full-width, 1M | 5.528 ms | 7.752 ms | 28.7% |
| bounded 16-bit, 10M | 55.706 ms | 79.134 ms | 29.6% |
| full-width, 10M | 49.229 ms | 67.879 ms | 27.5% |

Reduce changed much less at 10M: 19.301 versus 19.955 ms for bounded keys and
19.303 versus 19.961 ms for full-width keys. More TPCs therefore improve the
scatter bottleneck substantially without producing linear end-to-end scaling.

## Portable-path implication

The portable 2-bit path executes all 16 radix passes for both inputs. On both
hosts, the bounded 16-bit input was slower than the full-width input even
though its upper eight digits are zero:

- `dopey`: 77.483 versus 71.006 ms at 10M, 9.1% slower
- `grumpy`: 101.583 versus 90.254 ms at 10M, 12.6% slower

The constant upper digits make their scatter passes more expensive rather than
free. This strengthens the case for adaptive identity-pass elimination on the
portable path. That optimization still needs a GPU-resident way to establish
the active key width and must preserve stable ordering.

These numbers do not extend the RTX `wgpu_sort` comparison to Jetson;
`wgpu_sort` was not measured on either Orin host. They independently establish
portable Vulkan correctness, the current Orin baseline, and the next measured
optimization target.

## Reproduction

From a clean checkout of the measured revision:

```sh
sh benchmarks/run-jetson-validation.sh
```

Stop memory-resident GPU workloads first and record `free -h`,
`nvpmodel -q`, `jetson_clocks --show`, and the active TPC count. Jetson Linux
R39.2 emitted errors from `jetson_clocks --restore` on `grumpy`. On `dopey`, a
pre-existing difference between the live and configured TPC masks made the
power-mode reset require a reboot. Clock restoration must therefore be verified
explicitly rather than assumed from a tool's exit path.
