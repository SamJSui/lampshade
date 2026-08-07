# Jetson Explicit Byte-Pass Scheduling

This follow-up makes the NVIDIA Vulkan 8-bit radix path honor the caller's
declared key width directly. Bounds of 1-8, 9-16, 17-24, and 25-32 bits now
record one, two, three, and four scatter passes respectively. The change was
audited for pass parity, cached binding reuse, device limits, buffer aliasing,
and maximum dispatch size before physical validation on both Jetson systems.

## Revisions and method

- Source branch base revision: `be74b7fdf9262f22aa5704240bfe415054d46da3`
- Isolated benchmark snapshot revision:
  `d11e0c7b574867b515a5d624caeab6bbd3800f64`
- Benchmarked source archive SHA-256:
  `3ABDC65AD9B634C5F50602EAD0F165BE345591AA469E71F72A5BD44B508B5EC6`
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
R39.2 emitted persistence-mode errors after restoring, as in the preceding
run, but the frequency ranges and CPU idle states were restored.

The archive was committed in each disposable benchmark repository so the
runner could record a stable snapshot revision; that synthetic revision is not
the parent of the working branch. The benchmark used isolated temporary source
directories. The compact
[machine-readable snapshot](2026-08-07-jetson-explicit-byte-passes.json)
records the results and artifact hashes. The complete controlled runner outputs
used to produce it had these SHA-256 hashes:

- `dopey`: `3742B5CCC36A6AAE68D1C20B7801E23E94DD6545E74B1D2539AB9F74F3DABA25`
- `grumpy`: `3D4E06B1AD3FDB424136CD153DE0D2006BB5E1AFD93D86961F53D5426B66D985`

## Audit and correctness

The independent read-only audit identified four issues that were fixed before
benchmarking:

- Cache identity and pass routing now include the active pass count, so changing
  `key_bits` cannot reuse an incompatible binding layout and every odd/even pass
  count ends in the caller's output buffer.
- GPU-buffer entry points reject aliased input and output buffers. Odd pass
  counts cannot safely use one allocation for both storage bindings.
- Fast-path selection checks the shader's 256-invocation workgroup and 16,388-byte
  workgroup-storage requirements in addition to the subgroup contract.
- The accepted element count is capped by the device's 1D compute-dispatch
  limit. A device with 65,535 workgroups per dimension accepts at most
  117,438,720 pairs for this 1,792-pair tile layout.

Both hosts passed all 38 library and GPU integration tests. New coverage checks
the boundaries 0, 1, 8, 9, 16, 17, 24, 25, and 32; one- and three-pass output
parity; cache rebuilding across changing bounds; profiling span counts; the
zero-bit all-zero-key case; and buffer-alias rejection.

## Parameter screening

The explicit scheduling result was measured with the existing structural
parameters: 256 threads, seven pairs per thread, 1,792 pairs per tile, and a
2,048-workgroup histogram cap. A 10-million-pair RTX screening made both tested
tile alternatives worse:

| Pairs per thread | Bounded 16-bit | Full width |
| ---: | ---: | ---: |
| 6 | 0.949 ms | 1.716 ms |
| **7 (kept)** | **0.914 ms** | **1.647 ms** |
| 8 | 0.962 ms | 1.729 ms |

One-process screening of histogram caps 256, 512, and 1,024 produced only a
narrow 0.911-0.921 ms bounded range and 1.607-1.615 ms full-width range. It did
not establish a repeatable improvement worth adding another device-specific
parameter. A different workgroup width was not a free parameter: this shader
maps its 256 local invocations directly to the 256 radix buckets.

## Controlled comparison

Times are milliseconds. `Previous` is the immediately preceding integrated
subgroup result. `Change` compares explicit scheduling with it. `vs sort` is the
speedup over pinned `wgpu_sort`; values above 1 mean `wgpu-primitives` is faster.

| Host | Workload | Pairs | Previous | New | Change | `wgpu_sort` | vs sort |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dopey` (8 TPCs) | bounded 16-bit | 1M | 1.450 | 1.373 | -5.3% | 2.801 | 2.04x |
| `dopey` (8 TPCs) | full width | 1M | 2.575 | 2.597 | +0.8% | 2.904 | 1.12x |
| `dopey` (8 TPCs) | bounded 16-bit | 10M | 11.398 | 11.242 | -1.4% | 25.077 | 2.23x |
| `dopey` (8 TPCs) | full width | 10M | 21.505 | 21.774 | +1.3% | 26.392 | 1.21x |
| `grumpy` (4 TPCs) | bounded 16-bit | 1M | 1.478 | 1.401 | -5.2% | 2.833 | 2.02x |
| `grumpy` (4 TPCs) | full width | 1M | 2.617 | 2.649 | +1.2% | 2.972 | 1.12x |
| `grumpy` (4 TPCs) | bounded 16-bit | 10M | 11.524 | 11.427 | -0.8% | 25.020 | 2.19x |
| `grumpy` (4 TPCs) | full width | 10M | 21.832 | 22.042 | +1.0% | 26.302 | 1.19x |

Explicit pass scheduling improves bounded-key latency by 0.8-5.3%. Full-width
latency remains within 1.3% of the preceding run. More importantly, the bound
now controls recorded work for every fast-path call instead of relying on
full-width prefix logic to discover that the upper byte pair is constant.

## 100-million-pair stress result

Each `wgpu-primitives` row used two warmups, a two-second minimum warmup, seven
samples, one process, resident buffers, fixed clocks, and CPU result validation.

| Host | Active TPCs | Workload | Median | Throughput | Correct |
| --- | ---: | --- | ---: | ---: | --- |
| `dopey` | 8 | bounded 16-bit | 112.650 ms | 887.7 M pairs/s | yes |
| `dopey` | 8 | full width | 215.107 ms | 464.9 M pairs/s | yes |
| `grumpy` | 4 | bounded 16-bit | 113.957 ms | 877.5 M pairs/s | yes |
| `grumpy` | 4 | full width | 217.349 ms | 460.1 M pairs/s | yes |

The pinned `wgpu_sort` comparison could not produce a 100M result on the 8 GB
Jetson. Its independent process exited with code 101 while creating
`wgpu_sort Resident Value Backup`: wgpu reported `Not enough memory left`.
The captured stderr has SHA-256
`573FEA468537624C7CE3B5368F7BFA9AC3B34378C4B5ED75C4187691EF49C007`.
This is a capacity result, not a timing win: under this committed resident
benchmark configuration, `wgpu-primitives` completes and validates 100M pairs
where the pinned comparison does not fit.

## Conclusion

Keep the 8-bit path and its current seven-pair tile geometry. On the qualified
Jetsons, 16-bit bounds are the best default when the application can prove
them: they halve the scatter count and are about 2.2x faster than `wgpu_sort` at
10M. Use the 32-bit path when keys can occupy the full `u32`; it remains about
1.2x faster at 10M. The next optimization target is the full-width scatter
rather than a speculative 16-bit kernel split or another Jetson-only adapter.
