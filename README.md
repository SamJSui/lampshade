# wgpu-primitives

[![CI](https://github.com/samjsui/wgpu-primitives/actions/workflows/ci.yml/badge.svg)](https://github.com/samjsui/wgpu-primitives/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/wgpu-primitives.svg)](https://crates.io/crates/wgpu-primitives)
[![Docs.rs](https://docs.rs/wgpu-primitives/badge.svg)](https://docs.rs/wgpu-primitives)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Safe, composable GPU predicate masks, prefix scan, stream compaction, and unsigned integer radix sort for Rust applications using wgpu.

`wgpu-primitives` continues the package previously published as `wgpu-algorithms` beginning with version 0.2.

## Benchmark highlights

With data already resident on an NVIDIA RTX 4070 Ti SUPER, the GPU-buffer APIs delivered the following results at 100 million items (`u32` values or `KeyValue` pairs):

| Primitive | Best backend | GPU time | Resident throughput | Reference | Relative speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Inclusive prefix scan | DX12 | 5.568 ms | 17.96 billion elements/s | Scalar CPU | 5.02x |
| Exclusive prefix scan | DX12 | 6.238 ms | 16.03 billion elements/s | Scalar CPU | 4.58x |
| Predicate mask | Vulkan | 1.379 ms | 72.51 billion elements/s | Scalar CPU | 17.28x |
| Radix sort | Vulkan | 43.724 ms | 2.287 billion elements/s | Rayon | 6.35x |
| Stable key-value radix sort, 16-bit keys | Vulkan | 8.605 ms | 11.62 billion pairs/s | `wgpu_sort` | 1.73x |

These figures measure the composable resident-buffer path: command encoding, submission, primitive execution, and reusable workspace management are included, while host upload and readback are excluded. See [Performance](#performance) for smaller inputs and round-trip results.

The stable key-value row uses the NVIDIA Vulkan fast path added in version 0.4
and the median of three benchmark-process medians. It has 42.2% lower latency
(1.73x speedup) than `wgpu_sort` at 100 million pairs for this bounded-key
workload on the tested NVIDIA Vulkan system. With random full-width `u32` keys,
the same path measured 15.457 ms, a modest 2.8% latency reduction (1.03x
speedup). The remaining headline rows are from version 0.3.

## Features

- Inclusive and exclusive `u32` prefix scan.
- Reusable `u32` comparison predicates that generate compaction-ready masks for
  values or either field of `KeyValue` records.
- Stable `u32` and `KeyValue` stream compaction from caller-provided 0/1 masks.
- Stable 2-bit LSD radix sort for `u32` values.
- Stable LSD radix sort for `(u32 key, u32 value)` pairs, with a profiled NVIDIA Vulkan fast path.
- Opt-in significant-key-bit bounds that eliminate unnecessary portable and wide radix passes.
- Convenience slice APIs that upload, execute, and read back.
- GPU-buffer APIs that record into an existing command encoder.
- Reusable internal scratch storage.
- No `unsafe` blocks in library code.

## Usage

The convenience context is useful for standalone compute programs:

```rust
use wgpu_primitives::{Compactor, Context, Scanner, Sorter};

#[tokio::main]
async fn main() -> Result<(), wgpu_primitives::Error> {
    let context = Context::init().await?;
    let mut scanner = Scanner::from_context(&context);
    let mut compactor = Compactor::from_context(&context);
    let mut sorter = Sorter::from_context(&context);

    let prefixes = scanner.scan_exclusive(&[3, 1, 4, 1]).await?;
    let compacted = compactor.compact(&[40, 10, 30, 20], &[0, 1, 1, 0]).await?;
    let sorted = sorter.sort(&[10, 4, 7, 1]).await?;

    assert_eq!(prefixes, [0, 3, 4, 8]);
    assert_eq!(compacted, [10, 30]);
    assert_eq!(sorted, [1, 4, 7, 10]);
    Ok(())
}
```

Stable key-value sorting keeps payloads in their original order when keys compare equal:

```rust
use wgpu_primitives::{Context, KeyValue, KeyValueSorter};

#[tokio::main]
async fn main() -> Result<(), wgpu_primitives::Error> {
    let context = Context::init().await?;
    let mut sorter = KeyValueSorter::from_context(&context);
    let sorted = sorter
        .sort(&[
            KeyValue::new(2, 10),
            KeyValue::new(1, 20),
            KeyValue::new(2, 30),
        ])
        .await?;

    assert_eq!(
        sorted,
        [
            KeyValue::new(1, 20),
            KeyValue::new(2, 10),
            KeyValue::new(2, 30),
        ]
    );
    Ok(())
}
```

The same record type can be filtered without separating keys from values:

```rust
use wgpu_primitives::{Context, KeyValue, KeyValueCompactor};

#[tokio::main]
async fn main() -> Result<(), wgpu_primitives::Error> {
    let context = Context::init().await?;
    let mut compactor = KeyValueCompactor::from_context(&context);
    let records = [
        KeyValue::new(2, 10),
        KeyValue::new(1, 20),
        KeyValue::new(2, 30),
    ];

    let selected = compactor.compact(&records, &[1, 0, 1]).await?;
    assert_eq!(selected, [KeyValue::new(2, 10), KeyValue::new(2, 30)]);
    Ok(())
}
```

Predicate masks remove the need to build selection flags on the CPU:

```rust
use wgpu_primitives::{Compactor, Context, MaskGenerator, U32Predicate};

#[tokio::main]
async fn main() -> Result<(), wgpu_primitives::Error> {
    let context = Context::init().await?;
    let generator = MaskGenerator::from_context(&context);
    let mut compactor = Compactor::from_context(&context);
    let input = [4, 17, 9, 22, 11, 3];

    let mask = generator
        .mask(&input, U32Predicate::GreaterThanOrEqual(10))
        .await?;
    let selected = compactor.compact(&input, &mask).await?;

    assert_eq!(mask, [0, 1, 0, 1, 1, 0]);
    assert_eq!(selected, [17, 22, 11]);
    Ok(())
}
```

The slice example emphasizes semantics. Performance-sensitive applications
should record `MaskGenerator::record_mask` followed by `record_compact` into the
same command encoder, keeping the generated mask resident and submitting once.

Applications that already own a wgpu device should reuse it:

```rust,ignore
let generator = MaskGenerator::new(&device, &queue);
let mut scanner = Scanner::new(&device, &queue);
let mut compactor = Compactor::new(&device, &queue);
let mut sorter = Sorter::new(&device, &queue);

generator.record_mask(
    &mut encoder,
    &input,
    &mask,
    item_count,
    U32Predicate::GreaterThanOrEqual(10),
)?;
scanner.record_scan(&mut encoder, &scan_input, &scan_output, item_count)?;
compactor.record_compact(
    &mut encoder,
    &input,
    &mask,
    &compacted_output,
    &compacted_count,
    item_count,
)?;
sorter.record_sort(&mut encoder, &sort_input, &sort_output, item_count)?;
queue.submit(Some(encoder.finish()));
```

When an application knows that keys fit in a narrower range, it can explicitly
reduce radix work. For example, 16-bit keys need 8 passes instead of 16 on the
portable 2-bit path and 2 passes instead of 4 on the NVIDIA 8-bit path:

```rust,ignore
sorter.record_sort_with_key_bits(
    &mut encoder,
    &sort_input,
    &sort_output,
    item_count,
    16,
)?;
```

Slice methods validate every host key before upload. GPU-buffer methods trust
the declared bound to avoid a validation reduction or readback; a key outside
the bound can produce only partially sorted output. The existing methods remain
full-width by default.

`KeyValueSorter::new_for_adapter` enables measured adapter-specific kernels when
adapter metadata is available. `KeyValueSorter::new` retains the portable path.

`record_mask` requires `STORAGE` on the input and `STORAGE | COPY_SRC` on its output. Allocate the output with `MaskGenerator::mask_buffer_size`; its `num_items` flags feed compaction directly. `record_scan` requires `COPY_SRC` on the input and `COPY_DST | STORAGE` on the output. `record_compact` requires `STORAGE` on the input and output, `STORAGE | COPY_SRC` on its 0/1 mask, and `STORAGE | COPY_DST` on its four-byte resident count. Predicate and compaction buffers must not overlap where writes could race with reads. `record_sort` requires `STORAGE` on both buffers, and its input and output must be distinct allocations.

## Installation

Version `0.4` contains inclusive and exclusive scan, key-only radix sort,
stable key-value radix sort, GPU timestamp profiling, and the adapter-selected
NVIDIA Vulkan fast path:

```toml
[dependencies]
wgpu-primitives = "0.4"
```

## Algorithms

Predicate masking runs one thread per item. Each thread reads a `u32` value—or
the selected key/value field—evaluates an equality, ordering, or inclusive-range
comparison, and writes exactly one `0` or `1`. Recording mask generation before
compaction keeps the flags GPU-resident and lets one queue submission execute
the predicate, exclusive scan, and stable scatter in order.

The scan recursively computes per-workgroup inclusive prefixes, scans the workgroup totals, and propagates those totals back through the hierarchy.

Stream compaction exclusively scans a caller-provided 0/1 mask into stable
destination indices, scatters selected `u32` values or `KeyValue` records in their original order,
and leaves the selected-item count in a caller-owned GPU buffer. The resident
API performs no validation pass or readback, so GPU masks must contain only 0
or 1; the slice convenience API validates them before upload.

The portable radix sort processes two bits per pass. A full-width sort uses 16
passes; `*_with_key_bits` calls use only the passes required by the declared
width. Each pass builds four per-workgroup histograms, scans them into global
offsets, and stably scatters values between ping-pong buffers. Odd and even pass
counts both route the final scatter to the caller's output buffer.

`KeyValueSorter` moves values with their keys during every scatter, so equal keys retain their original value order. On NVIDIA Vulkan adapters with enabled, fixed 32-wide subgroups, 256-thread workgroups, and at least 16,388 bytes of workgroup storage, adapter-aware construction selects a dedicated 8-bit path, including compatible integrated devices. It builds the active byte histograms in one read, computes their prefixes together, and uses subgroup-assisted stable scatter with partition lookback. Explicit bounds schedule one to four byte passes; on a full-width call, indirect dispatch can still skip the upper pair when both bytes are constant. Discrete NVIDIA Vulkan devices without compatible subgroups retain the 4-bit path; other hardware and backends use the portable 2-bit path.

## Performance

Criterion measurements from an RTX 4070 Ti SUPER show why the GPU-buffer API is the primary interface. Resident execution keeps data on the GPU; round-trip execution includes upload, allocation, execution, and readback. GPU acceleration becomes valuable as the workload grows enough to amortize dispatch and transfer overhead.

| Primitive | Items | CPU | Best GPU resident | Resident speedup | Best GPU round trip |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefix scan | 1M | 0.220 ms | 0.170 ms (Vulkan) | 1.29x | 1.351 ms (DX12) |
| Prefix scan | 10M | 2.232 ms | 0.717 ms (DX12) | 3.11x | 11.108 ms (DX12) |
| Prefix scan | 100M | 27.949 ms | 5.568 ms (DX12) | 5.02x | 230.390 ms (DX12) |
| Exclusive prefix scan | 100M | 28.580 ms | 6.238 ms (DX12) | 4.58x | Not measured |
| Predicate mask | 10M | 1.045 ms | 0.218 ms (Vulkan) | 4.80x | Not measured |
| Predicate mask | 100M | 23.833 ms | 1.379 ms (Vulkan) | 17.28x | Not measured |
| Radix sort | 1M | 2.458 ms | 1.331 ms (Vulkan) | 1.85x | 2.453 ms (Vulkan) |
| Radix sort | 10M | 25.224 ms | 5.511 ms (Vulkan) | 4.58x | 15.783 ms (Vulkan) |
| Radix sort | 100M | 277.730 ms | 43.724 ms (Vulkan) | 6.35x | 253.760 ms (Vulkan) |

At 100M items, resident throughput reached 17.96 billion elements/s for inclusive scan, 16.03 billion elements/s for exclusive scan, and 2.287 billion elements/s for key-only sort. The version 0.4 key-value path compares as follows:

| Key width | Pairs | `wgpu-primitives` | `wgpu_sort` | Time change |
| ---: | ---: | ---: | ---: | ---: |
| 16 bits | 10M | 0.989 ms | 1.718 ms | -42.4% |
| 16 bits | 100M | 8.605 ms | 14.884 ms | -42.2% |
| 32 bits | 10M | 1.720 ms | 1.735 ms | -0.9% |
| 32 bits | 100M | 15.457 ms | 15.907 ms | -2.8% |

See the [base benchmark methodology](benchmarks/2026-08-05-windows.md),
[timestamp baseline](benchmarks/2026-08-05-gpu-timestamps.md), and
[direct `wgpu_sort` comparison](benchmarks/2026-08-05-wgpu-sort-comparison.md).
The comparison has a
[committed reproduction harness](benchmarks/wgpu-sort-comparison/README.md) and
[machine-readable aggregate snapshot](benchmarks/2026-08-05-wgpu-sort-comparison.json).
The [version 0.4 full-width profile](benchmarks/2026-08-07-full-width-profile.md)
records a clean-revision confirmation and the measured scatter optimization
budget. The [Jetson Orin Nano validation](benchmarks/2026-08-07-jetson-orin-nano.md)
adds physical portable-Vulkan correctness and 4-TPC/8-TPC performance results.
The follow-up [Jetson `wgpu_sort` comparison](benchmarks/2026-08-07-jetson-wgpu-sort-comparison.md)
shows that an explicit 16-bit bound halves portable-path latency on both
systems while leaving full-width performance unchanged within 0.2%.
The [integrated NVIDIA subgroup follow-up](benchmarks/2026-08-07-jetson-integrated-subgroup.md)
qualifies the existing 8-bit path on both Jetsons. At 10 million pairs it
reduced bounded-key latency to 11.398-11.524 ms versus 25.045-25.111 ms for
`wgpu_sort`, and full-width latency to 21.505-21.832 ms versus 26.253-26.423 ms.
The [explicit byte-pass follow-up](benchmarks/2026-08-07-jetson-explicit-byte-passes.md)
adds audited one-to-four-pass scheduling and a 100-million-pair stress result.
At 10 million pairs, bounded 16-bit latency is 11.242-11.427 ms versus
25.020-25.077 ms for `wgpu_sort`; full-width latency is 21.774-22.042 ms versus
26.302-26.392 ms. Both 8 GB Jetsons completed and validated 100 million pairs,
while the pinned `wgpu_sort` runner could not allocate its resident backup
buffer at that size.
The bounded [full-width scatter experiment](benchmarks/2026-08-07-full-width-scatter-experiments.md)
profiles the merged kernel on RTX and both Jetsons and rejects five isolated
variants that missed the 5% improvement gate. It retains the production kernel
and records why shared reorder and decoupled lookback remain necessary.
The [stable stream-compaction baseline](benchmarks/2026-08-07-stream-compaction.md)
validates 10 million items at five selectivities on the RTX and both Jetsons.
At 50% kept, resident wall time is 0.892 ms, 7.601 ms, and 10.428 ms
respectively; every workload matches a stable CPU reference.
The [structured compaction follow-up](benchmarks/2026-08-07-key-value-compaction.md)
adds stable eight-byte `KeyValue` records. At the same 10-million-item, 50%
workload, it measured 0.973 ms, 7.873 ms, and 10.903 ms. Same-source `u32`
controls remained within 0.5% of the published baseline on all three systems.
The [predicate-mask follow-up](benchmarks/2026-08-08-predicate-masks.md) adds
GPU-generated selection flags. On the RTX, resident wall time measured 0.218 ms
for 10 million values and 1.379 ms for 100 million values, while isolated GPU
timestamps measured 0.127 ms and 1.268 ms. CPU-reference validation also passed
at both sizes on 8-TPC and 4-TPC Jetson Orin Nano systems. Their nearly equal
predicate times identify memory bandwidth, rather than TPC count, as the main
limit; the full predicate-plus-compaction path still favored the 8-TPC system
by 1.33x.
The pinned [Massively 0.96 comparison](benchmarks/2026-08-08-massively-comparison.md)
uses isolated wgpu 28 and wgpu 30 processes across the RTX and both Jetsons.
`wgpu-primitives` wins every stable-sort case: the direct full-width advantage
ranges from 3.82x at 1M on Jetson to 11.09x at 100M on RTX, while the explicit
16-bit path reaches 19.75x on RTX. At 10M, however, Massively's scan and
compaction are 1.69x and 1.54x faster on the 4-TPC Jetson, identifying the
portable hierarchical scan as the next measured performance target. The same
harness exposed a scheduling-sensitive 100M scratch-binding race in scan and
scan-derived compaction; those two 100M comparisons are withheld pending a
correctness fix and rerun.

## GPU profiling

Capability-gated hardware timestamp queries are available for `MaskGenerator`,
`Scanner`, `Compactor`, `KeyValueCompactor`, `Sorter`, and `KeyValueSorter`. Normal execution does not allocate or resolve
queries. Profiled calls return labeled dispatch spans, total dispatch time, and
elapsed GPU time; the same compute passes carry stable labels for external tools
such as NVIDIA Nsight Graphics.

Run the steady-state profile from a source checkout:

```powershell
$env:WGPU_BACKEND = 'vulkan' # or 'dx12'
cargo run --release --example profile_primitives
```

Set `WGPU_PRIMITIVES_PROFILE_CASES=compact_50` to isolate stable compaction
with a deterministic 50%-selective mask. Any percentage from `compact_0` to
`compact_100` is accepted; set `WGPU_PRIMITIVES_PROFILE_VALIDATE=1` to compare
the profiled workload against a stable CPU reference afterward.
Use `predicate_50` (or any percentage from 0 to 100) to profile GPU-generated
selection masks with the same optional CPU-reference validation.
Use `key_value_compact_50` (or any percentage from 0 to 100) for structured
record compaction.

At 100M items, the baseline profile attributed 74.8% of key-value dispatch time to stable scatter. The latest bounded-key profile spends 83.6% in two scatter passes, 16.3% in the all-byte histogram, and 0.1% in prefix setup. The finalized direct comparison harness measures the specialized path at 8.605 ms.

## Roadmap

Version 0.4 adds per-dispatch GPU timestamp profiling, a measured NVIDIA Vulkan
subgroup fast path, explicit one-to-four-byte scheduling, a reproducible pinned
`wgpu_sort` comparison, and physical Vulkan validation on Jetson Orin Nano.
Reusable predicate masks and the pinned Massively comparison complete the first
post-0.4 evidence pass. The remaining work
is ordered by measured impact:

1. **Fix hierarchical scan scratch ranges:** bind every scan level to its exact
   logical byte range, add a multi-level regression, and rerun 100M scan and
   scan-derived compaction before publishing those comparisons.
2. **Improve integrated scan performance:** profile the corrected portable scan
   on the 4-TPC Jetson, where Massively leads by 1.69x at 10M; carry any win into
   compaction without regressing RTX.
3. **Validate more hardware:** measure the specialized path on additional NVIDIA Vulkan devices and driver versions beyond the qualified discrete RTX and integrated Orin systems.
4. **Improve portability:** build on the explicit non-subgroup key-width path
   with GPU-side identity-pass detection for resident inputs whose bounds are
   not already known by the application.
5. **Extend derived primitives from evidence:** use real workloads and predicate
   benchmarks to decide whether logical composition or fused predicate-compaction
   kernels justify their API and maintenance cost.
6. **Revisit full-width scatter only with new evidence:** use hardware shader counters or a different stable-scatter algorithm; the measured local variants did not clear the 5% gate.

New primitives should land with a GPU-buffer API, deterministic boundary tests, CPU-reference validation, and benchmark coverage.

## Development

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib --tests
cargo check --examples --benches
cargo package
cargo bench --bench scan -- --noplot
cargo bench --bench compact -- --noplot
cargo bench --bench key_value_compact -- --noplot
cargo bench --bench predicate -- --noplot
cargo bench --bench sort -- --noplot
cargo bench --bench key_value_sort -- --noplot
```

Run the independent, pinned `wgpu_sort` comparison on Windows with:

```powershell
& .\benchmarks\wgpu-sort-comparison\run.ps1
```

GPU integration tests skip when no compatible adapter is available. CI installs Mesa's Vulkan software adapter so the shader paths execute on Linux.

## License

MIT
