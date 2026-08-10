# wgpu-primitives

[![CI](https://github.com/samjsui/wgpu-primitives/actions/workflows/ci.yml/badge.svg)](https://github.com/samjsui/wgpu-primitives/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/wgpu-primitives.svg)](https://crates.io/crates/wgpu-primitives)
[![Docs.rs](https://docs.rs/wgpu-primitives/badge.svg)](https://docs.rs/wgpu-primitives)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Fast, composable GPU histograms, reduction, predicate masks, prefix scan,
stream compaction, and unsigned integer radix sort for Rust applications using
wgpu.

## Benchmarks

Resident GPU-buffer benchmarks include command recording, submission, execution,
and reusable workspace management. They exclude host upload and validation
readback. Reduction comparisons include the required four-byte scalar readback
for both libraries. Inputs are deterministic, outputs are validated, and reported
comparisons are medians of independent process medians.

The [published-release regression harness](benchmarks/release-regression/README.md)
runs identical resident workloads against crates.io 0.6 and the current checkout,
writes raw runs and process medians to JSON, and enforces a 2% regression budget.

### Against Massively 0.96

On an RTX 4070 Ti SUPER using Vulkan, `wgpu-primitives` was faster in every
overlapping 100-million-item workload:

| Workload | `wgpu-primitives` | Massively | Speedup |
| --- | ---: | ---: | ---: |
| Stable sort, 16-bit keys | 7.961 ms | 167.915 ms | 21.09x |
| Stable sort, full-width keys | 14.559 ms | 168.132 ms | 11.55x |
| Exclusive scan | 2.837 ms | 3.550 ms | 1.25x |
| Stable compaction, 50% selected | 3.717 ms | 5.662 ms | 1.52x |
| Wrapping sum reduction | 0.714 ms | 1.388 ms | 1.94x |

The same comparison at 10 million items also favored `wgpu-primitives` on two
Jetson Orin Nano systems:

| Workload | RTX 4070 Ti SUPER | Jetson, 8 TPC | Jetson, 4 TPC |
| --- | ---: | ---: | ---: |
| Stable sort, 16-bit keys | 7.98x | 8.48x | 8.55x |
| Stable sort, full-width keys | 4.51x | 4.53x | 4.55x |
| Exclusive scan | 2.81x | 1.53x | 1.48x |
| Stable compaction, 50% selected | 2.06x | 1.66x | 1.63x |

See the [Massively harness](benchmarks/massively-comparison/README.md) and
[wgpu 30 report](benchmarks/2026-08-09-wgpu30-runtime.md) for the method, exact
revisions, complete matrices, and machine-readable results.

### Intel Vulkan

On Intel Alder Lake-N integrated graphics at 10 million items,
`wgpu-primitives` led Massively in every workload. Sort uses the
capability-gated 4-bit radix path; reduction uses the portable kernel:

| Workload | `wgpu-primitives` | Massively | Speedup |
| --- | ---: | ---: | ---: |
| Stable sort, 16-bit keys | 129.879 ms | 562.704 ms | 4.33x |
| Stable sort, full-width keys | 262.103 ms | 587.898 ms | 2.24x |
| Exclusive scan | 12.450 ms | 34.210 ms | 2.75x |
| Stable compaction, 50% selected | 15.900 ms | 42.429 ms | 2.67x |
| Wrapping sum reduction | 3.776 ms | 4.585 ms | 1.21x |

All 74 release tests passed. At 100M, reduction measured 21.983 ms versus
22.836 ms for Massively, a 1.04x lead. The
[Intel wide-radix report](benchmarks/2026-08-09-intel-wide-radix.md) includes
1M-100M results, stage profiles, and measured regression controls. At 100M,
the same four speedups are 9.78x, 4.79x, 2.44x, and 2.52x respectively.

### Apple Metal

Upgrading from wgpu 28 to wgpu 30 removed the previous host-returning reduction
deficit on an M3 Pro. These are final-candidate medians of three independent
process medians:

| Items | `wgpu-primitives` | Massively | Speedup |
| ---: | ---: | ---: | ---: |
| 1M | 0.171 ms | 0.749 ms | 4.37x |
| 10M | 0.479 ms | 0.844 ms | 1.76x |
| 100M | 3.260 ms | 3.602 ms | 1.11x |

Massively 0.96 could not initialize these Metal pipelines: its generated layouts
requested 42 or 47 storage buffers against the adapter limit of 29. The harness
records this as an unsupported comparison, not an artificial speedup. Reduction
does run in both libraries and uses the same end-to-host scalar boundary. All 74
release GPU tests and every 100M benchmark validator pass on the M3 Pro. See the
[wgpu 30 report](benchmarks/2026-08-09-wgpu30-runtime.md), the earlier
[Apple report](benchmarks/2026-08-08-apple-metal-validation.md), and the
[upstream issue](https://github.com/massively-labs/massively/issues/62).

### Against wgpu_sort

On the tested NVIDIA Vulkan system at 100 million key/value pairs:

| Key width | `wgpu-primitives` | `wgpu_sort` | Speedup |
| ---: | ---: | ---: | ---: |
| 16 bits | 8.605 ms | 14.884 ms | 1.73x |
| 32 bits | 15.457 ms | 15.907 ms | 1.03x |

The [wgpu_sort report](benchmarks/2026-08-05-wgpu-sort-comparison.md) documents
the pinned baseline and reproduction harness.

## Features

- Portable 1-256-bin `u32` histograms with workgroup-private counters.
- Wrapping sum, minimum, and maximum reduction for `u32` values.
- Inclusive and exclusive `u32` prefix scan.
- Reusable comparison predicates that produce compaction-ready masks.
- Stable compaction of `u32` values and `KeyValue` records.
- Stable radix sort for `u32` values and `(u32 key, u32 value)` pairs.
- Explicit key-width bounds that skip unnecessary radix passes.
- Slice APIs for simple upload/execute/readback workflows.
- GPU-buffer APIs for composing work in one command encoder.
- Reusable scratch storage and no `unsafe` blocks in library code.

## Installation

Published version 0.7 contains histogram and reduction and uses wgpu 30. Tokio
is listed because the executable quick start below uses `#[tokio::main]`;
library development dependencies do not propagate to applications.

```toml
[dependencies]
wgpu-primitives = "0.7"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

Because wgpu types appear in the public GPU-buffer APIs, upgrading from 0.5 also
requires upgrading the application's wgpu dependency. The package continues
the crate previously published as `wgpu-algorithms`.

## Quick start

```rust
use wgpu_primitives::{
    Compactor, Context, MaskGenerator, Reducer, Scanner, Sorter, U32Predicate,
};

#[tokio::main]
async fn main() -> Result<(), wgpu_primitives::Error> {
    let context = Context::init().await?;
    let generator = MaskGenerator::from_context(&context);
    let mut reducer = Reducer::from_context(&context);
    let mut scanner = Scanner::from_context(&context);
    let mut compactor = Compactor::from_context(&context);
    let mut sorter = Sorter::from_context(&context);

    let input = [4, 17, 9, 22, 11, 3];
    let mask = generator
        .mask(&input, U32Predicate::GreaterThanOrEqual(10))
        .await?;

    assert_eq!(mask, [0, 1, 0, 1, 1, 0]);
    assert_eq!(reducer.sum(&input).await?, 66);
    assert_eq!(scanner.scan_exclusive(&[3, 1, 4, 1]).await?, [0, 3, 4, 8]);
    assert_eq!(compactor.compact(&input, &mask).await?, [17, 22, 11]);
    assert_eq!(sorter.sort(&input).await?, [3, 4, 9, 11, 17, 22]);
    Ok(())
}
```

See [`examples/`](examples/) for the composed resident pipeline, histograms,
key/value sorting, structured compaction, and standalone examples for every
primitive.

## GPU-resident composition

Applications that already own a wgpu device should reuse it and record multiple
primitives before submitting once:

```rust,ignore
let generator = MaskGenerator::new(&device, &queue);
let mut compactor = Compactor::new(&device, &queue);

generator.record_mask(
    &mut encoder,
    &input,
    &mask,
    item_count,
    U32Predicate::GreaterThanOrEqual(10),
)?;
compactor.record_compact(
    &mut encoder,
    &input,
    &mask,
    &output,
    &output_count,
    item_count,
)?;
queue.submit(Some(encoder.finish()));
```

[`resident_pipeline.rs`](examples/resident_pipeline.rs) extends this pattern
through predicate, compaction, sort, and reduction in one submission and maps
one final readback buffer.

The command encoder preserves GPU execution order. Rust borrows the encoder and
buffers only while recording; no input is cloned or read back. Use
`KeyValueSorter::new_for_adapter` when adapter metadata is available so compatible
fast paths can be selected.

The resident methods validate sizes and usages, but do not inspect GPU data.
Masks must contain only `0` or `1`; declared key-width bounds must contain every
key. Sort input and output must be distinct, and buffers must not overlap where a
write could race with a read. Full usage requirements are documented on each API
at [docs.rs](https://docs.rs/wgpu-primitives).

See the [architecture guide](docs/architecture.md) for the public convenience,
resident composition, and private kernel/runtime layers.

## How it works

- **Histogram:** each workgroup accumulates up to 256 counters in shared memory,
  then merges at most one count per bin into the global output. Values outside
  the requested range are ignored.
- **Reduction:** each workgroup combines a coalesced input range into one
  partial value; later passes repeat over the partials until one value remains.
- **Predicate mask:** one thread evaluates each value or `KeyValue` field and
  writes a `0` or `1`.
- **Scan:** workgroups scan local ranges, recursively scan block totals, then add
  those totals to produce global prefixes. Supported devices use subgroup
  operations; others use the portable shared-memory path.
- **Compaction:** an exclusive mask scan gives stable destination indices.
  Scatter combines block-local offsets with scanned block totals without
  materializing another full-size prefix pass.
- **Radix sort:** stable least-significant-digit passes ping-pong between buffers.
  Known key-width bounds reduce the pass count. Compatible NVIDIA Vulkan
  devices use 8-bit or 4-bit paths, capable Intel Vulkan devices use a 4-bit
  path, and other adapters retain the portable 2-bit path.

## Profiling

GPU timestamp spans are available for every primitive when the adapter supports
timestamp queries. Dispatches also carry stable labels for tools such as NVIDIA
Nsight Graphics.

```powershell
$env:WGPU_BACKEND = 'vulkan' # or 'dx12'
$env:WGPU_PRIMITIVES_PROFILE_CASES = 'compact_50'
$env:WGPU_PRIMITIVES_PROFILE_VALIDATE = '1'
cargo run --release --example profile_primitives
```

Cases include histogram, reduction, scan, sort, predicate, value compaction,
and key/value compaction at selectable sizes and selectivities.

## Roadmap

1. Validate AMD, more Intel and Apple GPUs, and additional driver versions.
2. Grow the CUB-like private kernel/workspace engine behind the existing safe,
   Thrust-like Rust APIs; split crates only when usage evidence justifies it.
3. Improve portable key-width detection for GPU-resident inputs.
4. Add derived primitives only when real workloads justify their API and cost.
5. Revisit full-width scatter when hardware counters or a new algorithm provide
   evidence for at least a 5% gain.

New primitives require a resident-buffer API, deterministic boundary tests,
CPU-reference validation, and reproducible benchmarks.

## Development

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --release --all-targets
cargo check --examples --benches
cargo package
```

Criterion benches cover `histogram`, `reduce`, `scan`, `compact`,
`key_value_compact`, `predicate`, `sort`, and `key_value_sort`. GPU integration
tests skip when no compatible adapter is available; CI uses Mesa's Vulkan
software adapter.

## License

MIT
