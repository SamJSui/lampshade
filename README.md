# wgpu-primitives

[![CI](https://github.com/samjsui/wgpu-primitives/actions/workflows/ci.yml/badge.svg)](https://github.com/samjsui/wgpu-primitives/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/wgpu-primitives.svg)](https://crates.io/crates/wgpu-primitives)
[![Docs.rs](https://docs.rs/wgpu-primitives/badge.svg)](https://docs.rs/wgpu-primitives)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Fast, composable GPU predicate masks, prefix scan, stream compaction, and unsigned integer radix sort for Rust applications using wgpu.

## Benchmarks

Resident GPU-buffer benchmarks include command recording, submission, execution,
and reusable workspace management. They exclude host upload and readback. Inputs
are deterministic, outputs are validated, and reported comparisons are the median
of three process medians.

### Against Massively 0.96

On an RTX 4070 Ti SUPER using Vulkan, `wgpu-primitives` was faster in every
overlapping 100-million-item workload:

| Workload | `wgpu-primitives` | Massively | Speedup |
| --- | ---: | ---: | ---: |
| Stable sort, 16-bit keys | 8.395 ms | 165.442 ms | 19.71x |
| Stable sort, full-width keys | 14.990 ms | 165.862 ms | 11.07x |
| Exclusive scan | 2.836 ms | 3.174 ms | 1.12x |
| Stable compaction, 50% selected | 3.736 ms | 5.695 ms | 1.52x |

The same comparison at 10 million items also favored `wgpu-primitives` on two
Jetson Orin Nano systems:

| Workload | RTX 4070 Ti SUPER | Jetson, 8 TPC | Jetson, 4 TPC |
| --- | ---: | ---: | ---: |
| Stable sort, 16-bit keys | 7.98x | 8.48x | 8.55x |
| Stable sort, full-width keys | 4.51x | 4.53x | 4.55x |
| Exclusive scan | 2.81x | 1.53x | 1.48x |
| Stable compaction, 50% selected | 2.06x | 1.66x | 1.63x |

See the [Massively harness](benchmarks/massively-comparison/README.md) and
[latest NVIDIA report](benchmarks/2026-08-08-fused-compaction-prefix.md) for the
method, exact revisions, complete matrices, and machine-readable results.

### Apple Metal

An M3 Pro completed all 64 release tests and every 100-million-item validator:

| Workload | Time | Throughput |
| --- | ---: | ---: |
| Stable sort, 16-bit keys | 147.804 ms | 0.68 billion pairs/s |
| Stable sort, full-width keys | 294.699 ms | 0.34 billion pairs/s |
| Exclusive scan | 13.736 ms | 7.28 billion items/s |
| Stable compaction, 50% selected | 18.302 ms | 5.46 billion items/s |

Massively 0.96 could not initialize these Metal pipelines: its generated layouts
requested 42 or 47 storage buffers against the adapter limit of 29. The harness
records this as an unsupported comparison, not an artificial speedup. See the
[Apple report](benchmarks/2026-08-08-apple-metal-validation.md) and
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

- Inclusive and exclusive `u32` prefix scan.
- Reusable comparison predicates that produce compaction-ready masks.
- Stable compaction of `u32` values and `KeyValue` records.
- Stable radix sort for `u32` values and `(u32 key, u32 value)` pairs.
- Explicit key-width bounds that skip unnecessary radix passes.
- Slice APIs for simple upload/execute/readback workflows.
- GPU-buffer APIs for composing work in one command encoder.
- Reusable scratch storage and no `unsafe` blocks in library code.

## Installation

```toml
[dependencies]
wgpu-primitives = "0.4"
```

Version 0.4 uses wgpu 28. The package continues the crate previously published
as `wgpu-algorithms`.

## Quick start

```rust
use wgpu_primitives::{Compactor, Context, MaskGenerator, Scanner, Sorter, U32Predicate};

#[tokio::main]
async fn main() -> Result<(), wgpu_primitives::Error> {
    let context = Context::init().await?;
    let generator = MaskGenerator::from_context(&context);
    let mut scanner = Scanner::from_context(&context);
    let mut compactor = Compactor::from_context(&context);
    let mut sorter = Sorter::from_context(&context);

    let input = [4, 17, 9, 22, 11, 3];
    let mask = generator
        .mask(&input, U32Predicate::GreaterThanOrEqual(10))
        .await?;

    assert_eq!(mask, [0, 1, 0, 1, 1, 0]);
    assert_eq!(scanner.scan_exclusive(&[3, 1, 4, 1]).await?, [0, 3, 4, 8]);
    assert_eq!(compactor.compact(&input, &mask).await?, [17, 22, 11]);
    assert_eq!(sorter.sort(&input).await?, [3, 4, 9, 11, 17, 22]);
    Ok(())
}
```

See [`examples/`](examples/) for key/value sorting, structured compaction, and
standalone examples for every primitive.

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

The command encoder preserves GPU execution order. Rust borrows the encoder and
buffers only while recording; no input is cloned or read back. Use
`KeyValueSorter::new_for_adapter` when adapter metadata is available so compatible
fast paths can be selected.

The resident methods validate sizes and usages, but do not inspect GPU data.
Masks must contain only `0` or `1`; declared key-width bounds must contain every
key. Sort input and output must be distinct, and buffers must not overlap where a
write could race with a read. Full usage requirements are documented on each API
at [docs.rs](https://docs.rs/wgpu-primitives).

## How it works

- **Predicate mask:** one thread evaluates each value or `KeyValue` field and
  writes a `0` or `1`.
- **Scan:** workgroups scan local ranges, recursively scan block totals, then add
  those totals to produce global prefixes. Supported devices use subgroup
  operations; others use the portable shared-memory path.
- **Compaction:** an exclusive mask scan gives stable destination indices.
  Scatter combines block-local offsets with scanned block totals without
  materializing another full-size prefix pass.
- **Radix sort:** stable least-significant-digit passes ping-pong between buffers.
  Known key-width bounds reduce the pass count. Adapter-aware NVIDIA Vulkan
  devices use a specialized 8-bit path; other adapters use portable paths.

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

Cases include scan, sort, predicate, value compaction, and key/value compaction
at selectable sizes and selectivities.

## Roadmap

1. Validate AMD, Intel, more Apple GPUs, and additional driver versions.
2. Improve portable key-width detection for GPU-resident inputs.
3. Add derived primitives only when real workloads justify their API and cost.
4. Revisit full-width scatter when hardware counters or a new algorithm provide
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

Criterion benches cover `scan`, `compact`, `key_value_compact`, `predicate`,
`sort`, and `key_value_sort`. GPU integration tests skip when no compatible
adapter is available; CI uses Mesa's Vulkan software adapter.

## License

MIT
