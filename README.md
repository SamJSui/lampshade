# wgpu-primitives

[![CI](https://github.com/samjsui/wgpu-primitives/actions/workflows/ci.yml/badge.svg)](https://github.com/samjsui/wgpu-primitives/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/wgpu-primitives.svg)](https://crates.io/crates/wgpu-primitives)
[![Docs.rs](https://docs.rs/wgpu-primitives/badge.svg)](https://docs.rs/wgpu-primitives)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Safe, composable GPU prefix scan and unsigned integer radix sort for Rust applications using wgpu.

`wgpu-primitives` continues the package previously published as `wgpu-algorithms` beginning with version 0.2.

## Benchmark highlights

With data already resident on an NVIDIA RTX 4070 Ti SUPER, the GPU-buffer APIs delivered the following results at 100 million items (`u32` values or `KeyValue` pairs):

| Primitive | Best backend | GPU time | Resident throughput | CPU baseline | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Inclusive prefix scan | DX12 | 5.568 ms | 17.96 billion elements/s | Scalar CPU | 5.02x |
| Exclusive prefix scan | DX12 | 6.238 ms | 16.03 billion elements/s | Scalar CPU | 4.58x |
| Radix sort | Vulkan | 43.724 ms | 2.287 billion elements/s | Rayon | 6.35x |
| Stable key-value radix sort | Vulkan | 61.429 ms | 1.628 billion pairs/s | Stable Rayon | 12.36x |

These figures measure the composable resident-buffer path: command encoding, submission, primitive execution, and reusable workspace management are included, while host upload and readback are excluded. See [Performance](#performance) for smaller inputs and round-trip results.

The stable key-value row uses the current unreleased NVIDIA Vulkan fast path. The
remaining headline rows are from version 0.3.

## Features

- Inclusive and exclusive `u32` prefix scan.
- Stable 2-bit LSD radix sort for `u32` values.
- Stable LSD radix sort for `(u32 key, u32 value)` pairs, with a profiled NVIDIA Vulkan fast path.
- Convenience slice APIs that upload, execute, and read back.
- GPU-buffer APIs that record into an existing command encoder.
- Reusable internal scratch storage.
- No `unsafe` blocks in library code.

## Usage

The convenience context is useful for standalone compute programs:

```rust
use wgpu_primitives::{Context, Scanner, Sorter};

#[tokio::main]
async fn main() -> Result<(), wgpu_primitives::Error> {
    let context = Context::init().await?;
    let mut scanner = Scanner::from_context(&context);
    let mut sorter = Sorter::from_context(&context);

    let prefixes = scanner.scan_exclusive(&[3, 1, 4, 1]).await?;
    let sorted = sorter.sort(&[10, 4, 7, 1]).await?;

    assert_eq!(prefixes, [0, 3, 4, 8]);
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

Applications that already own a wgpu device should reuse it:

```rust,ignore
let mut scanner = Scanner::new(&device, &queue);
let mut sorter = Sorter::new(&device, &queue);

scanner.record_scan(&mut encoder, &scan_input, &scan_output, item_count)?;
sorter.record_sort(&mut encoder, &sort_input, &sort_output, item_count)?;
queue.submit(Some(encoder.finish()));
```

`KeyValueSorter::new_for_adapter` enables measured adapter-specific kernels when
adapter metadata is available. `KeyValueSorter::new` retains the portable path.

`record_scan` requires `COPY_SRC` on the input and `COPY_DST | STORAGE` on the output. `record_sort` requires `STORAGE` on both buffers.

## Installation

Version `0.3` contains inclusive and exclusive scan, key-only radix sort, and stable key-value radix sort:

```toml
[dependencies]
wgpu-primitives = "0.3"
```

## Algorithms

The scan recursively computes per-workgroup inclusive prefixes, scans the workgroup totals, and propagates those totals back through the hierarchy.

The portable radix sort processes two bits per pass. Each of its 16 passes builds four per-workgroup histograms, scans them into global offsets, and stably scatters values between ping-pong buffers.

`KeyValueSorter` moves values with their keys during every scatter, so equal keys retain their original value order. On discrete NVIDIA Vulkan adapters, adapter-aware construction selects a profiled 4-bit kernel that halves the number of full-buffer passes. Other hardware and backends retain the portable 2-bit kernel.

## Performance

Criterion measurements from an RTX 4070 Ti SUPER show why the GPU-buffer API is the primary interface. Resident execution keeps data on the GPU; round-trip execution includes upload, allocation, execution, and readback. GPU acceleration becomes valuable as the workload grows enough to amortize dispatch and transfer overhead.

| Primitive | Items | CPU | Best GPU resident | Resident speedup | Best GPU round trip |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefix scan | 1M | 0.220 ms | 0.170 ms (Vulkan) | 1.29x | 1.351 ms (DX12) |
| Prefix scan | 10M | 2.232 ms | 0.717 ms (DX12) | 3.11x | 11.108 ms (DX12) |
| Prefix scan | 100M | 27.949 ms | 5.568 ms (DX12) | 5.02x | 230.390 ms (DX12) |
| Exclusive prefix scan | 100M | 28.580 ms | 6.238 ms (DX12) | 4.58x | Not measured |
| Radix sort | 1M | 2.458 ms | 1.331 ms (Vulkan) | 1.85x | 2.453 ms (Vulkan) |
| Radix sort | 10M | 25.224 ms | 5.511 ms (Vulkan) | 4.58x | 15.783 ms (Vulkan) |
| Radix sort | 100M | 277.730 ms | 43.724 ms (Vulkan) | 6.35x | 253.760 ms (Vulkan) |
| Stable key-value sort | 10M | 59.421 ms | 7.427 ms (Vulkan) | 8.00x | 59.726 ms (Vulkan) |
| Stable key-value sort | 100M | 759.420 ms | 61.429 ms (Vulkan) | 12.36x | 477.790 ms (Vulkan) |

At 100M items, resident throughput reached 17.96 billion elements/s for inclusive scan, 16.03 billion elements/s for exclusive scan, 2.287 billion elements/s for key-only sort, and 1.628 billion pairs/s for stable key-value sort. See the [base benchmark methodology](benchmarks/2026-08-05-windows.md) and [Vulkan key-value optimization report](benchmarks/2026-08-05-vulkan-key-value-radix.md).

## GPU profiling (unreleased)

The current development branch adds capability-gated hardware timestamp queries to `Scanner`, `Sorter`, and `KeyValueSorter`. Normal execution does not allocate or resolve queries. Profiled calls return labeled dispatch spans, total dispatch time, and elapsed GPU time; the same compute passes carry stable labels for external tools such as NVIDIA Nsight Graphics.

Run the steady-state profile from a source checkout:

```powershell
$env:WGPU_BACKEND = 'vulkan' # or 'dx12'
cargo run --release --example profile_primitives
```

At 100M items, the baseline profile attributed 67-75% of sort dispatch time to stable scatter and 1% or less to histogram scanning. The resulting Vulkan key-value fast path reduced full-buffer radix passes from 16 to 8 and improved controlled resident wall time by 41.9%; Criterion measured a 44.1% improvement. See the [timestamp baseline](benchmarks/2026-08-05-gpu-timestamps.md) and [optimization report](benchmarks/2026-08-05-vulkan-key-value-radix.md).

## Roadmap

Version 0.3 added exclusive scan and stable key-value radix sort. Current development adds per-dispatch GPU timestamp profiling and a measured NVIDIA Vulkan key-value fast path. The next work is ordered by measured impact:

1. **Reduce runtime overhead:** remove per-invocation uniform-buffer and bind-group allocation after preserving the measured kernel profile.
2. **Profile more hardware:** validate radix-width selection on integrated GPUs and additional Vulkan implementations.
3. **Build derived primitives:** implement stream compaction and selection on top of scan.

New primitives should land with a GPU-buffer API, deterministic boundary tests, CPU-reference validation, and benchmark coverage.

## Development

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib --tests
cargo check --examples --benches
cargo package
cargo bench --bench scan -- --noplot
cargo bench --bench sort -- --noplot
cargo bench --bench key_value_sort -- --noplot
```

GPU integration tests skip when no compatible adapter is available. CI installs Mesa's Vulkan software adapter so the shader paths execute on Linux.

## License

MIT
