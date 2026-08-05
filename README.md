# wgpu-primitives

[![CI](https://github.com/samjsui/wgpu-primitives/actions/workflows/ci.yml/badge.svg)](https://github.com/samjsui/wgpu-primitives/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/wgpu-primitives.svg)](https://crates.io/crates/wgpu-primitives)
[![Docs.rs](https://docs.rs/wgpu-primitives/badge.svg)](https://docs.rs/wgpu-primitives)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Safe, composable GPU prefix scan and unsigned integer radix sort for Rust applications using wgpu.

`wgpu-primitives` continues the package previously published as `wgpu-algorithms` beginning with version 0.2.

## Features

- Inclusive `u32` prefix scan.
- Stable 2-bit LSD radix sort for `u32` values.
- Convenience slice APIs that upload, execute, and read back.
- GPU-buffer APIs that record into an existing command encoder.
- Reusable internal scratch storage.
- No `unsafe` blocks in library code.

## Usage

The convenience context is useful for standalone compute programs:

```rust
use wgpu_primitives::{Context, Sorter};

#[tokio::main]
async fn main() -> Result<(), wgpu_primitives::Error> {
    let context = Context::init().await?;
    let mut sorter = Sorter::from_context(&context);
    let sorted = sorter.sort(&[10, 4, 7, 1]).await?;

    assert_eq!(sorted, [1, 4, 7, 10]);
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

`record_scan` requires `COPY_SRC` on the input and `COPY_DST | STORAGE` on the output. `record_sort` requires `STORAGE` on both buffers.

## Installation

```toml
[dependencies]
wgpu-primitives = "0.2"
```

## Algorithms

The scan recursively computes per-workgroup inclusive prefixes, scans the workgroup totals, and propagates those totals back through the hierarchy.

The radix sort processes two bits per pass. Each of its 16 passes builds four per-workgroup histograms, scans them into global offsets, and stably scatters values between ping-pong buffers.

## Performance

Criterion measurements from an RTX 4070 Ti SUPER show why the GPU-buffer API is the primary interface. Resident execution keeps data on the GPU; round trip execution includes upload, allocation, execution, and readback.

| Primitive | Items | CPU | Best GPU resident | Resident speedup | Best GPU round trip |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefix scan | 1M | 0.220 ms | 0.170 ms (Vulkan) | 1.29x | 1.351 ms (DX12) |
| Prefix scan | 10M | 2.232 ms | 0.717 ms (DX12) | 3.11x | 11.108 ms (DX12) |
| Prefix scan | 100M | 27.949 ms | 5.568 ms (DX12) | 5.02x | 230.390 ms (DX12) |
| Radix sort | 1M | 2.458 ms | 1.331 ms (Vulkan) | 1.85x | 2.453 ms (Vulkan) |
| Radix sort | 10M | 25.224 ms | 5.511 ms (Vulkan) | 4.58x | 15.783 ms (Vulkan) |
| Radix sort | 100M | 277.730 ms | 43.724 ms (Vulkan) | 6.35x | 253.760 ms (Vulkan) |

At 100M items, resident throughput reached 17.96 billion elements/s for scan and 2.287 billion elements/s for sort. See the [full methodology, confidence intervals, backend comparison, and memory accounting](benchmarks/2026-08-05-windows.md).

## Development

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib --tests
cargo check --examples --benches
cargo package
cargo bench --bench scan -- --noplot
cargo bench --bench sort -- --noplot
```

GPU integration tests skip when no compatible adapter is available. CI installs Mesa's Vulkan software adapter so the shader paths execute on Linux.

## License

MIT
