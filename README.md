# wgpu-algorithms

[![CI](https://github.com/samjsui/wgpu-algorithms/actions/workflows/ci.yml/badge.svg)](https://github.com/samjsui/wgpu-algorithms/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/wgpu-algorithms.svg)](https://crates.io/crates/wgpu-algorithms)
[![Docs.rs](https://docs.rs/wgpu-algorithms/badge.svg)](https://docs.rs/wgpu-algorithms)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Safe, composable GPU prefix scan and unsigned integer radix sort for Rust applications using wgpu.

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
use wgpu_algorithms::{Context, Sorter};

#[tokio::main]
async fn main() -> Result<(), wgpu_algorithms::Error> {
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
wgpu-algorithms = "0.2"
```

## Algorithms

The scan recursively computes per-workgroup inclusive prefixes, scans the workgroup totals, and propagates those totals back through the hierarchy.

The radix sort processes two bits per pass. Each of its 16 passes builds four per-workgroup histograms, scans them into global offsets, and stably scatters values between ping-pong buffers.

## Historical Performance

These v0.1 measurements were collected on an Apple M3 Max using the Metal backend. The upload-and-execute path excludes readback but still uploads input on every iteration.

| Items | CPU Rayon | GPU upload + execute | GPU round trip |
| ---: | ---: | ---: | ---: |
| 100k | 0.52 ms | 6.0 ms | 7.2 ms |
| 1M | 4.5 ms | 9.1 ms | 10.1 ms |
| 10M | 44.1 ms | 31.3 ms | 40.9 ms |
| 100M | 506 ms | 273 ms | 407 ms |

At 100M items, the measured scan throughput was approximately 5.2 billion elements per second and sort throughput was approximately 365 million elements per second. Re-run the current Criterion suite before attributing these historical results to the buffer-to-buffer API.

## Development

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --lib --tests
cargo check --examples --benches
cargo package
```

GPU integration tests skip when no compatible adapter is available. CI installs Mesa's Vulkan software adapter so the shader paths execute on Linux.

## License

MIT
