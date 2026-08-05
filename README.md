# wgpu-primitives

[![CI](https://github.com/samjsui/wgpu-primitives/actions/workflows/ci.yml/badge.svg)](https://github.com/samjsui/wgpu-primitives/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/wgpu-primitives.svg)](https://crates.io/crates/wgpu-primitives)
[![Docs.rs](https://docs.rs/wgpu-primitives/badge.svg)](https://docs.rs/wgpu-primitives)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Safe, composable GPU prefix scan and unsigned integer radix sort for Rust applications using wgpu.

`wgpu-primitives` continues the package previously published as `wgpu-algorithms` beginning with version 0.2.

## Benchmark highlights

With data already resident on an NVIDIA RTX 4070 Ti SUPER, the GPU-buffer APIs delivered the following results at 100 million `u32` values:

| Primitive | Best backend | GPU time | Resident throughput | CPU baseline | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Inclusive prefix scan | DX12 | 5.568 ms | 17.96 billion elements/s | Scalar CPU | 5.02x |
| Exclusive prefix scan | DX12 | 6.238 ms | 16.03 billion elements/s | Scalar CPU | 4.58x |
| Radix sort | Vulkan | 43.724 ms | 2.287 billion elements/s | Rayon | 6.35x |

These figures measure the composable resident-buffer path: command encoding, submission, primitive execution, and reusable workspace management are included, while host upload and readback are excluded. See [Performance](#performance) for smaller inputs and round-trip results.

## Features

- Inclusive and exclusive `u32` prefix scan.
- Stable 2-bit LSD radix sort for `u32` values.
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

At 100M items, resident throughput reached 17.96 billion elements/s for inclusive scan, 16.03 billion elements/s for exclusive scan, and 2.287 billion elements/s for sort. See the [full methodology, confidence intervals, backend comparison, and memory accounting](benchmarks/2026-08-05-windows.md).

## Roadmap

Version 0.2 established the public GPU-buffer APIs, deterministic GPU tests, reusable workspace, cross-backend benchmarks, and the `wgpu-primitives` package name. The next work is ordered by how much it improves the crate as a reusable primitive library:

1. **Complete the core APIs:** add stable key-value radix sort without forcing data through host memory.
2. **Measure kernels directly:** add GPU timestamp-query benchmarks and per-pass profiling so optimization decisions are separated from command submission and synchronization cost.
3. **Reduce runtime overhead:** remove the radix sort's per-invocation uniform-buffer allocation and tune workgroup/radix configurations from measurements across DX12, Vulkan, and Metal.
4. **Build derived primitives:** implement stream compaction and selection on top of scan after the lower-level APIs and performance contracts are stable.
5. **Broaden hardware evidence:** publish reproducible benchmark reports from integrated and discrete GPUs, including crossover sizes, memory use, and resident versus round-trip behavior.

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
```

GPU integration tests skip when no compatible adapter is available. CI installs Mesa's Vulkan software adapter so the shader paths execute on Linux.

## License

MIT
