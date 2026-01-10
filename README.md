# wgpu-algorithms

[![Crates.io](https://img.shields.io/crates/v/wgpu-algorithms.svg)](https://crates.io/crates/wgpu-algorithms)
[![Docs.rs](https://docs.rs/wgpu-algorithms/badge.svg)](https://docs.rs/wgpu-algorithms)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A high-performance, safe WebGPU sorting and scanning library for Rust.

**Safe Rust API.** All memory management, bind groups, and synchronization are handled internally. No `unsafe` blocks in library code.

## Performance Benchmarks

Benchmarks run on **Apple M3 Max** (Metal backend).

* **CPU:** Rayon Parallel Sort (`par_sort_unstable`)
* **GPU Resident:** Sorts data already on VRAM (Pipeline use-case)
* **GPU Round-Trip:** Upload -> Sort -> Download (Utility use-case)

| Items | CPU (Rayon) | GPU (Resident) | GPU (Round-Trip) | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **100k** | **0.52 ms** | 6.0 ms | 7.2 ms | ❌ **CPU Wins** (Driver Overhead) |
| **1M** | **4.5 ms** | 9.1 ms | 10.1 ms | ❌ **CPU Wins** |
| **10M** | 44.1 ms | **31.3 ms** | 40.9 ms | ✅ **GPU Wins** (1.4x) |
| **100M** | 506 ms | **273 ms** | 407 ms | 🚀 **GPU Domination** (1.85x) |

**Throughput (100M items):**
* **Scan:** ~5.2 Billion items/sec
* **Sort:** ~365 Million items/sec

### Prefix Scan (Inclusive Add)
Benchmarks include driver submission overhead (`queue.submit` + `device.poll`).

| Items | Time | Throughput | Bandwidth (Effective) |
| :--- | :--- | :--- | :--- |
| **100 M** | **19.2 ms** | **5.2 Gelem/s** | **~41.6 GB/s** |

> **Note:** Bandwidth calculated as Read + Write (4 bytes * 2 * items / time).

## Architecture
The library implements state-of-the-art parallel algorithms tailored for the WebGPU execution model:

* **LSD Radix Sort:** A 2-bit pass (4 bins) decoupling "Counting" and "Scattering" kernels.
* **Hierarchical Scan:** A "Reduce-Then-Scan" approach using 3 separate kernels (Downsweep, Scan-Aux, Upsweep) to handle arbitrary input sizes.
* **Vector Tiling (VT):** Automatically adjusts items-per-thread based on GPU capability (e.g., `VT=8` for Desktop, `VT=4` for Mobile) to saturate memory bandwidth.

## Features
* **Adaptive Sorting:** Automatically switches between CPU (latency-optimized) and GPU (throughput-optimized) based on input size (< 1M items uses CPU).
* **Zero-Allocation Hot Loop:** Reuses internal workspace buffers and pre-baked BindGroups to minimize driver pressure during animation loops.
* **WGPU Safe:** Runs on Metal, Vulkan, DX12, and WebGPU without experimental features.

## Usage

```rust
use wgpu_algorithms::{Context, Sorter};

#[tokio::main]
async fn main() {
    // 1. Initialize GPU Context
    let ctx = Context::init().await.unwrap();
    let mut sorter = Sorter::new(&ctx);

    // 2. Data
    let data: Vec<u32> = (0..50_000_000).map(|_| rand::random()).collect();

    // 3. Sort (Auto-selects CPU or GPU based on size)
    let sorted = sorter.sort(&data).await;
    
    // 4. Verification
    assert!(sorted[0] <= sorted[1]);
}
```

## Installation

Add this to your `Cargo.toml`:

```
[dependencies]
wgpu-algorithms = "0.1.0"
```