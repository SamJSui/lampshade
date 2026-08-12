# Lampshade

[![CI](https://github.com/samjsui/lampshade/actions/workflows/ci.yml/badge.svg)](https://github.com/samjsui/lampshade/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/lampshade.svg)](https://crates.io/crates/lampshade)
[![Docs.rs](https://docs.rs/lampshade/badge.svg)](https://docs.rs/lampshade)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Lampshade provides fast, composable GPU histograms, reduction, predicate masks,
prefix scan, stream compaction, run-length encoding, and unsigned integer radix
sort for Rust applications using wgpu and WGSL.

## Benchmarks

Resident GPU-buffer benchmarks include command recording, submission, execution,
and reusable workspace management. They exclude host upload and validation
readback. Reduction comparisons include the required four-byte scalar readback
for both libraries. Inputs are deterministic, outputs are validated, and reported
comparisons are medians of independent process medians.

The [published-release regression harness](benchmarks/release-regression/README.md)
runs identical resident workloads against crates.io 0.9 and the current checkout,
writes raw runs and process medians to JSON, and enforces a 2% regression budget.
The [0.8 typed-pipeline stabilization report](benchmarks/2026-08-10-typed-pipeline-stabilization.md)
records the final fixed-path gate and targeted rechecks.
The [Lampshade migration report](benchmarks/2026-08-11-lampshade-migration.md)
verifies the 0.8 rename against its published 0.7 predecessor.
The [key-only sort report](benchmarks/2026-08-11-key-only-sort.md) records the
fixed-length `u32` 8-bit path, large-input validation, and its unchanged
key/value control.
The [0.9 release report](benchmarks/2026-08-11-lampshade-0.9-release.md)
records the crates.io 0.8 regression gate, final key-sort/RLE characterization,
and identical-package validation on RTX, Jetson Orin, Intel, and Apple GPUs.
The [WGPU 29 candidate report](benchmarks/2026-08-11-wgpu29-release.md)
records an RTX pass and the accepted WGPU 29 Metal completion-boundary cost.
Starting with 0.10, WGPU 29 is Lampshade's compatibility baseline; the 0.9
release and `release/wgpu30` preserve the WGPU 30 line. The adjacent
[downstream spike report](benchmarks/2026-08-12-downstream-adoption-spikes.md)
separates promising Gaussian-splatting integrations from release readiness.
The [GPU-resident count report](benchmarks/2026-08-09-gpu-resident-counts.md)
separates isolated scheduling cost from full compaction-to-sort/reduction
results on RTX, Intel, and two Jetsons, plus fixed-path regression controls.

The comparison tables below are historical pre-0.10 measurements using the
runtime versions stated in their linked reports. They remain algorithmic
baselines, not evidence for the WGPU 29 release candidate; the 0.10 regression
and downstream integration reports supersede them where available.

### Against Massively 0.96

On an RTX 4070 Ti SUPER using Vulkan, Lampshade was faster in every
overlapping 100-million-item workload:

| Workload | Lampshade | Massively | Speedup |
| --- | ---: | ---: | ---: |
| Stable key/value sort, 16-bit keys | 7.961 ms | 167.915 ms | 21.09x |
| Stable key/value sort, full-width keys | 14.559 ms | 168.132 ms | 11.55x |
| Exclusive scan | 2.837 ms | 3.550 ms | 1.25x |
| Stable compaction, 50% selected | 3.717 ms | 5.662 ms | 1.52x |
| Wrapping sum reduction | 0.714 ms | 1.388 ms | 1.94x |

The same comparison at 10 million items also favored Lampshade on two
Jetson Orin Nano systems:

| Workload | RTX 4070 Ti SUPER | Jetson, 8 TPC | Jetson, 4 TPC |
| --- | ---: | ---: | ---: |
| Stable key/value sort, 16-bit keys | 7.98x | 8.48x | 8.55x |
| Stable key/value sort, full-width keys | 4.51x | 4.53x | 4.55x |
| Exclusive scan | 2.81x | 1.53x | 1.48x |
| Stable compaction, 50% selected | 2.06x | 1.66x | 1.63x |

See the [Massively harness](benchmarks/massively-comparison/README.md) and
[wgpu 30 report](benchmarks/2026-08-09-wgpu30-runtime.md) for the method, exact
revisions, complete matrices, and machine-readable results.

### Intel Vulkan

On Intel Alder Lake-N integrated graphics at 10 million items,
Lampshade led Massively in every workload. Sort uses the
capability-gated 4-bit radix path; reduction uses the portable kernel:

| Workload | Lampshade | Massively | Speedup |
| --- | ---: | ---: | ---: |
| Stable key/value sort, 16-bit keys | 129.879 ms | 562.704 ms | 4.33x |
| Stable key/value sort, full-width keys | 262.103 ms | 587.898 ms | 2.24x |
| Exclusive scan | 12.450 ms | 34.210 ms | 2.75x |
| Stable compaction, 50% selected | 15.900 ms | 42.429 ms | 2.67x |
| Wrapping sum reduction | 3.776 ms | 4.585 ms | 1.21x |

All 74 tests in the historical 0.6 release-candidate suite passed; the counted
and typed APIs added later in 0.8 were not part of that Intel run. At 100M,
reduction measured 21.983 ms versus 22.836 ms for Massively, a 1.04x lead. The
[Intel wide-radix report](benchmarks/2026-08-09-intel-wide-radix.md) includes
1M-100M results, stage profiles, and measured regression controls. At 100M,
the same four speedups are 9.78x, 4.79x, 2.44x, and 2.52x respectively.

### Apple Metal

Upgrading from wgpu 28 to wgpu 30 removed the previous host-returning reduction
deficit on an M3 Pro. These are final-candidate medians of three independent
process medians:

| Items | Lampshade | Massively | Speedup |
| ---: | ---: | ---: | ---: |
| 1M | 0.171 ms | 0.749 ms | 4.37x |
| 10M | 0.479 ms | 0.844 ms | 1.76x |
| 100M | 3.260 ms | 3.602 ms | 1.11x |

Massively 0.96 could not initialize these Metal pipelines: its generated layouts
requested 42 or 47 storage buffers against the adapter limit of 29. The harness
records this as an unsupported comparison, not an artificial speedup. Reduction
does run in both libraries and uses the same end-to-host scalar boundary. All 74
tests in the historical 0.6 release-candidate suite and every then-current 100M
benchmark validator passed on the M3 Pro; the later 0.8 counted and typed APIs
were not rerun there. See the
[wgpu 30 report](benchmarks/2026-08-09-wgpu30-runtime.md), the earlier
[Apple report](benchmarks/2026-08-08-apple-metal-validation.md), and the
[upstream issue](https://github.com/massively-labs/massively/issues/62).

### Against wgpu_sort

On the tested NVIDIA Vulkan system at 100 million key/value pairs:

| Key width | Lampshade | `wgpu_sort` | Speedup |
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
- Adjacent `u32` run-length encoding with a GPU-resident run count.
- Stable radix sort for `u32` values and `(u32 key, u32 value)` pairs.
- Explicit key-width bounds that skip unnecessary radix passes.
- Slice APIs for simple upload/execute/readback workflows.
- GPU-buffer APIs for composing work in one command encoder.
- Capacity-bounded sort and reduction driven by GPU-resident item counts.
- Typed buffer views and an ordered `pipeline` recorder that prepares shared
  GPU-count metadata automatically.
- Reusable scratch storage and no `unsafe` blocks in library code.

The Massively and `wgpu_sort` comparisons above were collected under the former
`wgpu-primitives` package name. The 0.8 rebrand changed package/import names but
not those kernels or timing boundaries; newer Lampshade measurements are linked
separately.

## Installation

Published Lampshade 0.9 uses wgpu 30. Lampshade 0.10 moves the public API to
wgpu 29 so its buffers compose directly with current graphics projects. Until
0.10 is published, use this checkout for integration testing. Tokio is listed
because the executable quick start
below uses `#[tokio::main]`; library development dependencies do not propagate
to applications.

```toml
[dependencies]
lampshade = { path = "../lampshade" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

The predecessor was published as `wgpu-primitives = "0.7"`. Existing users can
move to Lampshade and change Rust imports from `wgpu_primitives` to `lampshade`.
Choose Lampshade 0.9 or the `release/wgpu30` branch for wgpu 30. Use Lampshade
0.10 onward for wgpu 29; wgpu buffer types from different major versions are
not interchangeable.

## Quick start

```rust
use lampshade::{
    Compactor, Context, MaskGenerator, Reducer, RunLengthEncoder, Scanner, Sorter,
    U32Predicate,
};

#[tokio::main]
async fn main() -> Result<(), lampshade::Error> {
    let context = Context::init().await?;
    let generator = MaskGenerator::from_context(&context);
    let mut reducer = Reducer::from_context(&context);
    let mut scanner = Scanner::from_context(&context);
    let mut compactor = Compactor::from_context(&context);
    let mut run_length = RunLengthEncoder::from_context(&context);
    let mut sorter = Sorter::from_context(&context);

    let input = [4, 17, 9, 22, 11, 3];
    let mask = generator
        .mask(&input, U32Predicate::GreaterThanOrEqual(10))
        .await?;

    assert_eq!(mask, [0, 1, 0, 1, 1, 0]);
    assert_eq!(reducer.sum(&input).await?, 66);
    assert_eq!(scanner.scan_exclusive(&[3, 1, 4, 1]).await?, [0, 3, 4, 8]);
    assert_eq!(compactor.compact(&input, &mask).await?, [17, 22, 11]);
    assert_eq!(
        run_length.encode(&[3, 3, 7, 7, 7, 3]).await?,
        (vec![3, 7, 3], vec![2, 3, 1]),
    );
    assert_eq!(sorter.sort(&input).await?, [3, 4, 9, 11, 17, 22]);
    Ok(())
}
```

See [`examples/`](examples/) for standalone primitives and composed resident
pipelines. The [particle pipeline](examples/particle_pipeline.rs) filters,
stably compacts, and depth-sorts key/entity records with one submission and one
final readback.

## GPU-resident composition

The `pipeline` API below has been stable since Lampshade 0.8 and is not present in the
published `wgpu-primitives` 0.7 predecessor.

Applications that already own a wgpu device should reuse it and record multiple
primitives before submitting once. The stable `pipeline` API carries buffer
ranges, capacities, and fixed or GPU-resident extents between operations:

```rust,ignore
let mut primitives = pipeline::Primitives::new(&device, &queue);
let input_view = pipeline::GpuSlice::from_range(&input_buffer, 0..item_count)?;
let mask_output = pipeline::GpuSliceMut::from_range(&mask_buffer, 0..item_count)?;
let compacted = pipeline::GpuSliceMut::from_range(&compacted_buffer, 0..item_count)?;
let sorted = pipeline::GpuSliceMut::from_range(&sorted_buffer, 0..item_count)?;
let sum = pipeline::GpuSliceMut::from_range(&sum_buffer, 0..1)?;
let count = pipeline::GpuCount::new(&output_count)?;

primitives.reserve_workspace(
    pipeline::WorkspaceRequirements::new(item_count)
        .predicate()
        .compact()
        .counted_sort()
        .counted_reduce(),
)?;
primitives.reserve_count(count, item_count)?;

let mut recorder = primitives.record(&mut encoder);
let mask = recorder.mask(
    input_view,
    mask_output,
    U32Predicate::GreaterThanOrEqual(10),
)?;
let compacted = recorder.compact(input_view, mask, compacted, count)?;
let sorted = recorder.sort(compacted, sorted, pipeline::SortOptions::default())?;
recorder.reduce(sorted, sum, U32Reduction::Sum)?;
drop(recorder);
queue.submit(Some(encoder.finish()));
```

This recording boundary is especially important on WGPU 29 Metal. On the
tested M3 Pro, an explicit submit plus `Device::poll(Wait)` cost about 1.53 ms
even for an empty command buffer. Record the whole GPU workflow and synchronize
only for a required host readback; calling a host-returning convenience method
for every primitive repeatedly pays that runtime cost.

[`resident_pipeline.rs`](examples/resident_pipeline.rs) composes `u32`
predicate, compaction, sort, and reduction. The
[particle example](examples/particle_pipeline.rs) proves the same typed flow for
`KeyValue` records: predicate, stable compaction, and stable sort by key.
Compaction writes the selected count and later primitives consume it without a
CPU synchronization point. The recorder caches a `GpuCountPlan` internally and
schedules its preparation once after the count producer. Existing raw-buffer
and explicit-plan APIs remain available.

`GpuSlice` ranges use element indices and may start at aligned nonzero offsets.
Different read/write roles in one primitive must still use distinct underlying
buffer handles: WebGPU treats writable storage use as exclusive even for
disjoint static binding ranges. `reserve_workspace` prepares only the requested
pipelines and grows only their capacity-dependent workspaces; bind groups and
small uniform buffers may still be created while commands are recorded.

Plans default to `CountedSortDispatch::Indirect`, which sizes radix
reduce/scatter launches to the GPU-selected prefix and is the portable choice
for unknown or sparse counts. Its histogram scan remains capacity-sized.
`CountedSortDispatch::Capacity` trades inactive workgroups for lower dispatch
overhead and should be selected only with workload-specific benchmark evidence.

The command encoder preserves GPU execution order. Rust borrows the encoder and
buffers only while recording; no input is cloned or read back. Use
`KeyValueSorter::new_for_adapter` when adapter metadata is available so compatible
fast paths can be selected.

The resident methods validate sizes, ranges, alignment, and usages, but do not
inspect GPU data.
Masks must contain only `0` or `1`; declared key-width bounds must contain every
key. Primitive participants that read and write must use distinct buffer handles.
Full usage requirements are documented on each API
at [docs.rs](https://docs.rs/lampshade).

Applications that own the adapter as well as the device should construct the
facade with `Primitives::new_for_adapter(&device, &queue, &adapter_info)` so
measured hardware-specific paths remain available. The
[repository-only standalone consumer](https://github.com/samjsui/lampshade/tree/main/validation/particle-app)
validates this public API boundary and records typed-versus-raw overhead on
discrete NVIDIA and integrated Intel GPUs.

See the [architecture guide](docs/architecture.md) for the public convenience,
resident composition, and private kernel/runtime layers. The
[typed-pipeline guide](docs/typed-pipeline.md) records the API contract and
stabilization evidence.

## How it works

- **Histogram:** each workgroup accumulates up to 256 counters in shared memory,
  then merges at most one count per bin into the global output. Values outside
  the requested range are ignored.
- **Reduction:** each workgroup combines a coalesced input range into one
  partial value; later passes repeat over the partials until one value remains.
  A count plan builds the hierarchy and indirect dispatch arguments from a
  GPU-resident length.
- **Predicate mask:** one thread evaluates each value or `KeyValue` field and
  writes a `0` or `1`.
- **Scan:** workgroups scan local ranges, recursively scan block totals, then add
  those totals to produce global prefixes. Supported devices use subgroup
  operations; others use the portable shared-memory path.
- **Compaction:** an exclusive mask scan gives stable destination indices.
  Scatter combines block-local offsets with scanned block totals without
  materializing another full-size prefix pass.
- **Run-length encoding:** head flags and an exclusive scan assign each
  adjacent run an index. Ordered scatter/finalize dispatches write its value,
  length, and GPU-resident run count. Counted input is clamped to capacity and
  inactive scan lanes are zeroed without a host readback. Only
  `unique_values[..run_count]` and `run_lengths[..run_count]` are initialized;
  reused output-buffer tails are unspecified.
- **Radix sort:** stable least-significant-digit passes ping-pong between buffers.
  Known key-width bounds reduce the pass count. Compatible NVIDIA Vulkan
  devices use 8-bit or 4-bit paths, capable Intel Vulkan devices use a 4-bit
  path, and other adapters retain the portable 2-bit path. GPU-counted sorting
  uses the portable kernel with either count-proportional indirect dispatch or
  explicit capacity dispatch while preserving the same stable ordering contract.

## Profiling

GPU timestamp spans are available for every primitive when the selected device
enables timestamp queries. `Context::init` intentionally leaves them disabled
on Apple Metal and integrated NVIDIA Vulkan because those paths produced
incomplete timestamps or corrupted repeated dispatches. Dispatches still carry
stable labels for tools such as NVIDIA Nsight Graphics.

```powershell
$env:WGPU_BACKEND = 'vulkan' # or 'dx12'
$env:WGPU_PRIMITIVES_PROFILE_CASES = 'compact_50'
$env:WGPU_PRIMITIVES_PROFILE_VALIDATE = '1'
cargo run --release --example profile_primitives
```

Cases include histogram, reduction, scan, sort, predicate, value compaction,
and key/value compaction at selectable sizes and selectivities. Run-length
encoding exposes the same timestamp-span API and has a dedicated Criterion
bench across multiple average run lengths. Its dense GPU-counted control is
reported against capacity, because counted RLE deliberately scans that full
capacity even when the resident count is sparse. See the
[RTX RLE benchmark](benchmarks/2026-08-11-run-length-encoding.md) for the
source-pinned 1M-100M result.

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
cargo test --release --lib --tests
cargo check --release --examples --benches
cargo test --doc
cargo package
```

Criterion benches cover each primitive plus `counted_pipeline` and the
raw-versus-typed `particle_pipeline`. GPU integration tests skip when no
compatible adapter is available. Set `LAMPSHADE_REQUIRE_GPU_TESTS=1` to turn an
adapter miss into a test failure; CI sets it while using Mesa's Vulkan software
adapter.

## License

MIT
