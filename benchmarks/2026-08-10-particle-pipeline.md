# Typed particle pipeline and counted-sort regression

Date: 2026-08-10

This change validates the typed recorder against an
application-shaped visibility pipeline: generate a depth mask, stably compact
`KeyValue { key, value }` particle records, and stably sort the selected prefix
by key. The selected count remains GPU-resident between operations.

## Candidate and host

- Parent: `c98b42ebbf1e713feb2003c6370fdb97308b0d93`
- Candidate branch: `feat/particle-pipeline`
- Candidate state: dirty working tree; the PR commit containing this report is
  the reproducible candidate
- Pre-report library diff hash: `348e6f39c41ee3c5185afe2dd3051b27f596072b`
  from `git diff --binary HEAD -- src | git hash-object --stdin`
- Adapter: NVIDIA GeForce RTX 4070 Ti SUPER, Vulkan, driver 591.86

## Counted-path regression

The generalized counted radix shader now supports both `u32` and `KeyValue`.
The affected existing `u32` path was compared directly with the parent commit
in an isolated worktree. Each source ran in three alternating processes. Each
process used two seconds of warm-up and Criterion sampling; results are medians
of process medians.

Workload: 50%-selected compaction -> indirect GPU-counted full-width sort ->
GPU-counted reduction. Timing includes command recording, submission, GPU
execution, and completion waiting; it excludes upload and validation readback.

| Capacity | Parent | Candidate | Change | 2% gate |
| ---: | ---: | ---: | ---: | :---: |
| 1M | 1.6578 ms | 1.6687 ms | +0.66% | Pass |
| 10M | 3.9522 ms | 3.9549 ms | +0.07% | Pass |
| 100M | 27.021 ms | 27.029 ms | +0.03% | Pass |

The process medians and exact method are preserved in the adjacent JSON file.

## Existing fixed paths

The formal crates.io-0.7 release gate initially passed 12 of 15 rows. All 100M
rows passed between -0.66% and +0.75%. Three smaller rows landed in opposite
fast/slow GPU states, so they were repeated with nine alternating processes:

| Rechecked row | Published 0.7 | Candidate | Change | 2% gate |
| --- | ---: | ---: | ---: | :---: |
| Exclusive scan, 1M | 0.1149 ms | 0.1152 ms | +0.26% | Pass |
| Sum reduction, 10M | 0.0995 ms | 0.0984 ms | -1.11% | Pass |
| Full-width sort, 10M | 1.6215 ms | 1.6174 ms | -0.25% | Pass |

This resolves every initially failing fixed-path row under the repository's
established targeted-recheck protocol.

## Typed façade benchmark

`benches/particle_pipeline.rs` compares the raw explicit-plan calls with the
typed recorder using equivalent buffers, preparation, validation, submission,
and completion boundaries. Short runs were state-sensitive and sometimes
favored either ordering, so they establish a reproducible benchmark and
correctness baseline, not a typed/raw parity claim. The combined wall-clock
measurement does not isolate GPU execution, CPU recording overhead, or peak
workspace.

## Validation

- `cargo fmt --all --check`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`
- `cargo test --release --all-targets`
- `cargo package --allow-dirty`
- physical Vulkan and DX12 focused key/value and typed-pipeline suites
- one-submit particle example with one final map/readback
- zero, full, and partial selection; duplicate-key stability; oversized count
  clamping; aligned nonzero ranges; capacity, usage, and alias failures

All checks passed. The independent audit found no remaining blocker or
medium-severity code issue.
