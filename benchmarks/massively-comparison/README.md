# Massively comparison

This harness compares the overlapping public GPU primitives in wgpu-primitives and
[Massively 0.96.0](https://crates.io/crates/massively). The two implementations run in
separate processes because they use different wgpu versions (`28` and `30`). Massively is
pinned exactly in `Cargo.lock`; its `v0.96` source revision is
`ef9de55190529be98203aca207edab9d560d312e`.

## Workloads

- `sort_bounded16`: stable `u32` key/value radix sort where keys are known to fit in 16 bits.
- `sort_full_width`: stable `u32` key/value radix sort across the full 32-bit key range.
- `exclusive_scan`: wrapping unsigned 32-bit exclusive prefix sum.
- `compact_50`: stable copy-if using a precomputed `u32` mask with 50% selectivity.

Every runner generates the same deterministic input. Before timing, it reads the output back and
checks ordering, sort stability and permutation, exact scan values, or exact stable compaction.
Massively's sort-by-key API returns only the permuted values, whereas `KeyValueSorter`
writes both keys and values; the validator reconstructs Massively's keys through the
original value indices so both implementations must satisfy the same stable ordering.

## Timing boundary

The primary measurement is resident public-API wall time through confirmed GPU completion. Host
upload, readback, and correctness validation are excluded. This includes a real API difference:
wgpu-primitives reuses caller-owned output and workspace buffers, while Massively's public APIs
return a newly allocated owned output on each call. CubeCL may satisfy that allocation from its
cache after warmup. The JSON preserves this distinction instead of presenting the calls as
identical kernel-only measurements.

Each normal case warms for at least two seconds and four calls, then records 11 samples. A 100M
case uses two warmups and seven samples. The published value is the median of three independent
process medians. `wgpu_primitives_speedup` is `Massively time / wgpu-primitives time`, so values
above `1.0x` favor wgpu-primitives.

## Run

Windows smoke test:

```powershell
./benchmarks/massively-comparison/run.ps1 -Quick
```

Windows full matrix:

```powershell
./benchmarks/massively-comparison/run.ps1
```

Linux full matrix:

```sh
./benchmarks/massively-comparison/run.sh
```

Both scripts write machine-readable output to `results/latest.json` by default. PowerShell records
individual workload failures (including out-of-memory failures) and continues the remaining
matrix; the POSIX runner currently fails fast.

The first three-GPU result and the original 100M scan correctness finding are published in the
[2026-08-08 report](../2026-08-08-massively-comparison.md) with a compact
[machine-readable snapshot](../2026-08-08-massively-comparison.json).
The exact-range fix and the now-valid 100M scan and compaction results are in the
[fix follow-up](../2026-08-08-scan-scratch-binding-fix.md) and its compact
[machine-readable snapshot](../2026-08-08-scan-scratch-binding-fix.json).
