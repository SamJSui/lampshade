# Reproducible `wgpu_sort` comparison

This harness reproduces the direct stable key-value sort comparison without
mixing incompatible wgpu types in one process:

- `wgpu-primitives-runner` uses the current checkout and wgpu 28.
- `wgpu-sort-runner` pins `wgpu_sort` at
  `4cb640e8cae28eba0149d470c5168cc2853466dd` and wgpu 0.20.1.
- `common` owns the xorshift input generator, configuration schema, aggregation
  fields, and source-derived memory models used by both runners.

Both runners validate stable ordering against the same CPU reference before
timing. Resident timing includes command encoding, submission, GPU execution,
and waiting. It excludes initial allocation, upload, and readback. Because
`wgpu_sort` mutates its primary buffers, the orchestrator restores and waits for
its input before starting each resident sample. `wgpu-primitives` preserves its
input.

Round-trip timing exercises each public API's practical upload-to-readback path.
It is useful application context, but it is not a kernel-only comparison: the
crates expose different layouts and convenience APIs.

## Run

From the repository root on PowerShell:

```powershell
& .\benchmarks\wgpu-sort-comparison\run.ps1
```

The full run measures 1M, 10M, and 100M pairs; bounded 16-bit and full-width
keys; resident and round-trip modes; and three independent processes. Inputs
below 100M use four warmups and 11 samples. The 100M cases use two warmups and
seven samples. The aggregate is the median of the independent process medians.

For a quick correctness and harness smoke test:

```powershell
& .\benchmarks\wgpu-sort-comparison\run.ps1 -Quick
```

Generated JSON defaults to `results/latest.json`, which is ignored by Git.
Commit deliberately named result snapshots outside that directory when they
support a published report.

## Output contract

Each raw run records:

- implementation and exact revision;
- wgpu version;
- adapter, backend, and driver metadata;
- workload, timing mode, item count, warmups, samples, and process index;
- every measured duration and its median;
- stable-reference validation status;
- source-derived known buffer bytes and explicit exclusions.

Memory figures are allocation models derived from the pinned sources, not
driver telemetry. They exclude pipelines, bind groups, and driver-managed
allocations. Round-trip peaks additionally depend on transient upload and
readback staging buffers.
