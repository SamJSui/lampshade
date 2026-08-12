# Published-release regression gate

The executable harness is repository-only maintainer tooling. This overview is
packaged with the crate's benchmark documentation, but the scripts and runner
crates are not.

This harness compares the current Lampshade checkout against its last published
predecessor using the same deterministic inputs, public resident APIs,
completion boundary, correctness checks, and process-median aggregation. The
baseline runner depends on crates.io `lampshade = "=0.9.0"` and wgpu 30; the
candidate runner depends on the Lampshade repository checkout.

For the 0.10 migration, this is a cross-runtime characterization. Its Metal
timing failures are intentionally retained and accepted; do not relabel them as
a passing same-stack regression gate. After 0.10 is published, update the
baseline runner to crates.io 0.10 so future gates compare wgpu 29 with wgpu 29.

The default matrix covers the established reduction, fixed key/value sort,
scan, and compaction controls plus the counted full-width key/value path changed
by the 0.10 candidate.

Quick build, correctness, and artifact validation (timings are informational):

```powershell
benchmarks/release-regression/run.ps1 -Quick -Backend vulkan
```

Formal validation uses three independent processes at 1M, 10M, and 100M:

```powershell
benchmarks/release-regression/run.ps1 -Backend vulkan `
  -OutputPath benchmarks/release-regression/results/latest.json
```

On macOS or Linux, use the equivalent Python entry point:

```bash
python3 benchmarks/release-regression/run.py --quick
```

The Python entry point selects Metal on macOS and Vulkan elsewhere unless
`--backend` is supplied explicitly.

For the accepted 0.9/wgpu 30 to 0.10/wgpu 29 migration, preserve formal Metal
measurements without treating the different runtime stack as the next-release
threshold gate:

```bash
python3 benchmarks/release-regression/run.py --backend metal --characterize
```

The formal command writes all raw runs and comparisons to JSON. It exits
nonzero if a run fails or any candidate median exceeds the corresponding
published median by more than the default 2% budget. Use
`--threshold-percent` to state a different gate explicitly. Quick mode still
fails correctness or runner errors but does not treat its noisy one-process
timings as a regression gate. Characterization mode likewise requires complete
same-adapter runs but records rather than enforces cross-runtime timing deltas.
The shared runner source uses only APIs present
in the pinned release; new primitives need their own performance evidence until
they become the next baseline.
