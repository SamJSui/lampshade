# Lampshade in a 3D Gaussian Splatting renderer

On September 3, 2026, the optional Lampshade sorter reduced GPU
`preprocess + sort + render` time by **54.3%**, from **5.553 to 2.537 ms**, on a
695,779-splat scene at 1024x1024 using an RTX 4070 Ti SUPER and Vulkan.
Output was byte-identical in the checks before and after every benchmark process.
This is a fixed-workload result, not an interactive-FPS or general renderer claim.

## Primary result

Each entry below is the median of seven independent process medians, with
31 measured samples per backend per process. Ratios use the aggregate medians.

| Measurement | Embedded sorter | Lampshade | Reduction | Speedup |
|---|---:|---:|---:|---:|
| GPU encoder timestamps | 5.553152 ms | 2.537472 ms | 54.31% | 2.188x |
| Instrumented serialized wall time | 5.6556 ms | 2.6242 ms | 53.60% | 2.155x |

All seven primary processes are retained; none was discarded:

| Run | First backend | Wall ms, embedded / Lampshade | GPU ms, embedded / Lampshade | GPU reduction |
|---:|---|---:|---:|---:|
| 1 | baseline | 5.9769 / 2.6621 | 5.881856 / 2.575360 | 56.22% |
| 2 | candidate | 5.6653 / 2.6235 | 5.560320 / 2.537472 | 54.36% |
| 3 | baseline | 5.6317 / 2.6203 | 5.552128 / 2.537472 | 54.30% |
| 4 | candidate | 5.6391 / 2.6185 | 5.553152 / 2.536448 | 54.32% |
| 5 | baseline | 5.6556 / 2.6257 | 5.553152 / 2.537472 | 54.31% |
| 6 | candidate | 5.6426 / 2.6242 | 5.552128 / 2.536448 | 54.32% |
| 7 | baseline | 5.8524 / 2.7143 | 5.768192 / 2.631680 | 54.38% |

See [primary raw samples](results/2026-09-03-primary/),
[process medians](results/2026-09-03-primary/process-medians.csv), and
[aggregate summary](results/2026-09-03-primary/summary.json).
GPU clocks were not fixed: process telemetry is retained, and the first process
shows a clock-state transition. The median paired-process GPU reduction is
54.32%; the full process range is 54.30% to 56.22%.

A preceding seven-process batch gave 54.31% aggregate GPU reduction, but its
runner failed to record the local viewer revision. Its lockfile and benchmark
source hashes match the primary batch. Those [preliminary raw results](results/2026-09-03-preliminary/)
are retained separately and are not pooled into the headline. The primary runner
verified the clean viewer revision before running. In published metadata/logs,
only private absolute scene paths were replaced with the asset filename;
numerical samples are unchanged.

## What is measured

Both paths call the public `Viewer::render()` on the same device, queue, viewer,
scene buffers, camera, and render target. Only the sorter is swapped, before the
host timer starts. Preprocessing regenerates the depth keys and indices each time.
The candidate asserts that the native Lampshade plan is selected; the baseline
asserts that it is not. There is no CPU count readback in the timed workload.

- Twenty alternating warm-up pairs precede 31 alternating measured pairs.
- The first backend alternates within and between the seven fresh processes.
- GPU timestamps bracket preprocessing, sorting, and rendering. Query resolution
  and timestamp readback are outside that GPU interval.
- Host timing starts after command encoding and `encoder.finish()`, immediately
  before submission, and ends after blocking `device.poll`. It includes
  query-resolution/copy, map setup, and polling instrumentation.
- Exact 1024x1024 RGBA equality is checked before and after each process; the
  baseline also remains identical across the run. All 1,048,576 pixels differ
  from the black clear color. Equality is not checked after every timed sample.

This measures warmed, serialized GPU frame work for one scene and camera, not
cold initialization, presentation, pipelined application FPS, or CPU frame pacing.
The scene has 695,779 splats; this is not a claim that every splat is visible or
that exactly that many elements are sorted. The integration retains embedded
sorter resources for compatibility and therefore uses additional memory. No
memory-saving, other-GPU/backend, browser, or all-scene claim is made.

## Exact inputs

- Viewer integration: [`466025c8a9e566ac69ddea1f71388b1daf3a13e0`](https://github.com/SamJSui/wgpu-3dgs-viewer/commit/466025c8a9e566ac69ddea1f71388b1daf3a13e0), based on upstream `e4b3127f4043cb53a8ce19c56af49bc9cf942b4d`.
- Viewer/core: 0.8.0; core commit `d98ccc8f033ac060cf41401ee01087d5afd81124`.
- Lampshade: crates.io **0.13.0**, checksum `e812e6ee6701da8821e88648bcfeace908f91b4d84bb45044d75635bfd634923`.
- wgpu: **30.0.1**. The committed `Cargo.lock` pins the complete dependency graph.
- Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)`; Cargo: `1.97.1 (c980f4866 2026-06-30)`.
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER; driver **610.88**; backend **Vulkan**; timestamp period **1 ns**.
- Benchmark source SHA-256: `A0110D46C9943DEBC19382099500FCD84A6BF357B2686250E37ED8716BA97643`.
- Lockfile SHA-256: `42C4C17BF3442FCD037D8601229CB9AA0270FB1F8BB3C708C9A6AFABCCA01A88`.
- Scene: `cactus_splat3_30kSteps_719k_splats.ply`, **164,205,321 bytes**, **695,779 parsed splats** despite the filename.
- Scene SHA-256: `BF7C775739CDE020A45DA8CBFB1F274DDF2FE0A3424F29960A21947306751F09`.
- Scene attribution: [Steam Studio](https://www.steam-studio.jp). The source archive's `Readme.txt` declares CC0. A verified original download URL was not retained; supply the matching PLY locally. No scene data is redistributed here.
- Camera: yaw/pitch **0.1 radians**, vertical field of view **60 degrees**, near/far **0.1 / 10,000**, identity model transform; target **1024x1024 RGBA8**.

## Reproduce

Run from this directory on an eligible NVIDIA/Vulkan device. Supply the matching
scene yourself. The package is isolated from Lampshade's workspace and uses the
published crate plus the pinned viewer integration, not local path dependencies.

```powershell
$env:WGPU_BACKEND = 'vulkan'
$env:LAMPSHADE_3DGS_MODEL = (Resolve-Path '<matching-scene.ply>').Path
$env:LAMPSHADE_3DGS_WARMUPS = '20'
$env:LAMPSHADE_3DGS_SAMPLES = '31'
cargo +1.97.1 test --locked --release
& ./test_benchmark_statistics.ps1
for ($run = 1; $run -le 7; $run++) {
    $env:LAMPSHADE_3DGS_FIRST = if ($run % 2 -eq 0) { 'candidate' } else { 'baseline' }
    cargo +1.97.1 run --locked --release --quiet | Tee-Object -FilePath "run-$run.txt"
    if ($LASTEXITCODE -ne 0) { throw "Benchmark process $run failed" }
}
```

The harness emits raw timings and per-process medians. Use
`benchmark_statistics.ps1` for numeric aggregate medians; it includes the
floor-index fix for odd sample counts and a regression test. Do not combine
measurements from different source revisions, scenes, or environments.

The public harness passed formatting, its release-mode median test, strict
Clippy, and the PowerShell median tests. Native integration validation also
checks full/partial/zero visibility, exact RGBA parity, and unsupported-feature
fallback. Two upstream selection doctests fail on the unchanged base because of
enum-versus-struct construction, and a wasm32 check encounters pre-existing
core `Send`/`Sync` errors; this benchmark does not establish a passing browser build.
