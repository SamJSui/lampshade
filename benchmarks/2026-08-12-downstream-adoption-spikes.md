# WGPU 29 downstream adoption spikes

Date: 2026-08-12 (America/Chicago)

These experiments test whether Lampshade can replace application-owned radix
sorters without changing their GPU-resident data flow. They are PR preparation,
not merged downstream results.

## Ranking

| Target | Technical fit | Measured result | PR status |
|---|---|---|---|
| wgpu-3dgs-viewer | Native SoA keys/indices and exact GPU-visible count | 14.3-22.0x faster sorter stage; 9.8x synthetic full frame | Best first performance PR; public-scene frame test still required |
| Cuneus | Native SoA Gaussian depth keys/indices | 10.5-21.6x faster than a correct current path | Adapter backend and correctness fix ready; still needs a published dependency and public scene |
| splat-rs | Native SoA keys/payload and GPU count | Lampshade correct; 3.1-5.8x faster than current invalid output | Correctness test first; speed claim needs corrected baseline |

All RTX timings use Vulkan driver 591.86, 11 post-warmup samples, and
submit-to-completion wall time with encoding/reset/validation excluded. The
GPU changed performance states during some runs; reports preserve raw arrays.

## wgpu-3dgs-viewer

The spike reuses the Viewer's separate key/value buffers and exact visible
count. It creates and caches the Lampshade plan once, records no per-frame
allocations, and falls back to the embedded sorter without enabled subgroups.

- 1M active / 1M capacity: 10.440 ms to 0.721 ms, 14.48x.
- 10M active / 10M capacity: 48.309 ms to 2.195 ms, 22.01x.
- 1M active / 10M capacity: 9.741 ms to 0.681 ms, 14.30x.
- A 64-Gaussian overlapping/duplicate-depth scene produced byte-identical
  output through the legacy and adapter-aware renderers.
- A 100K synthetic full frame measured 10.438 ms versus 1.065 ms (9.80x) with
  byte-identical 1024x1024 RGBA output.
- Existing adapter-aware render and subgroup-disabled fallback tests pass.

Before PR: replace the local path dependency with a published or revision-pinned
0.10 compatibility dependency, split benchmark-only public API exposure from
the production patch, and run the prepared frame harness on a public PLY/SPZ
scene. The synthetic result proves the integration seam, not representative
application performance.

## Cuneus

The Gaussian example always chooses its 16-bit sorter, but exposes
`depth_shift = 1..=30`. Its generated key can exceed 16 bits when the shift is
below 16. The benchmark proves incorrect ordering at shift 8 and correct
ordering at shift 16.

At the valid shift-16 setting, the production facade measured 0.512 ms versus
5.359 ms (10.47x) at 1M and 1.262 ms versus 24.677 ms (19.56x) at 10M. At
shift 8, it measured 0.743/2.259 ms versus 10.551/48.781 ms for Cuneus's
correct full-width path (14.19x/21.59x); the current 16-bit output was invalid.

The example now selects 16- or 32-bit sorting from `depth_shift`, rebuilds when
the UI crosses that boundary, and uses Lampshade only when the adapter/device
matches its validated native path. Its focused test and upstream all-target
check pass. Before the performance PR, replace the local dependency and measure
a public Gaussian scene.

## splat-rs

The current sorter failed full key ordering at both 1M and 10M deterministic
inputs. Lampshade passed order, association, and duplicate-stability checks.
Its measured stage was 3.10x and 5.83x faster, but comparing speed against an
incorrect output is not sufficient for an upstream performance claim.

Before PR: land a minimal regression test that fails the current implementation,
then compare Lampshade against a corrected baseline or frame the PR primarily
as a correctness repair.

## Product implication

WGPU 29 unlocks unusually clean adoption seams in current Gaussian-splatting
projects. Lampshade 0.10 therefore adopts it as the compatibility baseline,
while explicitly accepting the measured WGPU 29 Metal completion cost for
host-synchronized calls. The 0.9 release and `release/wgpu30` preserve WGPU 30;
the migration is documented rather than presented as performance-neutral.
