# Public API contract

This document inventories the public surface Lampshade intends to carry into
1.0. It describes execution and allocation boundaries that Rust signatures
alone cannot express. The contract applies to the documented public facade;
WGSL kernels, bind-group layouts, capability routing, and workspace types under
private modules remain implementation details.

## API layers

| Layer | Primary entry points | Submission and synchronization | Resource behavior |
| --- | --- | --- | --- |
| Slice convenience | Async methods such as `Sorter::sort`, `Reducer::reduce`, and `Compactor::compact` | Uploads, submits, waits, and reads the result back to the CPU | May allocate upload, output, workspace, and readback buffers |
| Immediate GPU buffer | Methods ending in `_gpu_to_gpu` | Creates and submits its own command encoder; does not wait or read the result back unless the method is a profiling call | May grow workspace and create per-operation resources before submission |
| Raw recording | Methods beginning with `record_` | Appends commands to the caller's encoder; never submits, waits, maps, or reads back | May grow workspace or create bind groups, uniforms, and other retained command resources |
| Typed recording | `pipeline::{Primitives, Recorder, GpuSlice, GpuSliceMut, GpuCount}` | Has the same append-only command boundary as raw recording | `reserve_workspace` prevents requested pipeline/workspace growth; `reserve_count` creates reusable count metadata, but ordinary recording may still create lightweight operation resources |
| Prepared SoA recording | `KeyValueSoaSorter::prepare_*` followed by `record_reserved_*` | Appends commands only | The validated native backend performs no buffer, bind-group, or pipeline creation while recording; the portable bridge does not currently make that guarantee |
| Profiling | Methods beginning with `profile_` | Records, submits, waits for completion, and reads timestamp data back | Creates timestamp/query/readback resources and may also grow primitive workspace |

Calling `reserve`, `reserve_workspace`, or `reserve_count` is not a general
promise that all later recording is allocation-free. Only a method whose own
documentation states that its prepared `record_reserved_*` path creates no GPU
resources carries that stronger guarantee.

## Public families

### Preferred typed composition

`lampshade::pipeline` is the preferred interface for an ordered command stream
whose logical length may remain on the GPU. It currently covers:

| Operation | Element layout | Fixed extent | GPU-counted extent |
| --- | --- | :---: | :---: |
| Predicate mask | `u32`, `KeyValue` | yes | no |
| Stable compaction | `u32`, `KeyValue` | yes | produces a count |
| Stable radix sort | `u32`, AoS `KeyValue` | yes | yes |
| Reduction | `u32` | yes | yes |
| Run-length encoding | `u32` | yes | yes |
| Lexicographic argmin | `KeyValue` | yes | yes |

Scan and histogram remain direct primitive APIs because their present public
operations have CPU-known extents and do not need the shared counted-plan
vocabulary. Separate-buffer SoA sorting remains a prepared expert API because
it owns two in-place buffers and a buffer-identity-specific binding plan. These
omissions are deliberate scope boundaries, not missing aliases that must be
added before 1.0.

### Direct primitive facade

The root exports `Histogram`, `Reducer`, `ArgminByKey`, `MaskGenerator`,
`Scanner`, `Compactor`, `KeyValueCompactor`, `RunLengthEncoder`, `Sorter`, and
`KeyValueSorter`. Each primitive keeps its slice, immediate GPU-buffer,
recording, and profiling methods where those execution modes are meaningful.
These APIs are the stable escape hatch for applications that do not need typed
multi-primitive composition.

Supporting root exports are part of the same facade: `KeyValue` and
`RunLengthOutputBuffers` describe public storage layouts; `U32Predicate`,
`KeyValueField`, `U32Reduction`, and `CountedSortDispatch` select operation
policy; `GpuProfile` and `GpuTimestampSpan` report profiling results; and
`Error` reports construction, validation, submission, and readback failures.

### Expert resident metadata and layouts

`GpuCountPlan` exposes reusable metadata shared by counted sort and reduction.
`KeyValueSoaSorter` exposes fixed and GPU-counted in-place sorting for separate
key and value buffers. `KeyValueSoaRequirements` lets an application merge the
optional native backend requirements into its own device request. These are
public application integration APIs, not private kernels, and remain supported
even though they sit outside `pipeline`.

### Convenience context

`Context` is optional. Applications may supply their own `wgpu::Device` and
`wgpu::Queue`; the context merely requests the subset of optional features that
Lampshade has validated for the selected adapter.

## Ownership, counts, and output initialization

- All GPU buffers remain caller-owned. Lampshade clones ref-counted wgpu
  handles when a prepared plan must retain them.
- `GpuSliceMut` means shader-writable, not Rust-exclusive. A primitive rejects
  read/write aliasing at the underlying buffer-handle level even for disjoint
  static ranges when WebGPU binding rules require exclusivity.
- GPU-resident counts are clamped to their declared CPU-known capacity.
- Counted operations initialize only the output prefix described by the
  resulting count. Reused output tails are unspecified unless a method states
  otherwise.
- Recording order within one command encoder supplies producer-to-consumer
  visibility. Lampshade does not insert a CPU readback between counted stages.

## Compatibility policy

Lampshade exposes wgpu types directly. Lampshade 1.x therefore remains on the
wgpu 29 public type line; any incompatible wgpu upgrade requires a new
Lampshade major release. Compatible wgpu 29 patch releases remain allowed by
the Cargo dependency requirement.

The public `Error` enum is non-exhaustive beginning in 0.12 so validation can
become more precise without forcing a Lampshade major release. Consumers must
include a wildcard arm when matching it.

The deprecated `lampshade::v2` alias is removed in 0.12. Replace
`lampshade::v2::*` with `lampshade::pipeline::*`; the underlying types and
behavior are unchanged. See the [0.12 migration note](migration-0.12.md).
