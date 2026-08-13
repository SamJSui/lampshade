# Typed pipeline API

`lampshade::pipeline` is the stable recording-first API for resident GPU
composition. It uses a small common vocabulary instead of repeating raw
buffers, lengths, count plans, and preparation order in every primitive. The
deprecated `lampshade::v2` alias was removed in 0.12; see the
[migration note](migration-0.12.md).

## Implemented

- `GpuSlice<T>`: a borrowed buffer range with a physical capacity and either a
  CPU-fixed or GPU-resident logical extent. `T` is currently `u32` or the
  crate's `KeyValue` record.
- `GpuSliceMut<T>`: the writable counterpart. `Mut` describes shader access,
  not Rust-exclusive ownership; wgpu buffers are shared handles.
- `GpuCount`: a borrowed `u32` scalar at a byte offset. Construction checks
  bounds; reservation or primitive recording checks device alignment.
- `Primitives` and `Recorder`: reusable pipelines/workspace plus ordered command
  recording. A count produced by compaction is prepared once and then shared by
  counted sort and reduction. Predicate generation, stable key/value
  compaction, stable GPU-counted key/value sorting, and fixed or counted
  run-length encoding use the same vocabulary.
- `RunLengthOutput`: unique-value and run-length views that share one
  GPU-resident run count. The input may be unsorted; sorting first produces one
  run per distinct key.
- `Recorder::argmin_by_key`: reduces a fixed or GPU-counted `KeyValue` view to
  one lexicographic minimum in a caller-owned output view.
- `reserve_workspace` and `reserve_count`: explicit pre-recording pipeline,
  workspace, and count-metadata creation. A `WorkspaceRequirements` builder
  selects only the operations and fixed/counting modes the command stream
  needs. Per-operation bind groups and small uniform buffers may still be
  created while recording.

The typed recorder deliberately covers the operations that benefit from shared
GPU-resident extent metadata. Scan, histogram, and separate-buffer SoA sorting
remain direct primitive APIs. The complete execution, allocation, and scope
contract is recorded in the [public API inventory](public-api.md).

Views accept aligned nonzero offsets. Inputs, outputs, and counts participating
in one primitive must use distinct underlying buffer handles even when their
ranges do not overlap. WebGPU treats writable storage use as exclusive at the
buffer-handle level within a dispatch.

The existing slice, raw-buffer, and explicit `GpuCountPlan` APIs remain intact.
The typed layer is additive: callers can adopt it operation by operation while
retaining direct wgpu control.

Run-length output composes directly with counted primitives:

```rust,ignore
let encoded = recorder.run_length_encode(input, values, lengths, run_count)?;
recorder.reduce(encoded.run_lengths, total, U32Reduction::Sum)?;
```

## Key/payload decision

The first application-shaped path uses the existing array-of-structs
`KeyValue { key, value }`. Particle or visibility pipelines commonly move the
sort key and entity identifier together, so this layout keeps the API small and
makes stable duplicate-key ordering directly testable:

```rust,ignore
let mask = recorder.mask_key_values(
    particles,
    mask_output,
    KeyValueField::Key,
    U32Predicate::BetweenInclusive { min: near, max: far },
)?;
let visible = recorder.compact_key_values(particles, mask, compacted, count)?;
let sorted = recorder.sort_by_key(visible, sorted, SortOptions::default())?;
```

[`particle_pipeline.rs`](../examples/particle_pipeline.rs) runs this exact shape
with one submission and one final readback. Separate key and payload buffers
remain deliberately unsupported until a workload demonstrates that their
memory-access benefit justifies another public layout and another set of
kernels.

## Stabilization evidence

The in-repository particle pipeline now validates ordinary typed composition,
array-of-structs ownership, GPU-resident length propagation, and stable payload
ordering. A separate consumer crate now validates application-owned wgpu
devices, the public crate boundary, and typed-versus-raw overhead on RTX and
Intel. The stable namespace is accepted against objective repository gates:

1. formatting, strict Clippy, rustdoc, release tests, and package verification;
2. public crate-boundary correctness with an application-owned wgpu context;
3. typed-versus-raw total time within the 2% budget on discrete and integrated
   GPUs;
4. affected GPU-counted paths within the 2% pre-change budget at 1M, 10M, and
   100M items; and
5. existing fixed paths within the 2% published-release budget, using the
   documented targeted-recheck protocol for bimodal driver states.

The [standalone consumer report](../benchmarks/2026-08-10-external-particle-consumer.md)
and [particle pipeline report](../benchmarks/2026-08-10-particle-pipeline.md)
record the satisfied crate-boundary, multi-adapter overhead, counted-path, and
fixed-path gates. Real downstream adoption will guide future ergonomics and
layout additions, but is an outcome rather than a release requirement. Apple
and Jetson results remain useful additional coverage; no result is inferred
for adapters absent from these reports.
