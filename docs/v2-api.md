# Experimental typed recorder

`wgpu_primitives::v2` is an additive API experiment. It tests whether resident
GPU composition can use a small common vocabulary instead of repeating raw
buffers, lengths, count plans, and preparation order in every primitive.

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
  compaction, and stable GPU-counted key/value sorting use the same vocabulary.
- `reserve_workspace` and `reserve_count`: explicit pre-recording pipeline,
  workspace, and count-metadata creation. A `WorkspaceRequirements` builder
  selects only the operations and fixed/counting modes the command stream
  needs. Per-operation bind groups and small uniform buffers may still be
  created while recording.

Views accept aligned nonzero offsets. Inputs, outputs, and counts participating
in one primitive must use distinct underlying buffer handles even when their
ranges do not overlap. WebGPU treats writable storage use as exclusive at the
buffer-handle level within a dispatch.

The existing slice, raw-buffer, and explicit `GpuCountPlan` APIs remain intact
while this facade is evaluated.

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

## Promotion status

The in-repository particle pipeline now validates ordinary typed composition,
array-of-structs ownership, GPU-resident length propagation, and stable payload
ordering. Promotion out of `v2` still requires:

1. use by at least one external application;
2. measured reservation and recording overhead on multiple adapters;
3. existing fixed and GPU-counted paths staying within the 2% release
   regression budget.
