# Experimental typed recorder

`wgpu_primitives::v2` is an additive API experiment. It tests whether resident
GPU composition can use a small common vocabulary instead of repeating raw
buffers, lengths, count plans, and preparation order in every primitive.

## Implemented

- `GpuSlice<u32>`: a borrowed buffer range with a physical capacity and either
  a CPU-fixed or GPU-resident logical extent.
- `GpuSliceMut<u32>`: the writable counterpart. `Mut` describes shader access,
  not Rust-exclusive ownership; wgpu buffers are shared handles.
- `GpuCount`: a borrowed `u32` scalar at a byte offset. Construction checks
  bounds; reservation or primitive recording checks device alignment.
- `Primitives` and `Recorder`: reusable pipelines/workspace plus ordered command
  recording. A count produced by compaction is prepared once and then shared by
  counted sort and reduction.
- `reserve_workspace` and `reserve_count`: explicit pre-recording workspace and
  count-metadata creation. A `WorkspaceRequirements` builder selects only the
  operations and fixed/counting modes the pipeline needs. Per-operation bind
  groups and small uniform buffers may still be created while recording.

Views accept aligned nonzero offsets. Inputs, outputs, and counts participating
in one primitive must use distinct underlying buffer handles even when their
ranges do not overlap. WebGPU treats writable storage use as exclusive at the
buffer-handle level within a dispatch.

The existing slice, raw-buffer, and explicit `GpuCountPlan` APIs remain intact
while this facade is evaluated.

## Unresolved key/payload shape

The experiment currently implements only key-only `u32` operations. It does
not yet choose between array-of-structs and separate key/payload buffers. A
possible structure-of-arrays target looks like this pseudocode:

```rust,ignore
let input = KeyPayloadView {
    keys: GpuSlice::<u32>::from_range(&keys_in, range.clone())?,
    payloads: GpuSlice::<u32>::from_range(&payloads_in, range)?,
};
let output = KeyPayloadViewMut {
    keys: GpuSliceMut::<u32>::from_range(&keys_out, output_range.clone())?,
    payloads: GpuSliceMut::<u32>::from_range(&payloads_out, output_range)?,
};
let sorted = recorder.sort_by_key(input, output, SortOptions::default())?;
```

These names are deliberately non-compiling. The design should be promoted only
after a real application establishes whether separate buffers, `KeyValue`, or
both deserve first-class types without multiplying every primitive API.

## Promotion gates

1. At least one downstream application uses the recorder across multiple
   primitives without dropping to raw buffers for ordinary composition.
2. Key/payload ownership and layout are validated by a real workload.
3. Reservation semantics are measured and, if needed, extended to reusable
   pre-bound command plans.
4. Existing fixed and GPU-counted paths retain correctness and the 2% release
   regression budget.
