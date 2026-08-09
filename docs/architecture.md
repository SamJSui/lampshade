# Architecture

`wgpu-primitives` has one north star: make common data-parallel GPU work feel
native in Rust without giving up explicit wgpu composition or hardware-aware
performance.

The API should feel closer to Thrust: safe types, slices for simple programs,
and ordinary Rust errors. The implementation should grow more like CUB:
specialized kernels, reusable temporary storage, explicit capability routing,
and benchmarks that guard each backend. These are layers in one crate, not two
packages today. A crate split would add versioning and discovery cost before
there is evidence that users need independent release cycles.

## Three layers

### 1. Slice convenience

Methods such as `Sorter::sort(&mut self, input: &[u32])` and
`Reducer::sum(&mut self, input: &[u32])` accept borrowed CPU data, upload it,
run the primitive, and return an owned result. They are the smallest path from
ordinary Rust code to a correct GPU result.

The input borrow lasts across `.await` because the async function may still
need the slice before it finishes. The returned `Vec` owns its allocation,
while a returned `u32` is copied by value; either remains valid independently
of the input and primitive object.

### 2. Resident-buffer composition

The `*_gpu_to_gpu` methods operate on caller-owned `wgpu::Buffer` values and
submit immediately. The `record_*` methods instead borrow a caller-owned
`CommandEncoder` and append commands without submitting:

```rust,ignore
generator.record_mask(&mut encoder, &input, &mask, count, predicate)?;
compactor.record_compact(
    &mut encoder,
    &input,
    &mask,
    &output,
    &output_count,
    count,
)?;
queue.submit(Some(encoder.finish()));
```

`&mut encoder` means this call has exclusive CPU-side permission to mutate the
command list. It does not mutate the generator or clone GPU data. The buffer
borrows describe which handles must remain valid while commands are recorded;
actual reads and writes happen later on the GPU in submission order.

This layer is the stable composition boundary. Applications can keep data on
the GPU, combine primitives in one submission, and manage synchronization at
the wgpu level.

### 3. Private runtime, workspace, and kernels

Private code owns the mechanisms that public methods share:

- `CommandSession` owns an encoder until one immediate submission.
- `ProfileSession` adds optional timestamp queries and waits for completion.
- `ReusableBuffer` owns a lazily allocated buffer and its capacity.
- `AdapterCapabilities` records hardware facts separately from kernel policy.
- primitive-specific pipelines select and dispatch WGSL implementations.

The reusable state explains why reduction, scan, compaction, and sort generally
take `&mut self`: a call may grow scratch storage or refresh cached bindings.
A predicate mask does not own changing workspace, so its public methods can use
`&self`.

Kernel selection uses explicit enums rather than trait objects. That keeps the
selected path visible, avoids dynamic dispatch in command recording, and lets
each implementation own different workspace. Capable Intel Vulkan adapters
route to 4-bit key-value sort kernels, compatible NVIDIA Vulkan adapters can
route to wider or subgroup-specialized kernels, and other devices retain the
portable implementation.

## Performance contract

An internal refactor is accepted only when it preserves public behavior and
recorded GPU work. The minimum gate is:

1. formatting, Clippy, docs, package, and all release tests pass;
2. deterministic CPU-reference validation passes on affected physical GPUs;
3. identical pre/post resident benchmarks stay within a 2% regression budget;
4. any new optimized backend retains a portable fallback and direct routing
   tests.

Slice timing is useful for end-to-end applications, but primitive performance
claims use resident buffers. That boundary includes command recording,
submission, GPU execution, completion, and reusable-workspace management while
excluding upload, validation readback, and result download.

## Migration direction

The project will remain one crate while these boundaries mature. New work
should follow this order:

1. put common execution, profiling, capability, and workspace mechanics in the
   private engine layer;
2. keep existing public APIs source-compatible while adding physical GPU
   coverage;
3. add device-specialized kernels only behind measured capability gates;
4. add higher-level algorithms only when a real workload demonstrates the API
   and performance value;
5. consider separate low-level and high-level crates only when downstream use,
   compile time, or independent versioning provides concrete evidence for it.

This gives Rust users one obvious dependency now while preserving a path toward
a broader CUB/Thrust-like WebGPU ecosystem later.
