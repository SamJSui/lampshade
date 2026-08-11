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
run the primitive, and return an owned result. `Histogram::histogram` likewise
returns an owned bin vector while ignoring values outside its requested range.
These are the smallest paths from ordinary Rust code to a correct GPU result.

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

The `resident_pipeline` example records predicate, compaction, sort, and
reduction before one submission. Compaction writes its selected count to a
four-byte storage buffer. One `GpuCountPlan` borrows that buffer during
construction, clones its ref-counted wgpu handle, and prepares metadata shared
by sort and reduction, so the data-dependent pipeline needs no hidden readback
or CPU wait.

The CPU still provides a `capacity`. This is the memory-safety and allocation
contract: input, output, and private workspace are sized for at most that many
items. A preparation kernel clamps the GPU count to capacity and writes WebGPU
indirect-dispatch arguments. Later kernels read those arguments only when the
submitted command list executes:

```text
CPU capacity ──> validate buffers / allocate workspace
GPU count ─────> clamp + build dispatch arguments ──> sort ──> reduce
```

This separates two facts that are often conflated: capacity is CPU-known
storage, while count is GPU-known work. The fixed-length APIs remain available
when the CPU already knows both.

`&mut encoder` means this call has exclusive CPU-side permission to mutate the
command list. It does not mutate the generator or clone GPU data. The buffer
borrows describe which handles must remain valid while commands are recorded;
actual reads and writes happen later on the GPU in submission order.

This layer is the stable composition boundary. Applications can keep data on
the GPU, combine primitives in one submission, and manage synchronization at
the wgpu level.

The additive `v2` module prototypes a smaller recording-first vocabulary over
that boundary. `GpuSlice<T>` combines a typed buffer range, capacity, and either
a CPU-fixed or GPU-resident extent. `GpuSliceMut<T>` marks shader-writable use,
and `Recorder` transfers the resulting extent from compaction to sort and
reduction while preparing shared count metadata at most once per command
stream. The views borrow caller-owned buffers; they do not own allocations or
submit work.

The application-shaped `particle_pipeline` example extends those views to the
existing `KeyValue { key, value }` record. One recorder generates a depth mask,
stably compacts selected records, and performs a stable GPU-counted sort without
exposing the count to the CPU. This validates an array-of-structs key/payload
path while leaving a separate-buffer layout out of the public API until a
workload requires it.

Ranges support aligned nonzero offsets, but one primitive still requires
separate underlying handles for read and write roles. wgpu tracks writable
storage use at buffer-handle granularity within a dispatch, so disjoint static
bindings into one arena do not make simultaneous read/write use legal.
`Primitives::reserve_workspace` takes explicit operation and extent-mode
requirements, preparing lazy pipelines while avoiding unused fixed/counting
scratch. `reserve_count` creates count-specific metadata before recording.
Bind groups and small uniform buffers are still created during recording, so
this is not yet an allocation-free command-plan API.

### 3. Private runtime, workspace, and kernels

Private code owns the mechanisms that public methods share:

- `CommandSession` owns an encoder until one immediate submission.
- `ProfileSession` adds optional timestamp queries and waits for completion.
- `ReusableBuffer` owns a lazily allocated buffer and its capacity.
- `AdapterCapabilities` records hardware facts separately from kernel policy.
- primitive-specific pipelines select and dispatch WGSL implementations. The
  portable histogram privatizes 256 counters per workgroup before merging into
  global atomics.
- `GpuCountPlan` owns a cached preparation binding and bounded reduction/sort
  metadata for one count buffer and capacity. Counted sort defaults to indirect
  dispatch, while an explicit capacity strategy is available when benchmarks
  justify trading inactive workgroups for lower launch overhead.

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

New counted paths must also keep the fixed-length gate green. Their extra
preparation dispatch buys data-dependent composition; it is not inserted into
the established fixed-length command stream.

## Migration direction

The project will remain one crate while these boundaries mature. New work
should follow this order:

1. put common execution, profiling, capability, and workspace mechanics in the
   private engine layer;
2. keep existing public APIs source-compatible while adding physical GPU
   coverage;
3. validate the experimental typed recorder against real applications before
   promoting or replacing the explicit raw-buffer methods;
4. add device-specialized kernels only behind measured capability gates;
5. add higher-level algorithms only when a real workload demonstrates the API
   and performance value;
6. consider separate low-level and high-level crates only when downstream use,
   compile time, or independent versioning provides concrete evidence for it.

This gives Rust users one obvious dependency now while preserving a path toward
a broader CUB/Thrust-like WebGPU ecosystem later.
