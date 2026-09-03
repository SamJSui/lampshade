# Migrating to Lampshade 0.13

Lampshade 0.13 moves its public GPU runtime from wgpu 29 to wgpu 30. Lampshade
method names, buffer layouts, shaders, and execution boundaries are unchanged,
but the dependency update is a source-compatibility break because Lampshade
accepts and returns public wgpu types.

## Update both dependencies

After Lampshade 0.13 is published, change the application manifest from:

```toml
[dependencies]
lampshade = "0.12"
wgpu = "29"
```

to:

```toml
[dependencies]
lampshade = "0.13"
wgpu = "30"
```

Keeping both crates on the same wgpu major ensures that application-owned
`Device`, `Queue`, and `Buffer` values can be passed directly to Lampshade.
Applications that must remain on wgpu 29 can continue using Lampshade 0.12.

## Fallible mapped ranges

wgpu 30 makes `BufferSlice::get_mapped_range` fallible. Applications that map
their own readback buffers must now handle its result:

```rust,ignore
let mapped = slice.get_mapped_range()?;
```

Lampshade's host-returning methods propagate this failure as
`lampshade::Error::MapRange`.

## Adapter requests

Applications that construct `wgpu::RequestAdapterOptions` explicitly must set
the wgpu 30 `apply_limit_buckets` field or use a struct update. Lampshade's
`Context::init` disables limit bucketing so it continues requesting the
adapter's reported limits.
