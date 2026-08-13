# Migrating to Lampshade 0.12

Lampshade 0.12 contains two intentional pre-1.0 source-compatibility changes.
Neither changes GPU work or buffer layout.

## Typed pipeline namespace

The deprecated `v2` compatibility alias has been removed. Change imports from:

```rust,ignore
use lampshade::v2::{GpuSlice, GpuSliceMut, Primitives};
```

to:

```rust,ignore
use lampshade::pipeline::{GpuSlice, GpuSliceMut, Primitives};
```

The types have lived in `lampshade::pipeline` since 0.8; only the alias was
removed.

## Error matching

`lampshade::Error` is now non-exhaustive. Match the variants relevant to the
application and retain a fallback arm:

```rust,ignore
match error {
    lampshade::Error::BufferTooSmall { .. } => recover(),
    other => return Err(other),
}
```

This lets future releases report more precise validation failures without
breaking every exhaustive downstream match.
