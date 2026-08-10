// The release gate must exercise identical runner logic on both dependency
// sources. Keep the shared source restricted to APIs available in the pinned
// published baseline.
include!("../../../massively-comparison/wgpu-primitives-runner/src/main.rs");
