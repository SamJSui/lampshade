#![allow(deprecated)]

use lampshade::v2::{
    Extent, GpuCount, GpuElement, GpuSlice, GpuSliceMut, Primitives, Recorder, SortOptions,
    WorkspaceRequirements,
};

#[test]
fn deprecated_v2_namespace_reexports_the_typed_pipeline() {
    fn accept_types<'a>(
        _: Option<GpuCount<'a>>,
        _: Option<Extent<'a>>,
        _: Option<GpuSlice<'a, u32>>,
        _: Option<GpuSliceMut<'a, u32>>,
        _: Option<Primitives>,
        _: Option<Recorder<'a, 'a>>,
    ) {
    }

    fn accept_element<T: GpuElement>() {}

    accept_types(None, None, None, None, None, None);
    accept_element::<u32>();
    let _ = SortOptions::default();
    let _ = WorkspaceRequirements::new(0);
}
