use lampshade::{
    ArgminByKey, Compactor, Context, CountedSortDispatch, Error, GpuCountPlan, GpuProfile,
    GpuTimestampSpan, Histogram, KeyValue, KeyValueCompactor, KeyValueField,
    KeyValueSoaRequirements, KeyValueSoaSorter, KeyValueSorter, MaskGenerator, Reducer,
    RunLengthEncoder, RunLengthOutputBuffers, Scanner, Sorter, U32Predicate, U32Reduction,
    pipeline::{
        Extent, GpuCount, GpuElement, GpuSlice, GpuSliceMut, Primitives, Recorder, RunLengthOutput,
        SortOptions, WorkspaceRequirements,
    },
};

fn name_public_types<'a>(
    _: Option<ArgminByKey>,
    _: Option<Compactor>,
    _: Option<Context>,
    _: Option<CountedSortDispatch>,
    _: Option<Error>,
    _: Option<GpuCountPlan>,
    _: Option<GpuProfile>,
    _: Option<GpuTimestampSpan>,
    _: Option<Histogram>,
    _: Option<KeyValueCompactor>,
    _: Option<KeyValueField>,
    _: Option<KeyValueSoaRequirements>,
    _: Option<KeyValueSoaSorter>,
    _: Option<KeyValueSorter>,
    _: Option<MaskGenerator>,
    _: Option<Reducer>,
    _: Option<RunLengthEncoder>,
    _: Option<RunLengthOutputBuffers<'a>>,
    _: Option<Scanner>,
    _: Option<Sorter>,
    _: Option<GpuCount<'a>>,
    _: Option<Extent<'a>>,
    _: Option<GpuSlice<'a, u32>>,
    _: Option<GpuSliceMut<'a, u32>>,
    _: Option<Primitives>,
    _: Option<Recorder<'a, 'a>>,
    _: Option<RunLengthOutput<'a>>,
) {
}

fn name_gpu_element<T: GpuElement>() {}

fn main() {
    name_public_types(
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None, None,
    );
    name_gpu_element::<u32>();
    name_gpu_element::<KeyValue>();
    let _ = KeyValue::new(7, 11);
    let _ = SortOptions::default().key_bits(16);
    let _ = WorkspaceRequirements::new(1024)
        .predicate()
        .compact_key_values()
        .counted_key_value_sort()
        .counted_reduce()
        .run_length_encode()
        .argmin_by_key();
    let _ = U32Reduction::Sum.identity();
    let _ = U32Predicate::Equal(7);
}
