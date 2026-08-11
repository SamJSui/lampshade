# Standalone particle consumer

This crate is intentionally separate from the library package. It creates its
own wgpu instance, adapter, device, queue, buffers, and submission loop, then
uses only public `wgpu-primitives` APIs to filter, stably compact, and
depth-sort `KeyValue` particle records. The selected length stays GPU-resident
until one final validation readback.

Run the typed recorder:

```sh
cargo run --release --manifest-path validation/particle-app/Cargo.toml -- \
  --mode typed --items 1000000 --warmups 3 --iterations 10
```

Run the equivalent explicit-plan baseline:

```sh
cargo run --release --manifest-path validation/particle-app/Cargo.toml -- \
  --mode raw --items 1000000 --warmups 3 --iterations 10
```

Set `WGPU_BACKEND` to select a backend supported by the host, such as
`vulkan`, `dx12`, or `metal`. Each run prints one JSON report containing device
initialization, primitive construction/reservation, CPU command recording, and
submission-through-completion timings.

For alternating multi-process comparison and a machine-readable artifact:

```sh
python validation/particle-app/run.py --backend vulkan --processes 3 \
  --items 1000000,10000000 --output target/particle-consumer.json
```

The memory fields count application-owned GPU buffers exactly. Wgpu does not
expose portable physical-allocation or peak-driver-memory telemetry, so the
report labels internal primitive workspace as unobserved instead of presenting
an estimate as a measurement.
