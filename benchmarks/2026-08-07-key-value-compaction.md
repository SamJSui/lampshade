# Stable Key-Value Stream Compaction

This report validates stable stream compaction for the crate's existing
eight-byte `KeyValue { key: u32, value: u32 }` record. The implementation
reuses the `u32` mask scan and specializes only the scatter shader, so the
original four-byte path does not pay a dynamic record-width cost.

## Source and method

- Source base: `20271441a4bd8359c0e7da5a9996828a1d1030f6`
- Measured source package SHA-256:
  `9AB587C389ED92BC05CAA89A08B658D9CBEABE6ACE61EC45B1A72B46159214CB`
- Workload: 10 million resident records with periodic 0%, 10%, 50%, 90%, and
  100% selectivity masks
- Backend: Vulkan
- Timing: 11 samples after a two-second warmup; tables report medians
- Validation: every profiled workload was compared with a stable CPU reference

Resident wall time includes command encoding, submission, GPU execution, and
the GPU wait. GPU elapsed spans the recursive exclusive scan and stable
scatter. Allocation, upload, CPU reference generation, and readback are
outside the timed region. Documentation was added after measurement; the
measured library, shader, tests, example, benchmark, and profiler were not
changed.

## Systems

| Host | GPU | Driver | Active GPU TPCs | Controlled clocks |
| --- | --- | --- | ---: | --- |
| RTX desktop | NVIDIA GeForce RTX 4070 Ti SUPER | 591.86 | - | normal desktop state |
| `dopey` | NVIDIA Tegra Orin | 595.78 | 8 | CPU 1.728 GHz, GPU 1.020 GHz, EMC 3.199 GHz |
| `grumpy` | NVIDIA Tegra Orin | 595.78 | 4 | CPU 1.728 GHz, GPU 1.020 GHz, EMC 3.199 GHz |

Both Jetsons used `MAXN_SUPER`. After testing, their dynamic CPU, GPU, and EMC
ranges and WFI/c7 idle states were restored and verified.

## Resident results

Times are milliseconds. `Scan` and `scatter` are GPU dispatch time.

| Host | Kept | Wall | GPU elapsed | Dispatch | Scan | Scatter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RTX 4070 Ti SUPER | 0% | 0.802 | 0.617 | 0.588 | 0.513 | 0.075 |
| RTX 4070 Ti SUPER | 10% | 0.849 | 0.670 | 0.645 | 0.516 | 0.129 |
| RTX 4070 Ti SUPER | 50% | 0.973 | 0.782 | 0.754 | 0.516 | 0.239 |
| RTX 4070 Ti SUPER | 90% | 1.098 | 0.899 | 0.871 | 0.516 | 0.356 |
| RTX 4070 Ti SUPER | 100% | 1.172 | 0.918 | 0.889 | 0.515 | 0.374 |
| `dopey` (8 TPCs) | 0% | 6.951 | 5.051 | 5.014 | 4.085 | 0.929 |
| `dopey` (8 TPCs) | 10% | 7.464 | 5.584 | 5.547 | 4.083 | 1.464 |
| `dopey` (8 TPCs) | 50% | 7.873 | 5.978 | 5.941 | 4.082 | 1.858 |
| `dopey` (8 TPCs) | 90% | 8.424 | 6.520 | 6.483 | 4.086 | 2.397 |
| `dopey` (8 TPCs) | 100% | 8.768 | 6.655 | 6.618 | 4.083 | 2.535 |
| `grumpy` (4 TPCs) | 0% | 9.829 | 7.962 | 7.925 | 6.998 | 0.927 |
| `grumpy` (4 TPCs) | 10% | 10.394 | 8.510 | 8.473 | 6.999 | 1.472 |
| `grumpy` (4 TPCs) | 50% | 10.903 | 8.931 | 8.894 | 6.998 | 1.896 |
| `grumpy` (4 TPCs) | 90% | 11.508 | 9.469 | 9.432 | 7.000 | 2.432 |
| `grumpy` (4 TPCs) | 100% | 11.662 | 9.596 | 9.559 | 7.001 | 2.558 |

At 50% kept, widening the record from four to eight bytes increased resident
wall time by 8.6% on the RTX, 3.9% on `dopey`, and 4.6% on `grumpy`. The
additional cost is isolated to selected-record movement; the scan still
processes the same four-byte mask and dominates both integrated-GPU results.

The same-source `u32` controls measured 0.896 ms, 7.576 ms, and 10.423 ms at
50%. Compared with the published baseline, those changed by +0.45%, -0.33%,
and -0.05%, respectively, confirming that specialization preserved the
existing path within the 2% guardrail.

## Correctness and evidence

- All five 10-million-record selectivities matched stable CPU filtering on all
  three hosts.
- The full suite passed on both Jetsons: 9 unit tests and 43 GPU integration
  tests, including 6 key-value-compaction tests.
- Tests cover boundaries, whole-record movement, stable order, explicit logical
  length, resident counts, multiple recorded invocations, timestamp spans,
  invalid masks, eight-byte capacities, and forbidden aliases.
- Raw-output SHA-256 values are recorded in the machine-readable aggregate:
  RTX `FC58DAC2...D5046BD`, `dopey` `15CCD8AA...976F747`, and `grumpy`
  `E17D5278...60263F`.

See [`2026-08-07-key-value-compaction.json`](2026-08-07-key-value-compaction.json)
for the complete aggregate.

## Decision

Typed `KeyValue` compaction is ready to publish. The next derived primitive
should generate reusable masks from GPU predicates, then feed those masks into
the same stable scan/scatter path.
