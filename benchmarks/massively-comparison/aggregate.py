#!/usr/bin/env python3
"""Aggregate process-isolated wgpu-primitives and Massively runner results."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute the median of an empty collection")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2.0
    return ordered[middle]


def aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        config = run["config"]
        key = (run["implementation"], config["workload"], int(config["items"]))
        groups[key].append(run)

    aggregates = []
    for (implementation, workload, items), group in groups.items():
        process_medians = [float(run["median_ms"]) for run in group]
        aggregate_median = median(process_medians)
        aggregates.append(
            {
                "implementation": implementation,
                "workload": workload,
                "items": items,
                "process_medians_ms": process_medians,
                "median_of_process_medians_ms": aggregate_median,
                "throughput_items_per_second": items / (aggregate_median / 1_000.0),
                "memory": group[0]["memory"],
            }
        )
    return sorted(aggregates, key=lambda row: (row["items"], row["workload"], row["implementation"]))


def comparisons(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in aggregates:
        by_case[(row["workload"], row["items"])][row["implementation"]] = row

    rows = []
    for (workload, items), implementations in sorted(by_case.items(), key=lambda pair: (pair[0][1], pair[0][0])):
        primitives = implementations.get("wgpu-primitives")
        massively = implementations.get("massively")
        if primitives is None or massively is None:
            continue
        primitives_ms = float(primitives["median_of_process_medians_ms"])
        massively_ms = float(massively["median_of_process_medians_ms"])
        rows.append(
            {
                "workload": workload,
                "items": items,
                "wgpu_primitives_ms": primitives_ms,
                "massively_ms": massively_ms,
                "wgpu_primitives_speedup": massively_ms / primitives_ms,
            }
        )
    return rows


def load_failures(index_path: Path | None) -> list[dict[str, Any]]:
    if index_path is None or not index_path.exists():
        return []

    failures = []
    with index_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(
                    f"invalid failure index line {line_number}: expected 5 tab-separated fields"
                )
            implementation, workload, items, process_index, error_path = fields
            error = Path(error_path).read_text(encoding="utf-8", errors="replace").strip()
            failures.append(
                {
                    "implementation": implementation,
                    "workload": workload,
                    "items": int(items),
                    "process_index": int(process_index),
                    "error": error,
                }
            )
    return failures


def csv_values(value: str) -> list[str]:
    return [part for part in value.split(",") if part]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-revision", required=True)
    parser.add_argument("--repository-dirty", choices=("true", "false"), required=True)
    parser.add_argument("--package-version", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.runs.open(encoding="utf-8") as source:
        runs = [json.loads(line) for line in source if line.strip()]
    if not runs:
        raise ValueError("runner output did not contain any results")
    failures = load_failures(args.failures)
    aggregates = aggregate_runs(runs)
    result = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": {
            "revision": args.repository_revision,
            "dirty": args.repository_dirty == "true",
            "package_version": args.package_version,
        },
        "comparison": {
            "name": "massively",
            "version": "0.96.0",
            "revision": "ef9de55190529be98203aca207edab9d560d312e",
        },
        "host": {"hostname": args.host, "architecture": args.architecture},
        "config": {
            "backend": args.backend,
            "items": [int(value) for value in csv_values(args.items)],
            "workloads": csv_values(args.workloads),
            "processes": args.processes,
            "quick": args.quick,
        },
        "methodology": {
            "timing": "resident public API call through GPU completion",
            "excluded": ["host upload", "readback", "correctness validation"],
            "allocation_difference": "wgpu-primitives reuses caller-owned outputs; Massively public APIs allocate owned outputs and may reuse CubeCL allocator storage",
        },
        "runs": runs,
        "failures": failures,
        "aggregates": aggregates,
        "comparisons": comparisons(aggregates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("\nComparison medians:")
    print(f"{'workload':<20} {'items':>10} {'wgpu_ms':>11} {'massively_ms':>14} {'speedup':>10}")
    for row in result["comparisons"]:
        print(
            f"{row['workload']:<20} {row['items']:>10} {row['wgpu_primitives_ms']:>11.3f} "
            f"{row['massively_ms']:>14.3f} {row['wgpu_primitives_speedup']:>9.2f}x"
        )
    if failures:
        print(f"\n{len(failures)} run(s) failed; details are recorded in the result JSON.")
    print(f"Machine-readable results: {args.output}")


if __name__ == "__main__":
    main()
