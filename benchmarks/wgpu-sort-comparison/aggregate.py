#!/usr/bin/env python3
"""Aggregate process-isolated wgpu sort comparison runner results."""

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
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        config = run["config"]
        key = (
            run["implementation"],
            config["mode"],
            config["workload"],
            int(config["items"]),
        )
        groups[key].append(run)

    aggregates = []
    for (implementation, mode, workload, items), group in groups.items():
        process_medians = [float(run["median_ms"]) for run in group]
        aggregate_median = median(process_medians)
        aggregates.append(
            {
                "implementation": implementation,
                "mode": mode,
                "workload": workload,
                "items": items,
                "process_medians_ms": process_medians,
                "median_of_process_medians_ms": aggregate_median,
                "throughput_pairs_per_second": items / (aggregate_median / 1_000.0),
                "memory": group[0]["memory"],
            }
        )

    return sorted(
        aggregates,
        key=lambda row: (row["items"], row["workload"], row["mode"], row["implementation"]),
    )


def csv_values(value: str) -> list[str]:
    return [part for part in value.split(",") if part]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-revision", required=True)
    parser.add_argument("--repository-dirty", choices=("true", "false"), required=True)
    parser.add_argument("--package-version", required=True)
    parser.add_argument("--comparison-revision", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--workloads", required=True)
    parser.add_argument("--modes", required=True)
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

    result = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": {
            "revision": args.repository_revision,
            "dirty": args.repository_dirty == "true",
            "package_version": args.package_version,
        },
        "comparison": {
            "name": "wgpu_sort",
            "revision": args.comparison_revision,
        },
        "host": {
            "hostname": args.host,
            "architecture": args.architecture,
        },
        "config": {
            "backend": args.backend,
            "items": [int(value) for value in csv_values(args.items)],
            "workloads": csv_values(args.workloads),
            "modes": csv_values(args.modes),
            "processes": args.processes,
            "quick": args.quick,
        },
        "runs": runs,
        "aggregates": aggregate_runs(runs),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("\nAggregate medians:")
    print(f"{'implementation':<18} {'mode':<11} {'workload':<11} {'items':>10} {'median_ms':>12}")
    for row in result["aggregates"]:
        print(
            f"{row['implementation']:<18} {row['mode']:<11} {row['workload']:<11} "
            f"{row['items']:>10} {row['median_of_process_medians_ms']:>12.3f}"
        )
    print(f"Machine-readable results: {args.output}")


if __name__ == "__main__":
    main()
