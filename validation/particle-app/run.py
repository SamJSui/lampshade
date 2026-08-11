#!/usr/bin/env python3
"""Run alternating standalone particle-consumer processes and aggregate JSON."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "validation" / "particle-app" / "Cargo.toml"
BINARY_NAME = "wgpu-primitives-particle-app.exe" if os.name == "nt" else "wgpu-primitives-particle-app"
BINARY = MANIFEST.parent / "target" / "release" / BINARY_NAME
METRICS = (
    "command_recording",
    "submission_through_completion",
    "record_submit_completion",
)


def parse_csv_items(value: str) -> list[int]:
    items = [int(part.strip().replace("_", "")) for part in value.split(",")]
    if not items or any(item <= 0 or item > 0xFFFF_FFFF for item in items):
        raise argparse.ArgumentTypeError("items must be positive u32 values")
    return items


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_state() -> dict[str, Any]:
    revision = git_output("rev-parse", "HEAD")
    branch = git_output("branch", "--show-current")
    tracked_dirty = subprocess.run(
        ["git", "diff", "--quiet"], cwd=ROOT, check=False
    ).returncode != 0
    staged_dirty = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=ROOT, check=False
    ).returncode != 0
    untracked = git_output("ls-files", "--others", "--exclude-standard").splitlines()
    return {
        "revision": revision,
        "branch": branch,
        "dirty": tracked_dirty or staged_dirty or bool(untracked),
        "traceability": "the PR commit containing this artifact is the reproducible candidate",
    }


def build() -> None:
    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError("cargo is not available")
    subprocess.run(
        [cargo, "build", "--release", "--locked", "--manifest-path", str(MANIFEST)],
        cwd=ROOT,
        check=True,
    )


def run_once(
    mode: str,
    items: int,
    warmups: int,
    iterations: int,
    backend: str | None,
) -> dict[str, Any]:
    environment = os.environ.copy()
    if backend:
        environment["WGPU_BACKEND"] = backend
    command = [
        str(BINARY),
        "--mode",
        mode,
        "--items",
        str(items),
        "--warmups",
        str(warmups),
        "--iterations",
        str(iterations),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    report["command"] = command
    return report


def adapter_key(report: dict[str, Any]) -> str:
    adapter = report["adapter"]
    return "|".join(
        str(adapter[field])
        for field in ("name", "vendor", "device", "device_type", "backend")
    )


def aggregate(runs: list[dict[str, Any]], items: int, mode: str) -> dict[str, Any]:
    matching = [
        run for run in runs if run["config"]["items"] == items and run["mode"] == mode
    ]
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        process_medians = [run["timings_ms"][metric]["median"] for run in matching]
        metrics[metric] = {
            "process_medians_ms": process_medians,
            "median_of_process_medians_ms": statistics.median(process_medians),
        }
    keys = {adapter_key(run) for run in matching}
    return {
        "mode": mode,
        "items": items,
        "metrics": metrics,
        "adapter_key": next(iter(keys)) if len(keys) == 1 else None,
        "adapter_consistent": len(keys) == 1,
    }


def percent_change(baseline: float, candidate: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=parse_csv_items, default=[1_000_000, 10_000_000])
    parser.add_argument("--processes", type=int, choices=range(1, 21), default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--backend")
    parser.add_argument("--threshold-percent", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if args.warmups < 0 or args.iterations <= 0:
        parser.error("warmups must be non-negative and iterations must be positive")
    if args.threshold_percent < 0:
        parser.error("threshold-percent must be non-negative")

    if not args.skip_build:
        build()
    if not BINARY.exists():
        raise RuntimeError(f"consumer binary does not exist: {BINARY}")

    runs: list[dict[str, Any]] = []
    for process_index in range(args.processes):
        mode_order = ("raw", "typed") if process_index % 2 == 0 else ("typed", "raw")
        for items in args.items:
            for mode in mode_order:
                report = run_once(mode, items, args.warmups, args.iterations, args.backend)
                report["process_index"] = process_index + 1
                runs.append(report)

    aggregates = [
        aggregate(runs, items, mode)
        for items in args.items
        for mode in ("raw", "typed")
    ]
    comparisons: list[dict[str, Any]] = []
    for items in args.items:
        raw = next(row for row in aggregates if row["items"] == items and row["mode"] == "raw")
        typed = next(
            row for row in aggregates if row["items"] == items and row["mode"] == "typed"
        )
        adapter_match = (
            raw["adapter_consistent"]
            and typed["adapter_consistent"]
            and raw["adapter_key"] == typed["adapter_key"]
        )
        metric_changes = {}
        for metric in METRICS:
            raw_ms = raw["metrics"][metric]["median_of_process_medians_ms"]
            typed_ms = typed["metrics"][metric]["median_of_process_medians_ms"]
            metric_changes[metric] = {
                "raw_ms": raw_ms,
                "typed_ms": typed_ms,
                "change_percent": percent_change(raw_ms, typed_ms),
            }
        total_change = metric_changes["record_submit_completion"]["change_percent"]
        comparisons.append(
            {
                "items": items,
                "adapter_match": adapter_match,
                "metrics": metric_changes,
                "passed": adapter_match
                and (args.quick or total_change <= args.threshold_percent + 1e-9),
            }
        )

    artifact = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "architecture": platform.machine(),
        },
        "source": source_state(),
        "config": {
            "items": args.items,
            "processes": args.processes,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "backend": args.backend,
            "threshold_percent": args.threshold_percent,
            "quick": args.quick,
        },
        "methodology": {
            "ordering": "raw/typed alternates by process",
            "aggregation": "median of independent process medians",
            "timing": "CPU recording and submission-through-device-completion are measured separately",
            "validation": "every process validates selected count, predicate, ordering, and stable payloads",
        },
        "runs": runs,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "gate_evaluated": not args.quick,
        "gate_passed": all(row["passed"] for row in comparisons),
    }
    rendered = json.dumps(artifact, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if artifact["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
