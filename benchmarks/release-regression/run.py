#!/usr/bin/env python3
"""Compare the current checkout with the pinned published release."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

BASELINE_VERSION = "0.11.0"
DEFAULT_ITEMS = (1_000_000, 10_000_000, 100_000_000)
DEFAULT_WORKLOADS = (
    "reduce_sum",
    "sort_bounded16",
    "sort_full_width",
    "sort_counted_full_width",
    "exclusive_scan",
    "compact_50",
)
VALID_WORKLOADS = frozenset(DEFAULT_WORKLOADS)
REGRESSION_EPSILON_PERCENT = 1e-9
SOURCE_PATHS = (
    "Cargo.toml",
    "src",
    "benchmarks/massively-comparison/common",
    "benchmarks/massively-comparison/lampshade-runner",
    "benchmarks/release-regression",
)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    return float(statistics.median(values))


def adapter_identity(adapter: dict[str, Any]) -> dict[str, Any]:
    return {
        key: adapter[key]
        for key in ("name", "vendor", "device", "device_type", "backend")
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for run in runs:
        key = (run["source"], run["result"]["config"]["workload"], run["result"]["config"]["items"])
        grouped.setdefault(key, []).append(run["result"])

    aggregates = []
    for (source, workload, items), group in sorted(grouped.items(), key=lambda pair: pair[0]):
        process_medians = [float(run["median_ms"]) for run in group]
        adapters = [adapter_identity(run["adapter"]) for run in group]
        aggregates.append(
            {
                "source": source,
                "workload": workload,
                "items": items,
                "process_medians_ms": process_medians,
                "median_of_process_medians_ms": median(process_medians),
                "adapter": adapters[0],
                "adapter_consistent": all(adapter == adapters[0] for adapter in adapters),
            }
        )
    return aggregates


def compare_aggregates(
    aggregates: list[dict[str, Any]], threshold_percent: float
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in aggregates:
        grouped.setdefault((row["workload"], row["items"]), {})[row["source"]] = row

    comparisons = []
    for (workload, items), sources in sorted(grouped.items(), key=lambda pair: (pair[0][1], pair[0][0])):
        if "published" not in sources or "checkout" not in sources:
            continue
        baseline_ms = float(sources["published"]["median_of_process_medians_ms"])
        candidate_ms = float(sources["checkout"]["median_of_process_medians_ms"])
        change_percent = (candidate_ms / baseline_ms - 1.0) * 100.0
        adapter_match = (
            sources["published"]["adapter_consistent"]
            and sources["checkout"]["adapter_consistent"]
            and sources["published"]["adapter"] == sources["checkout"]["adapter"]
        )
        comparisons.append(
            {
                "workload": workload,
                "items": items,
                "published_ms": baseline_ms,
                "checkout_ms": candidate_ms,
                "change_percent": change_percent,
                "adapter_match": adapter_match,
                "passed": adapter_match
                and change_percent <= threshold_percent + REGRESSION_EPSILON_PERCENT,
            }
        )
    return comparisons


def passes_gate(
    failures: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    expected_comparisons: int,
    quick: bool,
) -> bool:
    adapters_passed = all(row["adapter_match"] for row in comparisons)
    regression_passed = quick or all(row["passed"] for row in comparisons)
    return (
        not failures
        and len(comparisons) == expected_comparisons
        and adapters_passed
        and regression_passed
    )


def command_output(command: list[str], cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def source_manifest(repo_root: Path) -> tuple[list[dict[str, str]], str]:
    listed = command_output(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            *SOURCE_PATHS,
        ],
        repo_root,
    )
    files = []
    manifest_lines = []
    for relative in sorted(line for line in listed.splitlines() if line):
        normalized = (repo_root / relative).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest = hashlib.sha256(normalized).hexdigest()
        files.append({"path": relative, "sha256": digest})
        manifest_lines.append(f"{digest}  {relative}\n")
    manifest = hashlib.sha256("".join(manifest_lines).encode()).hexdigest()
    return files, manifest


def build_runner(manifest: Path, target: Path) -> None:
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target)
    subprocess.run(
        ["cargo", "build", "--release", "--locked", "--manifest-path", str(manifest)],
        check=True,
        env=env,
    )


def resolved_wgpu_stack(manifest: Path) -> str:
    lock_text = manifest.with_name("Cargo.lock").read_text(encoding="utf-8")
    names = ("wgpu", "wgpu-core", "wgpu-hal", "wgpu-types")
    versions = {}
    for name in names:
        matched = re.search(
            rf'\[\[package\]\]\s+name = "{re.escape(name)}"\s+version = "([^"]+)"',
            lock_text,
        )
        if matched is None:
            raise ValueError(f"missing {name} in {manifest.with_name('Cargo.lock')}")
        versions[name] = matched.group(1)
    return "; ".join(f"{name} {versions[name]}" for name in names)


def package_version(manifest: Path) -> str:
    manifest_text = manifest.read_text(encoding="utf-8")
    package = re.search(r"(?ms)^\[package\]\s+(.*?)(?=^\[|\Z)", manifest_text)
    if package is None:
        raise ValueError(f"missing [package] in {manifest}")
    version = re.search(r'^version\s*=\s*"([^"]+)"', package.group(1), re.MULTILINE)
    if version is None:
        raise ValueError(f"missing package version in {manifest}")
    return version.group(1)


def sampling(items: int, quick: bool) -> tuple[int, int, int]:
    if quick:
        return 1, 0, 3
    if items >= 100_000_000:
        return 2, 2_000, 7
    return 4, 2_000, 11


def run_one(
    executable: Path,
    source: str,
    implementation: str,
    version: str,
    revision: str,
    runtime_stack: str,
    backend: str,
    items: int,
    workload: str,
    process_index: int,
    quick: bool,
) -> dict[str, Any]:
    warmups, warmup_ms, samples = sampling(items, quick)
    env = os.environ.copy()
    env.update(
        {
            "WGPU_BACKEND": backend,
            "MASSIVELY_BENCH_ITEMS": str(items),
            "MASSIVELY_BENCH_WORKLOAD": workload,
            "MASSIVELY_BENCH_WARMUPS": str(warmups),
            "MASSIVELY_BENCH_WARMUP_MS": str(warmup_ms),
            "MASSIVELY_BENCH_SAMPLES": str(samples),
            "MASSIVELY_BENCH_PROCESS_INDEX": str(process_index),
            "MASSIVELY_BENCH_IMPLEMENTATION_NAME": implementation,
            "MASSIVELY_BENCH_IMPLEMENTATION_VERSION": version,
            "MASSIVELY_BENCH_IMPLEMENTATION_REVISION": revision,
            "MASSIVELY_BENCH_RUNTIME_STACK": runtime_stack,
        }
    )
    completed = subprocess.run(executable, env=env, text=True, capture_output=True)
    if completed.returncode != 0:
        details = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
        raise RuntimeError(details or f"runner exited with {completed.returncode}")
    json_lines = [line for line in completed.stdout.splitlines() if line.lstrip().startswith("{")]
    if not json_lines:
        raise RuntimeError("runner completed without emitting a JSON result")
    return {"source": source, "result": json.loads(json_lines[-1])}


def csv_ints(value: str) -> list[int]:
    values = [int(part.replace("_", "")) for part in value.split(",") if part]
    if not values or any(item < 1 or item > 2**32 - 1 for item in values):
        raise argparse.ArgumentTypeError("items must be integers between 1 and u32::MAX")
    return values


def csv_workloads(value: str) -> list[str]:
    values = [part for part in value.split(",") if part]
    invalid = [part for part in values if part not in VALID_WORKLOADS]
    if not values or invalid:
        raise argparse.ArgumentTypeError(f"unsupported workloads: {invalid}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=csv_ints, default=list(DEFAULT_ITEMS))
    parser.add_argument("--workloads", type=csv_workloads, default=list(DEFAULT_WORKLOADS))
    parser.add_argument("--processes", type=int, choices=range(1, 21), default=3)
    parser.add_argument("--backend", default="metal" if sys.platform == "darwin" else "vulkan")
    parser.add_argument("--threshold-percent", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--characterize",
        action="store_true",
        help="record formal cross-runtime timings without enforcing the timing threshold",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threshold_percent < 0:
        raise ValueError("threshold-percent must be nonnegative")
    if args.quick:
        args.items = [1_000_000]
        args.processes = 1

    benchmark_root = Path(__file__).resolve().parent
    repo_root = benchmark_root.parents[1]
    target_root = repo_root / "target" / "release-regression"
    candidate_manifest = repo_root / "benchmarks" / "massively-comparison" / "lampshade-runner" / "Cargo.toml"
    baseline_manifest = benchmark_root / "published-runner" / "Cargo.toml"
    candidate_target = target_root / "checkout"
    baseline_target = target_root / "published"

    print("Building checkout runner...", file=sys.stderr)
    build_runner(candidate_manifest, candidate_target)
    print(f"Building crates.io {BASELINE_VERSION} runner...", file=sys.stderr)
    build_runner(baseline_manifest, baseline_target)
    published_runtime_stack = resolved_wgpu_stack(baseline_manifest)
    checkout_runtime_stack = resolved_wgpu_stack(candidate_manifest)
    suffix = ".exe" if os.name == "nt" else ""
    candidate_executable = candidate_target / "release" / f"lampshade-massively-comparison-runner{suffix}"
    baseline_executable = baseline_target / "release" / f"lampshade-release-baseline-runner{suffix}"
    revision = command_output(
        ["git", "-c", f"safe.directory={repo_root}", "rev-parse", "HEAD"], repo_root
    )
    dirty = bool(
        command_output(
            ["git", "-c", f"safe.directory={repo_root}", "status", "--porcelain"],
            repo_root,
        )
    )
    candidate_version = package_version(repo_root / "Cargo.toml")
    source_files, source_manifest_sha256 = source_manifest(repo_root)

    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for items in args.items:
        for workload in args.workloads:
            for process_index in range(1, args.processes + 1):
                sources = [
                    ("published", "lampshade", baseline_executable, f"crates.io-{BASELINE_VERSION}", f"v{BASELINE_VERSION}", published_runtime_stack),
                    ("checkout", "lampshade", candidate_executable, f"working-tree-{candidate_version}", revision, checkout_runtime_stack),
                ]
                if process_index % 2 == 0:
                    sources.reverse()
                for source, implementation, executable, version, source_revision, runtime_stack in sources:
                    print(f"{source} {workload} items={items} process={process_index}", file=sys.stderr)
                    try:
                        runs.append(
                            run_one(
                                executable,
                                source,
                                implementation,
                                version,
                                source_revision,
                                runtime_stack,
                                args.backend,
                                items,
                                workload,
                                process_index,
                                args.quick,
                            )
                        )
                    except Exception as error:  # preserve every case in the result artifact
                        failures.append(
                            {
                                "source": source,
                                "workload": workload,
                                "items": items,
                                "process_index": process_index,
                                "error": str(error),
                            }
                        )

    aggregates = aggregate_runs(runs)
    comparisons = compare_aggregates(aggregates, args.threshold_percent)
    expected_comparisons = len(args.items) * len(args.workloads)
    timing_gate_disabled = args.quick or args.characterize
    gate_passed = passes_gate(
        failures, comparisons, expected_comparisons, timing_gate_disabled
    )
    result = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"hostname": platform.node(), "architecture": platform.machine()},
        "baseline": {"source": "crates.io", "version": BASELINE_VERSION},
        "candidate": {
            "version": candidate_version,
            "revision": revision,
            "dirty": dirty,
            "source_manifest_algorithm": "SHA-256 of ordered '<LF-normalized file SHA-256>  <path>\\n' entries",
            "source_manifest_sha256": source_manifest_sha256,
            "source_files": source_files,
        },
        "config": {
            "backend": args.backend,
            "items": args.items,
            "workloads": args.workloads,
            "processes": args.processes,
            "threshold_percent": args.threshold_percent,
            "quick": args.quick,
            "characterize": args.characterize,
        },
        "methodology": {
            "timing": "identical public resident API and completion boundary per source",
            "aggregation": "median of independent process medians",
            "gate": (
                "not evaluated: cross-runtime characterization"
                if args.characterize
                else "candidate increase must not exceed threshold_percent"
            ),
        },
        "runs": runs,
        "failures": failures,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "gate_evaluated": not timing_gate_disabled,
        "gate_passed": gate_passed,
    }
    output = args.output or benchmark_root / "results" / "latest.json"
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"{'workload':<20} {'items':>10} {'published':>11} {'checkout':>11} {'change':>9} {'gate':>6}")
    for row in comparisons:
        gate = "n/a" if timing_gate_disabled else ("pass" if row["passed"] else "FAIL")
        print(
            f"{row['workload']:<20} {row['items']:>10} {row['published_ms']:>11.3f} "
            f"{row['checkout_ms']:>11.3f} {row['change_percent']:>8.2f}% "
            f"{gate:>6}"
        )
    print(f"Machine-readable results: {output}")
    if failures:
        print(f"{len(failures)} runner invocation(s) failed", file=sys.stderr)
    if not gate_passed:
        print("release regression gate failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
