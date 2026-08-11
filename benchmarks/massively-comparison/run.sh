#!/usr/bin/env sh
set -eu

usage() {
    cat <<'EOF'
Usage: benchmarks/massively-comparison/run.sh [options]

Options:
  --items CSV       Item counts (default: 1000000,10000000,100000000)
  --workloads CSV   reduce_sum,sort_bounded16,sort_full_width,exclusive_scan,compact_50
  --processes N     Independent processes, 1-20 (default: 3)
  --backend NAME    Lampshade backend (default: vulkan)
  --output PATH     Aggregate JSON path
  --quick           All workloads at 1M, one process
  --help            Show this help
EOF
}

require_value() {
    if [ "$#" -lt 2 ]; then
        echo "missing value for $1" >&2
        exit 2
    fi
}

items_csv=1000000,10000000,100000000
workloads_csv=reduce_sum,sort_bounded16,sort_full_width,exclusive_scan,compact_50
processes=3
backend=vulkan
output_path=
quick=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --items) require_value "$@"; items_csv=$2; shift 2 ;;
        --workloads) require_value "$@"; workloads_csv=$2; shift 2 ;;
        --processes) require_value "$@"; processes=$2; shift 2 ;;
        --backend) require_value "$@"; backend=$2; shift 2 ;;
        --output) require_value "$@"; output_path=$2; shift 2 ;;
        --quick) quick=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$processes" in
    ''|*[!0-9]*) echo "--processes must be an integer" >&2; exit 2 ;;
esac
if [ "$processes" -lt 1 ] || [ "$processes" -gt 20 ]; then
    echo "--processes must be between 1 and 20" >&2
    exit 2
fi
if [ "$quick" = true ]; then
    items_csv=1000000
    workloads_csv=reduce_sum,sort_bounded16,sort_full_width,exclusive_scan,compact_50
    processes=1
fi

command -v cargo >/dev/null 2>&1 || { echo "cargo is required" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }

benchmark_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(CDPATH= cd -- "$benchmark_root/../.." && pwd -P)
target_root=$repo_root/target/massively-comparison
if [ -z "$output_path" ]; then
    output_path=$benchmark_root/results/latest.json
elif [ "${output_path#/}" = "$output_path" ]; then
    output_path=$repo_root/$output_path
fi

primitives_manifest=$benchmark_root/lampshade-runner/Cargo.toml
massively_manifest=$benchmark_root/massively-runner/Cargo.toml
primitives_target=$target_root/lampshade
massively_target=$target_root/massively

echo "Building Lampshade runner..."
CARGO_TARGET_DIR=$primitives_target cargo build --release --locked --manifest-path "$primitives_manifest"
echo "Building Massively runner..."
CARGO_TARGET_DIR=$massively_target cargo build --release --locked --manifest-path "$massively_manifest"

executable_suffix=
case "${OS:-}" in Windows_NT) executable_suffix=.exe ;; esac
primitives_executable=$primitives_target/release/lampshade-massively-comparison-runner$executable_suffix
massively_executable=$massively_target/release/massively-comparison-runner$executable_suffix

repo_revision=$(git -c "safe.directory=$repo_root" -C "$repo_root" rev-parse HEAD)
if [ -n "$(git -c "safe.directory=$repo_root" -C "$repo_root" status --porcelain)" ]; then repo_dirty=true; else repo_dirty=false; fi
package_version=$(cargo metadata --no-deps --format-version 1 --manifest-path "$repo_root/Cargo.toml" | python3 -c 'import json, sys; print(json.load(sys.stdin)["packages"][0]["version"])')

temp_root=$(mktemp -d)
trap 'rm -rf "$temp_root"' EXIT HUP INT TERM
runs_path=$temp_root/runs.jsonl
failures_path=$temp_root/failures.tsv
: > "$runs_path"
: > "$failures_path"
failure_count=0

csv_words() { printf '%s\n' "$1" | tr ',' ' '; }

run_one() {
    executable=$1; implementation=$2; implementation_version=$3; implementation_revision=$4
    item_count=$5; workload=$6; warmups=$7; warmup_ms=$8; samples=$9; process_index=${10}
    echo "$implementation $workload items=$item_count process=$process_index" >&2
    runner_output=$temp_root/runner-output.log
    if WGPU_BACKEND=$backend \
        MASSIVELY_BENCH_ITEMS=$item_count \
        MASSIVELY_BENCH_WORKLOAD=$workload \
        MASSIVELY_BENCH_WARMUPS=$warmups \
        MASSIVELY_BENCH_WARMUP_MS=$warmup_ms \
        MASSIVELY_BENCH_SAMPLES=$samples \
        MASSIVELY_BENCH_PROCESS_INDEX=$process_index \
        MASSIVELY_BENCH_IMPLEMENTATION_NAME=$implementation \
        MASSIVELY_BENCH_IMPLEMENTATION_VERSION=$implementation_version \
        MASSIVELY_BENCH_IMPLEMENTATION_REVISION=$implementation_revision \
        "$executable" > "$runner_output" 2>&1
    then
        json_line=$(awk '/^[[:space:]]*\{/{line=$0} END{if(line != "") print line}' "$runner_output")
        if [ -n "$json_line" ]; then
            printf '%s\n' "$json_line" >> "$runs_path"
            rm -f "$runner_output"
            return
        fi
        printf '%s\n' 'Runner completed without emitting a JSON result.' >> "$runner_output"
    fi

    failure_count=$((failure_count + 1))
    failure_output=$temp_root/failure-$failure_count.log
    mv "$runner_output" "$failure_output"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$implementation" "$workload" "$item_count" "$process_index" "$failure_output" \
        >> "$failures_path"
    echo "$implementation failed for $workload/$item_count/process $process_index" >&2
    cat "$failure_output" >&2
}

for item_count in $(csv_words "$items_csv"); do
    case "$item_count" in ''|*[!0-9]*) echo "invalid item count: $item_count" >&2; exit 2 ;; esac
    if [ "$item_count" -lt 1 ] || [ "$item_count" -gt 4294967295 ]; then echo "item count out of range: $item_count" >&2; exit 2; fi
    if [ "$quick" = true ]; then
        warmups=1; warmup_ms=0; samples=3
    elif [ "$item_count" -ge 100000000 ]; then
        warmups=2; warmup_ms=2000; samples=7
    else
        warmups=4; warmup_ms=2000; samples=11
    fi
    for workload in $(csv_words "$workloads_csv"); do
        case "$workload" in reduce_sum|sort_bounded16|sort_full_width|exclusive_scan|compact_50) ;; *) echo "unsupported workload: $workload" >&2; exit 2 ;; esac
        process_index=1
        while [ "$process_index" -le "$processes" ]; do
            run_one "$primitives_executable" lampshade "lampshade-$package_version" "$repo_revision" "$item_count" "$workload" "$warmups" "$warmup_ms" "$samples" "$process_index"
            run_one "$massively_executable" massively 0.96.0 ef9de55190529be98203aca207edab9d560d312e "$item_count" "$workload" "$warmups" "$warmup_ms" "$samples" "$process_index"
            process_index=$((process_index + 1))
        done
    done
done

if [ ! -s "$runs_path" ]; then
    echo "all benchmark runs failed" >&2
    exit 1
fi

set -- \
    --runs "$runs_path" --failures "$failures_path" --output "$output_path" \
    --repository-revision "$repo_revision" --repository-dirty "$repo_dirty" \
    --package-version "$package_version" --backend "$backend" \
    --items "$items_csv" --workloads "$workloads_csv" --processes "$processes" \
    --host "$(hostname)" --architecture "$(uname -m)"
if [ "$quick" = true ]; then set -- "$@" --quick; fi
python3 "$benchmark_root/aggregate.py" "$@"
