#!/usr/bin/env sh
set -eu

usage() {
    cat <<'EOF'
Usage: benchmarks/wgpu-sort-comparison/run.sh [options]

Options:
  --items CSV       Item counts (default: 1000000,10000000,100000000)
  --workloads CSV   bounded16,full_width (default: both)
  --modes CSV       resident,round_trip (default: both)
  --processes N     Independent processes, 1-20 (default: 3)
  --backend NAME    wgpu backend (default: vulkan)
  --output PATH     Aggregate JSON path
  --quick           1M resident smoke test, one process
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
workloads_csv=bounded16,full_width
modes_csv=resident,round_trip
processes=3
backend=vulkan
output_path=
quick=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --items)
            require_value "$@"
            items_csv=$2
            shift 2
            ;;
        --workloads)
            require_value "$@"
            workloads_csv=$2
            shift 2
            ;;
        --modes)
            require_value "$@"
            modes_csv=$2
            shift 2
            ;;
        --processes)
            require_value "$@"
            processes=$2
            shift 2
            ;;
        --backend)
            require_value "$@"
            backend=$2
            shift 2
            ;;
        --output)
            require_value "$@"
            output_path=$2
            shift 2
            ;;
        --quick)
            quick=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$processes" in
    ''|*[!0-9]*)
        echo "--processes must be an integer" >&2
        exit 2
        ;;
esac
if [ "$processes" -lt 1 ] || [ "$processes" -gt 20 ]; then
    echo "--processes must be between 1 and 20" >&2
    exit 2
fi

if [ "$quick" = true ]; then
    items_csv=1000000
    workloads_csv=bounded16,full_width
    modes_csv=resident
    processes=1
fi

command -v cargo >/dev/null 2>&1 || { echo "cargo is required" >&2; exit 1; }
command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }

benchmark_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(CDPATH= cd -- "$benchmark_root/../.." && pwd -P)
target_root=$repo_root/target/wgpu-sort-comparison
if [ -z "$output_path" ]; then
    output_path=$benchmark_root/results/latest.json
elif [ "${output_path#/}" = "$output_path" ]; then
    output_path=$repo_root/$output_path
fi

primitives_manifest=$benchmark_root/wgpu-primitives-runner/Cargo.toml
comparison_manifest=$benchmark_root/wgpu-sort-runner/Cargo.toml
primitives_target=$target_root/wgpu-primitives
comparison_target=$target_root/wgpu-sort

echo "Building wgpu-primitives runner..."
CARGO_TARGET_DIR=$primitives_target cargo build --release --locked --manifest-path "$primitives_manifest"
echo "Building wgpu_sort runner..."
CARGO_TARGET_DIR=$comparison_target cargo build --release --locked --manifest-path "$comparison_manifest"

executable_suffix=
case "${OS:-}" in
    Windows_NT) executable_suffix=.exe ;;
esac
primitives_executable=$primitives_target/release/wgpu-primitives-comparison-runner$executable_suffix
comparison_executable=$comparison_target/release/wgpu-sort-comparison-runner$executable_suffix

repo_revision=$(git -c "safe.directory=$repo_root" -C "$repo_root" rev-parse HEAD)
if [ -n "$(git -c "safe.directory=$repo_root" -C "$repo_root" status --porcelain)" ]; then
    repo_dirty=true
else
    repo_dirty=false
fi
package_version=$(cargo metadata --no-deps --format-version 1 --manifest-path "$repo_root/Cargo.toml" | python3 -c 'import json, sys; print(json.load(sys.stdin)["packages"][0]["version"])')
pinned_wgpu_sort_revision=4cb640e8cae28eba0149d470c5168cc2853466dd

temp_root=$(mktemp -d)
trap 'rm -rf "$temp_root"' EXIT HUP INT TERM
runs_path=$temp_root/runs.jsonl
: > "$runs_path"

csv_words() {
    printf '%s\n' "$1" | tr ',' ' '
}

run_one() {
    executable=$1
    implementation_version=$2
    implementation_revision=$3
    item_count=$4
    workload=$5
    mode=$6
    warmups=$7
    warmup_ms=$8
    samples=$9
    process_index=${10}

    echo "$implementation_version $mode $workload items=$item_count process=$process_index" >&2
    WGPU_BACKEND=$backend \
    WGPU_SORT_BENCH_ITEMS=$item_count \
    WGPU_SORT_BENCH_WORKLOAD=$workload \
    WGPU_SORT_BENCH_MODE=$mode \
    WGPU_SORT_BENCH_WARMUPS=$warmups \
    WGPU_SORT_BENCH_WARMUP_MS=$warmup_ms \
    WGPU_SORT_BENCH_SAMPLES=$samples \
    WGPU_SORT_BENCH_PROCESS_INDEX=$process_index \
    WGPU_SORT_BENCH_IMPLEMENTATION_VERSION=$implementation_version \
    WGPU_SORT_BENCH_IMPLEMENTATION_REVISION=$implementation_revision \
    "$executable" >> "$runs_path"
}

for item_count in $(csv_words "$items_csv"); do
    case "$item_count" in
        ''|*[!0-9]*)
            echo "item counts must be positive integers, got: $item_count" >&2
            exit 2
            ;;
    esac
    if [ "$item_count" -lt 1 ] || [ "$item_count" -gt 4294967295 ]; then
        echo "item count must be between 1 and 4294967295, got: $item_count" >&2
        exit 2
    fi
    if [ "$quick" = true ]; then
        warmups=1
        warmup_ms=0
        samples=3
    elif [ "$item_count" -ge 100000000 ]; then
        warmups=2
        warmup_ms=2000
        samples=7
    else
        warmups=4
        warmup_ms=2000
        samples=11
    fi

    for workload in $(csv_words "$workloads_csv"); do
        case "$workload" in
            bounded16|full_width) ;;
            *) echo "unsupported workload: $workload" >&2; exit 2 ;;
        esac
        for mode in $(csv_words "$modes_csv"); do
            case "$mode" in
                resident|round_trip) ;;
                *) echo "unsupported mode: $mode" >&2; exit 2 ;;
            esac
            process_index=1
            while [ "$process_index" -le "$processes" ]; do
                run_one "$primitives_executable" "wgpu-primitives-$package_version" \
                    "$repo_revision" "$item_count" "$workload" "$mode" \
                    "$warmups" "$warmup_ms" "$samples" "$process_index"
                run_one "$comparison_executable" wgpu_sort-git \
                    "$pinned_wgpu_sort_revision" "$item_count" "$workload" "$mode" \
                    "$warmups" "$warmup_ms" "$samples" "$process_index"
                process_index=$((process_index + 1))
            done
        done
    done
done

set -- \
    --runs "$runs_path" \
    --output "$output_path" \
    --repository-revision "$repo_revision" \
    --repository-dirty "$repo_dirty" \
    --package-version "$package_version" \
    --comparison-revision "$pinned_wgpu_sort_revision" \
    --backend "$backend" \
    --items "$items_csv" \
    --workloads "$workloads_csv" \
    --modes "$modes_csv" \
    --processes "$processes" \
    --host "$(hostname)" \
    --architecture "$(uname -m)"
if [ "$quick" = true ]; then
    set -- "$@" --quick
fi
python3 "$benchmark_root/aggregate.py" "$@"
