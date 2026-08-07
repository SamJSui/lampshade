#!/usr/bin/env sh
set -eu

if [ "$(uname -m)" != "aarch64" ]; then
    echo "expected an aarch64 Jetson host, got $(uname -m)" >&2
    exit 1
fi

echo "hostname=$(hostname)"
echo "architecture=$(uname -m)"
if [ -r /etc/nv_tegra_release ]; then
    printf 'jetson_linux='
    head -n 1 /etc/nv_tegra_release
fi
rustc --version
cargo --version

export WGPU_BACKEND=vulkan

cargo test --lib --tests

export WGPU_PRIMITIVES_PROFILE_ITEMS=1000000,10000000
export WGPU_PRIMITIVES_PROFILE_CASES=key_value_bounded16,key_value_full_width
export WGPU_PRIMITIVES_PROFILE_SAMPLES=5
export WGPU_PRIMITIVES_PROFILE_WARMUP_MS=2000

cargo run --release --example profile_primitives
