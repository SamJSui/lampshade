use std::env;
use std::fmt;
use std::str::FromStr;

use serde::Serialize;

pub const SCHEMA_VERSION: u32 = 1;
pub const GENERATOR_NAME: &str = "xorshift32-v1";
pub const GENERATOR_BASE_SEED: u32 = 0x9E37_79B9;
pub const WGPU_SORT_REVISION: &str = "4cb640e8cae28eba0149d470c5168cc2853466dd";

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Workload {
    Bounded16,
    FullWidth,
}

impl Workload {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bounded16 => "bounded16",
            Self::FullWidth => "full_width",
        }
    }
}

impl FromStr for Workload {
    type Err = ConfigError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bounded16" => Ok(Self::Bounded16),
            "full_width" => Ok(Self::FullWidth),
            _ => Err(ConfigError(format!(
                "WGPU_SORT_BENCH_WORKLOAD must be bounded16 or full_width, got {value:?}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkMode {
    Resident,
    RoundTrip,
}

impl BenchmarkMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Resident => "resident",
            Self::RoundTrip => "round_trip",
        }
    }
}

impl FromStr for BenchmarkMode {
    type Err = ConfigError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "resident" => Ok(Self::Resident),
            "round_trip" => Ok(Self::RoundTrip),
            _ => Err(ConfigError(format!(
                "WGPU_SORT_BENCH_MODE must be resident or round_trip, got {value:?}"
            ))),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchmarkConfig {
    pub items: u32,
    pub workload: Workload,
    pub mode: BenchmarkMode,
    pub warmups: u32,
    pub warmup_ms: u64,
    pub samples: u32,
    pub process_index: u32,
}

impl BenchmarkConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        let items = parse_required("WGPU_SORT_BENCH_ITEMS")?;
        let warmups = parse_required("WGPU_SORT_BENCH_WARMUPS")?;
        let warmup_ms = parse_required("WGPU_SORT_BENCH_WARMUP_MS")?;
        let samples = parse_required("WGPU_SORT_BENCH_SAMPLES")?;
        let process_index = parse_required("WGPU_SORT_BENCH_PROCESS_INDEX")?;
        let workload = required("WGPU_SORT_BENCH_WORKLOAD")?.parse()?;
        let mode = required("WGPU_SORT_BENCH_MODE")?.parse()?;

        if items == 0 {
            return Err(ConfigError("item count must be nonzero".into()));
        }
        if samples == 0 {
            return Err(ConfigError("sample count must be nonzero".into()));
        }

        Ok(Self {
            items,
            workload,
            mode,
            warmups,
            warmup_ms,
            samples,
            process_index,
        })
    }
}

#[derive(Debug)]
pub struct ConfigError(String);

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

#[derive(Clone, Debug)]
pub struct LogicalInput {
    pub keys: Vec<u32>,
    pub values: Vec<u32>,
}

impl LogicalInput {
    pub fn generate(items: u32, workload: Workload) -> Self {
        let mut state = GENERATOR_BASE_SEED ^ items;
        let mut keys = Vec::with_capacity(items as usize);
        let mut values = Vec::with_capacity(items as usize);

        for index in 0..items {
            state = xorshift32(state);
            let key = match workload {
                Workload::Bounded16 => state & 0xffff,
                Workload::FullWidth => state,
            };
            keys.push(key);
            values.push(index);
        }

        Self { keys, values }
    }

    pub fn stable_sorted_pairs(&self) -> Vec<(u32, u32)> {
        let mut pairs: Vec<_> = self
            .keys
            .iter()
            .copied()
            .zip(self.values.iter().copied())
            .collect();
        pairs.sort_by_key(|pair| pair.0);
        pairs
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct AdapterMetadata {
    pub name: String,
    pub vendor: u32,
    pub device: u32,
    pub device_type: String,
    pub backend: String,
    pub driver: String,
    pub driver_info: String,
    pub subgroup_min_size: Option<u32>,
    pub subgroup_max_size: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MemoryEstimate {
    pub model: String,
    pub primary_input_output_bytes: Option<u64>,
    pub workspace_bytes: Option<u64>,
    pub total_known_buffer_bytes: Option<u64>,
    pub exclusions: Vec<String>,
}

impl MemoryEstimate {
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            model: reason.into(),
            primary_input_output_bytes: None,
            workspace_bytes: None,
            total_known_buffer_bytes: None,
            exclusions: vec!["driver-managed allocations".into()],
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchmarkRun {
    pub schema_version: u32,
    pub implementation: String,
    pub implementation_version: String,
    pub implementation_revision: String,
    pub wgpu_version: String,
    pub adapter: AdapterMetadata,
    pub config: BenchmarkConfig,
    pub generator: GeneratorMetadata,
    pub correctness_checked: bool,
    pub samples_ms: Vec<f64>,
    pub median_ms: f64,
    pub throughput_pairs_per_second: f64,
    pub memory: MemoryEstimate,
}

#[derive(Clone, Debug, Serialize)]
pub struct GeneratorMetadata {
    pub name: String,
    pub base_seed: u32,
}

impl GeneratorMetadata {
    pub fn current() -> Self {
        Self {
            name: GENERATOR_NAME.into(),
            base_seed: GENERATOR_BASE_SEED,
        }
    }
}

pub fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "median requires at least one value");
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

pub fn wgpu_primitives_eight_bit_memory(items: u32, uniform_stride: u64) -> MemoryEstimate {
    const ITEM_BYTES: u64 = 8;
    const GROWTH_BYTES: u64 = 16 * 1024 * 1024;
    const ITEMS_PER_TILE: u64 = 256 * 7;
    const BUCKETS: u64 = 256;
    const PASSES: u64 = 4;

    let requested = u64::from(items) * ITEM_BYTES;
    let scratch = if requested < GROWTH_BYTES {
        requested.max(ITEM_BYTES).next_power_of_two()
    } else {
        requested.div_ceil(GROWTH_BYTES) * GROWTH_BYTES
    };
    let max_items = scratch / ITEM_BYTES;
    let max_tiles = max_items.div_ceil(ITEMS_PER_TILE);
    let partition_entries = max_tiles * BUCKETS + PASSES;
    let partition_bytes = (partition_entries * 4).div_ceil(256) * 256;
    let digit_tables = 2 * PASSES * BUCKETS * 4;
    let dispatch_args = PASSES * 3 * 4;
    let aligned_uniforms = PASSES * uniform_stride.max(16);
    let workspace = scratch + partition_bytes + digit_tables + dispatch_args + aligned_uniforms;
    let primary = requested * 2;

    MemoryEstimate {
        model: "wgpu-primitives 8-bit source allocation formula".into(),
        primary_input_output_bytes: Some(primary),
        workspace_bytes: Some(workspace),
        total_known_buffer_bytes: Some(primary + workspace),
        exclusions: vec![
            "bind groups and pipelines".into(),
            "upload and readback staging buffers".into(),
            "driver-managed allocations".into(),
        ],
    }
}

pub fn wgpu_sort_pinned_memory(items: u32) -> MemoryEstimate {
    const BLOCK_ITEMS: u64 = 256 * 15;
    let items = u64::from(items);
    let padded_items = items.div_ceil(BLOCK_ITEMS) * BLOCK_ITEMS;
    let keys_each = padded_items * 16;
    let values_each = items * 4;
    let scatter_blocks = items.div_ceil(BLOCK_ITEMS);
    let internal = (4 + scatter_blocks) * 256 * 4;
    let state = 16;
    let primary = keys_each + values_each;
    let workspace = keys_each + values_each + internal + state;

    MemoryEstimate {
        model: format!("wgpu_sort source allocation formula at {WGPU_SORT_REVISION}"),
        primary_input_output_bytes: Some(primary),
        workspace_bytes: Some(workspace),
        total_known_buffer_bytes: Some(primary + workspace),
        exclusions: vec![
            "bind groups and pipelines".into(),
            "upload and readback staging buffers".into(),
            "driver-managed allocations".into(),
        ],
    }
}

fn xorshift32(mut value: u32) -> u32 {
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    value
}

fn required(name: &str) -> Result<String, ConfigError> {
    env::var(name).map_err(|_| ConfigError(format!("missing required environment variable {name}")))
}

fn parse_required<T>(name: &str) -> Result<T, ConfigError>
where
    T: FromStr,
    T::Err: fmt::Display,
{
    let value = required(name)?;
    value
        .parse()
        .map_err(|error| ConfigError(format!("invalid {name}={value:?}: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_is_deterministic_and_masks_bounded_keys() {
        let first = LogicalInput::generate(32, Workload::Bounded16);
        let second = LogicalInput::generate(32, Workload::Bounded16);
        assert_eq!(first.keys, second.keys);
        assert!(first.keys.iter().all(|key| *key <= u16::MAX.into()));
        assert_eq!(first.values, (0..32).collect::<Vec<_>>());
    }

    #[test]
    fn memory_models_match_the_published_100m_case() {
        let primitives = wgpu_primitives_eight_bit_memory(100_000_000, 256);
        assert_eq!(primitives.workspace_bytes, Some(862_838_064));
        assert_eq!(primitives.total_known_buffer_bytes, Some(2_462_838_064));

        let comparison = wgpu_sort_pinned_memory(100_000_000);
        assert_eq!(comparison.workspace_bytes, Some(2_026_691_600));
        assert_eq!(comparison.total_known_buffer_bytes, Some(4_026_712_080));
    }

    #[test]
    fn median_handles_odd_and_even_samples() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }
}
