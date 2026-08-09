use std::env;
use std::fmt;
use std::str::FromStr;

use serde::Serialize;

pub const SCHEMA_VERSION: u32 = 1;
pub const GENERATOR_NAME: &str = "xorshift32-v1";
pub const GENERATOR_BASE_SEED: u32 = 0x9E37_79B9;
pub const MASSIVELY_VERSION: &str = "0.96.0";
pub const MASSIVELY_REVISION: &str = "ef9de55190529be98203aca207edab9d560d312e";

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Workload {
    #[serde(rename = "reduce_sum")]
    ReduceSum,
    #[serde(rename = "sort_bounded16")]
    SortBounded16,
    #[serde(rename = "sort_full_width")]
    SortFullWidth,
    #[serde(rename = "exclusive_scan")]
    ExclusiveScan,
    #[serde(rename = "compact_50")]
    Compact50,
}

impl Workload {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReduceSum => "reduce_sum",
            Self::SortBounded16 => "sort_bounded16",
            Self::SortFullWidth => "sort_full_width",
            Self::ExclusiveScan => "exclusive_scan",
            Self::Compact50 => "compact_50",
        }
    }

    pub const fn output_items(self, input_items: u32) -> u32 {
        match self {
            Self::Compact50 => input_items.div_ceil(2),
            Self::ReduceSum => 1,
            _ => input_items,
        }
    }
}

impl FromStr for Workload {
    type Err = ConfigError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "reduce_sum" => Ok(Self::ReduceSum),
            "sort_bounded16" => Ok(Self::SortBounded16),
            "sort_full_width" => Ok(Self::SortFullWidth),
            "exclusive_scan" => Ok(Self::ExclusiveScan),
            "compact_50" => Ok(Self::Compact50),
            _ => Err(ConfigError(format!(
                "MASSIVELY_BENCH_WORKLOAD must be reduce_sum, sort_bounded16, sort_full_width, exclusive_scan, or compact_50; got {value:?}"
            ))),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchmarkConfig {
    pub items: u32,
    pub workload: Workload,
    pub warmups: u32,
    pub warmup_ms: u64,
    pub samples: u32,
    pub process_index: u32,
}

impl BenchmarkConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        let items = parse_required("MASSIVELY_BENCH_ITEMS")?;
        let warmups = parse_required("MASSIVELY_BENCH_WARMUPS")?;
        let warmup_ms = parse_required("MASSIVELY_BENCH_WARMUP_MS")?;
        let samples = parse_required("MASSIVELY_BENCH_SAMPLES")?;
        let process_index = parse_required("MASSIVELY_BENCH_PROCESS_INDEX")?;
        let workload = required("MASSIVELY_BENCH_WORKLOAD")?.parse()?;

        if items == 0 {
            return Err(ConfigError("item count must be nonzero".into()));
        }
        if samples == 0 {
            return Err(ConfigError("sample count must be nonzero".into()));
        }

        Ok(Self {
            items,
            workload,
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
pub struct SortInput {
    pub keys: Vec<u32>,
    pub values: Vec<u32>,
}

impl SortInput {
    pub fn generate(items: u32, workload: Workload) -> Self {
        assert!(matches!(
            workload,
            Workload::SortBounded16 | Workload::SortFullWidth
        ));
        let mut state = GENERATOR_BASE_SEED ^ items;
        let mut keys = Vec::with_capacity(items as usize);
        let mut values = Vec::with_capacity(items as usize);
        for index in 0..items {
            state = xorshift32(state);
            keys.push(match workload {
                Workload::SortBounded16 => state & 0xffff,
                Workload::SortFullWidth => state,
                _ => unreachable!(),
            });
            values.push(index);
        }
        Self { keys, values }
    }

    pub fn validate_values(&self, output_values: &[u32]) -> Result<(), String> {
        if output_values.len() != self.values.len() {
            return Err(format!(
                "sort output length mismatch: expected {}, got {}",
                self.values.len(),
                output_values.len()
            ));
        }
        let mut seen = vec![0_u64; output_values.len().div_ceil(64)];
        let mut previous: Option<(u32, u32)> = None;
        for (position, &value) in output_values.iter().enumerate() {
            let index = value as usize;
            if index >= self.keys.len() {
                return Err(format!(
                    "sort output value {value} at {position} is out of range"
                ));
            }
            let word = index / 64;
            let bit = 1_u64 << (index % 64);
            if seen[word] & bit != 0 {
                return Err(format!("sort output repeats value {value} at {position}"));
            }
            seen[word] |= bit;
            let key = self.keys[index];
            if let Some((previous_key, previous_value)) = previous
                && (key < previous_key || (key == previous_key && value < previous_value))
            {
                return Err(format!(
                    "sort output is not stable at {position}: ({previous_key}, {previous_value}) then ({key}, {value})"
                ));
            }
            previous = Some((key, value));
        }
        Ok(())
    }
}

pub fn generate_scan(items: u32) -> Vec<u32> {
    let mut state = GENERATOR_BASE_SEED ^ items ^ 0x5CA1;
    (0..items)
        .map(|_| {
            state = xorshift32(state);
            state & 15
        })
        .collect()
}

pub fn generate_reduction(items: u32) -> Vec<u32> {
    let mut state = GENERATOR_BASE_SEED ^ items ^ 0x5ED0_CE00;
    (0..items)
        .map(|_| {
            state = xorshift32(state);
            state
        })
        .collect()
}

pub fn validate_reduction_sum(input: &[u32], output: u32) -> Result<(), String> {
    let expected = input
        .iter()
        .fold(0_u32, |sum, value| sum.wrapping_add(*value));
    if output != expected {
        return Err(format!(
            "reduction sum mismatch: expected {expected}, got {output}"
        ));
    }
    Ok(())
}

pub fn validate_exclusive_scan(input: &[u32], output: &[u32]) -> Result<(), String> {
    if output.len() != input.len() {
        return Err(format!(
            "scan output length mismatch: expected {}, got {}",
            input.len(),
            output.len()
        ));
    }
    let mut expected = 0_u32;
    for (index, (&value, &actual)) in input.iter().zip(output).enumerate() {
        if actual != expected {
            return Err(format!(
                "exclusive scan mismatch at {index}: expected {expected}, got {actual}"
            ));
        }
        expected = expected.wrapping_add(value);
    }
    Ok(())
}

pub fn generate_compact(items: u32) -> (Vec<u32>, Vec<u32>) {
    let mut state = GENERATOR_BASE_SEED ^ items ^ 0xC04C_7A50;
    let mut input = Vec::with_capacity(items as usize);
    let mut mask = Vec::with_capacity(items as usize);
    for index in 0..items {
        state = xorshift32(state);
        input.push(state);
        mask.push(u32::from(index % 2 == 0));
    }
    (input, mask)
}

pub fn validate_compact(input: &[u32], mask: &[u32], output: &[u32]) -> Result<(), String> {
    let expected_len = mask.iter().map(|&flag| flag as usize).sum::<usize>();
    if output.len() != expected_len {
        return Err(format!(
            "compaction output length mismatch: expected {expected_len}, got {}",
            output.len()
        ));
    }
    let mut output_index = 0;
    for (&value, &flag) in input.iter().zip(mask) {
        if flag != 0 {
            if output[output_index] != value {
                return Err(format!(
                    "stable compaction mismatch at output {output_index}: expected {value}, got {}",
                    output[output_index]
                ));
            }
            output_index += 1;
        }
    }
    Ok(())
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
    pub primary_input_output_bytes: u64,
    pub workspace_bytes: Option<u64>,
    pub total_known_buffer_bytes: u64,
    pub exclusions: Vec<String>,
}

pub fn public_buffer_memory(
    implementation: &str,
    workload: Workload,
    items: u32,
) -> MemoryEstimate {
    let items = u64::from(items);
    let primary = match (implementation, workload) {
        ("wgpu-primitives", Workload::ReduceSum) => 4 * items + 8,
        ("massively", Workload::ReduceSum) => 4 * items,
        ("wgpu-primitives", Workload::SortBounded16 | Workload::SortFullWidth) => 16 * items,
        ("massively", Workload::SortBounded16 | Workload::SortFullWidth) => 12 * items,
        (_, Workload::ExclusiveScan) => 8 * items,
        (_, Workload::Compact50) => 12 * items + u64::from(implementation == "wgpu-primitives") * 4,
        _ => 0,
    };
    MemoryEstimate {
        model: "known public input/output and timed I/O allocations".into(),
        primary_input_output_bytes: primary,
        workspace_bytes: None,
        total_known_buffer_bytes: primary,
        exclusions: vec![
            "unexposed algorithm workspace, allocator caching, and Massively reduction readback storage".into(),
            "validation readback buffers".into(),
            "driver-managed allocations".into(),
        ],
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchmarkRun {
    pub schema_version: u32,
    pub implementation: String,
    pub implementation_version: String,
    pub implementation_revision: String,
    pub runtime_stack: String,
    pub adapter: AdapterMetadata,
    pub config: BenchmarkConfig,
    pub generator: GeneratorMetadata,
    pub timing_boundary: String,
    pub output_allocation: String,
    pub correctness_checked: bool,
    pub samples_ms: Vec<f64>,
    pub median_ms: f64,
    pub throughput_items_per_second: f64,
    pub output_items: u32,
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

pub fn runtime_metadata(name: &str, fallback: &str) -> String {
    env::var(name).unwrap_or_else(|_| fallback.into())
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
    fn sort_validation_checks_order_stability_and_permutation() {
        let input = SortInput {
            keys: vec![2, 1, 1, 3],
            values: vec![0, 1, 2, 3],
        };
        assert!(input.validate_values(&[1, 2, 0, 3]).is_ok());
        assert!(input.validate_values(&[2, 1, 0, 3]).is_err());
        assert!(input.validate_values(&[1, 1, 0, 3]).is_err());
    }

    #[test]
    fn compact_generator_selects_exactly_half_rounded_up() {
        let (input, mask) = generate_compact(7);
        let output: Vec<_> = input
            .iter()
            .zip(&mask)
            .filter_map(|(&value, &flag)| (flag != 0).then_some(value))
            .collect();
        assert_eq!(output.len(), 4);
        assert!(validate_compact(&input, &mask, &output).is_ok());
    }

    #[test]
    fn scan_validation_uses_wrapping_u32_addition() {
        assert!(validate_exclusive_scan(&[u32::MAX, 2], &[0, u32::MAX]).is_ok());
    }

    #[test]
    fn reduction_validation_uses_wrapping_u32_addition() {
        assert!(validate_reduction_sum(&[u32::MAX, 2], 1).is_ok());
        assert!(validate_reduction_sum(&[u32::MAX, 2], 2).is_err());
    }

    #[test]
    fn reduction_memory_counts_explicit_wgpu_scalar_buffers() {
        let memory = public_buffer_memory("wgpu-primitives", Workload::ReduceSum, 10);
        assert_eq!(memory.primary_input_output_bytes, 48);
        assert_eq!(memory.total_known_buffer_bytes, 48);
    }

    #[test]
    fn median_handles_odd_and_even_samples() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }
}
