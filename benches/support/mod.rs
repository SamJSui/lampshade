use criterion::{BenchmarkGroup, Throughput, measurement::WallTime};
use lampshade::Context;
use rand::{Rng, SeedableRng, rngs::StdRng};

pub const INPUT_SIZES: [usize; 4] = [100_000, 1_000_000, 10_000_000, 100_000_000];

pub fn seeded_input(item_count: usize, seed: u64) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(seed ^ item_count as u64);
    (0..item_count).map(|_| rng.random()).collect()
}

pub fn report_adapter(context: &Context) {
    let info = &context.adapter_info;
    eprintln!(
        "wgpu adapter: name={:?} backend={:?} device_type={:?} driver={:?} driver_info={:?}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );
}

pub fn configure_group(group: &mut BenchmarkGroup<'_, WallTime>, item_count: usize) {
    group.throughput(Throughput::Elements(item_count as u64));
    if item_count >= 10_000_000 {
        group.sample_size(10);
        group.measurement_time(std::time::Duration::from_secs(15));
    } else {
        group.sample_size(30);
        group.measurement_time(std::time::Duration::from_secs(5));
    }
}

pub fn wait_for_gpu(device: &wgpu::Device) {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("failed to wait for GPU work");
}
