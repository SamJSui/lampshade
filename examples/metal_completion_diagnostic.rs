use std::{
    hint::black_box,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use lampshade::{Context, Reducer, U32Reduction};
use wgpu::util::DeviceExt;

const ITEMS: u32 = 1_000_000;
const BATCH: usize = 32;
const WARMUPS: usize = 4;
const SAMPLES: usize = 11;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::init().await?;
    let input: Vec<_> = (0..ITEMS).map(|value| value.wrapping_mul(17)).collect();
    let gpu_input = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Metal completion diagnostic input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let gpu_output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Metal completion diagnostic output"),
        size: Reducer::output_buffer_size(),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut reducer = Reducer::from_context(&context);

    // Prepare reusable reduction scratch outside every measured region.
    reducer.reduce_gpu_to_gpu(&gpu_input, &gpu_output, ITEMS, U32Reduction::Sum)?;
    wait_all(&context.device)?;

    let empty_completion = sample(|| {
        let submission = context.queue.submit([]);
        context.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })?;
        Ok(())
    })?;

    let submit_and_wait = sample(|| {
        reducer.reduce_gpu_to_gpu(&gpu_input, &gpu_output, ITEMS, U32Reduction::Sum)?;
        wait_all(&context.device)
    })?;

    let submit_and_busy_poll = sample(|| {
        reducer.reduce_gpu_to_gpu(&gpu_input, &gpu_output, ITEMS, U32Reduction::Sum)?;
        wait_with_callback(&context.device, &context.queue)
    })?;

    let many_submits_one_wait = sample(|| {
        for _ in 0..BATCH {
            reducer.reduce_gpu_to_gpu(&gpu_input, &gpu_output, ITEMS, U32Reduction::Sum)?;
        }
        let submission = context.queue.submit([]);
        context.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })?;
        Ok(())
    })? / BATCH as f64;

    let one_submit_one_wait = sample(|| {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Metal completion diagnostic batch"),
            });
        for _ in 0..BATCH {
            reducer.record_reduce(
                &mut encoder,
                &gpu_input,
                &gpu_output,
                ITEMS,
                U32Reduction::Sum,
            )?;
        }
        let submission = context.queue.submit([encoder.finish()]);
        context.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })?;
        Ok(())
    })? / BATCH as f64;

    black_box(&gpu_output);
    println!(
        "adapter={:?} backend={:?} items={ITEMS} batch={BATCH}",
        context.adapter_info.name, context.adapter_info.backend
    );
    println!("empty_submit_wait_ms={empty_completion:.6}");
    println!("one_reduce_submit_wait_ms={submit_and_wait:.6}");
    println!("one_reduce_busy_poll_callback_ms={submit_and_busy_poll:.6}");
    println!("many_submits_one_wait_per_reduce_ms={many_submits_one_wait:.6}");
    println!("one_submit_one_wait_per_reduce_ms={one_submit_one_wait:.6}");
    Ok(())
}

fn sample(
    mut operation: impl FnMut() -> Result<(), lampshade::Error>,
) -> Result<f64, lampshade::Error> {
    for _ in 0..WARMUPS {
        operation()?;
    }
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let started = Instant::now();
        operation()?;
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[SAMPLES / 2])
}

fn wait_all(device: &wgpu::Device) -> Result<(), lampshade::Error> {
    device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    })?;
    Ok(())
}

fn wait_with_callback(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), lampshade::Error> {
    let finished = Arc::new(AtomicBool::new(false));
    let callback_flag = Arc::clone(&finished);
    queue.on_submitted_work_done(move || callback_flag.store(true, Ordering::Release));
    while !finished.load(Ordering::Acquire) {
        device.poll(wgpu::PollType::Poll)?;
        std::hint::spin_loop();
    }
    Ok(())
}
