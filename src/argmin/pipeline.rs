use crate::{common, profiling};

pub(super) const BLOCK_SIZE: u32 = 256;
pub(super) const MAX_WORKGROUPS_X: u32 = 65_535;

pub(super) struct ArgminPipelines {
    pub(super) fixed_layout: wgpu::BindGroupLayout,
    pub(super) counted_layout: wgpu::BindGroupLayout,
    pub(super) fixed: wgpu::ComputePipeline,
    pub(super) counted: wgpu::ComputePipeline,
}

impl ArgminPipelines {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let fixed_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Argmin-by-Key Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, true, true),
            ],
        });
        let counted_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Counted Argmin-by-Key Layout"),
            entries: &[
                common::buffers::bind_entry(0, true, false),
                common::buffers::bind_entry(1, false, false),
                common::buffers::bind_entry(2, true, false),
                common::buffers::bind_entry(3, true, true),
            ],
        });
        let fixed = create_pipeline(
            device,
            &fixed_layout,
            include_str!("argmin.wgsl"),
            "Argmin-by-Key Pipeline",
        );
        let counted = create_pipeline(
            device,
            &counted_layout,
            include_str!("argmin_counted.wgsl"),
            "Counted Argmin-by-Key Pipeline",
        );
        Self {
            fixed_layout,
            counted_layout,
            fixed,
            counted,
        }
    }

    pub(super) fn record_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        output_items: u32,
        level: u32,
        counted: bool,
        profiler: Option<&mut profiling::TimestampRecorder>,
    ) {
        let (groups_x, groups_y) = dispatch_dimensions(output_items);
        let pipeline = if counted { &self.counted } else { &self.fixed };
        let profile_label = profiler
            .is_some()
            .then(|| format!("argmin_by_key.level.{level}"));
        profiling::record_compute_pass(encoder, "Argmin by Key", profile_label, profiler, |pass| {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, 1);
        });
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source: &str,
    label: &str,
) -> wgpu::ComputePipeline {
    let source = source
        .replace("{{BLOCK_SIZE}}", &BLOCK_SIZE.to_string())
        .replace("{{MAX_WORKGROUPS_X}}", &MAX_WORKGROUPS_X.to_string());
    common::shader::create_compute_pipeline(device, layout, &source, label, "main", None)
}

pub(super) const fn output_items(input_items: u32) -> u32 {
    input_items.div_ceil(BLOCK_SIZE)
}

pub(super) fn pass_count(mut input_items: u32) -> u32 {
    let mut passes = 0;
    while input_items > 0 {
        passes += 1;
        input_items = output_items(input_items);
        if input_items == 1 {
            break;
        }
    }
    passes
}

fn dispatch_dimensions(output_items: u32) -> (u32, u32) {
    (
        output_items.min(MAX_WORKGROUPS_X),
        output_items.div_ceil(MAX_WORKGROUPS_X),
    )
}

#[cfg(test)]
mod tests {
    use super::{BLOCK_SIZE, MAX_WORKGROUPS_X, dispatch_dimensions, pass_count};

    #[test]
    fn hierarchy_and_two_dimensional_dispatch_cover_tails() {
        assert_eq!(pass_count(0), 0);
        assert_eq!(pass_count(1), 1);
        assert_eq!(pass_count(BLOCK_SIZE), 1);
        assert_eq!(pass_count(BLOCK_SIZE + 1), 2);
        assert_eq!(
            dispatch_dimensions(MAX_WORKGROUPS_X + 1),
            (MAX_WORKGROUPS_X, 2)
        );
    }
}
