use crate::{
    Error, KeyValue, common,
    common::buffers::BufferRange,
    common::runtime::{CommandSession, ProfileSession},
    context::Context,
    profiling::{GpuProfile, TimestampRecorder},
};

use super::{
    KeyValueField, U32Predicate,
    pipeline::{PredicateDispatch, PredicateItemKind, PredicatePipeline},
};

const MASK_ITEM_SIZE_BYTES: u64 = size_of::<u32>() as u64;

/// Generates reusable `0`/`1` selection masks for stream compaction.
pub struct MaskGenerator {
    value_pipeline: PredicatePipeline,
    key_value_pipeline: PredicatePipeline,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl MaskGenerator {
    /// Returns the required allocation size for `num_items` mask flags.
    pub fn mask_buffer_size(num_items: u32) -> Result<u64, Error> {
        common::math::checked_byte_size(u64::from(num_items), MASK_ITEM_SIZE_BYTES)
    }

    /// Creates a mask generator for an existing device and queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            value_pipeline: PredicatePipeline::new(device, PredicateItemKind::Value),
            key_value_pipeline: PredicatePipeline::new(device, PredicateItemKind::KeyValue),
            device: device.clone(),
            queue: queue.clone(),
        }
    }

    /// Creates a mask generator from the crate's convenience context.
    pub fn from_context(context: &Context) -> Self {
        Self::new(&context.device, &context.queue)
    }

    /// Uploads values, generates a mask on the GPU, and downloads the mask.
    pub async fn mask(&self, input: &[u32], predicate: U32Predicate) -> Result<Vec<u32>, Error> {
        self.mask_slice(
            input,
            KeyValueField::Key,
            predicate,
            PredicateItemKind::Value,
        )
        .await
    }

    /// Uploads records, tests one field on the GPU, and downloads the mask.
    pub async fn mask_key_values(
        &self,
        input: &[KeyValue],
        field: KeyValueField,
        predicate: U32Predicate,
    ) -> Result<Vec<u32>, Error> {
        self.mask_slice(input, field, predicate, PredicateItemKind::KeyValue)
            .await
    }

    /// Generates a mask in caller-owned GPU buffers and submits immediately.
    ///
    /// `input` requires `STORAGE`; `mask` requires `STORAGE | COPY_SRC` and
    /// [`Self::mask_buffer_size`] bytes. The buffers must be distinct.
    pub fn mask_gpu_to_gpu(
        &self,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        num_items: u32,
        predicate: U32Predicate,
    ) -> Result<(), Error> {
        self.submit(
            input,
            mask,
            num_items,
            KeyValueField::Key,
            predicate,
            PredicateItemKind::Value,
        )
    }

    /// Generates a key-value field mask in caller-owned buffers and submits immediately.
    ///
    /// `input` requires `STORAGE` and capacity for `num_items` [`KeyValue`]
    /// records. `mask` requires `STORAGE | COPY_SRC` and
    /// [`Self::mask_buffer_size`] bytes. The buffers must be distinct.
    pub fn mask_key_values_gpu_to_gpu(
        &self,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        num_items: u32,
        field: KeyValueField,
        predicate: U32Predicate,
    ) -> Result<(), Error> {
        self.submit(
            input,
            mask,
            num_items,
            field,
            predicate,
            PredicateItemKind::KeyValue,
        )
    }

    /// Records value-mask generation without submitting or waiting.
    ///
    /// The buffer contract matches [`Self::mask_gpu_to_gpu`].
    pub fn record_mask(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        num_items: u32,
        predicate: U32Predicate,
    ) -> Result<(), Error> {
        self.record_commands(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(mask),
            num_items,
            KeyValueField::Key,
            predicate,
            PredicateItemKind::Value,
            None,
        )
    }

    /// Records key-value field-mask generation without submitting or waiting.
    ///
    /// The buffer contract matches [`Self::mask_key_values_gpu_to_gpu`].
    pub fn record_key_value_mask(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        num_items: u32,
        field: KeyValueField,
        predicate: U32Predicate,
    ) -> Result<(), Error> {
        self.record_commands(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(mask),
            num_items,
            field,
            predicate,
            PredicateItemKind::KeyValue,
            None,
        )
    }

    pub(crate) fn record_mask_ranges(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        mask: BufferRange<'_>,
        num_items: u32,
        predicate: U32Predicate,
    ) -> Result<(), Error> {
        self.record_commands(
            encoder,
            input,
            mask,
            num_items,
            KeyValueField::Key,
            predicate,
            PredicateItemKind::Value,
            None,
        )
    }

    pub(crate) fn record_key_value_mask_ranges(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        mask: BufferRange<'_>,
        num_items: u32,
        field: KeyValueField,
        predicate: U32Predicate,
    ) -> Result<(), Error> {
        self.record_commands(
            encoder,
            input,
            mask,
            num_items,
            field,
            predicate,
            PredicateItemKind::KeyValue,
            None,
        )
    }

    /// Profiles value-mask generation using hardware timestamp queries.
    pub async fn profile_mask_gpu_to_gpu(
        &self,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        num_items: u32,
        predicate: U32Predicate,
    ) -> Result<GpuProfile, Error> {
        self.profile(
            input,
            mask,
            num_items,
            KeyValueField::Key,
            predicate,
            PredicateItemKind::Value,
        )
        .await
    }

    /// Profiles key-value field-mask generation using hardware timestamp queries.
    pub async fn profile_key_value_mask_gpu_to_gpu(
        &self,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        num_items: u32,
        field: KeyValueField,
        predicate: U32Predicate,
    ) -> Result<GpuProfile, Error> {
        self.profile(
            input,
            mask,
            num_items,
            field,
            predicate,
            PredicateItemKind::KeyValue,
        )
        .await
    }

    async fn mask_slice<T: bytemuck::Pod>(
        &self,
        input: &[T],
        field: KeyValueField,
        predicate: U32Predicate,
        item_kind: PredicateItemKind,
    ) -> Result<Vec<u32>, Error> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        debug_assert_eq!(size_of::<T>() as u64, item_kind.size_bytes());
        let num_items = common::math::checked_u32(input.len() as u64)?;
        let mask_bytes = Self::mask_buffer_size(num_items)?;
        let input_buffer = common::buffers::create_storage_buffer(&self.device, input);
        let mask_buffer = common::buffers::create_empty_storage_buffer(&self.device, mask_bytes);

        self.submit(
            &input_buffer,
            &mask_buffer,
            num_items,
            field,
            predicate,
            item_kind,
        )?;
        common::buffers::download_buffer(&self.device, &self.queue, &mask_buffer, input.len()).await
    }

    #[allow(clippy::too_many_arguments)]
    fn submit(
        &self,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        num_items: u32,
        field: KeyValueField,
        predicate: U32Predicate,
        item_kind: PredicateItemKind,
    ) -> Result<(), Error> {
        let mut commands = CommandSession::new(&self.device, None);
        self.record_commands(
            commands.encoder(),
            BufferRange::whole(input),
            BufferRange::whole(mask),
            num_items,
            field,
            predicate,
            item_kind,
            None,
        )?;
        commands.submit(&self.queue);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn profile(
        &self,
        input: &wgpu::Buffer,
        mask: &wgpu::Buffer,
        num_items: u32,
        field: KeyValueField,
        predicate: U32Predicate,
        item_kind: PredicateItemKind,
    ) -> Result<GpuProfile, Error> {
        if num_items == 0 {
            return Ok(GpuProfile::empty());
        }

        let mut profile =
            ProfileSession::new(&self.device, &self.queue, 1, "Profiled Predicate Mask")?;
        let (encoder, profiler) = profile.recording();
        self.record_commands(
            encoder,
            BufferRange::whole(input),
            BufferRange::whole(mask),
            num_items,
            field,
            predicate,
            item_kind,
            profiler,
        )?;
        profile.finish(&self.device, &self.queue).await
    }

    #[allow(clippy::too_many_arguments)]
    fn record_commands(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: BufferRange<'_>,
        mask: BufferRange<'_>,
        num_items: u32,
        field: KeyValueField,
        predicate: U32Predicate,
        item_kind: PredicateItemKind,
        profiler: Option<&mut TimestampRecorder>,
    ) -> Result<(), Error> {
        if num_items == 0 {
            return Ok(());
        }
        if input.buffer == mask.buffer {
            return Err(Error::BufferAlias {
                first: "predicate input",
                second: "predicate mask",
            });
        }

        let input_bytes =
            common::math::checked_byte_size(u64::from(num_items), item_kind.size_bytes())?;
        let mask_bytes = Self::mask_buffer_size(num_items)?;
        input.validate("predicate input", input_bytes, wgpu::BufferUsages::STORAGE)?;
        mask.validate(
            "predicate mask",
            mask_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        )?;
        input.validate_storage_offset(&self.device, "predicate input")?;
        mask.validate_storage_offset(&self.device, "predicate mask")?;
        let input = BufferRange {
            size: input_bytes,
            ..input
        };
        let mask = BufferRange {
            size: mask_bytes,
            ..mask
        };

        let pipeline = match item_kind {
            PredicateItemKind::Value => &self.value_pipeline,
            PredicateItemKind::KeyValue => &self.key_value_pipeline,
        };
        pipeline.dispatch(
            &self.device,
            &self.queue,
            encoder,
            PredicateDispatch {
                input,
                mask,
                num_items,
                field,
                predicate,
            },
            profiler,
        );
        Ok(())
    }
}
