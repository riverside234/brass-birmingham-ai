//! Synchronous native leaf evaluation on the caller's session thread.
use crate::{Control, Evaluator, Input, Result};
use brass_ai_core::{
    evaluation::EvaluationId,
    features::FeatureEncoder,
    search::{brass::Key, math::Statistics, Info, LeafEvaluator},
};

pub struct Neural<'a, E: Evaluator> {
    pub evaluator: &'a mut E,
    pub encoder: FeatureEncoder,
    pub control: Control,
    pub generation: u64,
    pub sequence: u64,
    pub encoding_ns: u64,
    pub inference_ns: u64,
}
impl<E: Evaluator> LeafEvaluator<Key, String> for Neural<'_, E> {
    fn evaluate(&mut self, info: &Info<Key, String>) -> Result<Statistics> {
        self.control.check()?;
        let k = &info.key;
        let view = k
            .information
            .observation
            .view
            .as_ref()
            .ok_or_else(|| crate::invalid("missing leaf view"))?;
        let manifest = self.evaluator.manifest();
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| crate::invalid("leaf request sequence overflow"))?;
        let id = EvaluationId {
            request_id: format!("gumbel-leaf-{}", self.sequence),
            generation: self.generation,
            model_id: k.context.model_id.clone(),
            export_sha256: manifest.export_sha256.clone(),
            reward_id: k.context.reward_id.clone(),
            profile_id: k.context.profile_id.clone(),
            revision: view.context.revision,
            observer: info.actor as u8,
        };
        let clock = std::time::Instant::now();
        let tensors = self
            .encoder
            .encode(view, &k.choices, &k.context, &mut || self.control.check())?;
        self.encoding_ns += crate::nanos(clock);
        let output = self.evaluator.evaluate(
            &[Input {
                id: &id,
                context: &k.context,
                choices: &k.choices,
                tensors,
            }],
            &self.control,
        )?;
        self.inference_ns += output.stats.inference_ns;
        self.control.check()?;
        let row = output
            .evaluations
            .into_iter()
            .next()
            .ok_or_else(|| crate::invalid("missing leaf response"))?;
        if row.id != id || row.choice_ids.as_slice() != info.actions.as_slice() {
            return Err(crate::invalid("mismatched leaf response"));
        }
        Statistics::new(
            row.logits.into_iter().map(f64::from).collect(),
            row.utility.map(f64::from),
        )
    }
}

/// Persistent session-owned tensor capacity. No executor or background task.
#[derive(Default)]
pub struct Buffers {
    encoders: [FeatureEncoder; 4],
}
pub struct Batched<'a, E: Evaluator> {
    pub evaluator: &'a mut E,
    pub buffers: &'a mut Buffers,
    pub control: Control,
    pub generation: u64,
    pub request: u64,
    pub owner: usize,
    pub sequence: u64,
    pub stats: brass_ai_core::evaluation::EvaluationStats,
}
impl<E: Evaluator> LeafEvaluator<Key, String> for Batched<'_, E> {
    fn evaluate(&mut self, info: &Info<Key, String>) -> Result<Statistics> {
        self.evaluate_batch(&[info])?
            .pop()
            .ok_or_else(|| crate::invalid("empty batch"))
    }
    fn evaluate_batch(&mut self, infos: &[&Info<Key, String>]) -> Result<Vec<Statistics>> {
        use brass_ai_core::work::{checkpoint, runtime_returned, Stage};
        crate::require(!infos.is_empty() && infos.len() <= 4, "leaf batch bound")?;
        self.control.check()?;
        let mut ids = Vec::with_capacity(infos.len());
        let start = std::time::Instant::now();
        for (slot, info) in infos.iter().enumerate() {
            checkpoint(Stage::Encoding)?;
            let k = &info.key;
            let view = k
                .information
                .observation
                .view
                .as_ref()
                .ok_or_else(|| crate::invalid("leaf view"))?;
            self.sequence = self
                .sequence
                .checked_add(1)
                .ok_or_else(|| crate::invalid("sequence overflow"))?;
            ids.push(EvaluationId {
                request_id: format!(
                    "cooperative-leaf-{}-{}-{}-{}",
                    self.generation, self.request, self.owner, self.sequence
                ),
                generation: self.generation,
                model_id: k.context.model_id.clone(),
                export_sha256: self.evaluator.manifest().export_sha256.clone(),
                reward_id: k.context.reward_id.clone(),
                profile_id: k.context.profile_id.clone(),
                revision: view.context.revision,
                observer: info.actor as u8,
            });
            self.buffers.encoders[slot]
                .encode(view, &k.choices, &k.context, &mut || self.control.check())?;
        }
        self.stats.encoding_ns += crate::nanos(start);
        let inputs: Vec<_> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| Input {
                id: &ids[i],
                context: &info.key.context,
                choices: &info.key.choices,
                tensors: self.buffers.encoders[i]
                    .tensors()
                    .expect("completed encoder"),
            })
            .collect();
        checkpoint(Stage::Inference)?;
        let evaluated = self.evaluator.evaluate(&inputs, &self.control);
        runtime_returned();
        self.control.check()?;
        let output = evaluated?;
        crate::require(output.evaluations.len() == infos.len(), "leaf reply count")?;
        let b = &output.stats;
        self.stats.packing_ns += b.packing_ns;
        self.stats.binding_ns += b.binding_ns;
        self.stats.inference_ns += b.inference_ns;
        self.stats.output_copy_ns += b.output_copy_ns;
        self.stats.postprocess_ns += b.postprocess_ns;
        self.stats.input_copy_bytes += b.input_copy_bytes;
        self.stats.output_copy_bytes += b.output_copy_bytes;
        self.stats.batches += b.batches;
        self.stats.positions += b.positions;
        self.stats.valid_cells += b.valid_cells;
        self.stats.padded_cells += b.padded_cells;
        self.stats.max_batch = self.stats.max_batch.max(b.max_batch);
        output
            .evaluations
            .into_iter()
            .zip(ids)
            .zip(infos)
            .map(|((row, id), info)| {
                crate::require(
                    row.id == id && row.choice_ids.as_slice() == info.actions.as_slice(),
                    "stale/misrouted cooperative reply",
                )?;
                Statistics::new(
                    row.logits.into_iter().map(f64::from).collect(),
                    row.utility.map(f64::from),
                )
            })
            .collect()
    }
}
