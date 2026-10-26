//! Reference-service adapter for parity/attribution and later GPU transport.
//! A transport must return only after its borrowed inputs are no longer in use.
use crate::{
    manifest::Manifest, math::log_softmax, nanos, require, BatchOutput, Control, Evaluator, Input,
    Result,
};
use brass_ai_core::evaluation::{Evaluation, EvaluationId, EvaluationStats};
use std::time::Instant;

pub struct RawRow {
    pub id: EvaluationId,
    pub choice_ids: Vec<String>,
    pub logits: Vec<f32>,
    pub values: [[f32; 4]; 4],
}
pub trait ReferenceTransport {
    fn evaluate(&mut self, inputs: &[Input<'_>], control: &Control) -> Result<Vec<RawRow>>;
}
pub struct ReferenceEvaluator<T> {
    pub manifest: Manifest,
    pub transport: T,
}
impl<T: ReferenceTransport> Evaluator for ReferenceEvaluator<T> {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn evaluate(&mut self, inputs: &[Input<'_>], control: &Control) -> Result<BatchOutput> {
        validate_batch(&self.manifest, inputs)?;
        control.check()?;
        let start = Instant::now();
        let raw = self.transport.evaluate(inputs, control)?;
        control.check()?;
        let stats = EvaluationStats {
            inference_ns: nanos(start),
            ..Default::default()
        };
        finish(inputs, raw, stats)
    }
}
pub fn validate_batch(manifest: &Manifest, inputs: &[Input<'_>]) -> Result<()> {
    require(
        !inputs.is_empty() && inputs.len() <= 4,
        "invalid evaluation batch count",
    )?;
    for (i, input) in inputs.iter().enumerate() {
        manifest.validate_input(input)?;
        require(
            input.id.generation == inputs[0].id.generation
                && inputs[..i]
                    .iter()
                    .all(|p| p.id.request_id != input.id.request_id),
            "mixed generation/duplicate request",
        )?;
    }
    Ok(())
}
pub fn finish(
    inputs: &[Input<'_>],
    raw: Vec<RawRow>,
    mut stats: EvaluationStats,
) -> Result<BatchOutput> {
    require(
        raw.len() == inputs.len(),
        "reference response row count mismatch",
    )?;
    let start = Instant::now();
    let mut evaluations = Vec::with_capacity(inputs.len());
    for (input, row) in inputs.iter().zip(raw) {
        require(
            row.id == *input.id
                && row
                    .choice_ids
                    .iter()
                    .map(String::as_str)
                    .eq(input.choices.iter().map(|c| c.id.as_str()))
                && row.logits.len() == input.choices.len(),
            "stale/misrouted evaluation reply",
        )?;
        require(
            row.values.iter().flatten().all(|x| x.is_finite()),
            "nonfinite value head",
        )?;
        let (log_probabilities, probabilities) = log_softmax(&row.logits)?;
        evaluations.push(Evaluation {
            id: row.id,
            choice_ids: row.choice_ids,
            logits: row.logits,
            log_probabilities,
            probabilities,
            win_share: row.values[0],
            placement: row.values[1],
            vp: row.values[2],
            utility: row.values[3],
        });
    }
    stats.postprocess_ns += nanos(start);
    stats.positions = inputs.len();
    stats.batches = 1;
    stats.max_batch = inputs.len();
    stats.valid_cells = inputs.iter().map(|i| i.choices.len()).sum();
    stats.padded_cells = inputs.len() * inputs.iter().map(|i| i.choices.len()).max().unwrap_or(0);
    Ok(BatchOutput { evaluations, stats })
}
