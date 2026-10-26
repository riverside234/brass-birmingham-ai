//! Versioned evaluation envelopes. No runtime dependency in the rules core.
use crate::{
    features::FeatureContext,
    state::{dto::Choice, observation::Observation},
};
use serde::{Deserialize, Serialize};

pub const FORMAT: &str = "brass-native-evaluation/1.0.0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationId {
    pub request_id: String,
    pub generation: u64,
    pub model_id: String,
    pub export_sha256: String,
    pub reward_id: String,
    pub profile_id: String,
    pub revision: u32,
    pub observer: u8,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FeatureSource {
    Permitted {
        view: Box<Observation>,
        choices: Vec<Choice>,
    },
    State {
        handle: u64,
        revision: u32,
    },
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationRequest {
    pub id: EvaluationId,
    pub context: Box<FeatureContext>,
    pub source: FeatureSource,
}
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchPolicy {
    pub cap: usize,
    pub pending_limit: usize,
    pub max_candidate_cells: usize,
    pub flush_us: u64,
}
impl Default for BatchPolicy {
    fn default() -> Self {
        Self {
            cap: 1,
            pending_limit: 1,
            max_candidate_cells: 8192,
            flush_us: 0,
        }
    }
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum InferenceCommand {
    Load {
        manifest: String,
        manifest_sha256: String,
    },
    Start {
        generation: u64,
        batch_id: String,
        budget_ms: u64,
        policy: BatchPolicy,
        requests: Vec<EvaluationRequest>,
    },
    Poll {
        ticket: u64,
    },
    Cancel {
        ticket: u64,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evaluation {
    pub id: EvaluationId,
    pub choice_ids: Vec<String>,
    /// All finite legal logits, in complete candidate order. No probability floor.
    pub logits: Vec<f32>,
    pub log_probabilities: Vec<f64>,
    pub probabilities: Vec<f64>,
    /// All four value heads are already in stable seat order in this export.
    pub win_share: [f32; 4],
    pub placement: [f32; 4],
    pub vp: [f32; 4],
    pub utility: [f32; 4],
}
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct EvaluationStats {
    /// Shared process monotonic origin, for overlap diagnostics only.
    pub work_started_ns: u64,
    pub work_finished_ns: u64,
    pub dispatch_ns: u64,
    pub encoding_ns: u64,
    pub queue_ns: u64,
    pub packing_ns: u64,
    pub binding_ns: u64,
    pub inference_ns: u64,
    pub output_copy_ns: u64,
    pub postprocess_ns: u64,
    pub input_copy_bytes: usize,
    pub output_copy_bytes: usize,
    pub batches: usize,
    pub positions: usize,
    pub valid_cells: usize,
    pub padded_cells: usize,
    pub max_batch: usize,
}
#[derive(Debug, Serialize)]
pub struct InferenceResponse {
    pub format: &'static str,
    pub state: &'static str,
    pub owner: String,
    pub ticket: u64,
    pub generation: u64,
    pub batch_id: String,
    pub evaluations: Vec<Evaluation>,
    pub stats: EvaluationStats,
    pub model: Option<serde_json::Value>,
}
