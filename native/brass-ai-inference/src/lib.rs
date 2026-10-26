//! Session-owned CPU evaluation. No inference threads or cross-session batching.
pub mod batch;
pub mod manifest;
pub mod math;
pub mod runtime;
pub mod search;
pub mod service;

use brass_ai_core::{
    evaluation::{Evaluation, EvaluationId, EvaluationStats},
    features::{FeatureContext, FeatureTensors},
    state::dto::Choice,
    CoreError,
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};

pub type Result<T> = std::result::Result<T, CoreError>;
pub fn invalid(message: impl Into<String>) -> CoreError {
    CoreError::Invalid(message.into())
}
pub(crate) fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(invalid(message))
    }
}
pub fn nanos(since: Instant) -> u64 {
    since.elapsed().as_nanos().min(u64::MAX as u128) as u64
}
pub fn process_clock_ns() -> u64 {
    static ORIGIN: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    nanos(*ORIGIN.get_or_init(Instant::now))
}

/// The caller owns the clock. Cancellation retires results, not an active ORT call.
#[derive(Clone)]
pub struct Control {
    pub deadline: Instant,
    pub cancelled: Arc<AtomicBool>,
}
impl Control {
    pub fn status(&self) -> Option<&'static str> {
        if self.cancelled.load(Ordering::Acquire) {
            Some("cancelled")
        } else if Instant::now() >= self.deadline {
            Some("expired")
        } else {
            None
        }
    }
    pub fn check(&self) -> Result<()> {
        self.status().map_or(Ok(()), |s| Err(invalid(s)))
    }
}

/// All borrows remain alive until the synchronous evaluator returns. Neither
/// implementation can retain them in a runtime task, cache or background thread.
pub struct Input<'a> {
    pub id: &'a EvaluationId,
    pub context: &'a FeatureContext,
    pub choices: &'a [Choice],
    pub tensors: &'a FeatureTensors,
}
pub struct BatchOutput {
    pub evaluations: Vec<Evaluation>,
    pub stats: EvaluationStats,
}
pub trait Evaluator {
    fn manifest(&self) -> &manifest::Manifest;
    fn evaluate(&mut self, inputs: &[Input<'_>], control: &Control) -> Result<BatchOutput>;
}
