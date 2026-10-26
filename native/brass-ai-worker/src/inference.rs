//! Owner-thread implementation and non-computing control-thread tickets.
use brass_ai_core::{
    evaluation::{
        BatchPolicy, EvaluationRequest, EvaluationStats, FeatureSource, InferenceCommand,
        InferenceResponse, FORMAT,
    },
    features::FeatureEncoder,
    protocol::Response,
    state::simulation::Simulator,
    CoreError,
};
use brass_ai_inference::{
    batch::{validate_policy, Pending, Queue},
    invalid,
    manifest::{label, Bundle},
    nanos,
    runtime::CpuEvaluator,
    Control, Evaluator, Input,
};
use crossbeam_channel::{bounded, Receiver, TryRecvError};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

pub struct Run {
    pub ticket: u64,
    pub generation: u64,
    pub batch_id: String,
    pub policy: BatchPolicy,
    pub requests: Vec<EvaluationRequest>,
    pub control: Control,
    pub submitted: Instant,
}
#[derive(Default)]
pub struct Engine {
    model: Option<CpuEvaluator>,
    encoders: [FeatureEncoder; 4],
}
pub fn response(
    state: &'static str,
    ticket: u64,
    generation: u64,
    batch_id: String,
    owner: &str,
) -> InferenceResponse {
    InferenceResponse {
        format: FORMAT,
        state,
        ticket,
        generation,
        batch_id,
        owner: owner.into(),
        evaluations: Vec::new(),
        stats: EvaluationStats::default(),
        model: None,
    }
}
impl Engine {
    pub fn load(
        &mut self,
        path: String,
        digest: String,
        owner: &str,
    ) -> Result<InferenceResponse, CoreError> {
        if let Some(model) = &self.model {
            if model.receipt()["manifest_sha256"] != digest {
                return Err(invalid("model is frozen; use a new worker to change it"));
            }
        } else {
            self.model = Some(CpuEvaluator::load(Bundle::load(
                Path::new(&path),
                &digest,
            )?)?);
        }
        let mut reply = response("loaded", 0, 0, String::new(), owner);
        reply.model = self.model.as_ref().map(CpuEvaluator::receipt);
        Ok(reply)
    }
    pub fn run(
        &mut self,
        run: Run,
        simulator: &Simulator,
        owner: &str,
    ) -> Result<InferenceResponse, CoreError> {
        let mut reply = response(
            "complete",
            run.ticket,
            run.generation,
            run.batch_id.clone(),
            owner,
        );
        reply.stats.dispatch_ns = nanos(run.submitted);
        reply.stats.work_started_ns = brass_ai_inference::process_clock_ns();
        let result = (|| {
            run.control.check()?;
            let model = self
                .model
                .as_mut()
                .ok_or_else(|| invalid("model must be loaded and warmed before timed work"))?;
            let mut rows = Vec::with_capacity(run.requests.len());
            for request in run.requests {
                let (view, choices) = match request.source {
                    FeatureSource::Permitted { view, choices } => (*view, choices),
                    FeatureSource::State { handle, revision } => {
                        simulator.feature_input(handle, revision)?
                    }
                };
                if request.id.revision != view.context.revision
                    || request.id.observer != u8::from(view.seat)
                {
                    return Err(invalid("evaluation revision/observer mismatch"));
                }
                rows.push((request.id, request.context, view, choices));
            }
            let mut queue = Queue::new(run.policy)?;
            for i in 0..rows.len() {
                run.control.check()?;
                let start = Instant::now();
                let (id, context, view, choices) = &rows[i];
                self.encoders[i].encode(view, choices, context, &mut || run.control.check())?;
                reply.stats.encoding_ns += nanos(start);
                model.manifest().validate_input(&Input {
                    id,
                    context,
                    choices,
                    tensors: self.encoders[i]
                        .tensors()
                        .ok_or_else(|| invalid("encoder incomplete"))?,
                })?;
                queue.push(Pending {
                    slot: i,
                    candidates: choices.len(),
                    submitted: Instant::now(),
                    deadline: run.control.deadline,
                })?;
                // Wait only while this same thread can produce another leaf.
                // The last producer always flushes; there is no batching sleep.
                loop {
                    let pending = queue.take_ready(Instant::now(), i + 1 < rows.len());
                    if pending.is_empty() {
                        break;
                    }
                    let now = Instant::now();
                    let inputs: Vec<_> = pending
                        .iter()
                        .map(|p| {
                            let (id, context, _, choices) = &rows[p.slot];
                            Input {
                                id,
                                context,
                                choices,
                                tensors: self.encoders[p.slot]
                                    .tensors()
                                    .expect("encoded pending slot"),
                            }
                        })
                        .collect();
                    let out = model.evaluate(&inputs, &run.control)?;
                    reply.stats.queue_ns += pending
                        .iter()
                        .map(|p| now.saturating_duration_since(p.submitted).as_nanos() as u64)
                        .sum::<u64>();
                    add(&mut reply.stats, &out.stats);
                    reply.evaluations.extend(out.evaluations);
                }
            }
            run.control.check()?;
            Ok(())
        })();
        if let Some(status) = run.control.status() {
            reply.state = status;
            reply.evaluations.clear();
        } else {
            result?;
        }
        reply.stats.work_finished_ns = brass_ai_inference::process_clock_ns();
        Ok(reply)
    }
}
fn add(a: &mut EvaluationStats, b: &EvaluationStats) {
    a.packing_ns += b.packing_ns;
    a.binding_ns += b.binding_ns;
    a.inference_ns += b.inference_ns;
    a.output_copy_ns += b.output_copy_ns;
    a.postprocess_ns += b.postprocess_ns;
    a.input_copy_bytes += b.input_copy_bytes;
    a.output_copy_bytes += b.output_copy_bytes;
    a.batches += b.batches;
    a.positions += b.positions;
    a.valid_cells += b.valid_cells;
    a.padded_cells += b.padded_cells;
    a.max_batch = a.max_batch.max(b.max_batch);
}

pub struct Active {
    ticket: u64,
    generation: u64,
    batch_id: String,
    control: Control,
    result: Receiver<Result<Response, CoreError>>,
}
impl Active {
    pub fn cancel(&self) {
        self.control.cancelled.store(true, Ordering::Release);
    }
}
impl super::Server {
    pub fn inference(
        &mut self,
        session: u8,
        command: InferenceCommand,
    ) -> Result<Response, CoreError> {
        let s = session as usize;
        if s >= 2 {
            return Err(invalid("invalid inference session"));
        }
        let owners = self.owners.as_ref().ok_or(CoreError::SessionFailed)?;
        match command {
            InferenceCommand::Load {
                manifest,
                manifest_sha256,
            } => {
                if self.active[s].is_some() {
                    return Err(invalid("session busy; drain its ticket before loading"));
                }
                owners[s].call(super::Command::LoadModel(manifest, manifest_sha256))
            }
            InferenceCommand::Start {
                generation,
                batch_id,
                budget_ms,
                policy,
                requests,
            } => {
                if self.active[s].is_some() {
                    return Err(invalid(
                        "inference backpressure: session busy until ticket is drained",
                    ));
                }
                validate_policy(policy)?;
                if generation <= self.generations[s]
                    || !label(&batch_id)
                    || budget_ms > 60_000
                    || requests.is_empty()
                    || requests.len() > policy.pending_limit
                    || requests.iter().any(|r| r.id.generation != generation)
                {
                    return Err(invalid("invalid inference generation/budget/batch"));
                }
                let mut ids = std::collections::HashSet::new();
                if !requests
                    .iter()
                    .all(|r| label(&r.id.request_id) && ids.insert(&r.id.request_id))
                {
                    return Err(invalid("duplicate/invalid evaluation request ID"));
                }
                self.sequence = self
                    .sequence
                    .checked_add(1)
                    .filter(|x| *x < u64::MAX / 2)
                    .ok_or_else(|| invalid("ticket exhausted"))?;
                let ticket = self.sequence * 2 + u64::from(session);
                let submitted = Instant::now();
                let control = Control {
                    deadline: submitted + Duration::from_millis(budget_ms),
                    cancelled: Arc::new(AtomicBool::new(false)),
                };
                let (sender, result) = bounded(1);
                let run = Run {
                    ticket,
                    generation,
                    batch_id: batch_id.clone(),
                    policy,
                    requests,
                    control: control.clone(),
                    submitted,
                };
                owners[s]
                    .sender
                    .as_ref()
                    .ok_or(CoreError::SessionFailed)?
                    .try_send((super::Command::Evaluate(Box::new(run)), sender))
                    .map_err(|_| CoreError::SessionFailed)?;
                self.generations[s] = generation;
                self.active[s] = Some(Active {
                    ticket,
                    generation,
                    batch_id: batch_id.clone(),
                    control,
                    result,
                });
                Ok(Response::Inference {
                    session,
                    data: Box::new(response("running", ticket, generation, batch_id, "control")),
                })
            }
            InferenceCommand::Poll { ticket } | InferenceCommand::Cancel { ticket } => {
                let active = self.active[s]
                    .as_ref()
                    .filter(|a| a.ticket == ticket)
                    .ok_or_else(|| invalid("stale or foreign inference ticket"))?;
                if matches!(command, InferenceCommand::Cancel { .. }) {
                    active.cancel();
                    return Ok(Response::Inference {
                        session,
                        data: Box::new(response(
                            "cancelling",
                            ticket,
                            active.generation,
                            active.batch_id.clone(),
                            "control",
                        )),
                    });
                }
                match active.result.try_recv() {
                    Ok(result) => {
                        let active = self.active[s].take().expect("matched active ticket");
                        let mut reply = result?;
                        if let Response::Inference { data, .. } = &mut reply {
                            if data.ticket != active.ticket
                                || data.generation != active.generation
                                || data.batch_id != active.batch_id
                            {
                                return Err(CoreError::SessionFailed);
                            }
                            if let Some(status) = active.control.status() {
                                data.state = status;
                                data.evaluations.clear();
                            }
                        } else {
                            return Err(CoreError::SessionFailed);
                        }
                        Ok(reply)
                    }
                    Err(TryRecvError::Empty) => Ok(Response::Inference {
                        session,
                        data: Box::new(response(
                            "running",
                            ticket,
                            active.generation,
                            active.batch_id.clone(),
                            "control",
                        )),
                    }),
                    Err(TryRecvError::Disconnected) => {
                        self.active[s] = None;
                        Err(CoreError::SessionFailed)
                    }
                }
            }
        }
    }
}
