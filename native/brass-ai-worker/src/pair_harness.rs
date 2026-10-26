//! Bounded L12 research harness. No live game actions or protocol admission.
use crate::pair::Pair;
use brass_ai_core::{
    evaluation::BatchPolicy,
    search::{
        self,
        brass::Input,
        session::{Identity, Mode},
        Config,
    },
    work::{Probe, Stage},
};
use brass_ai_inference::{
    manifest::{Bundle, Manifest},
    runtime::CpuEvaluator,
    service::{self, RawRow},
    BatchOutput, Control, Evaluator,
};
use serde::{Deserialize, Serialize};
use stats_alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM};
use std::{
    alloc::System,
    io::{BufWriter, Write},
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

// Diagnostic proofs can be large. Serialize a borrowed row directly, release
// it before the next admission, and publish the file only after every case.
struct PendingOutput(PathBuf);
impl Drop for PendingOutput {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum CaseResult {
    Complete {
        decision: Box<crate::pair::Decision>,
    },
    Retired {
        error: String,
    },
}
#[derive(Serialize)]
struct AllocationTraffic {
    allocations: usize,
    reallocations: usize,
    bytes_allocated: usize,
    bytes_deallocated: usize,
}
#[derive(Serialize)]
struct Measurement {
    name: String,
    result: CaseResult,
    elapsed_ns: u64,
    cancelled: bool,
    busy_after_cancel: bool,
    blocked_admission: bool,
    busy_after_retirement: bool,
    allocations: AllocationTraffic,
    allocation_probe: bool,
    policy: BatchPolicy,
    allowance_ms: u64,
}

#[cfg(test)]
mod tests;
type Result<T> = std::result::Result<T, brass_ai_core::CoreError>;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    manifest: PathBuf,
    manifest_sha256: String,
    #[serde(default)]
    scripted: bool,
    #[serde(default)]
    delay_ms: u64,
    cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    input: Input,
    config: Config,
    policy: BatchPolicy,
    mode: Mode,
    allowance_ms: u64,
    #[serde(default)]
    cancel_after_ms: Option<u64>,
    #[serde(default)]
    faults: [Option<Fault>; 2],
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fault {
    stage: Stage,
    after: u64,
}
enum Engine {
    Native(Box<CpuEvaluator>),
    Scripted { manifest: Box<Manifest>, delay: u64 },
}
impl Evaluator for Engine {
    fn manifest(&self) -> &Manifest {
        match self {
            Self::Native(e) => e.manifest(),
            Self::Scripted { manifest, .. } => manifest,
        }
    }
    fn evaluate(
        &mut self,
        inputs: &[brass_ai_inference::Input<'_>],
        control: &Control,
    ) -> Result<BatchOutput> {
        match self {
            Self::Native(e) => e.evaluate(inputs, control),
            Self::Scripted { manifest, delay } => {
                service::validate_batch(manifest, inputs)?;
                control.check()?;
                let begin = Instant::now();
                // Deliberately noncooperative runtime for retirement tests.
                if *delay > 0 {
                    std::thread::sleep(Duration::from_millis(*delay));
                }
                let rows = inputs
                    .iter()
                    .map(|i| RawRow {
                        id: i.id.clone(),
                        choice_ids: i.choices.iter().map(|c| c.id.clone()).collect(),
                        logits: i
                            .choices
                            .iter()
                            .map(|c| if c.id.starts_with("pass|") { 1. } else { 0. })
                            .collect(),
                        values: [[0.25; 4], [0.; 4], [0.; 4], [0.4, 0.3, 0.2, 0.1]],
                    })
                    .collect();
                service::finish(
                    inputs,
                    rows,
                    brass_ai_core::evaluation::EvaluationStats {
                        inference_ns: brass_ai_inference::nanos(begin),
                        ..Default::default()
                    },
                )
            }
        }
    }
}
pub fn run(instrumented: bool) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: gumbel-pair INPUT.json OUTPUT.json".into());
    }
    let source = PathBuf::from(&args[0]);
    search::check(
        source.metadata()?.len() <= 32 * 1024 * 1024,
        "request bound",
    )?;
    let request: Request = serde_json::from_slice(&std::fs::read(source)?)?;
    search::check(
        !request.cases.is_empty() && request.cases.len() <= 256 && request.delay_ms <= 5000,
        "case/delay bound",
    )?;
    let bundle = Bundle::load(&request.manifest, &request.manifest_sha256)?;
    let export = bundle.manifest.export_sha256.clone();
    let model = bundle.manifest.clone();
    drop(bundle);
    let manifest = request.manifest;
    let digest = request.manifest_sha256;
    let scripted = request.scripted;
    let delay = request.delay_ms;
    let mut pair = Pair::new(move |_| {
        if scripted {
            Ok(Engine::Scripted {
                manifest: Box::new(model.clone()),
                delay,
            })
        } else {
            Ok(Engine::Native(Box::new(CpuEvaluator::load(Bundle::load(
                &manifest, &digest,
            )?)?)))
        }
    })?;
    let destination = PathBuf::from(&args[1]);
    search::check(!destination.exists(), "output already exists")?;
    let staging = destination.with_extension("partial");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staging)?;
    let pending_output = PendingOutput(staging);
    let mut writer = BufWriter::with_capacity(65536, file);
    writer.write_all(b"{\"identity\":\"brass-gumbel-is-mcts/coordinator-v1\",\"scripted\":")?;
    serde_json::to_writer(&mut writer, &scripted)?;
    writer.write_all(b",\"results\":[")?;
    for (index, case) in request.cases.into_iter().enumerate() {
        search::check(
            case.name.len() <= 256 && case.allowance_ms <= 120000,
            "case bounds",
        )?;
        let identity = Identity {
            game_instance: format!("l12-{}", case.name),
            generation: index as u64 + 1,
            export_sha256: export.clone(),
        };
        let config = case.config.clone();
        let probes = case.faults.map(|f| {
            f.map(|f| {
                let counter = AtomicU64::new(0);
                Arc::new(move |stage| {
                    if stage == f.stage && counter.fetch_add(1, Ordering::Relaxed) + 1 == f.after {
                        Err(search::invalid("injected stage failure"))
                    } else {
                        Ok(())
                    }
                }) as Probe
            })
        });
        let region = Region::new(GLOBAL);
        let began = Instant::now();
        pair.admit(
            identity.clone(),
            &case.input,
            case.config,
            case.allowance_ms * 1_000_000,
        )?;
        let ticket = pair.start(
            case.input.clone(),
            case.mode.clone(),
            case.policy,
            case.allowance_ms * 1_000_000,
            probes,
        )?;
        let mut cancelled = false;
        let mut busy_after_cancel = false;
        let mut blocked_admission = false;
        let outcome = loop {
            if !cancelled
                && case
                    .cancel_after_ms
                    .is_some_and(|n| began.elapsed() >= Duration::from_millis(n))
            {
                pair.cancel(ticket)?;
                cancelled = true;
                busy_after_cancel = pair.busy();
                blocked_admission = pair
                    .admit(
                        identity.clone(),
                        &case.input,
                        config.clone(),
                        case.allowance_ms * 1_000_000,
                    )
                    .is_err();
            }
            match pair.poll(ticket) {
                Ok(None) => std::thread::sleep(Duration::from_millis(1)),
                Ok(Some(r)) => break Ok(r),
                Err(e) => break Err(e.to_string()),
            }
        };
        let elapsed_ns = brass_ai_inference::nanos(began);
        let allocated = region.change();
        search::check(!pair.busy(), "retirement leak")?;
        let status = if outcome.is_ok() {
            "complete"
        } else {
            "retired"
        };
        let result = match outcome {
            Ok(r) => CaseResult::Complete {
                decision: Box::new(r),
            },
            Err(error) => CaseResult::Retired { error },
        };
        let measurement = Measurement {
            name: case.name,
            result,
            elapsed_ns,
            cancelled,
            busy_after_cancel,
            blocked_admission,
            busy_after_retirement: pair.busy(),
            allocations: AllocationTraffic {
                allocations: allocated.allocations,
                reallocations: allocated.reallocations,
                bytes_allocated: allocated.bytes_allocated,
                bytes_deallocated: allocated.bytes_deallocated,
            },
            allocation_probe: instrumented,
            policy: case.policy,
            allowance_ms: case.allowance_ms,
        };
        if index > 0 {
            writer.write_all(b",")?;
        }
        serde_json::to_writer(&mut writer, &measurement)?;
        eprintln!("{}: {status}", measurement.name);
    }
    writer.write_all(b"]}")?;
    writer.flush()?;
    drop(writer);
    std::fs::rename(&pending_output.0, destination)?;
    Ok(())
}
