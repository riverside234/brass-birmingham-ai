//! Part 1 local transport. Research inputs, scripted inference and full states
//! are deliberately absent from this protocol. One Pair owns both CPU lanes.
use crate::pair::{Decision, Pair};
use brass_ai_core::{
    evaluation::BatchPolicy,
    framing::{read_frame, write_frame},
    protocol::BuildIdentity,
    search::{
        self,
        brass::Input,
        session::{Identity, Mode, Startup, TurnKey},
        Config,
    },
    CoreError,
};
use brass_ai_inference::{
    manifest::{self, Bundle, Manifest},
    runtime::CpuEvaluator,
};
use crossbeam_channel::{bounded, RecvTimeoutError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    io,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

pub const PROTOCOL: &str = "brass-local-native/1.0.0";
pub use crate::local_model::{BACKEND, MODEL, EXPORT, CHECKPOINT, PROFILE, PROFILE_PREFIX, INDUSTRY_FOCUSES};
pub use crate::local_model::REWARD;
// The selected allowance is entirely search work. A single additional second
// covers retirement, legal completion and host delivery on the original clock.
const DELIVERY_GRACE_MS: u64 = 1_000;
const DELIVERY_GRACE_NS: u64 = DELIVERY_GRACE_MS * 1_000_000;
type Result<T> = std::result::Result<T, CoreError>;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Acceptance {
    pub version: String,
    pub scope: String,
    pub decision: String,
    pub model_id: String,
    pub export_sha256: String,
    pub checkpoint_sha256: String,
    #[serde(deserialize_with = "required_calibration")]
    pub utility_calibration: Option<Value>,
    pub final_model: bool,
}
fn required_calibration<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<Value>, D::Error> {
    Option::<Value>::deserialize(d)
}
impl Acceptance {
    pub fn validate(&self, model: &Manifest) -> Result<()> {
        search::check(
            self.version == "brass-local-utility-acceptance/1.0.0"
                && self.scope == crate::local_model::SCOPE
                && self.decision == crate::local_model::DECISION
                && self.model_id == MODEL
                && self.export_sha256 == EXPORT
                && self.checkpoint_sha256 == CHECKPOINT
                && model.model_id == self.model_id
                && model.export_sha256 == self.export_sha256
                && model.checkpoint_sha256 == self.checkpoint_sha256
                && !self.final_model
                && self.utility_calibration == model.utility_calibration
                && if crate::local_model::BETA {
                    !model.mechanical_only && self.utility_calibration.as_ref().is_some_and(|r|
                        r["request_id"] == crate::local_model::CALIBRATION_REQUEST
                        && r["source_checkpoint_sha256"] == crate::local_model::SOURCE_CHECKPOINT
                        && r["source_model_id"] == MODEL && r["reward_id"] == REWARD
                        && r["mode"] == "read-only-validation" && r["ready"] == true
                        && r["completed"] == true && r["weights_unchanged"] == true
                        && r["steps"].as_u64() == Some(0))
                } else { model.mechanical_only && self.utility_calibration.is_none() },
            "model lacks exact local utility acceptance",
        )
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Package {
    pub protocol: String,
    pub backend: String,
    pub model_manifest_sha256: String,
    pub acceptance: Acceptance,
    pub config: Config,
    pub k_by_allowance_ms: BTreeMap<u64, usize>,
    pub startup_ns: u64,
    pub startup_source: String,
    pub warmup_sha256: String,
    #[serde(default)]
    pub method: search::session::focus::Method,
    #[serde(default = "default_temperature")]
    pub policy_temperature: f64,
}
fn default_temperature() -> f64 {
    1.0
}
impl Package {
    pub fn load(path: &Path, digest: &str) -> Result<(Self, PathBuf)> {
        search::check(
            std::fs::metadata(path)
                .map_err(|e| search::invalid(e.to_string()))?
                .len()
                <= 65536,
            "package size",
        )?;
        let bytes = std::fs::read(path).map_err(|e| search::invalid(e.to_string()))?;
        search::check(
            bytes.len() <= 65536 && manifest::hash(&bytes) == digest,
            "package identity",
        )?;
        let p: Self = serde_json::from_slice(&bytes).map_err(|e| search::invalid(e.to_string()))?;
        crate::local_temperature::validate(p.policy_temperature)?;
        search::check(
            p.protocol == PROTOCOL
                && p.backend == BACKEND
                && !p.config.trace
                && p.config.simulations == 64
                && p.config.k == 8
                && p.startup_ns >= 1_000_000
                && p.startup_ns <= 100_000_000
                && manifest::label(&p.startup_source),
            "unsupported local search settings",
        )?;
        p.config.validate()?;
        search::check(
            p.k_by_allowance_ms
                .keys()
                .copied()
                .eq([4000, 8000, 10000, 12000]),
            "missing or unsupported search allowance",
        )?;
        for &allowance in p.k_by_allowance_ms.keys() {
            p.search_config(allowance)?.validate()?;
        }
        let model = path
            .parent()
            .ok_or_else(|| search::invalid("package directory"))?
            .join("model/manifest.json");
        p.acceptance
            .validate(&Bundle::load(&model, &p.model_manifest_sha256)?.manifest)?;
        Ok((p, model))
    }

    fn search_config(&self, allowance_ms: u64) -> Result<Config> {
        let mut config = self.config.clone();
        // Use the seat's whole-turn setting, never its diminishing remainder.
        config.k = *self
            .k_by_allowance_ms
            .get(&allowance_ms)
            .ok_or_else(|| search::invalid("unsupported search allowance"))?;
        Ok(config)
    }
}
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Ticket {
    pub game_instance: String,
    pub game_id: String,
    pub generation: u64,
    pub revision: u32,
    pub actor: u8,
    pub profile: String,
    pub root_sha256: String,
    pub source_sha256: String,
}
#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Health,
    Search {
        ticket: Ticket,
        allowance_ms: u64,
        remaining_ms: u64,
        input_json: String,
        #[serde(default)]
        policy_temperature: Option<f64>,
    },
    Cancel {
        target: u64,
    },
    Shutdown,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub protocol: String,
    pub id: u64,
    pub request: Operation,
}
fn emit(id: u64, status: &str, data: Value) -> io::Result<()> {
    crate::local_profile::event(id, "reply_serialize_begin", crate::local_profile::now(), 0);
    let payload =
        serde_json::to_vec(&json!({"protocol":PROTOCOL,"id":id,"status":status,"data":data}))?;
    crate::local_profile::event(
        id,
        "reply_write_begin",
        crate::local_profile::now(),
        payload.len(),
    );
    let result = write_frame(&mut io::stdout().lock(), &payload);
    crate::local_profile::event(
        id,
        "reply_write_end",
        crate::local_profile::now(),
        payload.len(),
    );
    result
}
fn focus_context(profile: &str) -> Option<([f64; 4], [f64; 8])> {
    let name = profile
        .strip_prefix(PROFILE_PREFIX)?
        .strip_suffix("/1.0.0")?;
    let index = INDUSTRY_FOCUSES.iter().position(|f| *f == name)?;
    let mut style = [0.; 4];
    let mut intent = [0.; 8];
    if index > 0 {
        style[if index <= 3 { 0 } else { 2 }] = 1.;
        intent[index - 1] = 1.;
    }
    Some((style, intent))
}
fn validate_input(ticket: &Ticket, input: &Input, allowance: u64, remaining: u64) -> Result<()> {
    search::check(
        manifest::label(&ticket.game_instance)
            && manifest::label(&ticket.game_id)
            && ticket.generation > 0
            && ticket.revision == input.view.context.revision
            && ticket.actor == u8::from(input.view.seat)
            && ticket.actor < 4
            && ticket.profile == input.context.profile_id
            && manifest::digest(&ticket.root_sha256)
            && ticket.source_sha256 == BuildIdentity::current().source_sha256
            && [4000, 8000, 10000, 12000].contains(&allowance)
            && remaining > 0
            && remaining <= allowance.saturating_add(DELIVERY_GRACE_MS),
        "request identity/allowance mismatch",
    )?;
    search::check(
        input.context.model_id == MODEL
            && input.context.reward_id == REWARD
            && input.context.max_cost == crate::local_model::MAX_COST
            && focus_context(&input.context.profile_id)
                == Some((input.context.style, input.context.intent))
            && input.context.stable_descriptors.is_none()
            && input.shuffle_tape.is_none(),
        "unaccepted local model/profile or research input",
    )?;
    input.context.validate()?;
    search::check(
        input.reward.identity()? == input.context.reward_id,
        "reward identity",
    )
}
struct Active {
    id: u64,
    pair_id: u64,
    ticket: Ticket,
    deadline: Instant,
    requested_k: usize,
    temperature: f64,
}
fn compact(d: Decision, ticket: Ticket, requested_k: usize, temperature: f64) -> Value {
    let mut value = json!({"ticket":ticket,"backend":BACKEND,"model_id":MODEL,"export_sha256":EXPORT,
        "actions":d.result.action_ids,"provenance":d.result.provenance,
        "completed":d.lanes.iter().map(|l| l.outcome.completed).sum::<u64>(),
        "sessions":2,"threads_per_session":1,"retired":true,"requested_k":requested_k,
        "policy_temperature":temperature,"temperature_scope":"all_nn_policy_evaluations",
        "industry_focus":ticket.profile.strip_prefix(PROFILE_PREFIX).and_then(|s| s.strip_suffix("/1.0.0")),
        "replay":{
            "version":"brass-local-replay/1.0.0",
            "root_epoch":d.result.root_epoch,
            "supporting_root_epoch":d.result.supporting_root_epoch,
            "root_gumbels":d.result.gumbels,
            "owners":d.lanes.iter().enumerate().map(|(owner,lane)|json!({
                "owner":owner,"root_epoch":lane.outcome.root_epoch,
                "completed":lane.outcome.completed,"attempted":lane.outcome.attempted,
                "partial_completed":lane.outcome.partial_completed,"quota":lane.outcome.quota,
                "effective_k":lane.outcome.effective_k,
                "snapshot":lane.outcome.snapshot,"work_stopped_ns":lane.outcome.work_stopped_ns,
                "last_completion_ns":lane.outcome.last_completion_ns,
                "retired_pending":lane.outcome.retired_pending,
                "seed_xor":if owner==0 {0} else {0xd1b54a32d192ed03_u64},
                "sample_seed_xor":if owner==0 {0} else {0x9e3779b97f4a7c15_u64}
            })).collect::<Vec<_>>()},
        "response_ready_ns":d.response_ready_ns,"deadline_ns":d.result.budget.deadline_ns});
    if let Some(focus) = d.focus {
        value["search_revision"] = json!({"version":"brass-local-focus/2.0.0", "focus":focus,
            "focused_owners":d.focused_lanes.map(|lanes| lanes.map(|l| json!({
                "epoch":2,"completed":l.outcome.completed,"effective_k":l.outcome.effective_k,
                "cache_hits":l.outcome.cache_hits,"watermark":l.outcome.snapshot.as_ref().map(|s|s.support_watermark),
                "last_completion_ns":l.outcome.last_completion_ns,"work_stopped_ns":l.outcome.work_stopped_ns}))) });
    }
    value
}

pub fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: brass-ai-local PACKAGE.json SHA256".into());
    }
    let package_path = PathBuf::from(&args[0]);
    let (package, model_path) = Package::load(&package_path, &args[1].to_string_lossy())?;
    let digest = package.model_manifest_sha256.clone();
    let package_temperature = package.policy_temperature;
    let evaluation_temperature = Arc::new(AtomicU32::new((package_temperature as f32).to_bits()));
    let owner_temperature = evaluation_temperature.clone();
    let mut pair = Pair::new(move |_| {
        Ok(crate::local_temperature::Tempered::new(
            CpuEvaluator::load(Bundle::load(&model_path, &digest)?)?,
            owner_temperature.clone(),
        ))
    })?;
    // Exercise real features, both owners and the pinned export before exposing
    // readiness. This public fixture has no trusted full-state material.
    let warm_path = package_path.parent().unwrap().join("warmup.json");
    search::check(
        std::fs::metadata(&warm_path)?.len() <= 1024 * 1024,
        "warmup size",
    )?;
    let warm = std::fs::read(warm_path)?;
    search::check(
        manifest::hash(&warm) == package.warmup_sha256,
        "warmup identity",
    )?;
    let warm: Input = serde_json::from_slice(&warm)?;
    pair.admit(
        Identity {
            game_instance: "startup".into(),
            generation: 1,
            export_sha256: EXPORT.into(),
        },
        &warm,
        package.config.clone(),
        60_000_000_000,
    )?;
    let id = pair.start(
        warm,
        Mode::Count,
        BatchPolicy::default(),
        60_000_000_000,
        [None, None],
    )?;
    let warmup = loop {
        if let Some(d) = pair.poll(id)? {
            break d;
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    search::check(
        warmup.lanes.iter().all(|l| l.neural.positions > 0),
        "both owners must evaluate during warmup",
    )?;
    emit(
        0,
        "ready",
        json!({"backend":BACKEND,"build":BuildIdentity::current(),"model_id":MODEL,
        "export_sha256":EXPORT,"sessions":2,"threads_per_session":1,
        "provider":"CPUExecutionProvider","utility_acceptance":package.acceptance,
        "k_by_allowance_ms":package.k_by_allowance_ms,
        "k32_host_delivery_reserve_ns":DELIVERY_GRACE_NS,
        "host_delivery_reserve_policy":"search-plus-delivery/1.0.0",
        "delivery_grace_ms":DELIVERY_GRACE_MS,
        "host_delivery_reserve_ns_by_k":{"8":DELIVERY_GRACE_NS,"16":DELIVERY_GRACE_NS,"32":DELIVERY_GRACE_NS},
        "policy_temperature":package_temperature,"temperature_scope":"all_nn_policy_evaluations",
        "request_policy_temperature":true,
        "industry_focuses":INDUSTRY_FOCUSES,
        "measured_readiness":{"warmup_passed":true,"native_nn_positions":warmup.lanes.iter().map(|l| l.neural.positions).sum::<usize>()},
        "max_frame_bytes":brass_ai_core::framing::MAX_FRAME_BYTES}),
    )?;
    drop(warmup);
    let (tx, rx) = bounded(4);
    std::thread::Builder::new()
        .name("brass-control-reader".into())
        .spawn(move || {
            let mut reader = io::stdin().lock();
            while let Ok(Some(bytes)) = read_frame(&mut reader) {
                if tx.send((bytes, crate::local_profile::now())).is_err() {
                    break;
                }
            }
        })?;
    let mut active: Option<Active> = None;
    let mut binding: Option<(Identity, TurnKey, u64, f64)> = None;
    let mut last_id = 0;
    let mut shutdown = None;
    loop {
        if let Some(a) = &active {
            if Instant::now() >= a.deadline {
                let _ = pair.cancel(a.pair_id);
            }
            match pair.poll(a.pair_id) {
                Ok(None) => (),
                result => {
                    let a = active.take().unwrap();
                    crate::local_profile::event(
                        a.id,
                        "pair_retired",
                        crate::local_profile::now(),
                        0,
                    );
                    match result {
                        Ok(Some(d)) if Instant::now() < a.deadline => {
                            emit(a.id, "ok", compact(d, a.ticket, a.requested_k, a.temperature))?
                        }
                        Ok(_) => emit(a.id, "error", json!({"error":"expired","retired":true}))?,
                        Err(e) => emit(
                            a.id,
                            "error",
                            json!({"error":e.to_string(),"retired":!pair.busy()}),
                        )?,
                    }
                }
            }
        }
        if let Some(id) = shutdown.filter(|_| active.is_none()) {
            drop(pair);
            emit(id, "stopped", json!({"retired":true}))?;
            return Ok(());
        }
        let (bytes, read_at) = match rx.recv_timeout(Duration::from_millis(1)) {
            Ok(b) => b,
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        // The caller's remaining clock includes its serialization and scheduler
        // queue. Receiving/parsing also consumes, rather than resets, that clock.
        let received = Instant::now();
        let dequeued_at = crate::local_profile::now();
        let request: Request = match serde_json::from_slice(&bytes) {
            Ok(r) => r,
            Err(_) => {
                emit(0, "error", json!({"error":"invalid request"}))?;
                continue;
            }
        };
        crate::local_profile::event(request.id, "pipe_read_complete", read_at, bytes.len());
        crate::local_profile::event(request.id, "dequeued", dequeued_at, 0);
        crate::local_profile::event(request.id, "parsed", crate::local_profile::now(), 0);
        if request.protocol != PROTOCOL || request.id <= last_id || shutdown.is_some() {
            emit(
                request.id,
                "error",
                json!({"error":"protocol/stale request"}),
            )?;
            continue;
        }
        last_id = request.id;
        let result: Result<()> = (|| match request.request {
            Operation::Health => emit(
                request.id,
                "health",
                json!({"busy":pair.busy(),"backend":BACKEND}),
            )
            .map_err(|e| search::invalid(e.to_string())),
            Operation::Cancel { target } => {
                let a = active
                    .as_ref()
                    .filter(|a| a.id == target)
                    .ok_or_else(|| search::invalid("stale cancel"))?;
                pair.cancel(a.pair_id)?;
                emit(
                    request.id,
                    "cancelling",
                    json!({"target":target,"retired":false}),
                )
                .map_err(|e| search::invalid(e.to_string()))
            }
            Operation::Shutdown => {
                if let Some(a) = &active {
                    pair.cancel(a.pair_id)?;
                }
                shutdown = Some(request.id);
                Ok(())
            }
            Operation::Search {
                ticket,
                allowance_ms,
                remaining_ms,
                input_json,
                policy_temperature,
            } => {
                search::check(active.is_none(), "pair busy")?;
                let temperature = policy_temperature.unwrap_or(package_temperature);
                crate::local_temperature::validate(temperature)?;
                search::check(
                    manifest::hash(input_json.as_bytes()) == ticket.root_sha256,
                    "detached root fingerprint",
                )?;
                let input: Input = serde_json::from_str(&input_json)
                    .map_err(|e| search::invalid(e.to_string()))?;
                validate_input(&ticket, &input, allowance_ms, remaining_ms)?;
                crate::local_profile::event(
                    request.id,
                    "input_validated",
                    crate::local_profile::now(),
                    0,
                );
                let identity = Identity {
                    game_instance: ticket.game_instance.clone(),
                    generation: ticket.generation,
                    export_sha256: EXPORT.into(),
                };
                let turn = TurnKey::from_context(&input.view.context, ticket.actor as usize);
                let config = package.search_config(allowance_ms)?;
                let requested_k = config.k;
                if let Some((_, _, bound_allowance, bound_temperature)) = binding.as_ref().filter(
                    |(bound_identity, bound_turn, ..)| bound_identity == &identity && bound_turn == &turn,
                ) {
                    search::check(*bound_allowance == allowance_ms, "turn allowance changed")?;
                    search::check(*bound_temperature == temperature, "turn policy temperature changed")?;
                } else {
                    pair.admit(
                        identity.clone(),
                        &input,
                        config,
                        (allowance_ms + DELIVERY_GRACE_MS) * 1_000_000,
                    )?;
                    // Work deadline = original search allowance. Retirement and
                    // legal completion may use the additional second, not NN
                    // search. Retrying cannot repeat this fresh admission.
                    pair.configure_host_delivery(DELIVERY_GRACE_NS)?;
                    pair.method(package.method)?;
                    // Admission resets both search sessions. Set the shared
                    // temperature before dispatch, after earlier work retires.
                    evaluation_temperature.store((temperature as f32).to_bits(), Ordering::Relaxed);
                    binding = Some((identity, turn, allowance_ms, temperature));
                }
                let deadline = received + Duration::from_millis(remaining_ms);
                let remaining = deadline
                    .saturating_duration_since(Instant::now())
                    .as_nanos() as u64;
                let startup = Startup {
                    model_id: MODEL.into(),
                    export_sha256: EXPORT.into(),
                    reward_id: input.context.reward_id.clone(),
                    phase: input.view.context.phase,
                    estimate: search::session::clock::Estimate {
                        nanoseconds: package.startup_ns,
                        source: package.startup_source.clone(),
                    },
                };
                let pair_id = pair.start(
                    input,
                    Mode::Timed {
                        startup: Some(startup),
                    },
                    BatchPolicy::default(),
                    remaining,
                    [None, None],
                )?;
                crate::local_profile::event(
                    request.id,
                    "pair_started",
                    crate::local_profile::now(),
                    0,
                );
                active = Some(Active {
                    id: request.id,
                    pair_id,
                    ticket,
                    deadline,
                    requested_k,
                    temperature,
                });
                Ok(())
            }
        })();
        if let Err(e) = result {
            emit(
                request.id,
                "error",
                json!({"error":e.to_string(),"retired":!pair.busy()}),
            )?;
        }
    }
    // EOF/broken transport retires both threads synchronously before process exit.
    drop(pair);
    Ok(())
}

#[cfg(test)]
mod industry_focus_tests {
    use super::*;
    #[test]
    fn seven_exact_context_encodings() {
        for (i, name) in INDUSTRY_FOCUSES.iter().enumerate() {
            let (style, intent) = focus_context(&format!("{PROFILE_PREFIX}{name}/1.0.0")).unwrap();
            assert_eq!(
                style,
                [
                    if (1..=3).contains(&i) { 1. } else { 0. },
                    0.,
                    if i >= 4 { 1. } else { 0. },
                    0.
                ]
            );
            for (j, value) in intent.iter().enumerate() {
                assert_eq!(*value, if i > 0 && j == i - 1 { 1. } else { 0. });
            }
        }
        assert_eq!(focus_context(PROFILE), Some(([0.; 4], [0.; 8])));
    }
    #[test]
    fn unknown_profiles_and_versions_are_rejected() {
        for profile in [
            "alpha-test-g890-unknown/1.0.0",
            "alpha-test-g890-cotton/2.0.0",
            "alpha-test-g1340-cotton/1.0.0",
            "neutral",
            "",
        ] {
            assert!(focus_context(profile).is_none());
        }
    }
}
