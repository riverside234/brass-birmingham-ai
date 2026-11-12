//! In-process ONNX Runtime CPU inference for the lab's Rust workers.
//!
//! Adapted from the game's `brass-ai-inference` runtime: same `ort` binding and
//! session options (one intra/inter thread, sequential, no spinning, Level3,
//! exact GELU), same two warm-up forwards and output checks. The lab adds the
//! optional `search_delta_4` head output and a small per-process model cache.
//!
//! This file only exists in the detached CPU build. The qualified GPU binaries
//! are built from `lab-native/` and are never compiled with it.
use crate::{check, Result};
use brass_ai_core::features::{FeatureTensors, ACTION_DIM, STATE_DIM};
use brass_ai_core::search::math::{Parameters, Statistics};
use brass_ai_core::search::session::clock::Budget as SearchBudget;
use brass_ai_core::{
    features::FeatureContext,
    search::{brass::Key, reward::Ledger, Info},
    state::{
        observation::{InformationState, Observation, ObservationEnvelope, Recall},
        NativeState,
    },
    types::{Era, Phase},
};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    tensor::TensorElementType,
    value::{TensorRef, ValueType},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Instant,
};

pub const CONTRACT: &str = "brass-lab-cpu-inference/1.0.0";
pub const BUNDLE: &str = "brass-lab-cpu-bundle/1.0.0";
const INPUTS: [&str; 6] = [
    "states",
    "actions",
    "legal_mask",
    "stable_to_relative",
    "remaining",
    "descriptors",
];
const OUTPUTS: [&str; 5] = [
    "policy_logits",
    "win_share_4",
    "placement_4",
    "vp_4",
    "remaining_utility_4",
];
const SEARCH_DELTA: &str = "search_delta_4";
const MAX_CANDIDATES: usize = 8192;
/// A bundle or runtime is at most this large; the bound keeps a bad path from
/// reading an arbitrary file into memory.
const MAX_FILE: u64 = 512 * 1024 * 1024;
const MAX_MANIFEST: u64 = 1024 * 1024;
/// Fixed-count searches run to their quota; this only stops a stuck worker.
const COUNT_CAP_PER_TRAJECTORY_NS: u64 = 60_000_000;

static RUNTIME: OnceLock<std::result::Result<String, String>> = OnceLock::new();

fn ort_error(error: ort::Error) -> Box<dyn std::error::Error + Send + Sync> {
    format!("ONNX runtime: {error}").into()
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn bounded_read(path: &Path, limit: u64) -> Result<Vec<u8>> {
    check(
        std::fs::metadata(path)?.len() <= limit,
        "CPU inference file exceeds its bound",
    )?;
    Ok(std::fs::read(path)?)
}

fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    value[name]
        .as_str()
        .ok_or_else(|| format!("missing CPU inference field {name}").into())
}

/// Bind the shared library to its pinned hash once per process.
fn init_runtime(value: &Value) -> Result<String> {
    let path = PathBuf::from(text(value, "path")?);
    let expected = text(value, "sha256")?.to_owned();
    check(path.is_absolute(), "ONNX runtime path must be absolute")?;
    let result = RUNTIME.get_or_init(|| {
        let bytes = bounded_read(&path, MAX_FILE).map_err(|e| e.to_string())?;
        if sha256(&bytes) != expected {
            return Err("ONNX runtime bytes differ from their pin".to_owned());
        }
        drop(bytes);
        ort::init_from(path.to_string_lossy())
            .with_name("brass-lab-cpu")
            .with_telemetry(false)
            .commit()
            .map(|_| expected.clone())
            .map_err(|e| e.to_string())
    });
    match result {
        Ok(hash) if *hash == expected => Ok(hash.clone()),
        Ok(_) => Err("ONNX runtime identity changed inside one worker".into()),
        Err(error) => Err(format!("ONNX runtime initialization failed: {error}").into()),
    }
}

/// One position's raw float32 outputs.
pub struct Raw {
    pub logits: Vec<f32>,
    pub win_share: [f32; 4],
    pub placement: [f32; 4],
    pub vp: [f32; 4],
    pub utility: [f32; 4],
    pub delta: Option<[f32; 4]>,
    pub forward_ns: u64,
}

pub struct Model {
    session: Session,
    pub model_id: String,
    pub export_sha256: String,
    pub reward_id: String,
    pub manifest_sha256: String,
    pub max_cost: f64,
    pub load_ns: u64,
    pub warmup_ns: u64,
    has_delta: bool,
}

fn nanos(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

impl Model {
    /// `bundle` is `{"manifest": path, "sha256": hash}`. The manifest binds the
    /// exported graph by hash, and the session is checked before first use.
    pub fn load(bundle: &Value) -> Result<Self> {
        let start = Instant::now();
        let path = PathBuf::from(text(bundle, "manifest")?);
        let raw = bounded_read(&path, MAX_MANIFEST)?;
        let manifest_sha256 = sha256(&raw);
        check(
            manifest_sha256 == text(bundle, "sha256")?,
            "CPU bundle manifest differs from its pin",
        )?;
        let manifest: Value = serde_json::from_slice(&raw)?;
        check(manifest["contract"] == BUNDLE, "unknown CPU bundle contract")?;
        let file = text(&manifest, "export_file")?;
        check(
            Path::new(file).file_name().map(|n| n.to_string_lossy() == file) == Some(true),
            "CPU bundle export must sit beside its manifest",
        )?;
        let directory = path.parent().ok_or("CPU bundle directory")?;
        let bytes = bounded_read(&directory.join(file), MAX_FILE)?;
        let export_sha256 = sha256(&bytes);
        check(
            export_sha256 == text(&manifest, "export_sha256")?,
            "CPU bundle graph differs from its manifest",
        )?;
        let session = Session::builder()
            .map_err(ort_error)?
            .with_intra_threads(1)
            .map_err(ort_error)?
            .with_inter_threads(1)
            .map_err(ort_error)?
            .with_parallel_execution(false)
            .map_err(ort_error)?
            .with_intra_op_spinning(false)
            .map_err(ort_error)?
            .with_inter_op_spinning(false)
            .map_err(ort_error)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_error)?
            .with_config_entry("optimization.enable_gelu_approximation", "0")
            .map_err(ort_error)?
            .with_log_level(ort::logging::LogLevel::Error)
            .map_err(ort_error)?
            .commit_from_memory(&bytes)
            .map_err(ort_error)?;
        drop(bytes);
        let has_delta = validate_session(&session, &manifest)?;
        let max_cost = manifest["max_cost"]
            .as_f64()
            .filter(|x| x.is_finite() && *x >= 0.)
            .ok_or("invalid CPU bundle max_cost")?;
        let mut model = Self {
            session,
            model_id: text(&manifest, "model_id")?.to_owned(),
            export_sha256,
            reward_id: text(&manifest, "reward_id")?.to_owned(),
            manifest_sha256,
            max_cost,
            load_ns: nanos(start),
            warmup_ns: 0,
            has_delta,
        };
        let warm = Instant::now();
        let states = vec![0.; STATE_DIM];
        let actions = vec![0.; ACTION_DIM];
        let mut descriptors = vec![0.; 136];
        for seat in 0..4 {
            descriptors[seat * 34 + 4] = 1.;
        }
        let remaining = [max_cost as f32; 4];
        for _ in 0..2 {
            let raw = model.run(
                &states,
                &actions,
                &[true],
                &[0, 1, 2, 3],
                &remaining,
                &descriptors,
            )?;
            check(
                raw.logits.iter().all(|x| x.is_finite()),
                "nonfinite CPU warm-up output",
            )?;
        }
        model.warmup_ns = nanos(warm);
        Ok(model)
    }

    /// Singleton forward: borrowed tensors, exactly the game's fast path.
    fn run(
        &mut self,
        states: &[f32],
        actions: &[f32],
        masks: &[bool],
        mappings: &[i64],
        remaining: &[f32; 4],
        descriptors: &[f32],
    ) -> Result<Raw> {
        let n = masks.len();
        check(
            (1..=MAX_CANDIDATES).contains(&n)
                && states.len() == STATE_DIM
                && actions.len() == n * ACTION_DIM
                && mappings.len() == 4
                && descriptors.len() == 136,
            "CPU inference tensor shape",
        )?;
        let start = Instant::now();
        let outputs = self
            .session
            .run(ort::inputs![
                "states" => TensorRef::from_array_view(([1usize, STATE_DIM], states)).map_err(ort_error)?,
                "actions" => TensorRef::from_array_view(([1usize, n, ACTION_DIM], actions)).map_err(ort_error)?,
                "legal_mask" => TensorRef::from_array_view(([1usize, n], masks)).map_err(ort_error)?,
                "stable_to_relative" => TensorRef::from_array_view(([1usize, 4usize], mappings)).map_err(ort_error)?,
                "remaining" => TensorRef::from_array_view(([1usize, 4usize], &remaining[..])).map_err(ort_error)?,
                "descriptors" => TensorRef::from_array_view(([1usize, 4usize, 34usize], descriptors)).map_err(ort_error)?,
            ])
            .map_err(ort_error)?;
        let forward_ns = nanos(start);
        let (shape, logits) = outputs["policy_logits"]
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?;
        check(**shape == [1, n as i64], "CPU policy output shape")?;
        check(logits.iter().all(|x| x.is_finite()), "nonfinite CPU logits")?;
        let logits = logits.to_vec();
        let head = |name: &str| -> Result<[f32; 4]> {
            let (shape, values) = outputs[name]
                .try_extract_tensor::<f32>()
                .map_err(ort_error)?;
            check(**shape == [1, 4], "CPU head output shape")?;
            check(values.iter().all(|x| x.is_finite()), "nonfinite CPU head")?;
            let mut result = [0.; 4];
            result.copy_from_slice(values);
            Ok(result)
        };
        let raw = Raw {
            logits,
            win_share: head(OUTPUTS[1])?,
            placement: head(OUTPUTS[2])?,
            vp: head(OUTPUTS[3])?,
            utility: head(OUTPUTS[4])?,
            delta: if self.has_delta {
                Some(head(SEARCH_DELTA)?)
            } else {
                None
            },
            forward_ns,
        };
        Ok(raw)
    }

    pub fn forward(&mut self, t: &FeatureTensors, remaining: &[f32; 4]) -> Result<Raw> {
        check(t.legal_mask.iter().all(|x| *x), "CPU candidate mask")?;
        self.run(
            &t.state,
            &t.actions,
            &t.legal_mask,
            &t.stable_to_relative,
            remaining,
            &t.descriptors,
        )
    }

    fn receipt(&self) -> Value {
        json!({"model_id":self.model_id,"export_sha256":self.export_sha256,
            "manifest_sha256":self.manifest_sha256,"reward_id":self.reward_id,
            "max_cost":self.max_cost,"search_delta":self.has_delta,
            "load_ns":self.load_ns,"warmup_ns":self.warmup_ns})
    }
}

fn validate_session(session: &Session, manifest: &Value) -> Result<bool> {
    let initializers = session.overridable_initializers();
    let specs: Vec<_> = session
        .inputs
        .iter()
        .map(|x| (x.name.as_str(), &x.input_type))
        .chain(initializers.iter().map(|x| (x.name(), x.dtype())))
        .collect();
    let shapes: [&[i64]; 6] = [
        &[-1, 3128],
        &[-1, -1, 327],
        &[-1, -1],
        &[-1, 4],
        &[-1, 4],
        &[-1, 4, 34],
    ];
    check(specs.len() == INPUTS.len(), "unexpected ONNX input count")?;
    for (i, name) in INPUTS.iter().enumerate() {
        let ty = match i {
            2 => TensorElementType::Bool,
            3 => TensorElementType::Int64,
            _ => TensorElementType::Float32,
        };
        let value = specs
            .iter()
            .find(|(n, _)| n == name)
            .ok_or("missing ONNX input")?
            .1;
        check(
            matches!(value, ValueType::Tensor { ty: got, shape, .. } if *got == ty && &**shape == shapes[i]),
            "ONNX input dtype/shape mismatch",
        )?;
    }
    let names: Vec<&str> = manifest["outputs"]
        .as_array()
        .ok_or("missing CPU bundle outputs")?
        .iter()
        .map(|v| v.as_str().unwrap_or(""))
        .collect();
    let has_delta = names.len() == OUTPUTS.len() + 1 && names[OUTPUTS.len()] == SEARCH_DELTA;
    check(
        names.len() >= OUTPUTS.len()
            && names[..OUTPUTS.len()] == OUTPUTS
            && (names.len() == OUTPUTS.len() || has_delta),
        "CPU bundle output list",
    )?;
    check(
        session.outputs.len() == names.len(),
        "unexpected ONNX output count",
    )?;
    for (i, out) in session.outputs.iter().enumerate() {
        let shape: &[i64] = if i == 0 { &[-1, -1] } else { &[-1, 4] };
        check(
            out.name == names[i]
                && matches!(&out.output_type, ValueType::Tensor { ty, shape: s, .. }
                    if *ty == TensorElementType::Float32 && &**s == shape),
            "ONNX output dtype/shape mismatch",
        )?;
    }
    Ok(has_delta)
}

/// float32 blend, matching the GPU path's `torch.where(w == 0, u, u + w * d)`.
/// `w == 0` keeps the exact utility bytes and never needs the delta output.
pub fn blend(utility: [f32; 4], delta: Option<[f32; 4]>, w: f64) -> Result<[f32; 4]> {
    check(w.is_finite() && (0. ..=1.).contains(&w), "invalid search blend")?;
    if w == 0. {
        return Ok(utility);
    }
    let delta = delta.ok_or("search blend requires the search_delta_4 output")?;
    let weight = w as f32;
    Ok(std::array::from_fn(|i| utility[i] + weight * delta[i]))
}

/// Exact widening: identical to the GPU path's float32 -> Python float -> JSON.
pub fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|x| f64::from(*x)).collect()
}

/// The historical search constant, kept wherever no budget selects another.
const DEFAULT_C_SCALE: f64 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Budget {
    Timed,
    Count {
        full: u64,
        full_k: usize,
        fast: u64,
        fast_k: usize,
        c_scale: f64,
    },
}

fn budget_k(value: &Value, name: &str, default: usize) -> Result<usize> {
    match value.get(name) {
        None => Ok(default),
        Some(k) => Ok(k.as_u64().ok_or("invalid teacher search K")? as usize),
    }
}

fn budget_c_scale(value: &Value) -> Result<f64> {
    let c_scale = match value.get("c_scale") {
        None => DEFAULT_C_SCALE,
        Some(number) => number.as_f64().ok_or("invalid teacher c_scale")?,
    };
    check(
        c_scale == DEFAULT_C_SCALE || c_scale == 1.0,
        "teacher c_scale outside its bounds",
    )?;
    Ok(c_scale)
}

/// `{"mode":"count","full":N,"full_k":K,"fast":N,"fast_k":K,"c_scale":C}` or
/// `{"mode":"timed"}`. A count budget without K keeps the job's 16 and 4; one
/// without `c_scale` keeps the historical 0.1.
pub fn budget(value: &Value) -> Result<Budget> {
    match value["mode"].as_str() {
        Some("timed") => Ok(Budget::Timed),
        Some("count") => {
            let full = value["full"].as_u64().ok_or("missing full search count")?;
            let fast = value["fast"].as_u64().ok_or("missing fast search count")?;
            let full_k = budget_k(value, "full_k", 16)?;
            let fast_k = budget_k(value, "fast_k", 4)?;
            let c_scale = budget_c_scale(value)?;
            // A halving tournament over K needs K*log2(K) visits to stay at K.
            check(
                (64..=4096).contains(&full) && (8..=4096).contains(&fast),
                "teacher search counts outside their bounds",
            )?;
            // Labels are only ever produced by full searches, and require K=16.
            check(
                full_k == 16 && matches!(fast_k, 2 | 4 | 8 | 16),
                "teacher search K outside its bounds",
            )?;
            Ok(Budget::Count {
                full,
                full_k,
                fast,
                fast_k,
                c_scale,
            })
        }
        _ => Err("unknown teacher search budget".into()),
    }
}

struct Teacher {
    model: Model,
    budget: Budget,
    runtime_sha256: String,
    positions: u64,
    forward_ns: u64,
}

thread_local! {
    /// The session is owned by the search thread, like the game's `!Send`
    /// evaluator. Absent means this process is answering over the GPU wire.
    static TEACHER: RefCell<Option<Teacher>> = const { RefCell::new(None) };
    /// The job's reviewed g1491 revision and its per-table counters, if any.
    static REVISION: RefCell<Option<(Revision, Exploration)>> = const { RefCell::new(None) };
}

const REVISION_CONTRACT: &str = "training-revision-g1491/1.0.0";
const EXPLORATION_CONTRACT: &str = "root-visit-count-sampling/1.0.0";

/// `job["teacher_revision"]`: label short searches too, and sample the played
/// action from the root visit counts (temperature 1) for each seat's first
/// `opening_actions` actions of an era and, with `long_search`, every long search.
#[derive(Clone, Debug, PartialEq)]
pub struct Revision {
    job: Value,
    short_labels: bool,
    opening_actions: u32,
    long_search: bool,
    cotton: bool,
    cotton_develop: bool,
}

/// Per-table exploration state; index 0 counts long searches, 1 short ones.
#[derive(Default, Debug)]
pub struct Exploration {
    /// The action each seat is on: (era, round, actions remaining).
    current: [Option<(Era, u8, u8)>; 4],
    /// That seat's actions so far in the era, the current one included.
    count: [u32; 4],
    decisions: [u64; 2],
    sampled: [u64; 2],
    changed: [u64; 2],
    short_labels: u64,
    cotton: crate::cpu_cotton::State,
}

impl Exploration {
    fn for_revision(revision: &Revision) -> Self {
        let mut state = Self::default();
        state.cotton.develop = revision.cotton_develop;
        state
    }

    /// The seat's 1-based action index in its era; repeated decisions of one action keep it.
    fn action(&mut self, seat: usize, era: Era, round: u8, actions_remaining: u8) -> u32 {
        let key = (era, round, actions_remaining);
        if self.current[seat] != Some(key) {
            if self.current[seat].is_some_and(|(previous, _, _)| previous != era) {
                self.count[seat] = 0;
            }
            self.current[seat] = Some(key);
            self.count[seat] += 1;
        }
        self.count[seat]
    }
}

pub fn revision(job: &Value) -> Result<Option<Revision>> {
    let block = &job["teacher_revision"];
    if block.is_null() {
        return Ok(None);
    }
    let e = &block["exploration"];
    let opening = e["opening_actions_per_era"].as_u64().unwrap_or(u64::MAX);
    check(
        block["contract"] == REVISION_CONTRACT
            && block.as_object().is_some_and(|o| o.len() == 3)
            && block["short_labels"].is_boolean()
            && e.as_object().is_some_and(|o| o.len() == if e.get("cotton").is_some() {5} else {4})
            && e["contract"] == EXPLORATION_CONTRACT
            && e["temperature"].as_f64() == Some(1.0)
            && e["long_search"].is_boolean()
            && opening <= 64
            && job["seating"] == "all-seat-search",
        "unreviewed teacher revision",
    )?;
    let cotton = e.get("cotton").is_some();
    let cotton_develop = cotton && e["cotton"] == crate::cpu_cotton::settings(true);
    check(!cotton || ((cotton_develop || e["cotton"] == crate::cpu_cotton::settings(false)) && job["split"] == "train"
        && job["h2h"] != true && job["shadow_probe"] != true), "unreviewed cotton exploration")?;
    Ok(Some(Revision {
        job: block.clone(),
        short_labels: block["short_labels"] == true,
        opening_actions: opening as u32,
        long_search: e["long_search"] == true,
        cotton,
        cotton_develop,
    }))
}

/// splitmix64: the exploration stream, independent of the search and policy draws.
pub(crate) fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// An index drawn with probability `counts[i] / total`, keyed by the job seed and decision.
fn visit_sample(counts: &[u64], total: u64, seed: u64, decision: u64) -> Result<usize> {
    check(total > 0 && counts.iter().sum::<u64>() == total, "invalid root visit counts")?;
    let z = mix(seed ^ mix(decision ^ 0x1491_0000_0000_0000));
    let target = ((u128::from(z) * u128::from(total)) >> 64) as u64;
    let mut seen = 0;
    for (i, n) in counts.iter().enumerate() {
        seen += n;
        if target < seen {
            return Ok(i);
        }
    }
    Err("root visit sample out of range".into())
}

/// The action to play. A g1491 job samples it from the decision root's visit
/// counts for a seat's opening actions of each era and for every long search;
/// otherwise, and for any other job, the search's own choice is played.
pub fn teacher_explore(
    planned: String,
    counts: Option<&[u64]>,
    actual: &Info<Key, String>,
    view: &Observation,
    fast: bool,
    seed: u64,
    decision: u64,
) -> Result<(String, bool)> {
    REVISION.with(|cell| {
        let mut guard = cell.borrow_mut();
        let Some((revision, state)) = guard.as_mut() else {
            return Ok((planned, false));
        };
        let c = &view.context;
        if c.phase == Phase::Continuation && revision.cotton_develop {
            return Ok((state.cotton.sample_continuation(planned, &actual.key.choices, seed, decision), false));
        }
        if c.phase != Phase::Action {
            return Ok((planned, false));
        }
        check(actual.actor < 4, "invalid acting seat")?;
        let index = state.action(actual.actor, c.era, c.round, c.actions_remaining);
        let profile = usize::from(fast);
        state.decisions[profile] += 1;
        let eligible = index <= revision.opening_actions || (!fast && revision.long_search);
        let Some(counts) = counts.filter(|_| eligible && actual.actions.len() > 1) else {
            if revision.cotton { state.cotton.no_evidence(); }
            return Ok((planned, false));
        };
        check(counts.len() == actual.actions.len(), "root visit counts differ from the legal actions")?;
        let total: u64 = counts.iter().sum();
        if total == 0 {
            if revision.cotton { state.cotton.no_evidence(); }
            return Ok((planned, false));
        }
        let base = visit_sample(counts, total, seed, decision)?;
        let index = if revision.cotton {
            state.cotton.sample(base, &actual.key.choices, counts, total, seed, decision)?
        } else { base };
        let choice = actual.actions[index].clone();
        state.sampled[profile] += 1;
        if choice != planned {
            state.changed[profile] += 1;
        }
        Ok((choice, true))
    })
}

/// Pure admission before the search; the matching core root alone gets coverage.
/// A cached continuation keeps its existing evidence and never launches extra work.
pub fn teacher_cotton_prepare(view: &Observation, actual: &Info<Key, String>, fast: bool) -> (usize, bool) {
    REVISION.with(|cell| {
        let mut guard=cell.borrow_mut();
        let Some((revision,state))=guard.as_mut().filter(|(r,_)| r.cotton) else {return (0, false);};
        let c=&view.context;
        let index = match c.phase {
            Phase::Action => state.action(actual.actor,c.era,c.round,c.actions_remaining),
            Phase::Continuation if revision.cotton_develop => state.count[actual.actor],
            _ => 0,
        };
        let window = index > 0 && (index <= revision.opening_actions || !fast && revision.long_search);
        (state.cotton.prepare(view,&actual.key.choices,fast,window), revision.cotton_develop)
    })
}

pub fn teacher_cotton_commit(view: &Observation, choice: &brass_ai_core::state::dto::Choice, after_revision: u32) -> Result<()> {
    REVISION.with(|cell| {
        let mut guard=cell.borrow_mut();
        if let Some((_,state))=guard.as_mut().filter(|(r,_)| r.cotton) {
            state.cotton.commit(view,choice,after_revision)?;
        }
        Ok(())
    })
}

/// True when this table labels its short searches too.
pub fn teacher_short_labels() -> bool {
    REVISION.with(|cell| cell.borrow().as_ref().is_some_and(|(r, _)| r.short_labels))
}

/// Marks a short-search label with its own K and an explored played action.
/// Rows of jobs without the revision keep their exact bytes.
pub fn teacher_mark_label(row: &mut Value, fast: bool, explored: bool) -> Result<()> {
    let fast_k = teacher_k(true, false, 4);
    REVISION.with(|cell| {
        let mut guard = cell.borrow_mut();
        let Some((revision, state)) = guard.as_mut() else {
            check(!fast && !explored, "short-search label without the teacher revision")?;
            return Ok(());
        };
        check(!fast || revision.short_labels, "short-search label without short labels")?;
        let teacher = row.get_mut("teacher").ok_or("label row without teacher fields")?;
        if fast {
            teacher["profile"] = json!("fast");
            teacher["requested_k"] = json!(fast_k);
            state.short_labels += 1;
        }
        if explored {
            teacher["explored"] = json!(true);
        }
        if revision.cotton {
            if let Some(selection)=&state.cotton.selection {
                check(teacher["played_action"] == selection.choice, "cotton label choice differs")?;
                teacher["cotton_exploration"] = selection.label();
            }
        }
        Ok(())
    })
}

fn revision_receipt() -> Option<Value> {
    REVISION.with(|cell| {
        cell.borrow().as_ref().map(|(r, s)| {
            let mut value=json!({"job":r.job,"short_labels":s.short_labels,
                "decisions":{"long":s.decisions[0],"short":s.decisions[1]},
                "sampled":{"long":s.sampled[0],"short":s.sampled[1]},
                "changed":{"long":s.changed[0],"short":s.changed[1]}});
            if r.cotton { value["cotton"]=s.cotton.receipt(); }
            value
        })
    })
}

/// Load the frozen model named by `request["inference"]`, if any.
pub fn teacher_install(request: &Value, expected: &Value, max_cost: f64) -> Result<()> {
    let revision = revision(&request["job"])?;
    let block = &request["inference"];
    if block.is_null() {
        check(revision.is_none(), "the teacher revision needs CPU inference")?;
        REVISION.with(|cell| *cell.borrow_mut() = None);
        return Ok(());
    }
    check(block["contract"] == CONTRACT, "CPU inference contract")?;
    let runtime_sha256 = init_runtime(&block["runtime"])?;
    let model = Model::load(&block["bundle"])?;
    check(
        expected["model_id"] == model.model_id.as_str()
            && expected["reward_id"] == model.reward_id.as_str()
            && (model.max_cost - max_cost).abs() < 1e-12,
        "CPU bundle model/reward differs from the frozen job",
    )?;
    let budget = budget(&block["budget"])?;
    check(
        revision.is_none() || matches!(budget, Budget::Count { .. }),
        "the teacher revision needs fixed-count searches",
    )?;
    TEACHER.with(|cell| {
        *cell.borrow_mut() = Some(Teacher {
            model,
            budget,
            runtime_sha256,
            positions: 0,
            forward_ns: 0,
        })
    });
    REVISION.with(|cell| *cell.borrow_mut() = revision.map(|r| {
        let state = Exploration::for_revision(&r);
        (r, state)
    }));
    Ok(())
}

pub fn teacher_enabled() -> bool {
    TEACHER.with(|cell| cell.borrow().is_some())
}

pub fn teacher_count_mode() -> bool {
    TEACHER.with(|cell| {
        cell.borrow()
            .as_ref()
            .is_some_and(|t| matches!(t.budget, Budget::Count { .. }))
    })
}

/// Calibration probes stay at the existing eight trajectories.
pub fn teacher_simulations(fast: bool, calibration: bool) -> u64 {
    TEACHER.with(|cell| match cell.borrow().as_ref().map(|t| t.budget) {
        Some(Budget::Count {
            full, fast: quick, ..
        }) if !calibration => {
            if fast {
                quick
            } else {
                full
            }
        }
        _ => 8,
    })
}

fn count_cap_ns(simulations: u64, floor_ns: u64) -> u64 {
    floor_ns.max(simulations.saturating_mul(COUNT_CAP_PER_TRAJECTORY_NS))
}

/// The turn budget for a fixed-count search; `None` keeps the job's own limits.
pub fn teacher_turn_budget(
    started_ns: u64,
    fast: bool,
    calibration: bool,
) -> Result<Option<SearchBudget>> {
    let budget = TEACHER.with(|cell| cell.borrow().as_ref().map(|t| t.budget));
    let Some(Budget::Count {
        full, fast: quick, ..
    }) = budget
    else {
        return Ok(None);
    };
    if calibration {
        return Ok(None);
    }
    let (simulations, floor_ns, post_ns) = if fast {
        (quick, 800_000_000, 500_000_000)
    } else {
        (full, 4_000_000_000, 1_000_000_000)
    };
    let cap_ns = count_cap_ns(simulations, floor_ns);
    let deadline_ns = started_ns
        .checked_add(cap_ns.saturating_add(post_ns))
        .ok_or("turn clock overflow")?;
    Ok(Some(SearchBudget {
        started_ns,
        work_deadline_ns: deadline_ns - post_ns,
        deadline_ns,
        reserve_ns: post_ns,
    }))
}

/// The Gumbel `c_scale` (the core's `c2`) of this search; 0.1 unless a budget says otherwise.
pub fn teacher_c_scale() -> f64 {
    TEACHER.with(|cell| match cell.borrow().as_ref().map(|t| t.budget) {
        Some(Budget::Count { c_scale, .. }) => c_scale,
        _ => DEFAULT_C_SCALE,
    })
}

/// The improved-policy target for `teacher_c_scale()`.
pub fn teacher_policy(stats: &Statistics, actor: usize) -> Result<Vec<f64>> {
    let parameters = Parameters {
        c1: 50.,
        c2: teacher_c_scale(),
    };
    Ok(stats.policy(actor, parameters)?)
}

/// A fixed-count budget chooses K; every other mode keeps the job's own K.
pub fn teacher_k(fast: bool, calibration: bool, job_k: usize) -> usize {
    TEACHER.with(|cell| match cell.borrow().as_ref().map(|t| t.budget) {
        Some(Budget::Count { full_k, fast_k, .. }) if !calibration => {
            if fast {
                fast_k
            } else {
                full_k
            }
        }
        _ => job_k,
    })
}

/// The frozen importer refuses one legal state: the develop-bonus choice of a
/// multi-beer sale that still owes beer. Search rebuilds every root through that
/// importer, so it cannot start from this state.
pub fn unimportable_bonus(state: &NativeState) -> bool {
    state.context().phase == Phase::Bonus
        && NativeState::import(&state.export())
            .err()
            .is_some_and(|e| e.to_string().contains("bonus state mismatch"))
}

/// The root a search would build, taken directly from the true state: the same
/// permitted view, recall, sorted legal choices, context and settled ledger.
pub fn direct_root(
    state: &NativeState,
    view: &Observation,
    recall: &Recall,
    context: &FeatureContext,
    ledger: &Ledger,
) -> Result<Info<Key, String>> {
    let actor = usize::try_from(state.context().actor).map_err(|_| "invalid acting seat")?;
    check(
        usize::from(u8::from(view.seat)) == actor && view.context.actor == actor as i8,
        "direct root must be the acting permitted view",
    )?;
    let mut choices = state.legal_choices()?;
    choices.sort_by(|a, b| a.id.cmp(&b.id));
    let actions = choices.iter().map(|c| c.id.clone()).collect::<Vec<_>>().into();
    let mut context = context.clone();
    context.remaining = ledger.remaining();
    Ok(Info {
        key: std::sync::Arc::new(Key {
            information: InformationState {
                observation: ObservationEnvelope {
                    view: Some(view.clone()),
                    setup: false,
                    chance: None,
                },
                recall: recall.clone(),
            },
            choices,
            context,
            settled: ledger.state().settled,
        }),
        actor,
        actions,
    })
}

/// `search_value` carries `w` for the seat whose search is running.
pub fn teacher_evaluate(
    t: &FeatureTensors,
    search_value: Option<&Value>,
) -> Result<(Vec<f64>, [f64; 4])> {
    let w = match search_value {
        Some(value) => value["w"].as_f64().ok_or("search blend")?,
        None => 0.,
    };
    TEACHER.with(|cell| {
        let mut guard = cell.borrow_mut();
        let teacher = guard.as_mut().ok_or("CPU evaluator is not loaded")?;
        let raw = teacher.model.forward(t, &t.remaining)?;
        teacher.positions += 1;
        teacher.forward_ns += raw.forward_ns;
        let value = blend(raw.utility, raw.delta, w)?;
        Ok((widen(&raw.logits), value.map(f64::from)))
    })
}

pub fn teacher_receipt() -> Value {
    let mut receipt = TEACHER.with(|cell| match cell.borrow().as_ref() {
        Some(t) => json!({"contract":CONTRACT,"backend":"cpu-ort","runtime_sha256":t.runtime_sha256,
            "provider":"CPUExecutionProvider","intra_threads":1,"inter_threads":1,"sequential":true,
            "budget":match t.budget {
                Budget::Timed => json!({"mode":"timed"}),
                Budget::Count{full,full_k,fast,fast_k,c_scale} => json!({"mode":"count","full":full,"full_k":full_k,"fast":fast,"fast_k":fast_k,"c_scale":c_scale}),
            },
            "positions":t.positions,"forward_ns":t.forward_ns,"model":t.model.receipt()}),
        None => Value::Null,
    });
    // Only a g1491 job adds the field, so every other receipt keeps its exact bytes.
    if let (Some(revision), Some(object)) = (revision_receipt(), receipt.as_object_mut()) {
        object.insert("revision".into(), revision);
    }
    receipt
}

/// Ordinary collector models, most recently used last.
#[derive(Default)]
pub struct Cache {
    models: HashMap<String, Model>,
    order: VecDeque<String>,
    limit: usize,
    runtime_sha256: Option<String>,
    loads: u64,
    hits: u64,
    evictions: u64,
    positions: u64,
    forward_ns: u64,
    load_ns: u64,
}

impl Cache {
    pub fn configure(&mut self, request: &Value) -> Result<Value> {
        check(request["contract"] == CONTRACT, "CPU inference contract")?;
        let limit = request["model_cache"]
            .as_u64()
            .filter(|x| (1..=64).contains(x))
            .ok_or("invalid CPU model cache")? as usize;
        self.runtime_sha256 = Some(init_runtime(&request["runtime"])?);
        self.limit = limit;
        Ok(json!({"contract":CONTRACT,"runtime_sha256":self.runtime_sha256,"model_cache":limit}))
    }

    /// Idempotent for an already cached model; never reloads its session.
    pub fn load(&mut self, request: &Value) -> Result<Value> {
        check(self.limit > 0, "configure CPU inference before loading")?;
        let id = text(request, "model_id")?;
        if self.models.contains_key(id) {
            self.touch(id);
            self.hits += 1;
            return Ok(json!({"model_id":id,"loaded":false}));
        }
        let model = Model::load(&request["bundle"])?;
        check(model.model_id == id, "CPU bundle model identity")?;
        while self.models.len() >= self.limit {
            let old = self.order.pop_front().ok_or("empty CPU model order")?;
            self.models.remove(&old);
            self.evictions += 1;
        }
        self.loads += 1;
        self.load_ns += model.load_ns + model.warmup_ns;
        let receipt = model.receipt();
        self.order.push_back(id.to_owned());
        self.models.insert(id.to_owned(), model);
        Ok(json!({"model_id":id,"loaded":true,"receipt":receipt}))
    }

    fn touch(&mut self, id: &str) {
        if let Some(i) = self.order.iter().position(|x| x == id) {
            let key = self.order.remove(i).expect("position is valid");
            self.order.push_back(key);
        }
    }

    /// Little-endian float32: logits, then win share, placement, VP and utility.
    pub fn infer(&mut self, id: &str, t: &FeatureTensors) -> Result<(Vec<u8>, u64)> {
        self.touch(id);
        let model = self.models.get_mut(id).ok_or("CPU model is not loaded")?;
        let remaining = [model.max_cost as f32; 4];
        let raw = model.forward(t, &remaining)?;
        self.positions += 1;
        self.forward_ns += raw.forward_ns;
        let mut bytes = Vec::with_capacity((raw.logits.len() + 16) * 4);
        for value in raw
            .logits
            .iter()
            .chain(&raw.win_share)
            .chain(&raw.placement)
            .chain(&raw.vp)
            .chain(&raw.utility)
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Ok((bytes, raw.forward_ns))
    }

    pub fn report(&self) -> Value {
        json!({"contract":CONTRACT,"runtime_sha256":self.runtime_sha256,"loads":self.loads,"hits":self.hits,
            "evictions":self.evictions,"resident":self.models.len(),"positions":self.positions,
            "forward_ns":self.forward_ns,"load_ns":self.load_ns})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use brass_ai_core::search::{
        brass::{Brass, Input},
        reward::{Insolvency, LedgerState, Reward},
        Game,
    };
    use brass_ai_core::state::{
        dto::{Industry, Kind, StateDto},
        observation::RecallEntry,
    };
    use brass_ai_core::types::Seat;

    /// The lane tests' multi-beer sale: two beers owed to a Gloucester (develop bonus) merchant.
    fn multi_beer_sale() -> StateDto {
        let mut d: StateDto =
            serde_json::from_str(include_str!("../../native/brass-ai-core/tests/data/initial-state.json")).unwrap();
        let owner = usize::from(u8::from(d.context.turn_owner));
        let mut place = |industry, beer| {
            let mat = d.players[owner].mat.iter_mut().find(|m| m.industry == industry).unwrap();
            let index = mat
                .tiles
                .iter()
                .position(|t| match beer {
                    Some(n) => t.beer == n,
                    None => true,
                })
                .unwrap();
            let mut tile = mat.tiles.remove(index);
            if industry == Industry::Brewery {
                tile.resource = 1;
            }
            let slot = d.slots.iter_mut().find(|s| s.tile.is_none() && s.types.contains(&industry)).unwrap();
            slot.tile = Some(tile);
            slot.id.clone()
        };
        let target = place(Industry::Manufacturer, Some(2));
        place(Industry::Brewery, None);
        let merchant = d.merchants.iter().position(|m| m.id == "Gloucester:0").unwrap();
        let donor = d
            .merchants
            .iter()
            .position(|m| m.types.as_ref().is_some_and(|t| t.contains(&Industry::Manufacturer)))
            .unwrap();
        let types = d.merchants[merchant].types.clone();
        d.merchants[merchant].types = d.merchants[donor].types.clone();
        d.merchants[donor].types = types;
        for m in &mut d.merchants {
            m.beer = u8::from(m.types.as_ref().is_some_and(|t| !t.is_empty()));
        }
        d.context.phase = Phase::Payment;
        d.pending.kind = "sell".into();
        d.pending.target = target;
        d.pending.merchant = "Gloucester:0".into();
        d.pending.parts = 1;
        d.pending.beer = 2;
        d
    }

    fn reward() -> Reward {
        Reward {
            version: "brass-combined-return/2.0.0".into(),
            win: 1.,
            placement: 0.1,
            vp: 0.001,
            vp_cap: Some(180),
            insolvency: Insolvency { event_cost: 0.025, max_cost: 0.075 },
        }
    }

    /// The teacher's own root inputs for the acting seat of `state`.
    fn root(state: &NativeState) -> (Input, Ledger) {
        let seat = Seat::try_from(state.context().actor as u8).unwrap();
        let view = state.observation(seat).unwrap();
        let reward = reward();
        let mut ledger = Ledger::new(reward.clone(), LedgerState { spent: [0.; 4], settled: vec![] }).unwrap();
        ledger.observe(&view).unwrap();
        let context = FeatureContext {
            model_id: "frozen".into(),
            reward_id: reward.identity().unwrap(),
            profile_id: "brass-lab-teacher/1.1.0".into(),
            style: [0.; 4],
            intent: [0.; 8],
            remaining: ledger.remaining(),
            max_cost: reward.insolvency.max_cost,
            stable_descriptors: None,
        };
        let recall = vec![RecallEntry { revision: view.context.revision, hand: view.own_hand.clone() }];
        let input = Input {
            view,
            recall,
            context,
            reward,
            ledger: ledger.state(),
            shuffle_tape: None,
            sample_seed: 7,
        };
        (input, ledger)
    }

    fn direct(state: &NativeState) -> Info<Key, String> {
        let (input, ledger) = root(state);
        direct_root(state, &input.view, &input.recall, &input.context, &ledger).unwrap()
    }

    fn step(state: &NativeState, kind: Kind, target: Option<&str>) -> NativeState {
        let choice = state
            .legal_choices()
            .unwrap()
            .into_iter()
            .find(|c| c.kind == kind && (target.is_none() || c.target.as_deref() == target))
            .unwrap();
        state.after_choice(state.context().revision, &choice.id).unwrap()
    }

    #[test]
    fn a_direct_root_equals_the_search_root_wherever_search_can_build_one() {
        let sale = NativeState::import(&multi_beer_sale()).unwrap();
        let start: StateDto =
            serde_json::from_str(include_str!("../../native/brass-ai-core/tests/data/initial-state.json")).unwrap();
        for state in [NativeState::import(&start).unwrap(), sale] {
            assert!(!unimportable_bonus(&state));
            let (input, _) = root(&state);
            let searched = Brass::new(input, false).unwrap().root().unwrap();
            assert_eq!(direct(&state), searched);
        }
    }

    #[test]
    fn a_mid_sale_develop_bonus_is_detected_and_still_has_a_root() {
        for bonus in [Kind::BonusSkip, Kind::BonusDevelop] {
            let sale = NativeState::import(&multi_beer_sale()).unwrap();
            let state = step(&sale, Kind::Beer, Some("merchant:Gloucester:0"));
            assert_eq!(state.context().phase, Phase::Bonus);
            assert_eq!(state.export().pending.beer, 1);
            // This is the teacher's failure: search cannot rebuild the root.
            let (input, _) = root(&state);
            assert!(Brass::new(input, false).err().unwrap().to_string().contains("bonus state mismatch"));
            assert!(unimportable_bonus(&state));
            let info = direct(&state);
            let mut legal = state.legal_choices().unwrap();
            legal.sort_by(|a, b| a.id.cmp(&b.id));
            assert_eq!(info.key.choices, legal);
            assert_eq!(*info.actions, legal.iter().map(|c| c.id.clone()).collect::<Vec<_>>());
            assert_eq!(info.actor, usize::from(u8::from(state.observation(Seat::try_from(state.context().actor as u8).unwrap()).unwrap().seat)));
            assert!(info.key.choices.iter().any(|c| c.kind == bonus));
            // After the bonus the remaining beer is an ordinary searchable payment again.
            let paid = step(&state, bonus, None);
            assert!(!unimportable_bonus(&paid));
            let (input, _) = root(&paid);
            assert!(Brass::new(input, false).is_ok());
        }
    }

    /// The teacher keeps one search session per turn. A bonus decision that skips
    /// it must leave the session able to search the turn's next decision.
    #[cfg(feature = "teacher")]
    #[test]
    fn a_turn_session_continues_after_the_skipped_bonus_decision() {
        use brass_ai_core::search::{
            session::{clock::MonotonicClock, Identity, Mode, Session, Settings},
            Config, LeafEvaluator,
        };
        use std::sync::{atomic::AtomicBool, Arc};
        struct Uniform;
        impl LeafEvaluator<Key, String> for Uniform {
            fn evaluate(&mut self, info: &Info<Key, String>) -> brass_ai_core::search::Result<Statistics> {
                Statistics::new(vec![0.; info.actions.len()], [0.25; 4])
            }
        }
        let clock = MonotonicClock::new(Arc::new(AtomicBool::new(false)));
        let identity = Identity { game_instance: "bonus".into(), generation: 1, export_sha256: "a".repeat(64) };
        let sale = NativeState::import(&multi_beer_sale()).unwrap();
        let (first, _) = root(&sale);
        let mut settings = Settings::default();
        settings.search = Config { k: 4, simulations: 8, seed: 7, ..Config::default() };
        let budget = crate::teacher::profile_budget(brass_ai_core::search::session::clock::Clock::now_ns(&clock), true).unwrap();
        let mut owner = Session::new(identity.clone(), &first, settings, budget).unwrap();
        let mut evaluator = Uniform;
        crate::teacher::decide_turn(&mut owner, &identity, 1, first.clone(), Mode::Count, &clock, &mut evaluator).unwrap();
        // Whatever the search preferred, take the merchant beer that opens the bonus.
        let bonus = step(&sale, Kind::Beer, Some("merchant:Gloucester:0"));
        assert!(unimportable_bonus(&bonus));
        let (input, _) = root(&bonus);
        assert!(crate::teacher::decide_turn(&mut owner, &identity, 2, input, Mode::Count, &clock, &mut evaluator).is_err());
        // The patched teacher never asks the session here; it applies the policy's choice.
        let paid = step(&bonus, Kind::BonusSkip, None);
        let (mut next, _) = root(&paid);
        next.recall = first.recall.clone();
        let outcome = crate::teacher::decide_turn(&mut owner, &identity, 3, next, Mode::Count, &clock, &mut evaluator).unwrap();
        assert!(!outcome.action_ids.is_empty() || outcome.status == "completed");
    }

    #[test]
    fn zero_blend_keeps_exact_bytes_and_needs_no_delta() {
        let utility = [0.1f32, -0.2, 0.3, f32::MIN_POSITIVE];
        let out = blend(utility, None, 0.).unwrap();
        assert_eq!(out.map(f32::to_bits), utility.map(f32::to_bits));
    }

    #[test]
    fn blend_uses_float32_arithmetic_like_the_gpu_path() {
        let utility = [0.1f32, 0.2, 0.3, 0.4];
        let delta = [0.5f32, -0.25, 0.125, -0.0625];
        for w in [0.05f64, 0.1, 0.5, 1.] {
            let weight = w as f32;
            let expected: [f32; 4] = std::array::from_fn(|i| utility[i] + weight * delta[i]);
            let actual = blend(utility, Some(delta), w).unwrap();
            assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
        }
    }

    #[test]
    fn invalid_blends_are_rejected() {
        let utility = [0.; 4];
        assert!(blend(utility, None, 0.5).is_err());
        for w in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(blend(utility, Some([0.; 4]), w).is_err());
        }
    }

    #[test]
    fn widening_is_exact() {
        let values = [0.1f32, -3.25e-7, f32::MAX];
        let wide = widen(&values);
        for (a, b) in values.iter().zip(&wide) {
            assert_eq!(*b as f32, *a);
            assert_eq!(*b, f64::from(*a));
        }
    }

    #[test]
    fn budgets_parse_and_reject_out_of_bounds_counts() {
        assert_eq!(budget(&json!({"mode":"timed"})).unwrap(), Budget::Timed);
        assert_eq!(
            budget(&json!({"mode":"count","full":256,"fast":48})).unwrap(),
            Budget::Count { full: 256, full_k: 16, fast: 48, fast_k: 4, c_scale: DEFAULT_C_SCALE }
        );
        assert_eq!(
            budget(&json!({"mode":"count","full":640,"full_k":16,"fast":96,"fast_k":8})).unwrap(),
            Budget::Count { full: 640, full_k: 16, fast: 96, fast_k: 8, c_scale: DEFAULT_C_SCALE }
        );
        assert_eq!(
            budget(&json!({"mode":"count","full":640,"full_k":16,"fast":96,"fast_k":8,"c_scale":1.0})).unwrap(),
            Budget::Count { full: 640, full_k: 16, fast: 96, fast_k: 8, c_scale: 1.0 }
        );
        for bad in [0.5, 0.0, -1.0, 2.0] {
            let value = json!({"mode":"count","full":640,"fast":96,"c_scale":bad});
            assert!(budget(&value).is_err());
        }
        for bad in [
            json!({"mode":"count","full":8,"fast":48}),
            json!({"mode":"count","full":256,"fast":4}),
            json!({"mode":"count","full":256}),
            json!({"mode":"count","full":640,"full_k":8,"fast":96,"fast_k":8}),
            json!({"mode":"count","full":640,"full_k":16,"fast":96,"fast_k":3}),
            json!({"mode":"count","full":640,"full_k":"16","fast":96}),
            json!({"mode":"other"}),
        ] {
            assert!(budget(&bad).is_err());
        }
    }

    #[test]
    fn c_scale_keeps_the_historical_value_without_a_session() {
        assert_eq!(teacher_c_scale(), DEFAULT_C_SCALE);
    }

    #[test]
    fn simulations_default_to_the_existing_calibration_count_without_a_session() {
        assert_eq!(teacher_simulations(false, false), 8);
        assert_eq!(teacher_simulations(true, true), 8);
        assert!(!teacher_enabled() && !teacher_count_mode());
    }

    #[test]
    fn the_safety_cap_scales_with_the_quota_and_keeps_its_floor() {
        assert_eq!(count_cap_ns(640, 4_000_000_000), 38_400_000_000);
        assert_eq!(count_cap_ns(96, 800_000_000), 5_760_000_000);
        assert_eq!(count_cap_ns(8, 4_000_000_000), 4_000_000_000);
        assert_eq!(count_cap_ns(u64::MAX, 1), u64::MAX);
    }

    #[test]
    fn the_job_budget_stays_without_a_count_budget() {
        assert!(teacher_turn_budget(0, false, false).unwrap().is_none());
        assert!(teacher_turn_budget(0, true, true).unwrap().is_none());
    }

    #[test]
    fn search_k_defaults_to_the_job_without_a_count_budget() {
        assert_eq!(teacher_k(false, false, 16), 16);
        assert_eq!(teacher_k(true, false, 4), 4);
        assert_eq!(teacher_k(true, true, 4), 4);
    }

    #[test]
    fn model_load_rejects_bad_manifest_pins_before_touching_the_runtime() {
        let dir = std::env::temp_dir().join(format!("brass-cpu-eval-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("manifest.json");
        std::fs::write(&path, b"{}").unwrap();
        let bundle = json!({"manifest":path.to_string_lossy(),"sha256":"0".repeat(64)});
        let message = Model::load(&bundle).err().unwrap().to_string();
        assert!(message.contains("differs from its pin"), "{message}");
        let correct = sha256(b"{}");
        let bundle = json!({"manifest":path.to_string_lossy(),"sha256":correct});
        let message = Model::load(&bundle).err().unwrap().to_string();
        assert!(message.contains("unknown CPU bundle contract"), "{message}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn cache_requires_configuration_before_loading() {
        let mut cache = Cache::default();
        let message = cache
            .load(&json!({"model_id":"x","bundle":{}}))
            .err()
            .unwrap()
            .to_string();
        assert!(message.contains("configure CPU inference"), "{message}");
    }

    fn revision_job(opening: u64, long: bool) -> Value {
        json!({"seating":"all-seat-search","teacher_revision":{"contract":REVISION_CONTRACT,"short_labels":true,
            "exploration":{"contract":EXPLORATION_CONTRACT,"temperature":1.0,"opening_actions_per_era":opening,
                "long_search":long}}})
    }

    fn install_revision(job: Option<Value>) {
        let parsed = job.map(|j| revision(&j).unwrap().unwrap());
        REVISION.with(|cell| *cell.borrow_mut() = parsed.map(|r| {
            let state = Exploration::for_revision(&r);
            (r, state)
        }));
    }

    fn initial() -> NativeState {
        let start: StateDto =
            serde_json::from_str(include_str!("../../native/brass-ai-core/tests/data/initial-state.json")).unwrap();
        NativeState::import(&start).unwrap()
    }

    #[test]
    fn the_teacher_revision_parses_only_its_reviewed_shape() {
        assert!(revision(&json!({})).unwrap().is_none());
        let r = revision(&revision_job(7, true)).unwrap().unwrap();
        assert!(r.short_labels && r.long_search && r.opening_actions == 7);
        let mut cold = revision_job(7, true);
        cold["teacher_revision"]["exploration"]["temperature"] = json!(0.5);
        assert!(revision(&cold).is_err());
        let mut extra = revision_job(7, true);
        extra["teacher_revision"]["other"] = json!(1);
        assert!(revision(&extra).is_err());
        let mut reserved = revision_job(7, true);
        reserved["seating"] = json!("reserved");
        assert!(revision(&reserved).is_err());
        let mut contract = revision_job(7, true);
        contract["teacher_revision"]["contract"] = json!("training-revision-g1491/2.0.0");
        assert!(revision(&contract).is_err());
    }

    #[test]
    fn exploration_counts_each_seats_actions_per_era() {
        let mut e = Exploration::default();
        assert_eq!(e.action(0, Era::CanalEra, 1, 1), 1);
        // A later decision of the same action keeps its index.
        assert_eq!(e.action(0, Era::CanalEra, 1, 1), 1);
        assert_eq!(e.action(1, Era::CanalEra, 1, 1), 1);
        let mut last = 0;
        for round in 2..=4 {
            for remaining in [2, 1] {
                last = e.action(0, Era::CanalEra, round, remaining);
            }
        }
        // Round one has one action: the seventh canal action is the last of round four.
        assert_eq!(last, 7);
        assert_eq!(e.action(0, Era::CanalEra, 5, 2), 8);
        assert_eq!(e.action(0, Era::RailEra, 1, 2), 1);
        assert_eq!(e.action(1, Era::CanalEra, 2, 2), 2);
    }

    #[test]
    fn visit_samples_follow_the_counts_and_skip_unvisited_actions() {
        let counts = [0, 30, 10, 0, 60];
        let mut seen = [0u32; 5];
        for decision in 0..20_000 {
            seen[visit_sample(&counts, 100, 11, decision).unwrap()] += 1;
        }
        assert_eq!(seen[0] + seen[3], 0);
        for (i, expected) in [(1, 0.3), (2, 0.1), (4, 0.6)] {
            let p = f64::from(seen[i]) / 20_000.;
            assert!((p - expected).abs() < 0.02, "{i}: {p}");
        }
        assert_eq!(visit_sample(&counts, 100, 11, 5).unwrap(), visit_sample(&counts, 100, 11, 5).unwrap());
        assert!(visit_sample(&counts, 99, 11, 5).is_err());
    }

    #[test]
    fn only_a_revised_job_explores_its_opening_and_long_searches() {
        let state = initial();
        let (input, _) = root(&state);
        let info = Brass::new(input.clone(), false).unwrap().root().unwrap();
        assert!(info.actions.len() > 2);
        let planned = info.actions[0].clone();
        // All visits on the last action: a sample can only pick it.
        let mut counts = vec![0u64; info.actions.len()];
        *counts.last_mut().unwrap() = 40;
        let last = info.actions.last().unwrap().clone();
        let explore = |fast: bool, counts: Option<&[u64]>| {
            teacher_explore(planned.clone(), counts, &info, &input.view, fast, 3, 9).unwrap()
        };
        install_revision(None);
        assert_eq!(explore(true, Some(&counts)), (planned.clone(), false));
        // First canal action of this seat: inside the opening window, on either search.
        install_revision(Some(revision_job(7, false)));
        assert_eq!(explore(true, Some(&counts)), (last.clone(), true));
        assert_eq!(explore(true, None), (planned.clone(), false));
        // No opening window: only a long search explores, and only when asked to.
        install_revision(Some(revision_job(0, false)));
        assert_eq!(explore(false, Some(&counts)), (planned.clone(), false));
        install_revision(Some(revision_job(0, true)));
        assert_eq!(explore(true, Some(&counts)), (planned.clone(), false));
        assert_eq!(explore(false, Some(&counts)), (last.clone(), true));
        let receipt = revision_receipt().unwrap();
        assert_eq!(receipt["sampled"]["long"], 1);
        assert_eq!(receipt["changed"]["long"], 1);
        assert_eq!(receipt["decisions"]["short"], 1);
        assert!(teacher_explore(planned.clone(), Some(&counts[1..]), &info, &input.view, false, 3, 9).is_err());
        install_revision(None);
    }

    #[test]
    fn short_labels_carry_their_profile_and_k_and_old_rows_keep_their_bytes() {
        let row = json!({"teacher":{"requested_k":16,"played_action":"x"}});
        install_revision(None);
        let mut unchanged = row.clone();
        teacher_mark_label(&mut unchanged, false, false).unwrap();
        assert_eq!(unchanged, row);
        assert!(teacher_mark_label(&mut row.clone(), true, false).is_err());
        install_revision(Some(revision_job(7, true)));
        let mut long = row.clone();
        teacher_mark_label(&mut long, false, false).unwrap();
        assert_eq!(long, row);
        let mut short = row.clone();
        teacher_mark_label(&mut short, true, true).unwrap();
        // Without a count budget the job's own quick K (4) stands.
        assert_eq!(short["teacher"]["profile"], "fast");
        assert_eq!(short["teacher"]["requested_k"], 4);
        assert_eq!(short["teacher"]["explored"], true);
        assert_eq!(revision_receipt().unwrap()["short_labels"], 1);
        assert!(teacher_short_labels());
        install_revision(None);
        assert!(!teacher_short_labels());
    }

    #[test]
    fn cotton_is_training_only_and_the_old_revision_keeps_its_receipt() {
        let mut job=revision_job(7,true);
        job["split"]=json!("train");
        job["teacher_revision"]["exploration"]["cotton"]=crate::cpu_cotton::settings(false);
        assert!(revision(&job).unwrap().unwrap().cotton);
        for split in ["heldout","validation"] {
            let mut bad=job.clone();bad["split"]=json!(split);assert!(revision(&bad).is_err());
        }
        let mut bad=job.clone();bad["teacher_revision"]["exploration"]["cotton"]["probability"]=json!(0.5);
        assert!(revision(&bad).is_err());
        install_revision(Some(revision_job(7,true)));
        assert!(revision_receipt().unwrap().get("cotton").is_none());
        install_revision(Some(job));
        assert_eq!(revision_receipt().unwrap()["cotton"]["targeted_by_era_seat"],json!([[0,0,0,0],[0,0,0,0]]));
        install_revision(None);
    }

    /// A real short native root with a low cotton prior: coverage spends exactly
    /// the same 16 trajectories, and neither rewrites logits nor shapes labels.
    #[cfg(feature = "teacher")]
    #[test]
    fn cotton_coverage_keeps_budget_targets_and_stale_root_binding() {
        use brass_ai_core::search::{session::{clock::MonotonicClock, Identity, Mode, Session, Settings},Config,LeafEvaluator};
        use std::sync::{atomic::AtomicBool,Arc};
        struct LowCotton;
        impl LeafEvaluator<Key,String> for LowCotton {
            fn evaluate(&mut self,info:&Info<Key,String>)->brass_ai_core::search::Result<Statistics> {
                Statistics::new(info.key.choices.iter().map(|c| if crate::cpu_cotton::cotton(c) {-100.} else {0.}).collect(),[0.25;4])
            }
        }
        for (coverage,stale) in [(0,false),(2,false),(2,true)] {
            let mut job=revision_job(7,true);
            job["split"]=json!("train");job["teacher_revision"]["exploration"]["cotton"]=crate::cpu_cotton::settings(false);
            install_revision(Some(job));
            let clock=MonotonicClock::new(Arc::new(AtomicBool::new(false)));
            let state=initial();let (input,_)=root(&state);
            let info=Brass::new(input.clone(),false).unwrap().root().unwrap();
            let view=input.view.clone();
            assert_eq!(teacher_cotton_prepare(&view,&info,false),(2,false));
            let identity=Identity {game_instance:"cotton".into(),generation:1,export_sha256:"a".repeat(64)};
            let mut settings=Settings::default();settings.search=Config {k:4,simulations:16,seed:7,..Config::default()};
            let mut owner=Session::new(identity.clone(),&input,settings,SearchBudget {started_ns:0,work_deadline_ns:10_000_000_000,
                deadline_ns:11_000_000_000,reserve_ns:1_000_000_000}).unwrap();
            owner.enable_teacher_exports().unwrap();
            let mut context=input.view.context.clone();if stale {context.revision+=1;}
            owner.set_teacher_cotton(&context,coverage,false).unwrap();
            let out=crate::teacher::decide_turn(&mut owner,&identity,1,input,Mode::Count,&clock,&mut LowCotton).unwrap();
            let e=owner.take_teacher_export().unwrap();
            assert_eq!(out.completed,16);assert_eq!(e.stats.count.iter().sum::<u64>(),16);
            assert_eq!(out.effective_k,4);
            let cotton:Vec<_>=e.info.key.choices.iter().enumerate().filter(|(_,c)|crate::cpu_cotton::cotton(c)).map(|(i,_)|i).collect();
            assert!(!cotton.is_empty());
            let visits:u64=cotton.iter().map(|i|e.stats.count[*i]).sum();
            assert_eq!(visits>0,coverage>0 && !stale);
            let target=e.policy().unwrap();
            for i in cotton {assert_eq!(e.stats.logits[i],-100.);assert!(target[i]<1e-20);}
            if coverage>0 && !stale {
                let decision=(0..100).find(|d|u128::from(mix(7 ^ mix(*d ^ 0xc077_0a01)))*5<(1u128<<64)).unwrap();
                let (played,explored)=teacher_explore(out.action_ids[0][0].clone(),Some(&e.stats.count),&info,&view,false,7,decision).unwrap();
                let mut row=crate::teacher::label(&e,&info,&json!({}),&played,"root",true,owner.budget(),0).unwrap().unwrap();
                let target_before=row["teacher"]["target"].clone();
                teacher_mark_label(&mut row,false,explored).unwrap();
                assert_eq!(row["teacher"]["target"],target_before);
                assert_eq!(row["teacher"]["cotton_exploration"]["targeted"],true);
                assert_eq!(row["teacher"]["cotton_exploration"]["used_before"],0);
                let selected=info.key.choices.iter().find(|c|c.id==played).unwrap();
                let next=state.after_choice(view.context.revision,&played).unwrap();
                teacher_cotton_commit(&view,selected,next.context().revision).unwrap();
                assert_eq!(revision_receipt().unwrap()["cotton"]["long"]["targeted_builds"],1);
            }
            install_revision(None);
        }
    }

    /// An explored action leaves the turn's plan; the session must replan the
    /// action's next decision with the same budget instead of failing.
    #[cfg(feature = "teacher")]
    #[test]
    fn a_turn_session_replans_after_an_explored_action() {
        use brass_ai_core::search::{
            session::{clock::MonotonicClock, Identity, Mode, Session, Settings},
            Config, LeafEvaluator,
        };
        use std::sync::{atomic::AtomicBool, Arc};
        struct Uniform;
        impl LeafEvaluator<Key, String> for Uniform {
            fn evaluate(&mut self, info: &Info<Key, String>) -> brass_ai_core::search::Result<Statistics> {
                Statistics::new(vec![0.; info.actions.len()], [0.25; 4])
            }
        }
        let clock = MonotonicClock::new(Arc::new(AtomicBool::new(false)));
        let identity = Identity { game_instance: "explore".into(), generation: 1, export_sha256: "a".repeat(64) };
        let state = initial();
        let (first, _) = root(&state);
        let mut settings = Settings::default();
        settings.search = Config { k: 4, simulations: 16, seed: 7, ..Config::default() };
        let budget = crate::teacher::profile_budget(brass_ai_core::search::session::clock::Clock::now_ns(&clock), true).unwrap();
        let mut owner = Session::new(identity.clone(), &first, settings, budget).unwrap();
        let mut evaluator = Uniform;
        let outcome =
            crate::teacher::decide_turn(&mut owner, &identity, 1, first.clone(), Mode::Count, &clock, &mut evaluator).unwrap();
        let planned = outcome.action_ids.first().and_then(|a| a.first()).cloned().unwrap();
        // Another action whose next decision is still in this turn (a payment or continuation).
        let revision = state.context().revision;
        let next = state
            .legal_choices()
            .unwrap()
            .into_iter()
            .filter(|c| c.id != planned)
            .map(|c| state.after_choice(revision, &c.id).unwrap())
            .find(|s| s.context().phase != Phase::Action && s.context().actor == state.context().actor && !s.context().terminal)
            .expect("an action with a second decision");
        let (mut input, _) = root(&next);
        input.recall = vec![];
        let replanned = crate::teacher::decide_turn(&mut owner, &identity, 2, input, Mode::Count, &clock, &mut evaluator).unwrap();
        assert!(!replanned.provenance.starts_with("continuation/"), "{}", replanned.provenance);
        assert!(!replanned.action_ids.is_empty());
    }
}
