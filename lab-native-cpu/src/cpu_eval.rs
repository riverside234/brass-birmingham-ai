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
}

/// Load the frozen model named by `request["inference"]`, if any.
pub fn teacher_install(request: &Value, expected: &Value, max_cost: f64) -> Result<()> {
    let block = &request["inference"];
    if block.is_null() {
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
    TEACHER.with(|cell| {
        *cell.borrow_mut() = Some(Teacher {
            model,
            budget,
            runtime_sha256,
            positions: 0,
            forward_ns: 0,
        })
    });
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
    TEACHER.with(|cell| match cell.borrow().as_ref() {
        Some(t) => json!({"contract":CONTRACT,"backend":"cpu-ort","runtime_sha256":t.runtime_sha256,
            "provider":"CPUExecutionProvider","intra_threads":1,"inter_threads":1,"sequential":true,
            "budget":match t.budget {
                Budget::Timed => json!({"mode":"timed"}),
                Budget::Count{full,full_k,fast,fast_k,c_scale} => json!({"mode":"count","full":full,"full_k":full_k,"fast":fast,"fast_k":fast_k,"c_scale":c_scale}),
            },
            "positions":t.positions,"forward_ns":t.forward_ns,"model":t.model.receipt()}),
        None => Value::Null,
    })
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
}
