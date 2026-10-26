use crate::{
    batch::Packed,
    invalid,
    manifest::{Bundle, Manifest, INPUTS, OUTPUTS},
    nanos, require,
    service::{finish, validate_batch, RawRow},
    BatchOutput, Control, Evaluator, Input, Result,
};
use brass_ai_core::{
    evaluation::EvaluationStats,
    features::{ACTION_DIM, STATE_DIM},
};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    tensor::TensorElementType,
    value::{TensorRef, ValueType},
};
use std::{marker::PhantomData, rc::Rc, sync::OnceLock, time::Instant};

static RUNTIME: OnceLock<std::result::Result<String, String>> = OnceLock::new();
fn ort_error(error: ort::Error) -> brass_ai_core::CoreError {
    invalid(format!("ONNX runtime: {error}"))
}
struct Arrays<'a> {
    b: usize,
    n: usize,
    states: &'a [f32],
    actions: &'a [f32],
    masks: &'a [bool],
    mappings: &'a [i64],
    remaining: &'a [f32],
    descriptors: &'a [f32],
}

/// !Send/!Sync: construct, warm, run and drop on the session's owning thread.
/// ```compile_fail
/// use brass_ai_inference::runtime::CpuEvaluator;
/// fn require_send<T: Send>() {}
/// require_send::<CpuEvaluator>();
/// ```
pub struct CpuEvaluator {
    manifest: Manifest,
    manifest_sha256: String,
    session: Session,
    packed: Packed,
    failed: bool,
    pub load_ns: u64,
    pub warmup_ns: u64,
    _owner: PhantomData<Rc<()>>,
}
impl CpuEvaluator {
    pub fn load(bundle: Bundle) -> Result<Self> {
        let start = Instant::now();
        let runtime_key = bundle.manifest.runtime_sha256.clone();
        let initialized = RUNTIME.get_or_init(|| {
            ort::init_from(bundle.runtime_path.to_string_lossy())
                .with_name("brass-cpu")
                .with_telemetry(false)
                .commit()
                .map(|_| runtime_key.clone())
                .map_err(|e| e.to_string())
        });
        require(
            initialized
                .as_ref()
                .map(|k| k == &runtime_key)
                .unwrap_or(false),
            "runtime initialization failed or runtime identity changed",
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
            .commit_from_memory(&bundle.model_bytes)
            .map_err(ort_error)?;
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
        require(
            specs.len() == 6 && session.outputs.len() == 5,
            "unexpected ONNX tensor count",
        )?;
        for (i, name) in INPUTS.iter().enumerate() {
            let ty = if i == 2 {
                TensorElementType::Bool
            } else if i == 3 {
                TensorElementType::Int64
            } else {
                TensorElementType::Float32
            };
            require(
                specs.iter().filter(|(n, _)| n == name).count() == 1,
                "unexpected ONNX input name",
            )?;
            let value = specs
                .iter()
                .find(|(n, _)| n == name)
                .expect("checked name")
                .1;
            require(
                matches!(value,ValueType::Tensor{ty:got,shape,..} if *got==ty && &**shape==shapes[i]),
                "ONNX input dtype/shape mismatch",
            )?;
        }
        for (i, out) in session.outputs.iter().enumerate() {
            let shape = if i == 0 { &[-1, -1] } else { &[-1, 4] };
            require(
                out.name == OUTPUTS[i]
                    && matches!(&out.output_type,ValueType::Tensor{ty,shape:s,..} if *ty==TensorElementType::Float32 && &**s==shape),
                "ONNX output dtype/shape mismatch",
            )?;
        }
        let load_ns = nanos(start);
        let mut result = Self {
            manifest: bundle.manifest,
            manifest_sha256: bundle.manifest_sha256,
            session,
            packed: Packed::default(),
            failed: false,
            load_ns,
            warmup_ns: 0,
            _owner: PhantomData,
        };
        let start = Instant::now();
        let states = vec![0.; STATE_DIM];
        let actions = vec![0.; ACTION_DIM];
        let mut d = vec![0.; 136];
        for s in 0..4 {
            d[s * 34 + 4] = 1.;
        }
        let remaining = [result.manifest.max_cost as f32; 4];
        for _ in 0..2 {
            let a = Arrays {
                b: 1,
                n: 1,
                states: &states,
                actions: &actions,
                masks: &[true],
                mappings: &[0, 1, 2, 3],
                remaining: &remaining,
                descriptors: &d,
            };
            let (raw, _) = run(&mut result.session, a, None)?;
            require(
                raw.iter().flatten().all(|x| x.is_finite()),
                "nonfinite warm-up output",
            )?;
        }
        result.warmup_ns = nanos(start);
        Ok(result)
    }
    pub fn receipt(&self) -> serde_json::Value {
        serde_json::json!({"manifest_sha256":self.manifest_sha256,"model_id":self.manifest.model_id,
            "export_sha256":self.manifest.export_sha256,"runtime_sha256":self.manifest.runtime_sha256,
            "runtime_version":self.manifest.runtime_version,"binding":"ort/2.0.0-rc.10","runtime_build":ort::info(),
            "provider":"CPUExecutionProvider","intra_threads":1,"inter_threads":1,"sequential":true,
            "cached_models":1,"load_ns":self.load_ns,"warmup_ns":self.warmup_ns,"mechanical_only":true,"live_ready":false})
    }
}
fn run(
    session: &mut Session,
    a: Arrays<'_>,
    control: Option<&Control>,
) -> Result<(Vec<Vec<f32>>, EvaluationStats)> {
    let start = Instant::now();
    let states = TensorRef::from_array_view(([a.b, STATE_DIM], a.states)).map_err(ort_error)?;
    let actions =
        TensorRef::from_array_view(([a.b, a.n, ACTION_DIM], a.actions)).map_err(ort_error)?;
    let masks = TensorRef::from_array_view(([a.b, a.n], a.masks)).map_err(ort_error)?;
    let mappings = TensorRef::from_array_view(([a.b, 4], a.mappings)).map_err(ort_error)?;
    let remaining = TensorRef::from_array_view(([a.b, 4], a.remaining)).map_err(ort_error)?;
    let descriptors =
        TensorRef::from_array_view(([a.b, 4, 34], a.descriptors)).map_err(ort_error)?;
    let mut stats = EvaluationStats {
        binding_ns: nanos(start),
        ..Default::default()
    };
    if let Some(c) = control {
        c.check()?;
    }
    let start = Instant::now();
    let outputs = session
        .run(
            ort::inputs!["states"=>states,"actions"=>actions,"legal_mask"=>masks,
        "stable_to_relative"=>mappings,"remaining"=>remaining,"descriptors"=>descriptors],
        )
        .map_err(ort_error)?;
    stats.inference_ns = nanos(start);
    // Runtime work is synchronous. The views/inputs still live here on cancel.
    if let Some(c) = control {
        c.check()?;
    }
    let start = Instant::now();
    let mut raw = Vec::with_capacity(5);
    for (i, name) in OUTPUTS.iter().enumerate() {
        let (shape, values) = outputs[*name]
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?;
        require(
            **shape == [a.b as i64, if i == 0 { a.n as i64 } else { 4 }],
            "ONNX response shape mismatch",
        )?;
        stats.output_copy_bytes += values.len() * 4;
        raw.push(values.to_vec());
    }
    stats.output_copy_ns = nanos(start);
    Ok((raw, stats))
}
impl Evaluator for CpuEvaluator {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    fn evaluate(&mut self, inputs: &[Input<'_>], control: &Control) -> Result<BatchOutput> {
        require(!self.failed, "ONNX evaluator retired after runtime failure")?;
        validate_batch(&self.manifest, inputs)?;
        control.check()?;
        let n = inputs.iter().map(|i| i.choices.len()).max().unwrap_or(0);
        let start = Instant::now();
        let copied = if inputs.len() == 1 {
            0
        } else {
            self.packed.pack(inputs, n)
        };
        let packing_ns = nanos(start);
        let a = if inputs.len() == 1 {
            let t = inputs[0].tensors;
            Arrays {
                b: 1,
                n,
                states: &t.state,
                actions: &t.actions,
                masks: &t.legal_mask,
                mappings: &t.stable_to_relative,
                remaining: &t.remaining,
                descriptors: &t.descriptors,
            }
        } else {
            let p = &self.packed;
            Arrays {
                b: inputs.len(),
                n,
                states: &p.states,
                actions: &p.actions,
                masks: &p.masks,
                mappings: &p.mappings,
                remaining: &p.remaining,
                descriptors: &p.descriptors,
            }
        };
        let result = (|| {
            let (raw, mut stats) = run(&mut self.session, a, Some(control))?;
            stats.packing_ns = packing_ns;
            stats.input_copy_bytes = copied;
            let copy_start = Instant::now();
            let mut rows = Vec::with_capacity(inputs.len());
            for (i, input) in inputs.iter().enumerate() {
                let count = input.choices.len();
                require(
                    raw[0][i * n + count..(i + 1) * n]
                        .iter()
                        .all(|x| *x == f32::NEG_INFINITY),
                    "ONNX padding mask mismatch",
                )?;
                let mut values = [[0.; 4]; 4];
                for h in 0..4 {
                    values[h].copy_from_slice(&raw[h + 1][i * 4..i * 4 + 4]);
                }
                rows.push(RawRow {
                    id: input.id.clone(),
                    choice_ids: input.choices.iter().map(|c| c.id.clone()).collect(),
                    logits: raw[0][i * n..i * n + count].to_vec(),
                    values,
                });
                stats.output_copy_bytes += (count + 16) * 4;
            }
            stats.output_copy_ns += nanos(copy_start);
            control.check()?;
            finish(inputs, rows, stats)
        })();
        if result.is_err() && control.status().is_none() {
            self.failed = true;
        }
        result
    }
}
