//! Prepared-input two-owner probe. Research-only; bounded by the outer runner.
use brass_ai_core::{
    evaluation::EvaluationId,
    features::{FeatureEncoder, FeatureTensors},
    search::brass::Key,
};
use brass_ai_inference::{
    manifest::{hash, Bundle},
    nanos,
    runtime::CpuEvaluator,
    Control, Evaluator, Input,
};
use serde::{Deserialize, Serialize};
use std::{
    path::PathBuf,
    sync::{atomic::AtomicBool, Arc, Barrier},
    time::{Duration, Instant},
};

#[derive(Clone, Deserialize)]
struct Case {
    name: String,
    held_out: bool,
    keys: Vec<Key>,
}
#[derive(Deserialize)]
struct Request {
    manifest: PathBuf,
    cases: Vec<Case>,
    iterations: usize,
}
#[derive(Serialize)]
struct Sample {
    total_ns: u64,
    stats: brass_ai_core::evaluation::EvaluationStats,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert_eq!(args.len(), 2);
    let request: Request = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    assert!((1..=64).contains(&request.iterations) && request.cases.len() <= 24);
    let barrier = Arc::new(Barrier::new(3));
    let (ready, received) = std::sync::mpsc::channel();
    let mut owners = Vec::new();
    for owner in 0..2 {
        let manifest = request.manifest.clone();
        let cases = request.cases.clone();
        let iterations = request.iterations;
        let barrier = barrier.clone();
        let ready = ready.clone();
        owners.push(std::thread::spawn(move || {
          let outcome = std::panic::catch_unwind(|| {
            let bytes = std::fs::read(&manifest).unwrap();
            let bundle = Bundle::load(&manifest, &hash(&bytes)).unwrap();
            let m = bundle.manifest.clone();
            let mut evaluator = CpuEvaluator::load(bundle).unwrap();
            let mut prepared = Vec::new(); let mut first_shapes = Vec::new();
            for case in &cases {
                assert!(!case.keys.is_empty() && case.keys.len() <= 4);
                let mut features: Vec<FeatureTensors> = Vec::new(); let mut ids = Vec::new();
                for (i, key) in case.keys.iter().enumerate() {
                    let view = key.information.observation.view.as_ref().unwrap();
                    let mut encoder = FeatureEncoder::default();
                    let t=encoder.encode(view, &key.choices, &key.context, &mut || Ok(())).unwrap();
                    features.push(FeatureTensors { state:t.state.clone(),actions:t.actions.clone(),legal_mask:t.legal_mask.clone(),
                        stable_to_relative:t.stable_to_relative,remaining:t.remaining,descriptors:t.descriptors.clone(),stats:Default::default() });
                    ids.push(EvaluationId { request_id: format!("{}-{i}", case.name), generation: 1,
                        model_id: m.model_id.clone(), export_sha256: m.export_sha256.clone(),
                        reward_id: m.reward_id.clone(), profile_id: key.context.profile_id.clone(),
                        revision: view.context.revision, observer: view.seat.into() });
                }
                prepared.push((features, ids));
            }
            let control = Control { deadline: Instant::now()+Duration::from_secs(75), cancelled: Arc::new(AtomicBool::new(false)) };
            // Shape specialization is paid and reported before steady-state timing.
            for (case, (features, ids)) in cases.iter().zip(&prepared) {
                let inputs: Vec<_> = case.keys.iter().enumerate().map(|(i,k)| Input {
                    id:&ids[i],context:&k.context,choices:&k.choices,tensors:&features[i] }).collect();
                let start=Instant::now(); evaluator.evaluate(&inputs,&control).unwrap();
                first_shapes.push(serde_json::json!({"name":case.name,"first_ns":nanos(start)}));
                evaluator.evaluate(&inputs,&control).unwrap();
            }
            ready.send(true).unwrap(); barrier.wait(); barrier.wait();
            let mut rows = Vec::new();
            for (case, (features, ids)) in cases.iter().zip(&prepared) {
                let inputs: Vec<_> = case.keys.iter().enumerate().map(|(i,k)| Input {
                    id:&ids[i],context:&k.context,choices:&k.choices,tensors:&features[i] }).collect();
                let mut samples=Vec::new(); let mut outputs=None;
                for _ in 0..iterations {
                    let start=Instant::now(); let output=evaluator.evaluate(&inputs,&control).unwrap();
                    samples.push(Sample{total_ns:nanos(start),stats:output.stats});
                    if outputs.is_none() { outputs=Some(output.evaluations); }
                }
                rows.push(serde_json::json!({"name":case.name,"held_out":case.held_out,"batch":inputs.len(),
                    "candidates":case.keys.iter().map(|k| k.choices.len()).collect::<Vec<_>>(),
                    "samples":samples,"outputs":outputs}));
            }
            serde_json::json!({"owner":owner,"receipt":evaluator.receipt(),"first_shapes":first_shapes,"rows":rows})
          });
          if outcome.is_err() { let _ = ready.send(false); }
          outcome.expect("provider probe owner failed")
        }));
    }
    for _ in 0..2 {
        assert!(
            received.recv_timeout(Duration::from_secs(30))?,
            "provider failed before measurement"
        );
    }
    barrier.wait();
    println!("MEASURE");
    barrier.wait();
    let rows: Vec<_> = owners
        .into_iter()
        .map(|o| o.join().expect("owner failed"))
        .collect();
    println!("MEASURE_DONE");
    let path = PathBuf::from(&args[1]);
    assert!(!path.exists());
    std::fs::write(
        path,
        serde_json::to_vec(&serde_json::json!({"owners":rows}))?,
    )?;
    Ok(())
}
