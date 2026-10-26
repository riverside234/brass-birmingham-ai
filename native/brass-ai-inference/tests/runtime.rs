//! Requires immutable export and golden arrays; explicitly run by L09 qualification.
use brass_ai_core::{
    evaluation::EvaluationId,
    features::{FeatureContext, FeatureTensors},
    state::{dto::StateDto, NativeState},
};
use brass_ai_inference::{
    manifest::{hash, Bundle},
    runtime::CpuEvaluator,
    Control, Evaluator, Input,
};
use serde::Deserialize;
use std::{
    path::PathBuf,
    sync::{atomic::AtomicBool, Arc},
    time::{Duration, Instant},
};
#[derive(Deserialize)]
struct Row {
    observer: u8,
    state: Vec<f32>,
    actions: Vec<f32>,
    mapping: [i64; 4],
    remaining: [f32; 4],
    descriptors: Vec<f32>,
    stable_descriptors: Vec<Vec<f32>>,
    torch: Vec<Vec<f32>>,
    ort: Vec<Vec<f32>>,
}
#[test]
#[ignore = "requires BRASS_ONNX_MANIFEST and BRASS_ONNX_GOLDEN, run by qualifier"]
fn categorical_export_arrays_and_mixed_singletons_match_both_oracles() {
    let path = PathBuf::from(std::env::var("BRASS_ONNX_MANIFEST").unwrap());
    let bytes = std::fs::read(&path).unwrap();
    let bundle = Bundle::load(&path, &hash(&bytes)).unwrap();
    let m = bundle.manifest.clone();
    let mut runtime = CpuEvaluator::load(bundle).unwrap();
    let batches: Vec<Vec<Row>> = serde_json::from_slice(
        &std::fs::read(std::env::var("BRASS_ONNX_GOLDEN").unwrap()).unwrap(),
    )
    .unwrap();
    let dto: StateDto = serde_json::from_str(include_str!(
        "../../brass-ai-core/tests/data/initial-state.json"
    ))
    .unwrap();
    let template = NativeState::import(&dto).unwrap().legal_choices().unwrap()[0].clone();
    let mut max_error = [0_f64; 5];
    let mut count = 0;
    for rows in batches {
        let mut features = Vec::new();
        let mut contexts = Vec::new();
        let mut ids = Vec::new();
        let mut choices = Vec::new();
        for (i, r) in rows.iter().enumerate() {
            let n = r.actions.len() / 327;
            features.push(FeatureTensors {
                state: r.state.clone(),
                actions: r.actions.clone(),
                legal_mask: vec![true; n],
                stable_to_relative: r.mapping,
                remaining: r.remaining,
                descriptors: r.descriptors.clone(),
                stats: Default::default(),
            });
            contexts.push(FeatureContext {
                model_id: m.model_id.clone(),
                reward_id: m.reward_id.clone(),
                profile_id: "export-golden".into(),
                style: [0.; 4],
                intent: [0.; 8],
                remaining: r.remaining.map(f64::from),
                max_cost: m.max_cost,
                stable_descriptors: Some(r.stable_descriptors.clone()),
            });
            ids.push(EvaluationId {
                request_id: i.to_string(),
                generation: 1,
                model_id: m.model_id.clone(),
                export_sha256: m.export_sha256.clone(),
                reward_id: m.reward_id.clone(),
                profile_id: "export-golden".into(),
                revision: 0,
                observer: r.observer,
            });
            choices.push(
                (0..n)
                    .map(|j| {
                        let mut c = template.clone();
                        c.id = format!("golden-{j}");
                        c
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let inputs: Vec<_> = (0..rows.len())
            .map(|i| Input {
                id: &ids[i],
                context: &contexts[i],
                choices: &choices[i],
                tensors: &features[i],
            })
            .collect();
        let control = Control {
            deadline: Instant::now() + Duration::from_secs(30),
            cancelled: Arc::new(AtomicBool::new(false)),
        };
        let out = runtime.evaluate(&inputs, &control).unwrap();
        for (i, (actual, row)) in out.evaluations.iter().zip(&rows).enumerate() {
            let single = runtime
                .evaluate(std::slice::from_ref(&inputs[i]), &control)
                .unwrap();
            let s = &single.evaluations[0];
            let observed: [&[f32]; 5] = [
                &actual.logits,
                &actual.win_share,
                &actual.placement,
                &actual.vp,
                &actual.utility,
            ];
            let singleton: [&[f32]; 5] = [&s.logits, &s.win_share, &s.placement, &s.vp, &s.utility];
            for h in 0..5 {
                for reference in [&row.torch[h][..], &row.ort[h][..], singleton[h]] {
                    assert_eq!(observed[h].len(), reference.len());
                    for (a, b) in observed[h].iter().zip(reference) {
                        let d = (*a as f64 - *b as f64).abs();
                        max_error[h] = max_error[h].max(d);
                        assert!(
                            a.is_finite() && b.is_finite() && d <= 2e-4 + 2e-4 * (*b as f64).abs(),
                            "head {h}: {a} vs {b}"
                        );
                    }
                }
            }
            count += 1;
        }
    }
    println!(
        "{}",
        serde_json::json!({"positions":count,"max_errors":max_error,"runtime":runtime.receipt()})
    );
}
