use brass_ai_core::{
    evaluation::{EvaluationId, EvaluationStats},
    features::{FeatureContext, FeatureEncoder},
    state::{dto::StateDto, NativeState},
    types::Seat,
};
use brass_ai_inference::{
    batch::Packed,
    manifest::Manifest,
    service::{finish, RawRow, ReferenceEvaluator, ReferenceTransport},
    Control, Evaluator, Input, Result,
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

fn manifest() -> Manifest {
    serde_json::from_str(include_str!("data/manifest.json")).unwrap()
}
fn fixture() -> (Manifest, FeatureContext, EvaluationId, NativeState) {
    let m = manifest();
    let c = FeatureContext {
        model_id: m.model_id.clone(),
        reward_id: m.reward_id.clone(),
        profile_id: "neutral".into(),
        style: [0.; 4],
        intent: [0.; 8],
        remaining: [0., 0.01, 0.05, 0.075],
        max_cost: m.max_cost,
        stable_descriptors: None,
    };
    let dto: StateDto = serde_json::from_str(include_str!(
        "../../brass-ai-core/tests/data/initial-state.json"
    ))
    .unwrap();
    let id = EvaluationId {
        request_id: "root/1".into(),
        generation: 1,
        model_id: m.model_id.clone(),
        export_sha256: m.export_sha256.clone(),
        reward_id: m.reward_id.clone(),
        profile_id: c.profile_id.clone(),
        observer: dto.context.actor as u8,
        revision: dto.context.revision,
    };
    (m, c, id, NativeState::import(&dto).unwrap())
}
fn control() -> Control {
    Control {
        deadline: Instant::now() + Duration::from_secs(5),
        cancelled: Arc::new(AtomicBool::new(false)),
    }
}
fn raw(i: &Input<'_>) -> RawRow {
    RawRow {
        id: i.id.clone(),
        choice_ids: i.choices.iter().map(|c| c.id.clone()).collect(),
        logits: vec![0.; i.choices.len()],
        values: [[0.1, 0.2, 0.3, 0.4]; 4],
    }
}

#[test]
fn pinned_contract_rejects_false_admission_and_layouts() {
    let m = manifest();
    m.validate().unwrap();
    for mutation in 0..7 {
        let mut bad = m.clone();
        match mutation {
            0 => bad.intra_threads = 2,
            1 => bad.inter_threads = 2,
            2 => bad.sequential = false,
            3 => bad.feature_schema = "0".repeat(64),
            4 => bad.mechanical_only = false,
            5 => bad.utility_calibration = Some(serde_json::json!({"passed":true})),
            _ => bad.runtime_version = "unknown".into(),
        };
        assert!(bad.validate().is_err());
    }
}
#[test]
fn reference_rejects_stale_reordered_nonfinite_and_partial_outputs() {
    let (_, c, id, state) = fixture();
    let choices = state.legal_choices().unwrap();
    let view = state
        .observation(Seat::try_from(id.observer).unwrap())
        .unwrap();
    let mut encoder = FeatureEncoder::default();
    let t = encoder.encode(&view, &choices, &c, &mut || Ok(())).unwrap();
    let input = Input {
        id: &id,
        context: &c,
        choices: &choices,
        tensors: t,
    };
    for mutation in 0..6 {
        let mut r = raw(&input);
        match mutation {
            0 => r.id.generation += 1,
            1 => r.id.model_id = "foreign".into(),
            2 => r.id.observer = (r.id.observer + 1) % 4,
            3 => r.choice_ids.reverse(),
            4 => r.logits[0] = f32::NAN,
            _ => r.values[0][1] = f32::INFINITY,
        };
        assert!(finish(
            std::slice::from_ref(&input),
            vec![r],
            EvaluationStats::default()
        )
        .is_err());
    }
    assert!(finish(
        std::slice::from_ref(&input),
        vec![],
        EvaluationStats::default()
    )
    .is_err());
    let r = finish(
        std::slice::from_ref(&input),
        vec![raw(&input)],
        EvaluationStats::default(),
    )
    .unwrap();
    assert_eq!(r.evaluations[0].win_share, [0.1, 0.2, 0.3, 0.4]); // no second seat rotation
}
#[test]
fn input_context_mapping_descriptors_masks_and_model_are_checked() {
    let (m, c, id, state) = fixture();
    let choices = state.legal_choices().unwrap();
    let view = state
        .observation(Seat::try_from(id.observer).unwrap())
        .unwrap();
    let mut encoder = FeatureEncoder::default();
    encoder.encode(&view, &choices, &c, &mut || Ok(())).unwrap();
    let source = encoder.tensors().unwrap();
    let mut t = brass_ai_core::features::FeatureTensors {
        state: source.state.clone(),
        actions: source.actions.clone(),
        legal_mask: source.legal_mask.clone(),
        stable_to_relative: source.stable_to_relative,
        remaining: source.remaining,
        descriptors: source.descriptors.clone(),
        stats: Default::default(),
    };
    m.validate_input(&Input {
        id: &id,
        context: &c,
        choices: &choices,
        tensors: &t,
    })
    .unwrap();
    t.descriptors[4] = 0.;
    assert!(m
        .validate_input(&Input {
            id: &id,
            context: &c,
            choices: &choices,
            tensors: &t
        })
        .is_err());
    t.descriptors[4] = 1.;
    t.remaining[0] = 0.075;
    assert!(m
        .validate_input(&Input {
            id: &id,
            context: &c,
            choices: &choices,
            tensors: &t
        })
        .is_err());
    t.remaining = source.remaining;
    t.legal_mask[0] = false;
    assert!(m
        .validate_input(&Input {
            id: &id,
            context: &c,
            choices: &choices,
            tensors: &t
        })
        .is_err());
    t.legal_mask[0] = true;
    let mut wrong = id.clone();
    wrong.model_id = "wrong".into();
    assert!(m
        .validate_input(&Input {
            id: &wrong,
            context: &c,
            choices: &choices,
            tensors: &t
        })
        .is_err());
}
struct LateService {
    seen: Arc<AtomicBool>,
}
impl ReferenceTransport for LateService {
    fn evaluate(&mut self, inputs: &[Input<'_>], c: &Control) -> Result<Vec<RawRow>> {
        self.seen.store(true, Ordering::Release);
        c.cancelled.store(true, Ordering::Release);
        // A late transport still owns/reads every borrowed buffer until return.
        assert!(!inputs[0].tensors.actions.is_empty());
        Ok(inputs.iter().map(raw).collect())
    }
}
#[test]
fn cancelled_reference_cannot_publish_or_reuse_borrowed_result() {
    let (m, c, id, state) = fixture();
    let choices = state.legal_choices().unwrap();
    let view = state
        .observation(Seat::try_from(id.observer).unwrap())
        .unwrap();
    let mut encoder = FeatureEncoder::default();
    let t = encoder.encode(&view, &choices, &c, &mut || Ok(())).unwrap();
    let seen = Arc::new(AtomicBool::new(false));
    let mut evaluator = ReferenceEvaluator {
        manifest: m,
        transport: LateService { seen: seen.clone() },
    };
    assert!(evaluator
        .evaluate(
            &[Input {
                id: &id,
                context: &c,
                choices: &choices,
                tensors: t
            }],
            &control()
        )
        .is_err());
    assert!(seen.load(Ordering::Acquire));
    // This reborrow is legal only after synchronous transport retirement.
    encoder.encode(&view, &choices, &c, &mut || Ok(())).unwrap();
}
#[test]
fn packing_reuses_capacity_and_zeroes_padding_after_width_changes() {
    let (_, c, id, state) = fixture();
    let choices = state.legal_choices().unwrap();
    let view = state
        .observation(Seat::try_from(id.observer).unwrap())
        .unwrap();
    let mut e1 = FeatureEncoder::default();
    let mut e2 = FeatureEncoder::default();
    let a = e1.encode(&view, &choices, &c, &mut || Ok(())).unwrap();
    let b = e2.encode(&view, &choices[..1], &c, &mut || Ok(())).unwrap();
    let inputs = [
        Input {
            id: &id,
            context: &c,
            choices: &choices,
            tensors: a,
        },
        Input {
            id: &id,
            context: &c,
            choices: &choices[..1],
            tensors: b,
        },
    ];
    let mut p = Packed::default();
    p.pack(&inputs, choices.len());
    let ptr = p.actions.as_ptr();
    p.pack(&inputs, choices.len());
    assert_eq!(ptr, p.actions.as_ptr());
    assert!(p.actions[(choices.len() + 1) * 327..]
        .iter()
        .all(|x| *x == 0.));
    assert!(p.masks[choices.len() + 1..].iter().all(|x| !*x));
}
