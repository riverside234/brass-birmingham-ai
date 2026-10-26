use brass_ai_core::{
    features::{
        session::{FeatureCommand as F, FeatureSession},
        FeatureContext, FeatureEncoder, ACTION_DIM, STATE_DIM,
    },
    state::{
        dto::StateDto,
        simulation::{SimulationCommand as S, Simulator},
        NativeState,
    },
    types::Seat,
    CoreError,
};

fn fixture() -> StateDto {
    serde_json::from_str(include_str!("data/initial-state.json")).unwrap()
}
fn context() -> FeatureContext {
    FeatureContext {
        model_id: "test-model".into(),
        reward_id: "test-reward".into(),
        profile_id: "test-profile".into(),
        style: [0.; 4],
        intent: [0.; 8],
        remaining: [0., 0.01, 0.05, 0.075],
        max_cost: 0.075,
        stable_descriptors: None,
    }
}
#[test]
fn full_candidates_reuse_buffers_and_card_equivalent_effects() {
    let state = NativeState::import(&fixture()).unwrap();
    let view = state
        .observation(Seat::try_from(fixture().context.actor as u8).unwrap())
        .unwrap();
    let choices = state.legal_choices().unwrap();
    let mut encoder = FeatureEncoder::default();
    let t = encoder
        .encode(&view, &choices, &context(), &mut || Ok(()))
        .unwrap();
    assert_eq!(t.state.len(), STATE_DIM);
    assert_eq!(t.actions.len(), choices.len() * ACTION_DIM);
    assert_eq!(t.legal_mask, vec![true; choices.len()]);
    assert!(t.stats.effect_hits > 0);
    assert_eq!(t.stats.effect_hits + t.stats.effect_misses, choices.len());
    let original = t.actions.clone();
    let ptr = t.actions.as_ptr();
    let state_ptr = t.state.as_ptr();
    let t = encoder
        .encode(&view, &choices, &context(), &mut || Ok(()))
        .unwrap();
    assert_eq!(t.actions, original);
    assert_eq!(ptr, t.actions.as_ptr());
    assert_eq!(state_ptr, t.state.as_ptr());
    // Reordered candidates retain full identity and exact action rows.
    let mut reversed = choices.clone();
    reversed.reverse();
    let t = encoder
        .encode(&view, &reversed, &context(), &mut || Ok(()))
        .unwrap();
    for (a, b) in t
        .actions
        .chunks(ACTION_DIM)
        .zip(original.chunks(ACTION_DIM).rev())
    {
        assert_eq!(a, b);
    }
}
#[test]
fn cancellation_and_invalid_input_clear_published_tensors() {
    let state = NativeState::import(&fixture()).unwrap();
    let view = state
        .observation(Seat::try_from(fixture().context.actor as u8).unwrap())
        .unwrap();
    let choices = state.legal_choices().unwrap();
    let mut encoder = FeatureEncoder::default();
    encoder
        .encode(&view, &choices, &context(), &mut || Ok(()))
        .unwrap();
    let mut checkpoints = 0;
    assert!(encoder
        .encode(&view, &choices, &context(), &mut || {
            checkpoints += 1;
            if checkpoints > choices.len() + 2 {
                Err(CoreError::Invalid("cancelled".into()))
            } else {
                Ok(())
            }
        })
        .is_err());
    assert!(encoder.tensors().is_none());
    let duplicate = vec![choices[0].clone(), choices[0].clone()];
    assert!(encoder
        .encode(&view, &duplicate, &context(), &mut || Ok(()))
        .is_err());
    assert!(encoder.tensors().is_none());
    let mut forged = choices.clone();
    forged[0].id.push('x');
    assert!(encoder
        .encode(&view, &forged, &context(), &mut || Ok(()))
        .is_err());
    let mut context = context();
    context.remaining[0] = f64::NAN;
    assert!(encoder
        .encode(&view, &choices, &context, &mut || Ok(()))
        .is_err());
    assert!(encoder.tensors().is_none());
}
#[test]
fn descriptors_are_critic_only_and_relative_budgets_stay_stable() {
    let state = NativeState::import(&fixture()).unwrap();
    let view = state
        .observation(Seat::try_from(fixture().context.actor as u8).unwrap())
        .unwrap();
    let choices = state.legal_choices().unwrap();
    let mut encoder = FeatureEncoder::default();
    let t = encoder
        .encode(&view, &choices, &context(), &mut || Ok(()))
        .unwrap();
    for row in t.descriptors.chunks(34) {
        assert_eq!(row[4], 1.);
        assert_eq!(row.iter().sum::<f32>(), 1.);
    }
    let (s, a) = (t.state.clone(), t.actions.clone());
    let mut ctx = context();
    let d: Vec<Vec<f32>> = (0..4)
        .map(|i| {
            let mut v = vec![0.; 34];
            v[i] = 1.;
            v[5] = i as f32 / 4.;
            v
        })
        .collect();
    ctx.stable_descriptors = Some(d.clone());
    let t = encoder
        .encode(&view, &choices, &ctx, &mut || Ok(()))
        .unwrap();
    assert_eq!(t.state, s);
    assert_eq!(t.actions, a);
    assert_eq!(t.remaining, [0., 0.01, 0.05, 0.075]);
    for (stable, descriptor) in d.iter().enumerate() {
        let r = t.stable_to_relative[stable] as usize;
        assert_eq!(&t.descriptors[r * 34..(r + 1) * 34], descriptor);
    }
    ctx.stable_descriptors.as_mut().unwrap()[0][4] = 1.;
    assert!(ctx.validate().is_err());
    ctx.stable_descriptors = Some(vec![vec![0.; 34]; 4]);
    assert!(ctx.validate().is_err());
    ctx = context();
    ctx.remaining[0] = 0.076;
    assert!(ctx.validate().is_err());
    ctx = context();
    ctx.style[0] = 1.01;
    assert!(ctx.validate().is_err());
}
#[test]
fn pages_are_owner_local_immutable_and_fail_closed() {
    let mut sim = Simulator::default();
    let handle = sim
        .execute(
            S::Import {
                state: Box::new(fixture()),
            },
            0,
        )
        .unwrap()
        .handle
        .unwrap();
    let revision = fixture().context.revision;
    let mut features = FeatureSession::default();
    let first = features
        .execute(
            F::FromState {
                handle,
                revision,
                context: Box::new(context()),
            },
            &sim,
            0,
        )
        .unwrap();
    assert_eq!(first.encoding_id % 2, 0);
    assert!(first.tensors.total_candidates > 64);
    let page = features
        .execute(
            F::Page {
                encoding_id: first.encoding_id,
                start: 64,
                limit: 64,
            },
            &sim,
            0,
        )
        .unwrap();
    assert_eq!(page.tensors.candidate_offsets, [64, 128]);
    assert_eq!(page.tensors.state, first.tensors.state);
    for (start, limit) in [(0, 0), (0, 65), (usize::MAX, 1)] {
        assert!(features
            .execute(
                F::Page {
                    encoding_id: first.encoding_id,
                    start,
                    limit
                },
                &sim,
                0
            )
            .is_err());
    }
    let other = Simulator::default();
    let mut foreign = FeatureSession::default();
    assert!(foreign
        .execute(
            F::Page {
                encoding_id: first.encoding_id,
                start: 0,
                limit: 64
            },
            &other,
            1
        )
        .is_err());
    assert!(foreign
        .execute(
            F::FromState {
                handle,
                revision,
                context: Box::new(context())
            },
            &other,
            1
        )
        .is_err());
    assert!(features
        .execute(
            F::FromState {
                handle,
                revision: revision + 1,
                context: Box::new(context())
            },
            &sim,
            0
        )
        .is_err());
    assert!(features
        .execute(
            F::Page {
                encoding_id: first.encoding_id,
                start: 0,
                limit: 64
            },
            &sim,
            0
        )
        .is_err());
    let second = features
        .execute(
            F::FromState {
                handle,
                revision,
                context: Box::new(context()),
            },
            &sim,
            0,
        )
        .unwrap();
    assert!(second.encoding_id > first.encoding_id);
    assert!(features
        .execute(
            F::Page {
                encoding_id: first.encoding_id,
                start: 0,
                limit: 64
            },
            &sim,
            0
        )
        .is_err());
}
#[test]
fn typed_input_never_exposes_setup_or_nonacting_seats() {
    let mut sim = Simulator::default();
    let handle = sim.execute(S::New, 0).unwrap().handle.unwrap();
    assert!(sim.feature_input(handle, 0).is_err());
    let state = NativeState::import(&fixture()).unwrap();
    let mut encoder = FeatureEncoder::default();
    let actor = fixture().context.actor as u8;
    let choices = state.legal_choices().unwrap();
    for s in 0..4 {
        let view = state.observation(Seat::try_from(s).unwrap()).unwrap();
        assert_eq!(
            encoder
                .encode(&view, &choices, &context(), &mut || Ok(()))
                .is_ok(),
            s == actor
        );
    }
}
