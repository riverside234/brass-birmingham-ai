use super::*;
use crate::pair::Decision;
use brass_ai_core::{
    features::FeatureContext,
    search::reward::{Insolvency, LedgerState, Reward},
    state::{dto::StateDto, NativeState},
    types::Seat,
};

fn fixture() -> (Pair, Input, Identity) {
    let manifest: Manifest = serde_json::from_str(include_str!(
        "../../../brass-ai-inference/tests/data/manifest.json"
    ))
    .unwrap();
    let dto: StateDto = serde_json::from_str(include_str!(
        "../../../brass-ai-core/tests/data/initial-state.json"
    ))
    .unwrap();
    let view = NativeState::import(&dto)
        .unwrap()
        .observation(Seat::try_from(dto.context.actor as u8).unwrap())
        .unwrap();
    let reward = Reward {
        version: "brass-combined-return/2.0.0".into(),
        win: 1.,
        placement: 0.1,
        vp: 0.001,
        vp_cap: Some(180),
        insolvency: Insolvency {
            event_cost: 0.02,
            max_cost: 0.075,
        },
    };
    assert_eq!(reward.identity().unwrap(), manifest.reward_id);
    let input = Input {
        view,
        recall: vec![],
        context: FeatureContext {
            model_id: manifest.model_id.clone(),
            reward_id: manifest.reward_id.clone(),
            profile_id: "neutral".into(),
            style: [0.; 4],
            intent: [0.; 8],
            remaining: [0.075; 4],
            max_cost: 0.075,
            stable_descriptors: None,
        },
        reward,
        ledger: LedgerState {
            spent: [0.; 4],
            settled: vec![],
        },
        shuffle_tape: None,
        sample_seed: 7,
    };
    let identity = Identity {
        game_instance: "pair-unit".into(),
        generation: 1,
        export_sha256: manifest.export_sha256.clone(),
    };
    let pair = Pair::new(move |_| {
        Ok(Engine::Scripted {
            manifest: Box::new(manifest.clone()),
            delay: 0,
        })
    })
    .unwrap();
    (pair, input, identity)
}
fn config() -> Config {
    Config {
        simulations: 32,
        k: 4,
        ..Default::default()
    }
}

#[test]
fn composite_fallback_keeps_incumbent_and_never_invents_joint_support() {
    use brass_ai_core::search::{
        brass::Brass,
        session::{
            clock::{Budget, MonotonicClock},
            focus,
        },
    };
    use std::sync::atomic::AtomicBool;
    let (mut pair, mut input, id) = fixture();
    input.view.context.round = 2;
    input.view.context.actions_remaining = 2;
    let mut cfg = config();
    cfg.simulations = 128;
    pair.admit(id, &input, cfg, 4_000_000_000).unwrap();
    let ticket = pair
        .start(
            input.clone(),
            Mode::Count,
            BatchPolicy::default(),
            4_000_000_000,
            [None, None],
        )
        .unwrap();
    let decision = drain(&mut pair, ticket).unwrap();
    let clock = MonotonicClock::new(Arc::new(AtomicBool::new(false)));
    // Session admission records the initial own hand before any search proof.
    input.recall = decision.result.actions[0].action.steps[0]
        .precondition
        .key
        .information
        .recall
        .clone();
    let game = Brass::new(input, false).unwrap();
    let prepared = focus::prepare(
        &game,
        &decision.result.actions,
        Budget::new(0, 4_000_000_000).unwrap(),
        &clock,
    )
    .unwrap();
    let incumbent = prepared.incumbent.clone();
    let (mut coordinator, _) =
        crate::focus::Coordinator::new(focus::Method::SearchV2FocusOnly, true);
    coordinator.prepared = Some(Box::new(prepared));
    coordinator.first_selection = Some(decision.selection.clone());
    let lanes = [&decision.lanes[0], &decision.lanes[1]];
    let mut second = decision.selection.clone();
    second.actions = None;
    let fallback = coordinator.compose(&second, lanes).unwrap();
    assert_eq!(
        serde_json::to_value(fallback.actions.as_ref().unwrap()).unwrap(),
        serde_json::to_value(&incumbent).unwrap()
    );
    assert!(coordinator.metrics.fallback_used);
    assert!(coordinator
        .metrics
        .composite
        .as_ref()
        .unwrap()
        .joint_support
        .is_none());
    assert!(coordinator
        .metrics
        .composite
        .as_ref()
        .unwrap()
        .actions
        .iter()
        .all(|a| a.root_epoch == 1));
    let mut fresh = incumbent[1].clone();
    fresh.support = 3;
    second.actions = Some(vec![fresh.clone()]);
    let composite = coordinator.compose(&second, lanes).unwrap();
    assert!(!coordinator.metrics.fallback_used);
    assert!(composite
        .actions
        .unwrap()
        .iter()
        .all(|a| a.support == 0 && a.returns == [0.; 4]));
    let evidence = coordinator.metrics.composite.as_ref().unwrap();
    assert!(evidence.joint_support.is_none());
    assert_eq!(evidence.actions[1].root_epoch, 2);
    assert_eq!(evidence.actions[1].conditional_support, 3);
    fresh.action.steps[0].precondition.actor = (fresh.action.steps[0].precondition.actor + 1) % 4;
    second.actions = Some(vec![fresh]);
    assert_eq!(
        serde_json::to_value(
            coordinator
                .compose(&second, lanes)
                .unwrap()
                .actions
                .unwrap()
        )
        .unwrap(),
        serde_json::to_value(incumbent).unwrap()
    );
    assert!(coordinator.metrics.fallback_used);
}

#[test]
fn focus_method_is_bound_at_admission_and_cancellation_retires_both_owners() {
    use brass_ai_core::search::session::focus::Method;
    let (mut pair, input, id) = fixture();
    assert!(pair.method(Method::SearchV2FocusOnly).is_err());
    pair.admit(id, &input, config(), 4_000_000_000).unwrap();
    pair.method(Method::SearchV2FocusOnly).unwrap();
    let ticket = pair
        .start(
            input,
            Mode::Count,
            BatchPolicy::default(),
            4_000_000_000,
            [None, None],
        )
        .unwrap();
    assert!(pair.method(Method::SearchV1).is_err());
    pair.cancel(ticket).unwrap();
    assert!(drain(&mut pair, ticket).is_err());
    assert!(!pair.busy());
}
fn drain(pair: &mut Pair, ticket: u64) -> Result<Decision> {
    let limit = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(result) = pair.poll(ticket)? {
            return Ok(result);
        }
        assert!(Instant::now() < limit, "owner failed to retire");
        std::thread::sleep(Duration::from_millis(1));
    }
}
#[test]
fn pooled_continuation_is_adopted_by_both_owners_without_a_new_clock() {
    let (mut pair, input, id) = fixture();
    pair.admit(id.clone(), &input, config(), 4_000_000_000)
        .unwrap();
    let ticket = pair
        .start(
            input.clone(),
            Mode::Count,
            BatchPolicy::default(),
            4_000_000_000,
            [None, None],
        )
        .unwrap();
    assert!(pair
        .start(
            input.clone(),
            Mode::Count,
            BatchPolicy::default(),
            4_000_000_000,
            [None, None]
        )
        .is_err());
    assert!(pair.cancel(ticket + 1).is_err());
    let first = drain(&mut pair, ticket).unwrap();
    assert!(!pair.busy());
    assert!(pair.admit(id, &input, config(), 4_000_000_000).is_err());
    let step = &first.result.actions[0].action.steps[1];
    let mut next = input;
    next.view = step
        .precondition
        .key
        .information
        .observation
        .view
        .clone()
        .unwrap();
    next.context = step.precondition.key.context.clone();
    next.recall.clear();
    let ticket = pair
        .start(
            next,
            Mode::Count,
            BatchPolicy::default(),
            4_000_000_000,
            [None, None],
        )
        .unwrap();
    let continuation = drain(&mut pair, ticket).unwrap();
    assert!(continuation.result.provenance.starts_with("continuation/"));
    assert_eq!(
        continuation.result.budget.deadline_ns,
        first.result.budget.deadline_ns
    );
    assert!(continuation
        .lanes
        .iter()
        .all(|l| l.outcome.completed == 0 && l.neural.positions == 0));
}
#[test]
fn narrowed_allowance_includes_root_preparation_and_cannot_be_replenished() {
    let (mut pair, input, id) = fixture();
    pair.admit(id, &input, config(), 4_000_000_000).unwrap();
    let ticket = pair
        .start(
            input.clone(),
            Mode::Count,
            BatchPolicy::default(),
            2_000_000_000,
            [None, None],
        )
        .unwrap();
    let first = drain(&mut pair, ticket).unwrap();
    assert!(first.result.budget.deadline_ns < 2_100_000_000);
    let step = &first.result.actions[0].action.steps[1];
    let mut next = input;
    next.view = step
        .precondition
        .key
        .information
        .observation
        .view
        .clone()
        .unwrap();
    next.context = step.precondition.key.context.clone();
    next.recall.clear();
    let ticket = pair
        .start(
            next,
            Mode::Count,
            BatchPolicy::default(),
            4_000_000_000,
            [None, None],
        )
        .unwrap();
    let second = drain(&mut pair, ticket).unwrap();
    assert!(second.result.budget.deadline_ns <= first.result.budget.deadline_ns);
}
#[test]
fn delivery_reserve_can_prepare_bounded_completion_without_search_or_nn_work() {
    let (mut pair, input, id) = fixture();
    pair.admit(id, &input, config(), 12_000_000_000).unwrap();
    let ticket = pair
        .start(
            input,
            Mode::Timed { startup: None },
            BatchPolicy::default(),
            450_000_000,
            [None, None],
        )
        .unwrap();
    let decision = drain(&mut pair, ticket).unwrap();
    assert!(!decision.result.actions.is_empty());
    assert_eq!(decision.result.budget.work_deadline_ns, 0);
    assert!(decision.result.budget.deadline_ns < 500_000_000);
    assert!(decision.response_ready_ns < decision.result.budget.deadline_ns);
    assert!(decision
        .lanes
        .iter()
        .all(|l| l.outcome.completed == 0 && l.neural.positions == 0));
    assert!(!pair.busy());
}
#[test]
fn owner_panic_and_subsequent_partial_dispatch_retire_before_idle() {
    let (mut pair, input, mut id) = fixture();
    pair.admit(id.clone(), &input, config(), 4_000_000_000)
        .unwrap();
    let probe: Probe = Arc::new(|stage| {
        assert_ne!(stage, Stage::Sampling, "injected owner panic");
        Ok(())
    });
    let ticket = pair
        .start(
            input.clone(),
            Mode::Count,
            BatchPolicy::default(),
            4_000_000_000,
            [None, Some(probe)],
        )
        .unwrap();
    assert!(drain(&mut pair, ticket).is_err());
    assert!(!pair.busy());
    id.generation += 1;
    pair.admit(id, &input, config(), 4_000_000_000).unwrap();
    let ticket = pair
        .start(
            input,
            Mode::Count,
            BatchPolicy::default(),
            4_000_000_000,
            [None, None],
        )
        .unwrap();
    assert!(pair.busy());
    assert!(drain(&mut pair, ticket).is_err());
    assert!(!pair.busy());
}
