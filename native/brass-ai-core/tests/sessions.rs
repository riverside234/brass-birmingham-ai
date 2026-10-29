use brass_ai_core::{
    features::FeatureContext,
    search::{
        brass::{Brass, Input, Key},
        math::{Parameters, Statistics},
        reward::{Insolvency, LedgerState, Reward},
        session::{
            self,
            clock::{Budget, Clock, Costs, Estimate, FrozenQuota},
            plans,
            tournament::Rounds,
            Identity, Mode, Session, Settings,
        },
        Config, Game, Info, LeafEvaluator,
    },
    state::{dto::StateDto, NativeState},
    types::Seat,
    CoreError,
};
use std::cell::Cell;

#[derive(Default)]
struct Fake {
    now: Cell<u64>,
    cancelled: Cell<bool>,
    reads: Cell<u64>,
    cancel_at_read: Cell<u64>,
}
impl Clock for Fake {
    fn now_ns(&self) -> u64 {
        self.now.get()
    }
    fn cancelled(&self) -> bool {
        let read = self.reads.get() + 1;
        self.reads.set(read);
        self.cancelled.get() || (self.cancel_at_read.get() > 0 && read >= self.cancel_at_read.get())
    }
}
fn input() -> Input {
    let dto: StateDto = serde_json::from_str(include_str!("data/initial-state.json")).unwrap();
    let state = NativeState::import(&dto).unwrap();
    let view = state
        .observation(Seat::try_from(dto.context.actor as u8).unwrap())
        .unwrap();
    let reward = Reward {
        version: "brass-combined-return/2.0.0".into(),
        win: 1.,
        placement: 0.1,
        vp: 0.001,
        vp_cap: Some(180),
        insolvency: Insolvency {
            event_cost: 0.025,
            max_cost: 0.075,
        },
    };
    Input {
        view,
        recall: vec![],
        context: FeatureContext {
            model_id: "model".into(),
            reward_id: reward.identity().unwrap(),
            profile_id: "profile".into(),
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
    }
}
fn identity() -> Identity {
    Identity {
        game_instance: "game".into(),
        generation: 1,
        export_sha256: "a".repeat(64),
    }
}

fn ordinary_input() -> Input {
    let mut dto: StateDto = serde_json::from_str(include_str!("data/initial-state.json")).unwrap();
    dto.context.round = 2;
    dto.context.actions_remaining = 2;
    let mut input = input();
    input.view = NativeState::import(&dto)
        .unwrap()
        .observation(input.view.seat)
        .unwrap();
    input
}

#[test]
fn focus_stability_requires_both_owners_and_the_entire_proof() {
    use session::focus::Stability;
    let mut game = Brass::new(ordinary_input(), false).unwrap();
    let info = game.root().unwrap();
    let first = info
        .actions
        .iter()
        .find(|a| a.starts_with("pass|"))
        .unwrap();
    let clock = Fake::default();
    let budget = Budget::new(0, 4_000_000_000).unwrap();
    let path = plans::complete(&game, first, budget, &clock).unwrap();
    let mut gate = Stability::default();
    assert!(!gate.observe(Some(&path), [8, 8]));
    assert!(!gate.observe(Some(&path), [8, 8]));
    assert!(!gate.observe(Some(&path), [16, 8]));
    assert!(gate.observe(Some(&path), [16, 16]));
    let mut changed = path.clone();
    changed.steps.last_mut().unwrap().choice.push_str("changed");
    assert!(!gate.observe(Some(&changed), [24, 24]));
    assert!(!gate.observe(None, [32, 32]));
    assert!(!gate.observe(Some(&changed), [40, 40]));
}

#[test]
fn focus_affordability_is_conservative_without_changing_quota_costs() {
    let mut costs = Costs::default();
    for value in [1, 1, 1, 1, 100] {
        costs.observe(value);
    }
    assert_eq!(costs.estimate(), Some(1));
    assert_eq!(costs.conservative(7), 100);
    assert_eq!(costs.conservative(200), 200);
    assert!(
        !session::focus::affordable(100, 0, 8, 400, 20)
            .unwrap()
            .useful
    );
    assert!(
        !session::focus::affordable(21, 1, 8, 400, 20)
            .unwrap()
            .useful
    );
    assert!(
        session::focus::affordable(22, 1, 8, 400, 20)
            .unwrap()
            .useful
    );
    assert_eq!(
        session::focus::affordable(44, 1, 8, 400, 20)
            .unwrap()
            .effective_k,
        8
    );
    assert!(
        !session::focus::affordable(1000, 1, 8, 1, 20)
            .unwrap()
            .useful
    );
}

#[test]
fn permitted_projection_is_identical_in_all_24_hidden_burn_allocations() {
    use brass_ai_core::search::{Draws, Position};
    let mut dto: StateDto = serde_json::from_str(include_str!("data/initial-state.json")).unwrap();
    dto.context.round = 2;
    dto.context.actions_remaining = 2;
    let actor = dto.context.actor as usize;
    // Tractable full sampler support: the only unseen cards are the four
    // distinct Canal setup burns. All 4! allocations are enumerated, not seeded
    // spot checks. Public discards and the actor's known hand conserve 64 cards.
    for s in 0..4 {
        if s != actor {
            let cards = std::mem::take(&mut dto.cards.hands[s]);
            dto.cards.discards[s].extend(cards);
        }
    }
    let deck = std::mem::take(&mut dto.cards.deck);
    dto.cards.discards[0].extend(deck);
    let mut input = ordinary_input();
    input.view = NativeState::import(&dto)
        .unwrap()
        .observation(input.view.seat)
        .unwrap();
    let tapes = (0..4)
        .flat_map(|a| (0..3).flat_map(move |b| (0..2).map(move |c| vec![a, b, c])))
        .collect::<Vec<_>>();
    let clock = Fake::default();
    let budget = Budget::new(0, 4_000_000_000).unwrap();
    for family in ["pass|", "loan|", "develop|", "build|"] {
        input.shuffle_tape = Some(tapes.clone());
        let mut game = Brass::new(input.clone(), false).unwrap();
        let info = game.root().unwrap();
        let Some(first) = info.actions.iter().find(|a| a.starts_with(family)) else {
            continue;
        };
        let path = plans::complete(&game, first, budget, &clock).unwrap();
        let (_, projected) = game.project(&path, budget, &clock).unwrap();
        let mut draws = Draws::new(42, None, false);
        for _ in &tapes {
            let mut world = game.sample(&mut draws).unwrap();
            for step in &path.steps {
                let Position::Decision(actual) = game.inspect(&mut world).unwrap() else {
                    panic!("unexpected chance")
                };
                assert_eq!(actual, step.precondition);
                game.step(&mut world, &step.choice).unwrap();
            }
            let Position::Decision(actual) = game.inspect(&mut world).unwrap() else {
                panic!("unexpected chance")
            };
            assert_eq!(actual, projected);
        }
        game.finish().unwrap();
    }
}

#[test]
fn projection_preserves_live_identity_and_completes_a_legal_incumbent() {
    let input = ordinary_input();
    let clock = Fake::default();
    let budget = Budget::new(0, 4_000_000_000).unwrap();
    let mut live = Session::new(identity(), &input, settings(32), budget).unwrap();
    let (game, info) = live.preview(input.clone()).unwrap();
    for family in ["pass|", "loan|", "develop|", "build|"] {
        let Some(first) = info.actions.iter().find(|a| a.starts_with(family)) else {
            continue;
        };
        let action = plans::complete(&game, first, budget, &clock).unwrap();
        let prepared = session::focus::prepare(
            &game,
            &[plans::SupportedAction {
                action: action.clone(),
                support: 2,
                returns: [0.; 4],
            }],
            budget,
            &clock,
        )
        .unwrap();
        assert_eq!(prepared.incumbent.len(), 2);
        assert_eq!(prepared.incumbent[1].support, 0);
        assert_eq!(prepared.input.view.context.actions_remaining, 1);
        assert_eq!(prepared.input.view.seat, input.view.seat);
        assert!(prepared.input.shuffle_tape.is_none());
        let hypothetical = live.projected_owner(&prepared.input).unwrap();
        assert_eq!(hypothetical.budget().deadline_ns, budget.deadline_ns);
        assert_eq!(live.preview(input.clone()).unwrap().1, info);
        assert!(live
            .recall()
            .iter()
            .all(|r| r.revision <= input.view.context.revision));
        let mut corrupt = action;
        corrupt.steps[0].precondition.actions = vec!["invalid".into()].into();
        assert!(game.project(&corrupt, budget, &clock).is_err());
    }
    clock.now.set(budget.work_deadline_ns);
    let action = plans::complete(&game, &info.actions[0], budget, &clock).unwrap();
    assert!(game.project(&action, budget, &clock).is_err());
}

#[test]
fn no_switch_checkpoint_resumes_the_exact_owner_tournament() {
    use session::{
        cooperative::SharedRoot,
        focus::{Observer, Progress},
    };
    struct Watch(Vec<Progress>);
    impl Observer for Watch {
        fn checkpoint(&mut self, _: &Brass, p: Progress) -> Result<bool, CoreError> {
            self.0.push(p);
            Ok(false)
        }
    }
    let input = ordinary_input();
    let clock = Fake::default();
    let config = Settings {
        search: Config {
            simulations: 64,
            k: 8,
            ..Default::default()
        },
        ..Default::default()
    };
    let budget = Budget::new(0, 4_000_000_000).unwrap();
    let mut a = Session::new(identity(), &input, config.clone(), budget).unwrap();
    let mut b = Session::new(identity(), &input, config, budget).unwrap();
    b.enable_descendant_reuse().unwrap(); // Tracking alone must not alter any draw/count.
    let info = a.preview(input.clone()).unwrap().1;
    let mut e = Eval {
        clock: &clock,
        calls: 0,
        late_at: 0,
        cancel: false,
    };
    let stats = e.evaluate(&info).unwrap();
    let mut draws = brass_ai_core::search::Draws::new(a.next_root_seed(), None, false);
    let shared = SharedRoot {
        uniforms: (0..info.actions.len())
            .map(|_| draws.uniform().unwrap())
            .collect(),
        info,
        stats: Some(stats),
    };
    let policy = brass_ai_core::evaluation::BatchPolicy {
        cap: 1,
        pending_limit: 1,
        max_candidate_cells: 8192,
        flush_us: 0,
    };
    let baseline = a
        .decide_cooperative(
            &identity(),
            1,
            input.clone(),
            budget.deadline_ns,
            Mode::Count,
            &clock,
            &mut e,
            None,
            &shared,
            policy,
        )
        .unwrap();
    let mut watch = Watch(vec![]);
    let observed = b
        .decide_observed(
            &identity(),
            1,
            input,
            budget.deadline_ns,
            Mode::Count,
            &clock,
            &mut e,
            &shared,
            policy,
            &mut watch,
        )
        .unwrap();
    assert_eq!(watch.0.len(), 3);
    for field in [
        "snapshot",
        "gumbels",
        "actions",
        "action_ids",
        "completed",
        "attempted",
        "partial_completed",
        "plan_events",
        "eligible_plans",
    ] {
        assert_eq!(
            serde_json::to_value(&baseline).unwrap()[field],
            serde_json::to_value(&observed).unwrap()[field],
            "{field}"
        );
    }
}
fn retired_for_reuse(enabled: bool, clock: &Fake) -> (Session, session::focus::Prepared) {
    use session::{
        cooperative::SharedRoot,
        focus::{Observer, Progress},
    };
    struct Stop;
    impl Observer for Stop {
        fn checkpoint(&mut self, _: &Brass, p: Progress) -> Result<bool, CoreError> {
            Ok(p.outcome.snapshot.as_ref().unwrap().completed_rounds >= 2)
        }
    }
    let input = ordinary_input();
    let budget = Budget::new(0, 4_000_000_000).unwrap();
    let mut config = settings(96);
    config.search.k = 8;
    let mut session = Session::new(identity(), &input, config, budget).unwrap();
    if enabled {
        session.enable_descendant_reuse().unwrap();
    }
    let (game, info) = session.preview(input.clone()).unwrap();
    let mut evaluator = Eval {
        clock,
        calls: 0,
        late_at: 0,
        cancel: false,
    };
    let shared = SharedRoot {
        stats: Some(evaluator.evaluate(&info).unwrap()),
        uniforms: vec![0.5; info.actions.len()],
        info,
    };
    let outcome = session
        .decide_observed(
            &identity(),
            1,
            input,
            budget.deadline_ns,
            Mode::Count,
            clock,
            &mut evaluator,
            &shared,
            brass_ai_core::evaluation::BatchPolicy {
                cap: 1,
                pending_limit: 1,
                max_candidate_cells: 8192,
                flush_us: 0,
            },
            &mut Stop,
        )
        .unwrap();
    assert_eq!(outcome.status, "policy_round_stop");
    let prepared = session::focus::prepare(&game, &outcome.actions, budget, clock).unwrap();
    (session, prepared)
}
#[test]
fn both_one_and_neither_owner_hit_without_sharing_counts_or_root_credit() {
    use session::cooperative::SharedRoot;
    for enabled in [[true, true], [true, false], [false, false]] {
        let clock = Fake::default();
        let (mut a, pa) = retired_for_reuse(enabled[0], &clock);
        let (mut b, pb) = retired_for_reuse(enabled[1], &clock);
        assert_eq!(pa.boundary, pb.boundary);
        assert_eq!(pa.incumbent[0].action, pb.incumbent[0].action);
        let mut projected = [
            a.projected_owner_reusing(&pa.input, &pa.incumbent[0].action, &clock)
                .unwrap(),
            b.projected_owner_reusing(&pa.input, &pa.incumbent[0].action, &clock)
                .unwrap(),
        ];
        for lane in 0..2 {
            let p = &mut projected[lane];
            let metrics = p.reuse_metrics().unwrap();
            assert_eq!(metrics.hit, enabled[lane], "{}", metrics.reason);
            if enabled[lane] {
                assert!(metrics.counts.retained_nodes > 1);
                assert!(metrics.counts.inherited_edge_visits > 0);
            }
            let mut evaluator = Eval {
                clock: &clock,
                calls: 0,
                late_at: 0,
                cancel: false,
            };
            let stats = p
                .evaluate_projected_root(&pa.boundary, &mut evaluator, &clock)
                .unwrap();
            assert!(stats.count.iter().all(|n| *n == 0));
            let shared = SharedRoot {
                info: pa.boundary.clone(),
                stats: Some(stats),
                uniforms: vec![0.5; pa.boundary.actions.len()],
            };
            let outcome = p
                .decide_projected(
                    1,
                    pa.input.clone(),
                    Mode::Count,
                    &clock,
                    &mut evaluator,
                    &shared,
                    brass_ai_core::evaluation::BatchPolicy {
                        cap: 1,
                        pending_limit: 1,
                        max_candidate_cells: 8192,
                        flush_us: 0,
                    },
                )
                .unwrap();
            assert_eq!(outcome.completed, 96);
            assert_eq!(
                outcome
                    .snapshot
                    .as_ref()
                    .unwrap()
                    .stats
                    .count
                    .iter()
                    .sum::<u64>(),
                96
            );
            assert_eq!(outcome.snapshot.as_ref().unwrap().support_watermark, 96);
            assert!(outcome.actions[0].support <= 96);
            assert_eq!(outcome.budget.deadline_ns, 4_000_000_000);
            if enabled[lane] {
                assert!(outcome.schedule.reused_selections > 0);
            }
        }
        assert_eq!(projected[0].next_root_seed(), projected[1].next_root_seed());
    }
}
#[test]
fn reused_owner_rejects_changed_value_sampler_information_and_cancellation() {
    for variant in 0..7 {
        let clock = Fake::default();
        let (mut session, mut p) = retired_for_reuse(true, &clock);
        match variant {
            0 => p.input.context.style[0] += 0.1,
            1 => p.input.ledger.spent[0] += 0.001,
            2 => p.input.shuffle_tape = Some(vec![]),
            3 => p.input.view.context.revision += 1,
            4 => p.incumbent[0]
                .action
                .steps
                .last_mut()
                .unwrap()
                .choice
                .push_str("wrong"),
            5 => clock.cancelled.set(true),
            _ => clock.now.set(3_600_000_000),
        }
        let projected = session
            .projected_owner_reusing(&p.input, &p.incumbent[0].action, &clock)
            .unwrap();
        assert!(!projected.reuse_metrics().unwrap().hit);
        assert_eq!(projected.budget().deadline_ns, 4_000_000_000);
        assert_eq!(
            projected.reuse_metrics().unwrap().diagnostic_max_depth,
            Config::default().max_depth
        );
    }
}
fn settings(n: u64) -> Settings {
    Settings {
        search: Config {
            simulations: n,
            k: 1,
            ..Default::default()
        },
        ..Default::default()
    }
}
struct Eval<'a> {
    clock: &'a Fake,
    calls: u64,
    late_at: u64,
    cancel: bool,
}
impl LeafEvaluator<Key, String> for Eval<'_> {
    fn evaluate(&mut self, info: &Info<Key, String>) -> Result<Statistics, CoreError> {
        self.calls += 1;
        if self.calls == self.late_at {
            if self.cancel {
                self.clock.cancelled.set(true);
            } else {
                self.clock.now.set(3_600_000_000);
            }
        }
        // Pass/finish gives deterministic short complete-action proofs.
        let preferred = info
            .key
            .choices
            .iter()
            .find(|c| c.kind == brass_ai_core::state::dto::Kind::Pass)
            .map(|c| &c.id);
        Statistics::new(
            info.key
                .choices
                .iter()
                .map(|c| {
                    if Some(&c.id) == preferred || c.kind == brass_ai_core::state::dto::Kind::Finish
                    {
                        100.
                    } else {
                        -100.
                    }
                })
                .collect(),
            [0.3; 4],
        )
    }
}
fn decide(s: &mut Session, i: Input, c: &Fake, e: &mut Eval<'_>) -> session::Outcome {
    s.decide(&identity(), 1, i, 4_000_000_000, Mode::Count, c, e, None)
        .unwrap()
}

#[test]
fn recorded_round_extension_oracle() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/GumbelISMCTS/fixtures/session-v1.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let logits = serde_json::from_value(case["logits"].clone()).unwrap();
        let value = serde_json::from_value(case["value"].clone()).unwrap();
        let uniforms: Vec<f64> = serde_json::from_value(case["uniforms"].clone()).unwrap();
        let mut stats = Statistics::new(logits, value).unwrap();
        let mut rounds = Rounds::new(
            &stats.logits,
            &uniforms,
            case["k"].as_u64().unwrap() as usize,
            case["budget"].as_u64().unwrap(),
        )
        .unwrap();
        let values: Vec<[f64; 4]> = serde_json::from_value(case["values"].clone()).unwrap();
        let expected = &case["expected"];
        let mut selected = Vec::new();
        for n in 0..case["stop_after"]
            .as_u64()
            .unwrap()
            .min(case["budget"].as_u64().unwrap())
        {
            let edge = rounds.tournament.next_edge().unwrap().unwrap();
            selected.push(edge);
            stats.count[edge] += 1;
            for (sum, v) in stats.returns[edge]
                .iter_mut()
                .zip(values[n as usize % values.len()])
            {
                *sum += v;
            }
            rounds
                .complete(edge, &stats, 0, Parameters::default())
                .unwrap();
        }
        assert_eq!(
            serde_json::to_value(selected).unwrap(),
            expected["selected"],
            "{}",
            case["name"]
        );
        assert_eq!(
            serde_json::to_value(rounds.ranking(&stats, 0, Parameters::default()).unwrap())
                .unwrap(),
            expected["ranking"]
        );
        if let Some(s) = &rounds.eligible {
            let got = serde_json::to_value(s).unwrap();
            let want = &expected["snapshot"];
            for field in [
                "survivors",
                "support_watermark",
                "completed_rounds",
                "remainder_simulations",
                "reason",
            ] {
                assert_eq!(got[field], want[field]);
            }
            assert_eq!(got["stats"]["count"], want["count"]);
            for (a, b) in s.stats.returns.iter().flatten().zip(
                want["returns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|r| r.as_array().unwrap())
                    .map(|v| v.as_f64().unwrap()),
            ) {
                assert!((a - b).abs() < 1e-12);
            }
        } else {
            assert!(expected["snapshot"].is_null());
        }
        // A launch abandoned after the boundary cannot change a snapshot.
        if rounds.tournament.completed < case["budget"].as_u64().unwrap() {
            let before = serde_json::to_value(&rounds.eligible).unwrap();
            rounds.tournament.next_edge().unwrap();
            assert!(rounds.tournament.abandon());
            assert!(!rounds.tournament.abandon());
            assert_eq!(before, serde_json::to_value(&rounds.eligible).unwrap());
        }
    }
}
#[test]
fn budgets_costs_and_missing_estimate() {
    for seconds in [4, 8, 10, 12] {
        let mut b = Budget::new(100, seconds * 1_000_000_000).unwrap();
        assert_eq!(b.reserve_ns, (seconds * 100_000_000).min(1_200_000_000));
        let original = b.deadline_ns;
        b.narrow(1_000_000_000, seconds * 1_000_000_000);
        assert_eq!(b.deadline_ns, original);
        b.narrow(2_000_000_000, 10);
        assert_eq!(b.deadline_ns, 2_000_000_010);
    }
    let mut costs = Costs::default();
    for v in [100, 200, 300] {
        costs.observe(v);
    }
    assert!(costs.estimate().is_none());
    costs.observe(400);
    assert_eq!(costs.estimate(), Some(100));
    for v in [500, 600, 700, 800, 900, 1000] {
        costs.observe(v);
    }
    assert_eq!(costs.estimate(), Some(400));
    assert!(!costs.admit(399));
    assert!(costs.admit(400));
    assert_eq!(FrozenQuota::timed(3600, None).unwrap().simulations, 0);
    let q = FrozenQuota::timed(
        3600,
        Some(Estimate {
            nanoseconds: 10,
            source: "measured fixture".into(),
        }),
    )
    .unwrap();
    assert_eq!(q.simulations, 360);
    costs.observe(1);
    assert_eq!(q.simulations, 360);
}
#[test]
fn partial_leaf_has_no_complete_plan_support() {
    let i = input();
    let c = Fake::default();
    let mut e = Eval {
        clock: &c,
        calls: 0,
        late_at: 0,
        cancel: false,
    };
    let mut s = Session::new(
        identity(),
        &i,
        settings(1),
        Budget::new(0, 4_000_000_000).unwrap(),
    )
    .unwrap();
    let r = decide(&mut s, i, &c, &mut e);
    assert_eq!(r.completed, 1);
    assert_eq!(r.plan_events, 0);
    assert_eq!(r.actions[0].support, 0);
    assert_eq!(r.provenance, "bounded_policy_completion");
    assert!(r.action_ids[0].last().unwrap().starts_with("finish|"));
}
#[test]
fn support_watermark_excludes_future_and_partial_paths() {
    let mut game = Brass::new(input(), false).unwrap();
    let info = game.root().unwrap();
    let first = info
        .actions
        .iter()
        .find(|a| a.starts_with("pass|"))
        .unwrap();
    let c = Fake::default();
    let b = Budget::new(0, 4_000_000_000).unwrap();
    let a = plans::complete(&game, first, b, &c).unwrap();
    let mut ledger = plans::Ledger::default();
    ledger
        .record(std::slice::from_ref(&a), 1, [0.2; 4])
        .unwrap();
    ledger
        .record(std::slice::from_ref(&a), 3, [0.4; 4])
        .unwrap();
    assert!(ledger.select(first, 2, 2, info.actor).is_empty());
    let selected = ledger.select(first, 3, 2, info.actor);
    assert_eq!(selected[0].support, 2);
    assert!((selected[0].returns[0] - 0.6).abs() < 1e-12);
    let mut partial = a.clone();
    partial.steps.pop();
    assert!(plans::validate(&game, &[partial], b, &c).is_err());
    let mut wrong = a.clone();
    std::sync::Arc::make_mut(&mut wrong.steps[0].precondition.key)
        .context
        .profile_id = "other".into();
    assert!(plans::validate(&game, &[wrong], b, &c).is_err());
    let mut multiple = a.clone();
    multiple.steps.extend(a.steps);
    assert!(plans::validate(&game, &[multiple], b, &c).is_err());
}
#[test]
fn late_evaluation_does_not_publish_support_and_cancel_applies_nothing() {
    for cancel in [false, true] {
        let i = input();
        let c = Fake::default();
        let mut e = Eval {
            clock: &c,
            calls: 0,
            late_at: 2,
            cancel,
        };
        let mut s = Session::new(
            identity(),
            &i,
            settings(20),
            Budget::new(0, 4_000_000_000).unwrap(),
        )
        .unwrap();
        let r = decide(&mut s, i, &c, &mut e);
        assert_eq!(r.completed, 0);
        assert_eq!(r.retired_pending, 1);
        assert_eq!(r.plan_events, 0);
        assert!(r.snapshot.is_none());
        if cancel {
            assert_eq!(r.status, "cancelled");
            assert!(r.actions.is_empty());
        } else {
            assert_eq!(r.actions[0].support, 0);
        }
    }
}
#[test]
fn complete_support_continuation_and_no_second_clock() {
    let i = input();
    let c = Fake::default();
    let mut e = Eval {
        clock: &c,
        calls: 0,
        late_at: 0,
        cancel: false,
    };
    let mut s = Session::new(
        identity(),
        &i,
        settings(8),
        Budget::new(0, 4_000_000_000).unwrap(),
    )
    .unwrap();
    let r = decide(&mut s, i.clone(), &c, &mut e);
    assert_eq!(r.provenance, "search_supported");
    assert!(r.actions[0].support >= 2);
    let step = &r.actions[0].action.steps[1];
    let mut next = i.clone();
    next.view = step
        .precondition
        .key
        .information
        .observation
        .view
        .clone()
        .unwrap();
    next.recall = vec![];
    next.context = step.precondition.key.context.clone();
    c.now.set(3_700_000_000);
    let calls = e.calls;
    let continued = s
        .decide(
            &identity(),
            2,
            next,
            4_000_000_000,
            Mode::Count,
            &c,
            &mut e,
            None,
        )
        .unwrap();
    assert!(continued.provenance.starts_with("continuation/"));
    assert_eq!(continued.completed, 0);
    assert_eq!(e.calls, calls);
    assert_eq!(continued.budget.deadline_ns, r.budget.deadline_ns);
    assert!(s.recall().len() >= 2);
}
#[test]
fn stale_identity_repeat_and_context_cannot_restart() {
    let i = input();
    let c = Fake::default();
    let mut e = Eval {
        clock: &c,
        calls: 0,
        late_at: 0,
        cancel: false,
    };
    let mut s = Session::new(
        identity(),
        &i,
        settings(1),
        Budget::new(0, 4_000_000_000).unwrap(),
    )
    .unwrap();
    let _ = decide(&mut s, i.clone(), &c, &mut e);
    for (id, mut x) in [
        (
            Identity {
                generation: 2,
                ..identity()
            },
            i.clone(),
        ),
        (identity(), i.clone()),
    ] {
        if id.generation == 1 {
            x.context.style[0] = 0.5;
        }
        assert!(s
            .decide(&id, 2, x, 4_000_000_000, Mode::Count, &c, &mut e, None)
            .is_err());
    }
    assert!(s
        .decide(
            &identity(),
            3,
            i,
            4_000_000_000,
            Mode::Count,
            &c,
            &mut e,
            None
        )
        .is_err());
    assert_eq!(s.budget().deadline_ns, 4_000_000_000);
}
#[test]
fn completion_reserve_cancel_and_expiry_are_bounded() {
    let mut game = Brass::new(input(), false).unwrap();
    let info = game.root().unwrap();
    let first = &info.actions[0];
    let c = Fake::default();
    let b = Budget::new(0, 4_000_000_000).unwrap();
    c.cancel_at_read.set(3);
    assert!(plans::complete(&game, first, b, &c).is_err());
    c.cancel_at_read.set(0);
    c.now.set(b.deadline_ns);
    assert!(plans::complete(&game, first, b, &c).is_err());
}

#[test]
fn action_two_has_its_own_complete_proof_and_revalidates() {
    let mut i = input();
    // Authored two-action root; the production corpus separately covers C# roots.
    i.view.context.round = 2;
    i.view.context.actions_remaining = 2;
    i.ledger.settled = vec![(brass_ai_core::types::Era::CanalEra, 1)];
    let c = Fake::default();
    let mut e = Eval {
        clock: &c,
        calls: 0,
        late_at: 0,
        cancel: false,
    };
    let mut s = Session::new(
        identity(),
        &i,
        settings(12),
        Budget::new(0, 4_000_000_000).unwrap(),
    )
    .unwrap();
    let r = decide(&mut s, i.clone(), &c, &mut e);
    assert_eq!(r.actions.len(), 2);
    assert!(r.actions.iter().all(|a| a.support >= 2));
    assert!(r.actions[0].support >= r.actions[1].support);
    let second = &r.actions[1].action.steps[0].precondition;
    let mut next = i;
    next.view = second.key.information.observation.view.clone().unwrap();
    next.recall = vec![];
    next.context = second.key.context.clone();
    // Cash changes legality and invalidates the saved second action's proof.
    next.view.players[second.actor].cash = 0;
    c.now.set(3_700_000_000);
    let calls = e.calls;
    let out = s
        .decide(
            &identity(),
            2,
            next,
            4_000_000_000,
            Mode::Count,
            &c,
            &mut e,
            None,
        )
        .unwrap();
    assert_eq!(out.root_epoch, 2);
    assert_eq!(out.provenance, "bounded_legal_fallback");
    assert_eq!(out.completed, 0);
    assert_eq!(out.actions[0].support, 0);
    assert_eq!(e.calls, calls);
    assert_eq!(out.budget.deadline_ns, 4_000_000_000);
}

#[test]
fn timed_quota_is_frozen_and_missing_or_wrong_estimates_are_explicit() {
    use session::{clock::Estimate, Startup};
    let i = input();
    let c = Fake::default();
    let startup = Startup {
        model_id: i.context.model_id.clone(),
        export_sha256: identity().export_sha256,
        reward_id: i.context.reward_id.clone(),
        phase: i.view.context.phase,
        estimate: Estimate {
            nanoseconds: 500_000_000,
            source: "separately measured fixture".into(),
        },
    };
    let mut e = Eval {
        clock: &c,
        calls: 0,
        late_at: 0,
        cancel: false,
    };
    let mut s = Session::new(
        identity(),
        &i,
        settings(1000),
        Budget::new(0, 4_000_000_000).unwrap(),
    )
    .unwrap();
    let r = s
        .decide(
            &identity(),
            1,
            i.clone(),
            4_000_000_000,
            Mode::Timed {
                startup: Some(startup.clone()),
            },
            &c,
            &mut e,
            None,
        )
        .unwrap();
    assert_eq!(r.quota.as_ref().unwrap().simulations, 7);
    assert_eq!(r.completed, 7);
    assert!(r.unused_work_ns > 0);
    let mut s = Session::new(
        identity(),
        &i,
        settings(1000),
        Budget::new(0, 4_000_000_000).unwrap(),
    )
    .unwrap();
    let r = s
        .decide(
            &identity(),
            1,
            i.clone(),
            4_000_000_000,
            Mode::Timed { startup: None },
            &c,
            &mut e,
            None,
        )
        .unwrap();
    assert_eq!(r.quota.unwrap().simulations, 0);
    assert_eq!(r.completed, 0);
    assert!(r.snapshot.is_none());
    assert_eq!(r.provenance, "gumbel_prior_fallback");
    assert_eq!(r.actions[0].support, 0);
    let mut wrong = startup;
    wrong.export_sha256 = "b".repeat(64);
    let mut s = Session::new(
        identity(),
        &i,
        settings(1000),
        Budget::new(0, 4_000_000_000).unwrap(),
    )
    .unwrap();
    assert!(s
        .decide(
            &identity(),
            1,
            i,
            4_000_000_000,
            Mode::Timed {
                startup: Some(wrong)
            },
            &c,
            &mut e,
            None
        )
        .is_err());
}

#[test]
fn invalid_request_cannot_shorten_clock_or_poison_revision() {
    let i = input();
    let c = Fake::default();
    let mut e = Eval {
        clock: &c,
        calls: 0,
        late_at: 0,
        cancel: false,
    };
    let mut s = Session::new(
        identity(),
        &i,
        settings(1),
        Budget::new(0, 4_000_000_000).unwrap(),
    )
    .unwrap();
    let stale = Identity {
        generation: 0,
        ..identity()
    };
    assert!(s
        .decide(&stale, 1, i.clone(), 0, Mode::Count, &c, &mut e, None)
        .is_err());
    let mut bad = i.clone();
    bad.view.players[0].cash = -1;
    assert!(s
        .decide(&identity(), 99, bad, 0, Mode::Count, &c, &mut e, None)
        .is_err());
    assert_eq!(s.budget().deadline_ns, 4_000_000_000);
    let r = decide(&mut s, i, &c, &mut e);
    assert!(!r.actions.is_empty());
}

#[test]
fn native_interruptions_before_and_after_a_balanced_round() {
    // N=24,K=4: first balanced round ends at 12 completed trajectories.
    for (late_at, completed, watermark) in [(13, 11, 0), (14, 12, 12), (15, 13, 12)] {
        let i = input();
        let c = Fake::default();
        let mut e = Eval {
            clock: &c,
            calls: 0,
            late_at,
            cancel: false,
        };
        let mut settings = settings(24);
        settings.search.k = 4;
        let mut s = Session::new(
            identity(),
            &i,
            settings,
            Budget::new(0, 4_000_000_000).unwrap(),
        )
        .unwrap();
        let r = decide(&mut s, i, &c, &mut e);
        assert_eq!(r.completed, completed);
        assert_eq!(r.retired_pending, 1);
        assert_eq!(
            r.snapshot.as_ref().map_or(0, |s| s.support_watermark),
            watermark
        );
        assert_eq!(r.partial_completed, completed - watermark);
        if let Some(s) = r.snapshot {
            assert_eq!(s.stats.count.iter().sum::<u64>(), watermark);
            assert_eq!(s.completed_rounds, 1);
        }
    }
}

#[test]
fn complete_action_acknowledges_intermediate_recall_before_action_two() {
    let mut i = input();
    i.view.context.round = 2;
    i.view.context.actions_remaining = 2;
    i.ledger.settled = vec![(brass_ai_core::types::Era::CanalEra, 1)];
    let c = Fake::default();
    let mut e = Eval {
        clock: &c,
        calls: 0,
        late_at: 0,
        cancel: false,
    };
    let mut s = Session::new(
        identity(),
        &i,
        settings(12),
        Budget::new(0, 4_000_000_000).unwrap(),
    )
    .unwrap();
    let r = decide(&mut s, i.clone(), &c, &mut e);
    assert_eq!(r.actions.len(), 2);
    let expected = &r.actions[1].action.steps[0].precondition.key;
    let mut next = i;
    next.view = expected.information.observation.view.clone().unwrap();
    next.context = expected.context.clone();
    next.recall = vec![];
    let calls = e.calls;
    c.now.set(3_700_000_000);
    let second = s
        .decide(
            &identity(),
            2,
            next,
            4_000_000_000,
            Mode::Count,
            &c,
            &mut e,
            None,
        )
        .unwrap();
    assert_eq!(second.actions.len(), 1);
    assert!(second.provenance.starts_with("continuation/"));
    assert_eq!(second.action_ids[0], r.action_ids[1]);
    assert_eq!(second.supporting_root_epoch, r.root_epoch);
    assert_eq!(s.recall(), &expected.information.recall);
    assert_eq!(e.calls, calls);
    assert_eq!(second.budget.deadline_ns, r.budget.deadline_ns);
}

#[test]
fn changed_root_reuses_only_nn_cache_and_starts_with_fresh_counts() {
    let mut i = input();
    i.view.context.round = 2;
    i.view.context.actions_remaining = 2;
    i.ledger.settled = vec![(brass_ai_core::types::Era::CanalEra, 1)];
    let c = Fake::default();
    let mut e = Eval {
        clock: &c,
        calls: 0,
        late_at: 0,
        cancel: false,
    };
    let mut s = Session::new(
        identity(),
        &i,
        settings(2),
        Budget::new(0, 4_000_000_000).unwrap(),
    )
    .unwrap();
    let first = decide(&mut s, i.clone(), &c, &mut e);
    assert_eq!(first.actions.len(), 1);
    assert_eq!(first.actions[0].support, 0);
    i.recall = s.recall().clone();
    let game = Brass::new(i.clone(), false).unwrap();
    let next = plans::validate(&game, &[first.actions[0].action.clone()], s.budget(), &c)
        .unwrap()
        .unwrap();
    i.view = next.key.information.observation.view.clone().unwrap();
    i.context = next.key.context.clone();
    // Explicit known recall is also accepted after a complete-action acknowledgment.
    i.recall = next.key.information.recall.clone();
    let second = s
        .decide(
            &identity(),
            2,
            i,
            4_000_000_000,
            Mode::Count,
            &c,
            &mut e,
            None,
        )
        .unwrap();
    assert_eq!(second.root_epoch, 2);
    assert_eq!(second.completed, 2);
    assert!(second.cache_hits >= 1);
    let snapshot = second.snapshot.unwrap();
    assert_eq!(snapshot.stats.count.iter().sum::<u64>(), 2);
    assert_eq!(snapshot.support_watermark, 2);
    assert_eq!(second.plan_events, 1);
}
