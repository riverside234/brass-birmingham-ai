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
