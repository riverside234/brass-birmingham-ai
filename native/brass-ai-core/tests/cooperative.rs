use brass_ai_core::{
    evaluation::BatchPolicy,
    features::FeatureContext,
    search::{
        self,
        brass::{Brass, Input, Key},
        halving::Tournament,
        math::{Parameters, Statistics},
        reward::{Insolvency, LedgerState, Reward},
        session::{
            clock::{Budget, Clock},
            cooperative::SharedRoot,
            merge,
            tournament::Snapshot,
            Identity, Mode, Session, Settings,
        },
        trajectory,
        tree::Tree,
        Config, Draws, Game, Info, LeafEvaluator,
    },
    state::{dto::StateDto, NativeState},
    types::Seat,
    work::{self, Stage},
    CoreError,
};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

#[derive(Default)]
struct Fake {
    now: AtomicU64,
    cancelled: AtomicBool,
}
impl Clock for Fake {
    fn now_ns(&self) -> u64 {
        self.now.load(Ordering::Relaxed)
    }
    fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}
fn input() -> Input {
    let dto: StateDto = serde_json::from_str(include_str!("data/initial-state.json")).unwrap();
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
        game_instance: "test".into(),
        generation: 1,
        export_sha256: "a".repeat(64),
    }
}
struct Plain;
impl LeafEvaluator<Key, String> for Plain {
    fn evaluate(&mut self, info: &Info<Key, String>) -> Result<Statistics, CoreError> {
        Statistics::new(
            info.actions
                .iter()
                .map(|a| if a.starts_with("pass|") { 1. } else { 0. })
                .collect(),
            [0.4, 0.3, 0.2, 0.1],
        )
    }
}
fn prepared(config: Config) -> (Session, SharedRoot) {
    let session = Session::new(
        identity(),
        &input(),
        Settings {
            search: config.clone(),
            ..Default::default()
        },
        Budget::new(0, 4_000_000_000).unwrap(),
    )
    .unwrap();
    let (_, info) = session.preview(input()).unwrap();
    let stats = Some(Plain.evaluate(&info).unwrap());
    let mut draws = Draws::new(config.seed, None, false);
    let uniforms = (0..info.actions.len())
        .map(|_| draws.uniform().unwrap())
        .collect();
    (
        session,
        SharedRoot {
            info,
            stats,
            uniforms,
        },
    )
}
#[test]
fn recorded_merger_uses_pooled_q_and_union_of_survivors() {
    let data: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/GumbelISMCTS/fixtures/coordinator-v1.json"
    ))
    .unwrap();
    for case in data["cases"].as_array().unwrap() {
        let logits: Vec<f64> = serde_json::from_value(case["logits"].clone()).unwrap();
        let value: [f64; 4] = serde_json::from_value(case["value"].clone()).unwrap();
        let root = Statistics::new(logits, value).unwrap();
        let g: Vec<_> = case["uniforms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|u| search::math::gumbel(u.as_f64().unwrap()).unwrap())
            .collect();
        let snapshots: Vec<_> = case["snapshots"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                if s.is_null() {
                    None
                } else {
                    let mut stats = root.clone();
                    stats.count = serde_json::from_value(s["count"].clone()).unwrap();
                    stats.returns = serde_json::from_value(s["returns"].clone()).unwrap();
                    Some(Snapshot {
                        stats,
                        survivors: serde_json::from_value(s["survivors"].clone()).unwrap(),
                        completed_rounds: 1,
                        support_watermark: s["watermark"].as_u64().unwrap(),
                        remainder_simulations: 0,
                        reason: "balanced_round",
                    })
                }
            })
            .collect();
        let p = merge::pool(
            &root,
            &g,
            0,
            Parameters::default(),
            [snapshots[0].as_ref(), snapshots[1].as_ref()],
        )
        .unwrap();
        let expected = &case["expected"];
        assert_eq!(
            serde_json::to_value(&p.stats.count).unwrap(),
            expected["count"]
        );
        assert_eq!(
            serde_json::to_value(&p.ranking).unwrap(),
            expected["ranking"]
        );
        assert_eq!(
            serde_json::to_value(&p.eligible).unwrap(),
            expected["eligible"]
        );
        for (score, want) in p.scores.iter().zip(expected["scores"].as_array().unwrap()) {
            assert!((score - want.as_f64().unwrap()).abs() < 1e-12);
        }
    }
}
#[test]
fn reservations_are_bounded_and_barrier_is_completed_only() {
    let mut t = Tournament::new(&[0., 0.], &[0.4, 0.6], 2, 4).unwrap();
    assert_eq!(
        (0..4)
            .map(|_| t.reserve().unwrap().unwrap())
            .collect::<Vec<_>>(),
        [0, 1, 0, 1]
    );
    assert!(t.reserve().unwrap().is_none());
    assert_eq!(t.completed, 0);
    let mut s = Statistics::new(vec![0., 0.], [0.; 4]).unwrap();
    for (i, edge) in [0, 1, 0, 1].into_iter().enumerate() {
        s.count[edge] += 1;
        t.completed_backup(edge, &s, 0, Parameters::default())
            .unwrap();
        assert_eq!(t.rounds.len(), usize::from(i == 3));
    }
    assert_eq!(t.pending(), 0);
    assert!(t.completed_backup(1, &s, 0, Parameters::default()).is_err());
}
#[test]
fn retirement_releases_each_slot_once_and_never_adds_visits() {
    let mut t = Tournament::new(&[0., 0.], &[0.4, 0.6], 2, 4).unwrap();
    t.reserve().unwrap();
    t.reserve().unwrap();
    assert_eq!(t.retire_all(), 2);
    assert_eq!(t.retire_all(), 0);
    assert_eq!(t.completed, 0);
    assert!(t.rounds.is_empty());
}
#[test]
fn singleton_preserves_scalar_counts_returns_plans_and_draws() {
    let config = Config {
        simulations: 64,
        k: 4,
        ..Default::default()
    };
    let (mut scalar, _) = prepared(config.clone());
    let (mut batch, shared) = prepared(config);
    let clock = Fake::default();
    let a = scalar
        .decide(
            &identity(),
            1,
            input(),
            4_000_000_000,
            Mode::Count,
            &clock,
            &mut Plain,
            None,
        )
        .unwrap();
    let b = batch
        .decide_cooperative(
            &identity(),
            1,
            input(),
            4_000_000_000,
            Mode::Count,
            &clock,
            &mut Plain,
            None,
            &shared,
            BatchPolicy::default(),
        )
        .unwrap();
    assert_eq!(a.completed, b.completed);
    assert_eq!(a.gumbels, b.gumbels);
    assert_eq!(a.action_ids, b.action_ids);
    assert_eq!(
        serde_json::to_value(a.snapshot).unwrap(),
        serde_json::to_value(b.snapshot).unwrap()
    );
    assert_eq!(
        serde_json::to_value(a.actions).unwrap(),
        serde_json::to_value(b.actions).unwrap()
    );
}
#[test]
fn fixed_wave_interleaving_is_repeatable_and_deduplicates_expansions() {
    let config = Config {
        simulations: 64,
        k: 1,
        ..Default::default()
    };
    let policy = BatchPolicy {
        cap: 4,
        pending_limit: 4,
        flush_us: 2000,
        max_candidate_cells: 32768,
    };
    let run = || {
        let (mut s, r) = prepared(config.clone());
        s.decide_cooperative(
            &identity(),
            1,
            input(),
            4_000_000_000,
            Mode::Count,
            &Fake::default(),
            &mut Plain,
            None,
            &r,
            policy,
        )
        .unwrap()
    };
    let a = run();
    let b = run();
    assert_eq!(a.completed, 64);
    assert_eq!(a.action_ids, b.action_ids);
    assert_eq!(a.schedule.max_pending, 4);
    assert!(a.schedule.expansion_followers > 0);
    assert_eq!(
        serde_json::to_value(a.snapshot).unwrap(),
        serde_json::to_value(b.snapshot).unwrap()
    );
}
#[test]
fn late_batch_has_no_counts_no_plan_support_and_no_reservation_leak() {
    struct Late<'a>(&'a Fake);
    impl LeafEvaluator<Key, String> for Late<'_> {
        fn evaluate(&mut self, i: &Info<Key, String>) -> Result<Statistics, CoreError> {
            self.0.now.store(3_600_000_000, Ordering::Relaxed);
            Plain.evaluate(i)
        }
    }
    let (mut s, r) = prepared(Config {
        simulations: 32,
        k: 4,
        ..Default::default()
    });
    let clock = Fake::default();
    let out = s
        .decide_cooperative(
            &identity(),
            1,
            input(),
            4_000_000_000,
            Mode::Count,
            &clock,
            &mut Late(&clock),
            None,
            &r,
            BatchPolicy {
                cap: 4,
                pending_limit: 4,
                flush_us: 2000,
                max_candidate_cells: 32768,
            },
        )
        .unwrap();
    assert_eq!(out.completed, 0);
    assert_eq!(out.retired_pending, 4);
    assert!(out.snapshot.is_none());
    assert!(out.actions.iter().all(|a| a.support == 0));
}
#[test]
fn cancellation_at_each_publication_point_rolls_back_edges_and_expansion() {
    for (stage, at) in [
        (Stage::Backup, 1),
        (Stage::Expansion, 1),
        (Stage::Publication, 1),
        (Stage::Publication, 2),
    ] {
        let mut game = Brass::new(input(), false).unwrap();
        let info = game.root().unwrap();
        let mut tree = Tree::new(100, 10000).unwrap();
        let root = tree
            .insert(info.clone(), Plain.evaluate(&info).unwrap())
            .unwrap();
        let pending = trajectory::traverse(
            &mut game,
            &tree,
            root,
            0,
            &Config::default(),
            &mut Draws::new(7, None, false),
        )
        .unwrap();
        let evaluation = pending.leaf_info().map(|i| Plain.evaluate(i).unwrap());
        let count = Arc::new(AtomicU64::new(0));
        let probe_count = count.clone();
        let scope = work::Scope::enter(work::Control {
            deadline: Instant::now() + Duration::from_secs(10),
            cancelled: Arc::new(AtomicBool::new(false)),
            probe: Some(Arc::new(move |s| {
                if s == stage && probe_count.fetch_add(1, Ordering::Relaxed) + 1 == at {
                    Err(search::invalid("injected cancellation"))
                } else {
                    Ok(())
                }
            })),
        });
        assert!(
            pending
                .complete_checked(&mut tree, evaluation, &mut || Ok(()))
                .is_err(),
            "{stage:?}/{at}"
        );
        drop(scope);
        assert_eq!(tree.len(), 1);
        assert!(tree.get(root).unwrap().stats.count.iter().all(|n| *n == 0));
    }
}
#[test]
fn foreign_tree_cannot_receive_pending_backup() {
    let mut game = Brass::new(input(), false).unwrap();
    let info = game.root().unwrap();
    let mut a = Tree::new(100, 10000).unwrap();
    let root = a
        .insert(info.clone(), Plain.evaluate(&info).unwrap())
        .unwrap();
    let pending = trajectory::traverse(
        &mut game,
        &a,
        root,
        0,
        &Config::default(),
        &mut Draws::new(7, None, false),
    )
    .unwrap();
    let evaluation = pending.leaf_info().map(|i| Plain.evaluate(i).unwrap());
    let mut b = Tree::new(100, 10000).unwrap();
    b.insert(info.clone(), Plain.evaluate(&info).unwrap())
        .unwrap();
    assert!(pending
        .complete_checked(&mut b, evaluation, &mut || Ok(()))
        .is_err());
    assert_eq!(b.len(), 1);
    assert!(b.ordered().all(|n| n.stats.count.iter().all(|c| *c == 0)));
}
#[test]
fn deep_generation_cancellation_is_a_fallible_result() {
    let game_input = input();
    let count = Arc::new(AtomicU64::new(0));
    let n = count.clone();
    let _scope = work::Scope::enter(work::Control {
        deadline: Instant::now() + Duration::from_secs(10),
        cancelled: Arc::new(AtomicBool::new(false)),
        probe: Some(Arc::new(move |s| {
            if s == Stage::Generation && n.fetch_add(1, Ordering::Relaxed) == 3 {
                Err(search::invalid("generation cancelled"))
            } else {
                Ok(())
            }
        })),
    });
    assert!(Brass::new(game_input, false).is_err());
    assert_eq!(count.load(Ordering::Relaxed), 4);
}

#[test]
fn merger_rejects_malformed_snapshots_without_panicking() {
    let root = Statistics::new(vec![0.; 2], [0.; 4]).unwrap();
    let snapshot = Snapshot {
        stats: root.clone(),
        survivors: vec![0],
        completed_rounds: 1,
        support_watermark: 2,
        remainder_simulations: 0,
        reason: "balanced_round",
    };
    for mutation in 0..5 {
        let mut s = snapshot.clone();
        s.stats.count = vec![1, 1];
        match mutation {
            0 => s.stats.count = vec![u64::MAX, 1],
            1 => s.stats.returns[0][3] = f64::NAN,
            2 => s.survivors = vec![2],
            3 => s.stats.value[0] = 1.,
            _ => s.support_watermark = 3,
        }
        assert!(merge::pool(&root, &[0., 0.], 0, Parameters::default(), [Some(&s), None]).is_err());
    }
}
