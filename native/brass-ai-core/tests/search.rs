use brass_ai_core::search::{
    self,
    halving::{effective_k, Tournament},
    math::{self, Parameters, Statistics},
    tree::{PathEdge, Tree},
    Config, Info,
};

#[test]
fn completed_q_is_actor_prior_weighted() {
    let mut s = Statistics::new(
        vec![0.2f64.ln(), 0.3f64.ln(), 0.5f64.ln()],
        [9., 8., 0.6, 7.],
    )
    .unwrap();
    assert_eq!(s.completed_q(2).unwrap().0, vec![0.6; 3]);
    s.count = vec![2, 1, 0];
    s.returns = vec![[100., -100., 0.4, 50.], [100., -100., 0.8, 50.], [0.; 4]];
    let (q, m) = s.completed_q(2).unwrap();
    for (a, b) in q.iter().zip([0.2, 0.8, 0.57]) {
        assert!((a - b).abs() < 1e-14);
    }
    assert!((m - 0.57).abs() < 1e-14);
}
#[test]
fn underflow_ties_and_nonfinite() {
    let mut s = Statistics::new(vec![0., -1000., -1001.], [0.25; 4]).unwrap();
    s.count = vec![0, 1, 3];
    s.returns = vec![[0.; 4], [0.2; 4], [2.4; 4]];
    let mean = (0.2 + (-1f64).exp() * 0.8) / (1. + (-1f64).exp());
    assert!((s.completed_q(0).unwrap().1 - (0.25 + 4. * mean) / 5.).abs() < 1e-14);
    assert_eq!(math::argmax(&[1., 1.]).unwrap(), 0);
    assert_eq!(math::argmax(&[1., 1. + f64::EPSILON]).unwrap(), 1);
    assert!(Statistics::new(vec![f64::NAN], [0.; 4]).is_err());
    assert!(Statistics::new(vec![0.], [f64::INFINITY; 4]).is_err());
    for x in [0., 1., f64::NAN, -0.1] {
        assert!(math::gumbel(x).is_err());
    }
}
proptest::proptest! {
    #[test]
    fn equation_three_matches_literal_squared_loss(entries in proptest::collection::vec((-200i32..200,0u32..20,-100i32..100),2..18)) {
        let mut s=Statistics::new(entries.iter().map(|e|e.0 as f64/100.).collect(),[0.2;4]).unwrap();
        s.count=entries.iter().map(|e|e.1 as u64).collect();s.returns=entries.iter().map(|e|[e.1 as f64*e.2 as f64/100.;4]).collect();
        let p=s.policy(1,Parameters::default()).unwrap();let d=1.+s.count.iter().sum::<u64>() as f64;
        let loss:Vec<_>=(0..p.len()).map(|a|-p.iter().enumerate().map(|(b,p)|(p-(s.count[b] as f64+f64::from(a==b))/d).powi(2)).sum::<f64>()).collect();
        let selected=s.select(1,Parameters::default()).unwrap();let best=math::argmax(&loss).unwrap();
        // Near ties are tested explicitly above; floating cancellation in the
        // literal squared expression must not define a different tie policy.
        proptest::prop_assert!((loss[selected]-loss[best]).abs()<1e-12);
    }
}
#[test]
fn post_backup_halving_and_small_budgets() {
    for width in [1, 3, 5, 16] {
        for budget in [0, 1, 2, 3, 7, 8, 31, 64, 200] {
            let mut s = Statistics::new(vec![0.; width], [0.; 4]).unwrap();
            let mut t = Tournament::new(&s.logits, &vec![0.5; width], 32, budget).unwrap();
            while let Some(i) = t.next_edge().unwrap() {
                assert!(t.next_edge().is_err());
                s.count[i] += 1;
                s.returns[i] = [i as f64 * s.count[i] as f64; 4];
                t.completed_backup(i, &s, 0, Parameters::default()).unwrap();
            }
            assert!(t.winner().is_ok());
            assert_eq!(s.count.iter().sum::<u64>(), budget);
            assert!(t.rounds.iter().all(|r| r.quota_per_action > 0));
        }
    }
    assert_eq!(effective_k(16, 16, 23).unwrap(), 4);
    assert_eq!(effective_k(16, 16, 24).unwrap(), 8);
    let mut s = Statistics::new(vec![0.; 2], [0.; 4]).unwrap();
    let mut t = Tournament::new(&s.logits, &[0.5; 2], 2, 2).unwrap();
    assert_eq!(t.next_edge().unwrap(), Some(0));
    s.count[0] = 1;
    t.completed_backup(0, &s, 0, Parameters::default()).unwrap();
    assert!(t.rounds.is_empty());
    assert_eq!(t.next_edge().unwrap(), Some(1));
    s.count[1] = 1;
    s.returns[1] = [10., -5., 2., 1.];
    t.completed_backup(1, &s, 0, Parameters::default()).unwrap();
    assert_eq!(t.winner().unwrap(), 1);
}
fn info(key: u8) -> Info<u8, u8> {
    Info {
        key: std::sync::Arc::new(key),
        actor: 0,
        actions: vec![0, 1].into(),
    }
}
#[test]
fn arena_identity_stale_collision_and_cost_backup() {
    let mut t = Tree::new(4, 8).unwrap();
    let h = t
        .insert(info(1), Statistics::new(vec![0.; 2], [0.; 4]).unwrap())
        .unwrap();
    let mut bad = info(1);
    bad.actions = vec![1, 2].into();
    assert!(t.find(&bad).is_err());
    let foreign = Tree::<u8, u8>::new(4, 8).unwrap();
    assert!(foreign.get(h).is_err());
    let path = vec![
        PathEdge {
            node: h,
            edge: 0,
            spent: [0., 0.02, 0., 0.],
        },
        PathEdge {
            node: h,
            edge: 0,
            spent: [0.02; 4],
        },
    ];
    t.backup(&path, [0.1, 0.2, 0.3, 0.4], [0.04; 4]).unwrap();
    assert_eq!(t.get(h).unwrap().stats.count, vec![2, 0]);
    assert!((t.get(h).unwrap().stats.returns[0][0] - 0.14).abs() < 1e-14);
    let before = t.get(h).unwrap().stats.count.clone();
    let mut invalid = path;
    invalid.push(PathEdge {
        node: h,
        edge: 9,
        spent: [0.; 4],
    });
    assert!(t.backup(&invalid, [1.; 4], [0.04; 4]).is_err());
    assert_eq!(t.get(h).unwrap().stats.count, before);
    t.clear();
    let _new = t
        .insert(info(2), Statistics::new(vec![0.; 2], [0.; 4]).unwrap())
        .unwrap();
    assert!(t.get(h).is_err());
}
#[test]
fn toy_first_leaf_terminal_and_exact_budget() {
    let r = search::run(
        &mut search::toy::HiddenOffer { players: 4 },
        &mut search::toy::Evaluator,
        Config {
            trace: true,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(r.metrics.completed, 200);
    assert_eq!(
        r.rounds
            .iter()
            .map(|r| r.candidates.len() as u64 * r.quota_per_action)
            .collect::<Vec<_>>(),
        vec![48, 48, 48, 50]
    );
    assert_eq!(r.remainder_simulations, 6);
    assert_eq!(r.trajectories[0].path.len(), 1);
    assert_eq!(r.nodes[1].info.actor, 1);
    assert!(r.metrics.terminal_leaves > 0);
    for (id, n) in r.nodes.iter().enumerate() {
        for e in 0..n.stats.count.len() {
            let paths: Vec<_> = r
                .trajectories
                .iter()
                .filter(|t| t.path.contains(&(id, e)))
                .collect();
            assert_eq!(n.stats.count[e], paths.len() as u64);
            for s in 0..4 {
                assert!(
                    (n.stats.returns[e][s] - paths.iter().map(|t| t.value[s]).sum::<f64>()).abs()
                        < 1e-12
                );
            }
        }
    }
}

#[test]
fn failed_or_dropped_leaf_never_publishes_visits() {
    use search::{Game, LeafEvaluator};
    let mut game = search::toy::HiddenOffer { players: 4 };
    let mut evaluator = search::toy::Evaluator;
    let info = game.root().unwrap();
    let stats = evaluator.evaluate(&info).unwrap();
    let mut tree = Tree::new(4, 64).unwrap();
    let root = tree.insert(info, stats).unwrap();
    let mut draws = search::Draws::new(7, None, false);
    let pending =
        search::trajectory::traverse(&mut game, &tree, root, 0, &Config::default(), &mut draws)
            .unwrap();
    assert!(pending
        .complete(
            &mut tree,
            Some(Statistics::new(vec![0.], [0.; 4]).unwrap()),
            false
        )
        .is_err());
    assert_eq!(tree.len(), 1);
    assert_eq!(tree.get(root).unwrap().stats.count, vec![0; 16]);
    let pending =
        search::trajectory::traverse(&mut game, &tree, root, 0, &Config::default(), &mut draws)
            .unwrap();
    drop(pending);
    assert_eq!(tree.len(), 1);
    assert_eq!(tree.get(root).unwrap().stats.count, vec![0; 16]);
    assert!(search::trajectory::traverse(
        &mut game,
        &tree,
        root,
        0,
        &Config {
            max_depth: 1,
            ..Default::default()
        },
        &mut draws
    )
    .is_err());
    assert_eq!(tree.get(root).unwrap().stats.count, vec![0; 16]);
}

#[test]
fn hash_collision_uses_semantic_equality() {
    use std::hash::{Hash, Hasher};
    #[derive(Clone, Eq, PartialEq)]
    struct Collision(u8);
    impl Hash for Collision {
        fn hash<H: Hasher>(&self, h: &mut H) {
            0u8.hash(h);
        }
    }
    let mut t = Tree::new(4, 8).unwrap();
    for key in [1, 2] {
        t.insert(
            Info {
                key: std::sync::Arc::new(Collision(key)),
                actor: 0,
                actions: vec![0u8].into(),
            },
            Statistics::new(vec![0.], [key as f64; 4]).unwrap(),
        )
        .unwrap();
    }
    for key in [1, 2] {
        let h = t
            .find(&Info {
                key: std::sync::Arc::new(Collision(key)),
                actor: 0,
                actions: vec![0u8].into(),
            })
            .unwrap()
            .unwrap();
        assert_eq!(t.get(h).unwrap().stats.value, [key as f64; 4]);
    }
}

#[test]
fn exact_terminal_ties_and_vp_cap() {
    use brass_ai_core::{
        search::reward::{Insolvency, Reward},
        state::dto::{ResultDto, ResultSeat},
        types::Seat,
    };
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
    let result = ResultDto {
        objective: "fixture".into(),
        seats: [
            (200, 10, 20, 1, 0.5),
            (200, 10, 20, 1, 0.5),
            (200, 10, 19, 3, 0.),
            (199, 20, 90, 4, 0.),
        ]
        .map(|(vp, income, cash, rank, utility)| ResultSeat {
            seat: Seat::Zero,
            vp,
            income,
            cash,
            rank,
            utility,
        }),
    };
    let mut result = result;
    for (i, s) in result.seats.iter_mut().enumerate() {
        s.seat = Seat::try_from(i as u8).unwrap();
    }
    let value = reward.terminal(&result).unwrap();
    assert!((value[0] - (0.5 + 0.1 * 2.5 / 3. + 0.18)).abs() < 1e-14);
    assert_eq!(value[0], value[1]);
    assert!((value[2] - (0.1 / 3. + 0.18)).abs() < 1e-14);
    assert_eq!(value[3], 0.18);
    assert_eq!(
        reward.identity().unwrap(),
        "a84a3fccf2421d40aefc71c80d3b8839cf3ff22900eb6c5b917b821b5bb8d72e"
    );
}
