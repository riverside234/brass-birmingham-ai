//! Independent small-tree oracle: full equality despite deliberate hash collisions,
//! completed-watermark rollback, node-relative costs and fresh promoted counts.
use brass_ai_core::search::{
    self,
    math::Statistics,
    tree::{Handle, PathEdge, Tree},
    Info,
};
use std::{
    hash::{Hash, Hasher},
    sync::Arc,
};

#[derive(Clone, Eq, PartialEq)]
struct Collision(u8);
impl Hash for Collision {
    fn hash<H: Hasher>(&self, h: &mut H) {
        0u8.hash(h);
    }
}
fn info(n: u8) -> Info<Collision, u8> {
    Info {
        key: Arc::new(Collision(n)),
        actor: 0,
        actions: Arc::new(vec![0, 1]),
    }
}
fn fresh() -> Statistics {
    Statistics::new(vec![0., 1.], [0.4; 4]).unwrap()
}
fn edge(h: Handle, e: usize, cost: f64) -> PathEdge {
    PathEdge {
        node: h,
        edge: e,
        spent: [cost; 4],
    }
}
fn fixture() -> (Tree<Collision, u8>, [Handle; 7]) {
    let mut t = Tree::new(16, 32).unwrap();
    let r = t.insert(info(0), fresh()).unwrap();
    t.enable_retention();
    let a = t.insert_child(info(1), fresh(), &edge(r, 0, 0.)).unwrap();
    let b = t.insert_child(info(2), fresh(), &edge(a, 0, 0.01)).unwrap();
    let c = t.insert_child(info(3), fresh(), &edge(b, 0, 0.02)).unwrap();
    let d = t.insert_child(info(4), fresh(), &edge(c, 1, 0.03)).unwrap();
    let sibling = t.insert_child(info(5), fresh(), &edge(r, 1, 0.)).unwrap();
    for _ in 0..3 {
        t.backup(
            &[
                edge(r, 0, 0.),
                edge(a, 0, 0.01),
                edge(b, 0, 0.02),
                edge(c, 1, 0.03),
            ],
            [1.; 4],
            [0.05; 4],
        )
        .unwrap();
    }
    t.checkpoint_round(3);
    t.backup(
        &[edge(r, 0, 0.), edge(b, 0, 0.02), edge(c, 1, 0.03)],
        [9.; 4],
        [0.05; 4],
    )
    .unwrap();
    let partial = t.insert_child(info(6), fresh(), &edge(c, 0, 0.03)).unwrap();
    (t, [r, a, b, c, d, sibling, partial])
}
#[test]
fn promotion_matches_scalar_reference_and_never_recredits_old_root_work() {
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/GumbelISMCTS/fixtures/reuse-v1.json"
    ))
    .unwrap();
    let (mut t, [r, a, b, c, d, sibling, partial]) = fixture();
    let (root, m) = t
        .promote(
            r,
            &[(info(0), 0), (info(1), 0)],
            &info(2),
            3,
            &mut || Ok(()),
        )
        .unwrap();
    assert_eq!(root, b);
    assert_eq!(m.retained_nodes, 3);
    assert_eq!(m.discarded_nodes, 4);
    assert_eq!(m.inherited_edge_visits, 3);
    assert_eq!(m.reset_root_visits, 3);
    assert_eq!(m.rolled_back_edge_updates, 3);
    for h in [r, a, sibling, partial] {
        assert!(t.get(h).is_err());
    }
    assert_eq!(t.get(d).unwrap().stats.count, vec![0, 0]);
    for (h, expected) in [(b, &reference["promoted"]), (c, &reference["descendant"])] {
        let actual = serde_json::to_value(&t.get(h).unwrap().stats).unwrap();
        assert_eq!(actual["count"], expected["count"]);
        for e in 0..2 {
            for s in 0..4 {
                assert!(
                    (actual["returns"][e][s].as_f64().unwrap()
                        - expected["returns"][e][s].as_f64().unwrap())
                    .abs()
                        < 1e-12
                );
            }
        }
    }
    assert!(
        (t.get(c).unwrap().stats.returns[1][0]
            - reference["descendant"]["returns"][1][0].as_f64().unwrap())
        .abs()
            < 1e-12
    );
    t.backup(&[edge(b, 1, 0.02), edge(c, 0, 0.03)], [0.5; 4], [0.05; 4])
        .unwrap();
    assert_eq!(t.get(b).unwrap().stats.count, vec![0, 1]);
    assert_eq!(t.get(c).unwrap().stats.count, vec![1, 3]);
    assert!((t.get(b).unwrap().stats.returns[1][0] - 0.47).abs() < 1e-12);
    assert!((t.get(c).unwrap().stats.returns[0][0] - 0.48).abs() < 1e-12);
    // Old work is only descendant edge evidence; the root gained exactly one.
    assert_eq!(t.get(b).unwrap().stats.count.iter().sum::<u64>(), 1);
}
#[test]
fn incompatible_prefix_legal_order_watermark_and_foreign_handles_miss() {
    for variant in 0..6 {
        let (mut t, [r, _, b, _, _, _, _]) = fixture();
        let before = t.get(b).unwrap().stats.count.clone();
        let foreign = Tree::<Collision, u8>::new(4, 8).unwrap();
        assert!(foreign.get(b).is_err());
        let mut prefix = vec![(info(0), 0), (info(1), 0)];
        let mut boundary = info(2);
        let mut watermark = 3;
        match variant {
            0 => prefix[1].1 = 1,
            1 => boundary = info(7),
            2 => boundary.actor = 1,
            3 => boundary.actions = Arc::new(vec![0, 2]),
            4 => watermark = 2,
            _ => boundary = info(6),
        }
        assert!(t
            .promote(r, &prefix, &boundary, watermark, &mut || Ok(()))
            .is_err());
        assert_eq!(t.get(b).unwrap().stats.count, before);
    }
}
#[test]
fn cancellation_during_rollback_and_pruning_does_not_publish_a_new_root() {
    for stop in [1, 5, 9, 14] {
        let (mut t, [r, _, _, _, _, _, _]) = fixture();
        let mut calls = 0;
        let result = t.promote(r, &[(info(0), 0), (info(1), 0)], &info(2), 3, &mut || {
            calls += 1;
            search::check(calls < stop, "cancelled retention")
        });
        assert!(result.is_err());
        // Failed moved arenas are dropped, never handed back to a live session.
        drop(t);
    }
}
#[test]
fn reroot_depth_is_a_fresh_diagnostic_guard_not_a_value_horizon() {
    use search::{Config, Draws, Game, Position};
    struct Chain;
    impl Game for Chain {
        type Key = Collision;
        type Action = u8;
        type World = u8;
        fn root(&mut self) -> search::Result<Info<Collision, u8>> {
            Ok(info(2))
        }
        fn sample(&mut self, _: &mut Draws) -> search::Result<u8> {
            Ok(2)
        }
        fn inspect(&self, w: &mut u8) -> search::Result<Position<Collision, u8>> {
            Ok(if *w == 5 {
                Position::Terminal([0.8; 4])
            } else {
                Position::Decision(info(*w))
            })
        }
        fn step(&self, w: &mut u8, _: &u8) -> search::Result<()> {
            *w += 1;
            Ok(())
        }
        fn spent(&self, _: &u8) -> [f64; 4] {
            [0.; 4]
        }
    }
    let (mut t, [r, _, _, c, _, _, _]) = fixture();
    let (root, _) = t
        .promote(
            r,
            &[(info(0), 0), (info(1), 0)],
            &info(2),
            3,
            &mut || Ok(()),
        )
        .unwrap();
    let before = t.get(c).unwrap().stats.count.clone();
    let mut game = Chain;
    let mut draws = Draws::new(7, None, false);
    assert!(search::trajectory::traverse(
        &mut game,
        &t,
        root,
        0,
        &Config {
            max_depth: 1,
            ..Default::default()
        },
        &mut draws
    )
    .is_err());
    assert_eq!(t.get(c).unwrap().stats.count, before);
    // Three decisions plus terminal; the old two-edge prefix is not charged.
    let pending = search::trajectory::traverse(
        &mut game,
        &t,
        root,
        0,
        &Config {
            max_depth: 4,
            ..Default::default()
        },
        &mut draws,
    )
    .unwrap();
    assert_eq!(pending.steps(), 3);
    pending.complete(&mut t, None, false).unwrap();
    assert_eq!(t.get(root).unwrap().stats.count.iter().sum::<u64>(), 1);
}
