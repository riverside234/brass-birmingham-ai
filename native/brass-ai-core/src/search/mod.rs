//! Scalar Gumbel IS-MCTS oracle and the separately versioned whole-turn adapter.
pub mod brass;
pub mod halving;
pub mod math;
pub mod reward;
pub mod session;
pub mod toy;
pub mod trajectory;
pub mod tree;
use crate::CoreError;
use math::{Parameters, Statistics};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use std::{hash::Hash, time::Instant};
use tree::Tree;

pub const IDENTITY: &str = "brass-gumbel-is-mcts/count-v1";
pub type Result<T> = std::result::Result<T, CoreError>;
pub fn invalid(s: impl Into<String>) -> CoreError {
    CoreError::Invalid(s.into())
}
pub fn check(ok: bool, s: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(invalid(s))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Info<K, A> {
    /// Immutable typed identities are shared within one private tree/cache/plan.
    /// Equality and hashing still inspect semantic contents, never pointer IDs.
    pub key: std::sync::Arc<K>,
    pub actor: usize,
    pub actions: std::sync::Arc<Vec<A>>,
}
impl<K, A: Ord> Info<K, A> {
    pub fn validate(&self) -> Result<()> {
        check(
            self.actor < 4
                && !self.actions.is_empty()
                && self.actions.windows(2).all(|p| p[0] < p[1]),
            "invalid actor/canonical legal set",
        )
    }
}
pub enum Position<K, A> {
    Decision(Info<K, A>),
    Chance(Vec<(A, f64)>),
    Terminal([f64; 4]),
}
pub trait Game {
    type Key: Eq + Hash + Clone;
    type Action: Eq + Ord + Clone;
    type World;
    fn root(&mut self) -> Result<Info<Self::Key, Self::Action>>;
    fn sample(&mut self, draws: &mut Draws) -> Result<Self::World>;
    fn inspect(&self, world: &mut Self::World) -> Result<Position<Self::Key, Self::Action>>;
    fn step(&self, world: &mut Self::World, action: &Self::Action) -> Result<()>;
    fn spent(&self, world: &Self::World) -> [f64; 4];
}
pub trait LeafEvaluator<K, A> {
    fn evaluate(&mut self, info: &Info<K, A>) -> Result<Statistics>;
    fn evaluate_batch(&mut self, infos: &[&Info<K, A>]) -> Result<Vec<Statistics>> {
        infos.iter().map(|info| self.evaluate(info)).collect()
    }
}
pub struct Draws {
    rng: ChaCha12Rng,
    tape: Option<Vec<f64>>,
    index: usize,
    pub used: Vec<f64>,
    record: bool,
}
impl Draws {
    pub fn new(seed: u64, tape: Option<Vec<f64>>, record: bool) -> Self {
        Self {
            rng: ChaCha12Rng::seed_from_u64(seed),
            tape,
            index: 0,
            used: vec![],
            record,
        }
    }
    pub fn uniform(&mut self) -> Result<f64> {
        let u = if let Some(tape) = &self.tape {
            *tape
                .get(self.index)
                .ok_or_else(|| invalid("replay tape exhausted"))?
        } else {
            ((self.rng.next_u64() >> 12) as f64 + 0.5) / 4503599627370496.
        };
        math::gumbel(u)?;
        self.index += 1;
        if self.record {
            self.used.push(u);
        }
        Ok(u)
    }
    pub fn categorical<A: Clone>(&mut self, outcomes: &[(A, f64)]) -> Result<A> {
        check(
            !outcomes.is_empty() && outcomes.iter().all(|(_, p)| p.is_finite() && *p >= 0.),
            "invalid chance distribution",
        )?;
        let total: f64 = outcomes.iter().map(|(_, p)| p).sum();
        check(total.is_finite() && total > 0., "empty chance mass")?;
        let target = self.uniform()? * total;
        let mut cumulative = 0.;
        for (a, p) in outcomes {
            cumulative += p;
            if target < cumulative {
                return Ok(a.clone());
            }
        }
        outcomes
            .iter()
            .rev()
            .find(|(_, p)| *p > 0.)
            .map(|(a, _)| a.clone())
            .ok_or_else(|| invalid("empty chance mass"))
    }
    pub fn finish(&self) -> Result<()> {
        check(
            self.tape.as_ref().is_none_or(|t| t.len() == self.index),
            "unused replay draws",
        )
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub simulations: u64,
    pub k: usize,
    pub seed: u64,
    pub c1: f64,
    pub c2: f64,
    pub max_depth: usize,
    pub max_nodes: usize,
    pub max_edges: usize,
    pub trace: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            simulations: 200,
            k: 16,
            seed: 7,
            c1: 50.,
            c2: 0.1,
            max_depth: 256,
            max_nodes: 16384,
            max_edges: 2_000_000,
            trace: false,
        }
    }
}
impl Config {
    pub fn parameters(&self) -> Parameters {
        Parameters {
            c1: self.c1,
            c2: self.c2,
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.parameters().validate()?;
        check(
            self.simulations <= 1_000_000
                && self.k > 0
                && (1..=5000).contains(&self.max_depth)
                && (1..=65536).contains(&self.max_nodes)
                && (1..=4_000_000).contains(&self.max_edges),
            "search bounds",
        )
    }
}
#[derive(Default, Debug, Serialize)]
pub struct Metrics {
    pub completed: u64,
    pub nn_leaves: u64,
    pub terminal_leaves: u64,
    pub decisions: u64,
    pub chance_steps: u64,
    pub max_depth: usize,
    pub nodes: usize,
    pub edges: usize,
    pub root_prepare_ns: u64,
    pub first_leaf_ns: u64,
    pub total_ns: u64,
    pub traversal_ns: u64,
    pub evaluation_ns: u64,
    pub backup_ns: u64,
    pub sampling_ns: u64,
    pub information_ns: u64,
    pub lookup_ns: u64,
    pub selection_ns: u64,
    pub application_ns: u64,
}
pub fn nanos(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Leaf {
    Node(usize),
    Terminal(String),
}
#[derive(Debug, Serialize)]
pub struct Trace<A> {
    pub path: Vec<(usize, usize)>,
    pub leaf: Leaf,
    pub value: [f64; 4],
    pub chance: Vec<A>,
    pub edge_values: Vec<[f64; 4]>,
    pub leaf_spent: [f64; 4],
}
#[derive(Debug, Serialize)]
pub struct NodeReport<K, A> {
    pub info: Info<K, A>,
    #[serde(flatten)]
    pub stats: Statistics,
    pub q: Vec<f64>,
    pub v_mix: f64,
    pub improved_policy: Vec<f64>,
}
#[derive(Debug, Serialize)]
pub struct Report<K, A> {
    pub version: &'static str,
    pub config: Config,
    pub selected_action: A,
    pub status: &'static str,
    pub effective_k: usize,
    pub gumbels: Vec<f64>,
    pub rounds: Vec<halving::Round>,
    pub remainder_simulations: u64,
    pub nodes: Vec<NodeReport<K, A>>,
    pub trajectories: Vec<Trace<A>>,
    pub draws: Vec<f64>,
    pub metrics: Metrics,
}

pub fn run<G: Game, E: LeafEvaluator<G::Key, G::Action>>(
    game: &mut G,
    evaluator: &mut E,
    config: Config,
    tape: Option<Vec<f64>>,
) -> Result<Report<G::Key, G::Action>> {
    config.validate()?;
    let start = Instant::now();
    let mut metrics = Metrics::default();
    let mut tree = Tree::new(config.max_nodes, config.max_edges)?;
    let info = game.root()?;
    info.validate()?;
    let stats = evaluator.evaluate(&info)?;
    let root = tree.insert(info, stats)?;
    let mut draws = Draws::new(config.seed, tape, config.trace);
    let uniforms = (0..tree.get(root)?.info.actions.len())
        .map(|_| draws.uniform())
        .collect::<Result<Vec<_>>>()?;
    let mut tournament = halving::Tournament::new(
        &tree.get(root)?.stats.logits,
        &uniforms,
        config.k,
        config.simulations,
    )?;
    metrics.root_prepare_ns = nanos(start);
    let mut trajectories = Vec::new();
    while let Some(edge) = tournament.next_edge()? {
        let clock = Instant::now();
        let pending = trajectory::traverse(game, &tree, root, edge, &config, &mut draws)?;
        metrics.traversal_ns += nanos(clock);
        metrics.decisions += pending.path_len() as u64;
        metrics.chance_steps += pending.chance_len() as u64;
        metrics.max_depth = metrics.max_depth.max(pending.steps());
        let work = pending.work();
        metrics.sampling_ns += work.sampling_ns;
        metrics.information_ns += work.information_ns;
        metrics.lookup_ns += work.lookup_ns;
        metrics.selection_ns += work.selection_ns;
        metrics.application_ns += work.application_ns;
        let clock = Instant::now();
        let evaluated = if let Some(info) = pending.leaf_info() {
            metrics.nn_leaves += 1;
            Some(evaluator.evaluate(info)?)
        } else {
            metrics.terminal_leaves += 1;
            None
        };
        metrics.evaluation_ns += nanos(clock);
        let clock = Instant::now();
        let trace = pending.complete(&mut tree, evaluated, config.trace)?;
        metrics.backup_ns += nanos(clock);
        tournament.completed_backup(
            edge,
            &tree.get(root)?.stats,
            tree.get(root)?.info.actor,
            config.parameters(),
        )?;
        metrics.completed += 1;
        if metrics.completed == 1 {
            metrics.first_leaf_ns = nanos(start);
        }
        if let Some(trace) = trace {
            trajectories.push(trace);
        }
    }
    draws.finish()?;
    let selected_action = tree.get(root)?.info.actions[tournament.winner()?].clone();
    metrics.nodes = tree.len();
    metrics.edges = tree.edges();
    metrics.total_ns = nanos(start);
    let nodes = if config.trace {
        tree.ordered()
            .map(|n| {
                let (q, v_mix) = n.stats.completed_q(n.info.actor)?;
                Ok(NodeReport {
                    info: n.info.clone(),
                    stats: n.stats.clone(),
                    q,
                    v_mix,
                    improved_policy: n.stats.policy(n.info.actor, config.parameters())?,
                })
            })
            .collect::<Result<_>>()?
    } else {
        vec![]
    };
    Ok(Report {
        version: IDENTITY,
        status: if config.simulations == 0 {
            "prior_fallback"
        } else {
            "completed"
        },
        config,
        selected_action,
        effective_k: tournament.effective_k,
        gumbels: tournament.gumbels,
        rounds: tournament.rounds,
        remainder_simulations: tournament.remainder_simulations,
        nodes,
        trajectories,
        draws: draws.used,
        metrics,
    })
}
