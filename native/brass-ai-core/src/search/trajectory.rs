//! Owned leaf suspension: no tree/world borrow survives evaluator work.
use super::{
    tree::{Handle, PathEdge, Tree},
    *,
};
pub struct Pending<G: Game> {
    // Own the sampled world until evaluation/backup completes or this is dropped.
    world: G::World,
    info: Option<Info<G::Key, G::Action>>,
    terminal: Option<[f64; 4]>,
    path: Vec<PathEdge>,
    chance: Vec<G::Action>,
    spent: [f64; 4],
    steps: usize,
    work: Work,
}
#[derive(Clone, Copy, Default)]
pub struct Work {
    pub sampling_ns: u64,
    pub information_ns: u64,
    pub lookup_ns: u64,
    pub selection_ns: u64,
    pub application_ns: u64,
}
impl<G: Game> Pending<G> {
    /// Read completed decision edges while the suspension still owns its world.
    /// A caller may stage plan evidence here, but must publish it only after backup.
    pub fn path(&self) -> &[PathEdge] {
        &self.path
    }
    pub fn leaf_value(&self, evaluation: Option<&Statistics>) -> Result<[f64; 4]> {
        match (evaluation, self.terminal) {
            (Some(s), None) => Ok(s.value),
            (None, Some(v)) => Ok(v),
            _ => Err(invalid("leaf response does not match pending expansion")),
        }
    }
    pub fn root_return(&self, evaluation: Option<&Statistics>) -> Result<[f64; 4]> {
        let value = self.leaf_value(evaluation)?;
        let first = self
            .path
            .first()
            .ok_or_else(|| invalid("empty trajectory"))?;
        Ok(std::array::from_fn(|s| {
            value[s] - (self.spent[s] - first.spent[s])
        }))
    }
    pub fn work(&self) -> Work {
        self.work
    }
    pub fn leaf_info(&self) -> Option<&Info<G::Key, G::Action>> {
        self.info.as_ref()
    }
    pub fn path_len(&self) -> usize {
        self.path.len()
    }
    pub fn chance_len(&self) -> usize {
        self.chance.len()
    }
    pub fn steps(&self) -> usize {
        self.steps
    }
    /// A wave may contain followers of the same first-new information set.
    /// Only its first completed owner inserts it. Followers use that exact reply.
    pub fn complete_checked(
        self,
        tree: &mut Tree<G::Key, G::Action>,
        evaluation: Option<Statistics>,
        check_time: &mut impl FnMut() -> Result<()>,
    ) -> Result<()> {
        let value = self.leaf_value(evaluation.as_ref())?;
        let staged = tree.prepare_backup(&self.path, value, self.spent)?;
        check_time()?;
        let inserted = if let (Some(info), Some(stats)) = (self.info, evaluation) {
            if let Some(h) = tree.find(&info)? {
                let old = &tree.get(h)?.stats;
                check(
                    old.logits == stats.logits && old.value == stats.value,
                    "expansion follower reply mismatch",
                )?;
                None
            } else {
                Some(tree.insert(info, stats)?)
            }
        } else {
            None
        };
        tree.commit_checked(staged, inserted, check_time)
    }
    /// Consuming self makes a second publication impossible. Failure publishes no counts.
    pub fn complete(
        self,
        tree: &mut Tree<G::Key, G::Action>,
        evaluation: Option<Statistics>,
        trace: bool,
    ) -> Result<Option<Trace<G::Action>>> {
        let value = match (&evaluation, self.terminal) {
            (Some(s), None) => s.value,
            (None, Some(v)) => v,
            _ => return Err(invalid("leaf response does not match pending expansion")),
        };
        let staged = tree.prepare_backup(&self.path, value, self.spent)?;
        let (leaf, value) = match (self.info, evaluation, self.terminal) {
            (Some(info), Some(stats), None) => {
                let value = stats.value;
                let h = tree.insert(info, stats)?;
                (Leaf::Node(tree.get(h)?.ordinal), value)
            }
            (None, None, Some(value)) => (Leaf::Terminal("terminal".into()), value),
            _ => return Err(invalid("leaf response does not match pending expansion")),
        };
        let edge_values = tree.publish_backup(staged);
        drop(self.world);
        if !trace {
            return Ok(None);
        }
        let path = self
            .path
            .iter()
            .map(|e| Ok((tree.get(e.node)?.ordinal, e.edge)))
            .collect::<Result<_>>()?;
        Ok(Some(Trace {
            path,
            leaf,
            value,
            chance: self.chance,
            edge_values,
            leaf_spent: self.spent,
        }))
    }
}
pub fn traverse<G: Game>(
    game: &mut G,
    tree: &Tree<G::Key, G::Action>,
    root: Handle,
    root_edge: usize,
    config: &Config,
    draws: &mut Draws,
) -> Result<Pending<G>> {
    let mut work = Work::default();
    let clock = Instant::now();
    let mut world = game.sample(draws)?;
    work.sampling_ns = nanos(clock);
    let clock = Instant::now();
    let initial = game.inspect(&mut world)?;
    work.information_ns = nanos(clock);
    match &initial {
        Position::Decision(info) => check(
            *info == tree.get(root)?.info,
            "sample changed permitted root/context/legal set",
        )?,
        _ => return Err(invalid("sampled root is not a decision")),
    }
    let mut path = Vec::new();
    let mut chance = Vec::new();
    let mut initial = Some(initial);
    for step in 0..config.max_depth {
        crate::work::checkpoint(crate::work::Stage::Traversal)?;
        let position = if let Some(position) = initial.take() {
            position
        } else {
            let clock = Instant::now();
            let p = game.inspect(&mut world)?;
            work.information_ns += nanos(clock);
            p
        };
        let action = match position {
            Position::Terminal(value) => {
                let spent = game.spent(&world);
                return Ok(Pending {
                    world,
                    info: None,
                    terminal: Some(value),
                    path,
                    chance,
                    spent,
                    steps: step,
                    work,
                });
            }
            Position::Chance(outcomes) => {
                let a = draws.categorical(&outcomes)?;
                chance.push(a.clone());
                a
            }
            Position::Decision(info) => {
                info.validate()?;
                let clock = Instant::now();
                let handle = tree.find(&info)?;
                work.lookup_ns += nanos(clock);
                let Some(h) = handle else {
                    let spent = game.spent(&world);
                    return Ok(Pending {
                        world,
                        info: Some(info),
                        terminal: None,
                        path,
                        chance,
                        spent,
                        steps: step,
                        work,
                    });
                };
                let node = tree.get(h)?;
                let clock = Instant::now();
                let edge = if h == root {
                    root_edge
                } else {
                    node.stats.select(info.actor, config.parameters())?
                };
                work.selection_ns += nanos(clock);
                let action = info
                    .actions
                    .get(edge)
                    .ok_or_else(|| invalid("root edge outside legal set"))?
                    .clone();
                path.push(PathEdge {
                    node: h,
                    edge,
                    spent: game.spent(&world),
                });
                action
            }
        };
        let clock = Instant::now();
        game.step(&mut world, &action)?;
        work.application_ns += nanos(clock);
    }
    Err(invalid("trajectory exceeded diagnostic depth guard"))
}
