//! L11 scalar whole-turn adapter. Two-session coordination/batching belongs to L12.
pub mod clock;
pub mod cooperative;
pub mod merge;
pub mod plans;
pub mod tournament;
use super::{
    brass::{Brass, Input, Key},
    check, invalid,
    math::{self, Statistics},
    trajectory,
    tree::Tree,
    Config, Draws, Game, Info, LeafEvaluator, Result,
};
use crate::{
    features::FeatureContext,
    state::observation::{Recall, RecallEntry, ViewContext},
    types::{Era, Phase, Seat},
};
use clock::{Budget, Clock, Costs, Estimate, FrozenQuota};
use plans::SupportedAction;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tournament::{Rounds, Snapshot};

pub const IDENTITY: &str = "brass-gumbel-is-mcts/session-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TurnKey {
    pub era: Era,
    pub round: u8,
    pub turn: u8,
    pub turn_owner: Seat,
    pub actor: usize,
    pub debt: bool,
}
impl TurnKey {
    pub fn from_context(c: &ViewContext, actor: usize) -> Self {
        Self {
            era: c.era,
            round: c.round,
            turn: c.turn,
            turn_owner: c.turn_owner,
            actor,
            debt: c.phase == Phase::Debt,
        }
    }
    pub fn of(info: &Info<Key, String>) -> Result<Self> {
        let v = info
            .key
            .information
            .observation
            .view
            .as_ref()
            .ok_or_else(|| invalid("missing actor view"))?;
        Ok(Self::from_context(&v.context, info.actor))
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub game_instance: String,
    pub generation: u64,
    pub export_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Startup {
    pub model_id: String,
    pub export_sha256: String,
    pub reward_id: String,
    pub phase: Phase,
    pub estimate: Estimate,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Mode {
    Count,
    Timed { startup: Option<Startup> },
}

#[derive(Debug, Clone)]
pub struct Settings {
    pub search: Config,
    pub minimum_plan_support: u64,
    pub cache_entries: usize,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            search: Config::default(),
            minimum_plan_support: 2,
            cache_entries: 128,
        }
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    pub version: &'static str,
    pub identity: Identity,
    pub request_id: u64,
    pub revision: u32,
    pub root_epoch: u64,
    pub supporting_root_epoch: u64,
    pub turn: TurnKey,
    pub budget: Budget,
    pub context: FeatureContext,
    pub start_phase: Phase,
    pub status: String,
    pub provenance: String,
    pub actions: Vec<SupportedAction>,
    /// Same list-of-semantic-ID-lists consumed by the existing C# validator.
    pub action_ids: Vec<Vec<String>>,
    pub quota: Option<FrozenQuota>,
    pub gumbels: Vec<f64>,
    pub effective_k: usize,
    pub snapshot: Option<Snapshot>,
    pub completed: u64,
    pub attempted: u64,
    pub partial_completed: u64,
    pub retired_pending: u64,
    pub plan_events: usize,
    pub cache_hits: u64,
    pub last_completion_ns: Option<u64>,
    pub work_stopped_ns: u64,
    pub response_ready_ns: u64,
    pub unused_work_ns: u64,
    pub eligible_plans: Vec<Vec<SupportedAction>>,
    pub schedule: cooperative::Metrics,
}
struct CachedEvaluation {
    info: Info<Key, String>,
    stats: Statistics,
}
#[derive(Default)]
struct Cache {
    rows: HashMap<std::sync::Arc<Key>, CachedEvaluation>,
    order: VecDeque<std::sync::Arc<Key>>,
    hits: u64,
}
struct Cached<'a, E, C> {
    cache: &'a mut Cache,
    limit: usize,
    evaluator: &'a mut E,
    clock: &'a C,
    budget: Budget,
}
impl<E: LeafEvaluator<Key, String>, C: Clock> LeafEvaluator<Key, String> for Cached<'_, E, C> {
    fn evaluate(&mut self, info: &Info<Key, String>) -> Result<Statistics> {
        check(
            self.budget.work_open(self.clock),
            "session work deadline/cancellation",
        )?;
        if let Some(row) = self.cache.rows.get(&info.key) {
            check(row.info == *info, "NN cache legal/actor mismatch")?;
            self.cache.hits += 1;
            return Ok(row.stats.clone());
        }
        let stats = self.evaluator.evaluate(info)?;
        check(self.budget.work_open(self.clock), "late session evaluation")?;
        check(
            stats.count.len() == info.actions.len()
                && stats.count.iter().all(|n| *n == 0)
                && stats.returns.iter().all(|r| r.iter().all(|v| *v == 0.)),
            "evaluation contains search statistics",
        )?;
        let fresh = Statistics::new(stats.logits, stats.value)?;
        check(
            fresh.logits.len() == info.actions.len(),
            "evaluation legal width mismatch",
        )?;
        if self.limit > 0 {
            if self.cache.rows.len() == self.limit {
                if let Some(key) = self.cache.order.pop_front() {
                    self.cache.rows.remove(&key);
                }
            }
            self.cache.order.push_back(info.key.clone());
            self.cache.rows.insert(
                info.key.clone(),
                CachedEvaluation {
                    info: info.clone(),
                    stats: fresh.clone(),
                },
            );
        }
        Ok(fresh)
    }
    fn evaluate_batch(&mut self, infos: &[&Info<Key, String>]) -> Result<Vec<Statistics>> {
        check(self.budget.work_open(self.clock), "batch cache deadline")?;
        let mut values = vec![None; infos.len()];
        let mut missing = Vec::new();
        let mut indices = Vec::new();
        for (i, info) in infos.iter().enumerate() {
            if let Some(row) = self.cache.rows.get(&info.key) {
                check(row.info == **info, "batch cache identity")?;
                values[i] = Some(row.stats.clone());
                self.cache.hits += 1;
            } else {
                missing.push(*info);
                indices.push(i);
            }
        }
        if !missing.is_empty() {
            let replies = self.evaluator.evaluate_batch(&missing)?;
            check(
                replies.len() == missing.len() && self.budget.work_open(self.clock),
                "late/invalid batch",
            )?;
            let fresh = replies
                .into_iter()
                .zip(&missing)
                .map(|(s, info)| {
                    check(
                        s.count.len() == info.actions.len()
                            && s.returns.len() == info.actions.len()
                            && s.count.iter().all(|n| *n == 0)
                            && s.returns.iter().flatten().all(|v| *v == 0.),
                        "batch contains visits",
                    )?;
                    let fresh = Statistics::new(s.logits, s.value)?;
                    check(
                        fresh.logits.len() == info.actions.len(),
                        "batch legal width",
                    )?;
                    Ok(fresh)
                })
                .collect::<Result<Vec<_>>>()?;
            check(self.budget.work_open(self.clock), "late cache publication")?;
            for ((info, i), fresh) in missing.into_iter().zip(indices).zip(fresh) {
                if self.limit > 0 {
                    if self.cache.rows.len() == self.limit {
                        if let Some(key) = self.cache.order.pop_front() {
                            self.cache.rows.remove(&key);
                        }
                    }
                    self.cache.order.push_back(info.key.clone());
                    self.cache.rows.insert(
                        info.key.clone(),
                        CachedEvaluation {
                            info: info.clone(),
                            stats: fresh.clone(),
                        },
                    );
                }
                values[i] = Some(fresh);
            }
        }
        values
            .into_iter()
            .map(|s| s.ok_or_else(|| invalid("missing batch value")))
            .collect()
    }
}
struct CostBucket {
    phase: Phase,
    context: FeatureContext,
    costs: Costs,
}
pub struct Session {
    identity: Identity,
    turn: TurnKey,
    context: FeatureContext,
    recall: Recall,
    revision: u32,
    request: u64,
    epoch: u64,
    settings: Settings,
    budget: Budget,
    cache: Cache,
    costs: Vec<CostBucket>,
    previous: Option<Info<Key, String>>,
    continuation: Option<Outcome>,
    expected_after: Option<Info<Key, String>>,
    retired: bool,
}
fn same_context(a: &FeatureContext, b: &FeatureContext) -> bool {
    let mut a = a.clone();
    a.remaining = b.remaining;
    a == *b
}
impl Session {
    pub fn new(
        identity: Identity,
        input: &Input,
        settings: Settings,
        budget: Budget,
    ) -> Result<Self> {
        settings.search.validate()?;
        input.context.validate()?;
        check(
            settings.minimum_plan_support >= 2 && settings.cache_entries <= 256,
            "session bounds",
        )?;
        check(
            !identity.game_instance.is_empty()
                && identity.game_instance.len() <= 512
                && identity.export_sha256.len() == 64
                && identity
                    .export_sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit()),
            "session identity",
        )?;
        crate::state::observation::validate_recall(&input.view, &input.recall)?;
        let mut recall = input.recall.clone();
        if recall.last().is_none_or(|r| r.hand != input.view.own_hand) {
            recall.push(RecallEntry {
                revision: input.view.context.revision,
                hand: input.view.own_hand.clone(),
            });
        }
        Ok(Self {
            identity,
            turn: TurnKey::from_context(&input.view.context, u8::from(input.view.seat) as usize),
            context: input.context.clone(),
            recall,
            revision: input.view.context.revision,
            request: 0,
            epoch: 0,
            settings,
            budget,
            cache: Cache::default(),
            costs: vec![],
            previous: None,
            continuation: None,
            expected_after: None,
            retired: false,
        })
    }
    pub fn budget(&self) -> Budget {
        self.budget
    }
    pub fn next_root_seed(&self) -> u64 {
        self.settings.search.seed.wrapping_add(self.epoch)
    }
    pub fn recall(&self) -> &Recall {
        &self.recall
    }
    pub fn retire(&mut self) {
        self.retired = true;
        self.continuation = None;
        self.expected_after = None;
        self.cache = Cache::default();
    }

    fn canonical_input(&self, mut input: Input) -> Result<Input> {
        let mut expected = self.recall.clone();
        // A live host can commit a complete action between requests. Preserve
        // the intermediate own-hand revision only when the newly observed full
        // permitted state acknowledges a precondition from that validated plan.
        // Such proofs never cross chance or contain an unknown future own hand.
        if let Some(saved) = self
            .continuation
            .iter()
            .flat_map(|out| &out.actions)
            .flat_map(|a| &a.action.steps)
            .map(|s| &s.precondition.key)
            .chain(self.expected_after.iter().map(|i| &i.key))
            .find(|k| {
                k.information.observation.view.as_ref() == Some(&input.view)
                    && k.context == input.context
                    && k.information.recall.starts_with(&self.recall)
            })
        {
            expected = saved.information.recall.clone();
        }
        if expected
            .last()
            .is_none_or(|r| r.hand != input.view.own_hand)
        {
            expected.push(RecallEntry {
                revision: input.view.context.revision,
                hand: input.view.own_hand.clone(),
            });
        }
        // Empty means the caller delegates recall ownership to this session.
        check(
            input.recall.is_empty() || input.recall == expected,
            "caller recall differs from session",
        )?;
        crate::state::observation::validate_recall(&input.view, &expected)?;
        input.recall = expected.clone();
        Ok(input)
    }
    pub fn preview(&self, input: Input) -> Result<(Brass, Info<Key, String>)> {
        let mut game = Brass::new(self.canonical_input(input)?, false)?;
        let info = game.root()?;
        Ok((game, info))
    }
    pub fn continuation_matches(&self, info: &Info<Key, String>) -> bool {
        self.continuation.as_ref().is_some_and(|out| {
            out.actions
                .iter()
                .any(|a| a.action.steps.iter().any(|s| &s.precondition == info))
        })
    }
    /// Coordinator adoption keeps both private session lifetimes aligned with
    /// the single validated recommendation, without merging either private tree.
    pub fn adopt(
        &mut self,
        input: Input,
        mut outcome: Outcome,
        actions: Vec<SupportedAction>,
        clock: &impl Clock,
    ) -> Result<Outcome> {
        check(
            outcome.identity == self.identity
                && outcome.request_id == self.request
                && outcome.root_epoch == self.epoch,
            "stale coordinator adoption",
        )?;
        let (game, info) = self.preview(input)?;
        check(
            self.previous.as_ref() == Some(&info),
            "coordinator root changed",
        )?;
        outcome.actions = actions;
        self.deliver(&game, outcome, clock)
    }
    fn accept(
        &mut self,
        identity: &Identity,
        request: u64,
        mut input: Input,
        request_deadline_ns: u64,
        clock: &impl Clock,
    ) -> Result<(Brass, Info<Key, String>)> {
        check(
            !self.retired && !clock.cancelled(),
            "retired/cancelled session",
        )?;
        check(
            identity == &self.identity && request > self.request,
            "stale request identity",
        )?;
        let c = &input.view.context;
        check(
            TurnKey::from_context(c, u8::from(input.view.seat) as usize) == self.turn,
            "different turn requires a new admitted session",
        )?;
        check(c.revision >= self.revision, "stale revision")?;
        check(
            same_context(&input.context, &self.context)
                && input
                    .context
                    .remaining
                    .iter()
                    .zip(self.context.remaining)
                    .all(|(a, b)| *a <= b),
            "turn context/utility changed",
        )?;
        input = self.canonical_input(input)?;
        let expected = input.recall.clone();
        let revision = input.view.context.revision;
        let remaining = input.context.remaining;
        let mut game = Brass::new(input, false)?;
        let info = game.root()?;
        check(
            self.previous.as_ref() != Some(&info),
            "root already attempted; no tournament restart",
        )?;
        if let Some(previous) = &self.previous {
            let old = previous
                .key
                .information
                .observation
                .view
                .as_ref()
                .expect("admitted view");
            check(
                revision > old.context.revision,
                "changed root at unchanged revision",
            )?;
        }
        // Invalid/stale input cannot alter the admitted session's clock or recall.
        self.budget.narrow(0, request_deadline_ns);
        self.recall = expected;
        self.request = request;
        self.revision = revision;
        self.context.remaining = remaining;
        Ok((game, info))
    }
    fn empty(&self, phase: Phase, clock: &impl Clock) -> Outcome {
        Outcome {
            version: IDENTITY,
            identity: self.identity.clone(),
            request_id: self.request,
            revision: self.revision,
            root_epoch: self.epoch,
            supporting_root_epoch: self.epoch,
            turn: self.turn.clone(),
            budget: self.budget,
            context: self.context.clone(),
            start_phase: phase,
            status: "completed".into(),
            provenance: String::new(),
            actions: vec![],
            action_ids: vec![],
            quota: None,
            gumbels: vec![],
            effective_k: 0,
            snapshot: None,
            completed: 0,
            attempted: 0,
            partial_completed: 0,
            retired_pending: 0,
            plan_events: 0,
            cache_hits: 0,
            last_completion_ns: None,
            work_stopped_ns: clock.now_ns(),
            response_ready_ns: clock.now_ns(),
            unused_work_ns: 0,
            eligible_plans: vec![],
            schedule: Default::default(),
        }
    }
    fn deliver(
        &mut self,
        game: &Brass,
        mut result: Outcome,
        clock: &impl Clock,
    ) -> Result<Outcome> {
        self.expected_after = None;
        if clock.cancelled() {
            self.retire();
            result.actions.clear();
            result.status = "cancelled".into();
            result.provenance = "none".into();
        } else if !self.budget.delivery_open(clock) {
            result.actions.clear();
            result.status = "completion_deadline".into();
            result.provenance = "none".into();
        } else if !result.actions.is_empty() {
            match plans::validate(
                game,
                &result
                    .actions
                    .iter()
                    .map(|a| a.action.clone())
                    .collect::<Vec<_>>(),
                self.budget,
                clock,
            ) {
                Ok(next) => self.expected_after = next,
                Err(e) => {
                    if self.budget.delivery_open(clock) {
                        return Err(e);
                    }
                    result.actions.clear();
                    result.status = if clock.cancelled() {
                        "cancelled"
                    } else {
                        "completion_deadline"
                    }
                    .into();
                    result.provenance = "none".into();
                }
            }
        }
        result.action_ids = result.actions.iter().map(|a| a.action.ids()).collect();
        result.response_ready_ns = clock.now_ns();
        if clock.cancelled() || result.response_ready_ns >= self.budget.deadline_ns {
            self.expected_after = None;
            result.actions.clear();
            result.action_ids.clear();
            result.status = if clock.cancelled() {
                "cancelled"
            } else {
                "completion_deadline"
            }
            .into();
            result.provenance = "none".into();
        }
        result.unused_work_ns = self.budget.remaining_work(result.work_stopped_ns);
        self.continuation = (!result.actions.is_empty()).then(|| result.clone());
        Ok(result)
    }

    /// One scalar decision-root tournament. The caller retains this Session
    /// across sub-choices, action two and replanning; there is no fresh clock.
    #[allow(clippy::too_many_arguments)]
    pub fn decide<E: LeafEvaluator<Key, String>>(
        &mut self,
        identity: &Identity,
        request: u64,
        input: Input,
        remaining_ns: u64,
        mode: Mode,
        clock: &impl Clock,
        evaluator: &mut E,
        tape: Option<Vec<f64>>,
    ) -> Result<Outcome> {
        self.decide_impl(
            identity,
            request,
            input,
            clock.now_ns().saturating_add(remaining_ns),
            mode,
            clock,
            evaluator,
            tape,
            None,
        )
    }
    #[allow(clippy::too_many_arguments)]
    /// `request_deadline_ns` is absolute on the admitted turn clock, including
    /// dispatch, root preparation and input validation. It can only narrow time.
    pub fn decide_cooperative<E: LeafEvaluator<Key, String>>(
        &mut self,
        identity: &Identity,
        request: u64,
        input: Input,
        request_deadline_ns: u64,
        mode: Mode,
        clock: &impl Clock,
        evaluator: &mut E,
        tape: Option<Vec<f64>>,
        shared: &cooperative::SharedRoot,
        policy: crate::evaluation::BatchPolicy,
    ) -> Result<Outcome> {
        cooperative::validate_policy(policy)?;
        self.decide_impl(
            identity,
            request,
            input,
            request_deadline_ns,
            mode,
            clock,
            evaluator,
            tape,
            Some((shared, policy)),
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn decide_impl<E: LeafEvaluator<Key, String>>(
        &mut self,
        identity: &Identity,
        request: u64,
        input: Input,
        request_deadline_ns: u64,
        mode: Mode,
        clock: &impl Clock,
        evaluator: &mut E,
        tape: Option<Vec<f64>>,
        cooperative: Option<(&cooperative::SharedRoot, crate::evaluation::BatchPolicy)>,
    ) -> Result<Outcome> {
        let (mut game, info) = self.accept(identity, request, input, request_deadline_ns, clock)?;
        self.epoch += 1;
        self.previous = Some(info.clone());
        if let Some((shared, _)) = cooperative {
            check(shared.info == info, "shared root identity mismatch")?;
        }
        let phase = plans::phase(&info)?;
        let mut out = self.empty(phase, clock);
        if let Some(previous) = self.continuation.take() {
            for (i, action) in previous.actions.iter().enumerate() {
                if let Some(j) = action
                    .action
                    .steps
                    .iter()
                    .position(|s| s.precondition == info)
                {
                    out.actions = previous.actions[i..].to_vec();
                    out.actions[0].action.steps.drain(..j);
                    out.provenance = format!("continuation/{}", previous.provenance);
                    out.snapshot = previous.snapshot;
                    out.supporting_root_epoch = previous.supporting_root_epoch;
                    return self.deliver(&game, out, clock);
                }
            }
            // A stale continuation is discarded; replanning keeps the same budget.
        }
        if !self.budget.delivery_open(clock) {
            return self.deliver(&game, out, clock);
        }
        let bypass = info.actions.len() == 1 || phase == Phase::Debt;
        let mut draws = Draws::new(
            self.settings.search.seed.wrapping_add(self.epoch - 1),
            tape,
            false,
        );
        let mut cached = Cached {
            cache: &mut self.cache,
            limit: self.settings.cache_entries,
            evaluator,
            clock,
            budget: self.budget,
        };
        let hits = cached.cache.hits;
        let root_stats = if info.actions.len() == 1 || !self.budget.work_open(clock) {
            None
        } else if let Some((shared, _)) = cooperative {
            shared.stats.clone()
        } else {
            match cached.evaluate(&info) {
                Ok(s) => Some(s),
                Err(e) => {
                    if self.budget.work_open(clock) {
                        return Err(e);
                    }
                    None
                }
            }
        };
        let mut first = info.actions[0].clone();
        if bypass {
            if let Some(stats) = &root_stats {
                first = info.actions[math::argmax(&stats.logits)?].clone();
            }
            out.provenance = if phase == Phase::Debt {
                "debt_policy"
            } else {
                "forced_completion"
            }
            .into();
        } else if let Some(stats) = root_stats {
            let costs_index = self
                .costs
                .iter()
                .position(|b| b.phase == phase && b.context == info.key.context)
                .unwrap_or_else(|| {
                    self.costs.push(CostBucket {
                        phase,
                        context: info.key.context.clone(),
                        costs: Costs::default(),
                    });
                    self.costs.len() - 1
                });
            let costs = &mut self.costs[costs_index].costs;
            let available = self.budget.remaining_work(clock.now_ns());
            let quota = match mode {
                Mode::Count => FrozenQuota::count(self.settings.search.simulations, available)?,
                Mode::Timed { startup } => {
                    let estimate = if let Some(nanoseconds) = costs.estimate() {
                        Some(Estimate {
                            nanoseconds,
                            source: "last-eight-completed/lower-quartile".into(),
                        })
                    } else if let Some(startup) = startup {
                        check(
                            startup.model_id == info.key.context.model_id
                                && startup.export_sha256 == self.identity.export_sha256
                                && startup.reward_id == info.key.context.reward_id
                                && startup.phase == phase,
                            "startup estimate identity mismatch",
                        )?;
                        Some(startup.estimate)
                    } else {
                        None
                    };
                    FrozenQuota::timed(available, estimate)?
                }
            };
            let uniforms = (0..info.actions.len())
                .map(|_| draws.uniform())
                .collect::<Result<Vec<_>>>()?;
            let uniforms = cooperative.map_or(uniforms, |(shared, _)| shared.uniforms.clone());
            let mut rounds = Rounds::new(
                &stats.logits,
                &uniforms,
                self.settings.search.k,
                quota.simulations,
            )?;
            let mut tree = Tree::new(
                self.settings.search.max_nodes,
                self.settings.search.max_edges,
            )?;
            let root = tree.insert(info.clone(), stats.clone())?;
            let mut ledger = plans::Ledger::default();
            out.quota = Some(quota.clone());
            if let Some((_, policy)) = cooperative {
                let _scope = crate::work::Scope::narrow_remaining(
                    self.budget.remaining_work(clock.now_ns()),
                );
                cooperative::run(
                    &mut game,
                    &mut tree,
                    root,
                    &self.settings.search,
                    &mut draws,
                    &mut rounds,
                    &mut ledger,
                    &mut cached,
                    costs,
                    self.budget,
                    clock,
                    policy,
                    &mut out,
                )?;
            } else {
                while rounds.tournament.completed < quota.simulations {
                    if !self.budget.work_open(clock)
                        || !costs.admit(self.budget.remaining_work(clock.now_ns()))
                    {
                        out.status = "work_deadline".into();
                        break;
                    }
                    let began = clock.now_ns();
                    let edge = rounds.tournament.next_edge()?.expect("remaining quota");
                    out.attempted += 1;
                    let work = (|| {
                        let pending = trajectory::traverse(
                            &mut game,
                            &tree,
                            root,
                            edge,
                            &self.settings.search,
                            &mut draws,
                        )?;
                        check(self.budget.work_open(clock), "late traversal")?;
                        let evaluation = pending
                            .leaf_info()
                            .map(|i| cached.evaluate(i))
                            .transpose()?;
                        let value = pending.root_return(evaluation.as_ref())?;
                        let actions = plans::completed_prefixes(&tree, pending.path())?;
                        check(self.budget.work_open(clock), "late backup")?;
                        pending.complete(&mut tree, evaluation, false)?;
                        ledger.record(&actions, rounds.tournament.completed + 1, value)?;
                        rounds.complete(
                            edge,
                            &tree.get(root)?.stats,
                            info.actor,
                            self.settings.search.parameters(),
                        )?;
                        Ok(())
                    })();
                    if let Err(e) = work {
                        out.retired_pending += u64::from(rounds.tournament.abandon());
                        if self.budget.work_open(clock) {
                            return Err(e);
                        }
                        out.status = "work_deadline".into();
                        break;
                    }
                    out.last_completion_ns = Some(clock.now_ns());
                    costs.observe(clock.now_ns().saturating_sub(began));
                }
            }
            let ranking = rounds.ranking(&stats, info.actor, self.settings.search.parameters())?;
            first = info.actions[ranking[0]].clone();
            out.completed = rounds.tournament.completed;
            out.effective_k = rounds.tournament.effective_k;
            out.partial_completed =
                out.completed - rounds.eligible.as_ref().map_or(0, |s| s.support_watermark);
            out.plan_events = ledger.events();
            out.gumbels = rounds.tournament.gumbels;
            out.snapshot = rounds.eligible;
            if let Some(s) = &out.snapshot {
                for candidate in ranking {
                    let actions = ledger.select(
                        &info.actions[candidate],
                        s.support_watermark,
                        self.settings.minimum_plan_support,
                        info.actor,
                    );
                    if !actions.is_empty() {
                        if out.actions.is_empty() {
                            out.actions = actions.clone();
                        }
                        if cooperative.is_some() {
                            out.eligible_plans.push(actions);
                        } else {
                            break;
                        }
                    }
                }
            }
            out.provenance = if out.actions.is_empty() {
                if out.snapshot.is_some() {
                    "bounded_policy_completion"
                } else {
                    "gumbel_prior_fallback"
                }
            } else {
                "search_supported"
            }
            .into();
        } else {
            out.provenance = "bounded_legal_fallback".into();
        }
        out.cache_hits = cached.cache.hits - hits;
        out.work_stopped_ns = clock.now_ns();
        if out.actions.is_empty() && self.budget.delivery_open(clock) {
            match plans::complete(&game, &first, self.budget, clock) {
                Ok(action) => out.actions.push(SupportedAction {
                    action,
                    support: 0,
                    returns: [0.; 4],
                }),
                Err(e) => {
                    if self.budget.delivery_open(clock) {
                        return Err(e);
                    }
                }
            }
        }
        self.deliver(&game, out, clock)
    }
}
