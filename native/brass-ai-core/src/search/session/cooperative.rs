//! Bounded waves on one compute thread. Publication is FIFO in launch order.
//! Pending paths never influence non-root statistics; duplicate expansions share
//! a single first-new-leaf reply and retain separate completed path contributions.
use super::*;
use crate::evaluation::BatchPolicy;
use crate::work::{checkpoint, Stage};
use std::time::Instant;

#[derive(Clone)]
pub struct SharedRoot {
    pub info: Info<Key, String>,
    pub stats: Option<Statistics>,
    pub uniforms: Vec<f64>,
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct Metrics {
    pub reused_selections: u64,
    pub waves: u64,
    pub evaluator_batches: u64,
    pub unique_leaves: u64,
    pub expansion_followers: u64,
    pub max_pending: usize,
    pub retired: usize,
    pub queue_ns: u64,
    pub max_queue_ns: u64,
    pub time_to_leaf_ns: u64,
    pub sampling_ns: u64,
    pub information_ns: u64,
    pub lookup_ns: u64,
    pub selection_ns: u64,
    pub application_ns: u64,
    pub evaluation_ns: u64,
    pub publication_ns: u64,
    pub nodes: usize,
    pub edges: usize,
}
pub fn validate_policy(p: BatchPolicy) -> Result<()> {
    check(
        [1, 2, 4].contains(&p.cap)
            && (p.cap..=4).contains(&p.pending_limit)
            && (8192..=32768).contains(&p.max_candidate_cells)
            && [0, 500, 1000, 2000].contains(&p.flush_us),
        "cooperative policy bounds",
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run<E: LeafEvaluator<Key, String>>(
    game: &mut Brass,
    tree: &mut Tree<Key, String>,
    root: super::super::tree::Handle,
    config: &Config,
    draws: &mut Draws,
    rounds: &mut Rounds,
    ledger: &mut plans::Ledger,
    evaluator: &mut E,
    costs: &mut Costs,
    budget: Budget,
    clock: &impl Clock,
    policy: BatchPolicy,
    out: &mut Outcome,
    observer: &mut Option<&mut dyn focus::Observer>,
    minimum_support: u64,
) -> Result<()> {
    validate_policy(policy)?;
    let actor = tree.get(root)?.info.actor;
    let n = out.quota.as_ref().expect("frozen quota").simulations;
    let mut published_round = 0;
    while rounds.tournament.completed < n {
        let mut pending = Vec::with_capacity(policy.pending_limit);
        let wave_started = clock.now_ns();
        let wave = (|| {
            while pending.len() < policy.pending_limit {
                check(budget.work_open(clock), "work deadline")?;
                if !costs.admit(budget.remaining_work(clock.now_ns())) {
                    break;
                }
                let Some(edge) = rounds.tournament.reserve()? else {
                    break;
                };
                out.attempted += 1;
                let began = clock.now_ns();
                let path = trajectory::traverse(game, tree, root, edge, config, draws)?;
                let work = path.work();
                out.schedule.reused_selections += work.reused_selections;
                out.schedule.sampling_ns += work.sampling_ns;
                out.schedule.information_ns += work.information_ns;
                out.schedule.lookup_ns += work.lookup_ns;
                out.schedule.selection_ns += work.selection_ns;
                out.schedule.application_ns += work.application_ns;
                let queued = clock.now_ns();
                out.schedule.time_to_leaf_ns += queued.saturating_sub(began);
                pending.push((edge, path, began, queued));
                out.schedule.max_pending = out.schedule.max_pending.max(pending.len());
                if policy.flush_us == 0
                    || clock.now_ns().saturating_sub(wave_started) >= policy.flush_us * 1000
                {
                    break;
                }
            }
            if pending.is_empty() {
                return Ok(false);
            }
            check(budget.work_open(clock), "late traversal")?;
            out.schedule.waves += 1;
            let mut unique: Vec<&Info<Key, String>> = Vec::with_capacity(pending.len());
            let slots: Vec<_> = pending
                .iter()
                .map(|(_, p, _, _)| {
                    p.leaf_info().map(|info| {
                        if let Some(i) = unique.iter().position(|old| **old == *info) {
                            out.schedule.expansion_followers += 1;
                            i
                        } else {
                            unique.push(info);
                            unique.len() - 1
                        }
                    })
                })
                .collect();
            let mut replies = Vec::with_capacity(unique.len());
            let mut offset = 0;
            while offset < unique.len() {
                check(budget.work_open(clock), "expired queued leaf")?;
                let mut end = offset;
                let mut width = 0;
                while end < unique.len() && end - offset < policy.cap {
                    let next = width.max(unique[end].actions.len());
                    if next * (end - offset + 1) > policy.max_candidate_cells {
                        break;
                    }
                    width = next;
                    end += 1;
                }
                check(end > offset, "leaf exceeds batch cell bound")?;
                let now = clock.now_ns();
                for ((_, _, _, queued), slot) in pending.iter().zip(&slots) {
                    if slot.is_some_and(|i| i >= offset && i < end) {
                        let waited = now.saturating_sub(*queued);
                        out.schedule.queue_ns += waited;
                        out.schedule.max_queue_ns = out.schedule.max_queue_ns.max(waited);
                    }
                }
                let began = Instant::now();
                let values = evaluator.evaluate_batch(&unique[offset..end])?;
                out.schedule.evaluation_ns += super::super::nanos(began);
                check(budget.work_open(clock), "late batch reply")?;
                check(values.len() == end - offset, "batch reply width")?;
                replies.extend(values);
                out.schedule.evaluator_batches += 1;
                offset = end;
            }
            out.schedule.unique_leaves += unique.len() as u64;
            drop(unique);
            for ((edge, path, began, _), slot) in pending.drain(..).zip(slots) {
                let evaluation = slot.map(|i| replies[i].clone());
                let value = path.root_return(evaluation.as_ref())?;
                let actions = plans::completed_prefixes(tree, path.path())?;
                let publication = Instant::now();
                // prepare+commit journals only touched edges. Cancellation even
                // after the last edge restores all counts and the new expansion.
                let mut completed_at = 0;
                path.complete_checked(tree, evaluation, &mut || {
                    let now = clock.now_ns();
                    check(
                        !clock.cancelled() && now < budget.work_deadline_ns,
                        "late backup publication",
                    )?;
                    completed_at = now;
                    Ok(())
                })?;
                ledger.record(&actions, rounds.tournament.completed + 1, value)?;
                rounds.complete(edge, &tree.get(root)?.stats, actor, config.parameters())?;
                if rounds
                    .eligible
                    .as_ref()
                    .is_some_and(|s| s.support_watermark == rounds.tournament.completed)
                {
                    tree.checkpoint_round(rounds.tournament.completed);
                }
                out.schedule.publication_ns += super::super::nanos(publication);
                out.last_completion_ns = Some(completed_at);
                costs.observe(completed_at.saturating_sub(began));
            }
            Ok(true)
        })();
        match wave {
            Ok(true) => {}
            Ok(false) => {
                out.status = "launch_stop".into();
                break;
            }
            Err(error) => {
                let retired = rounds.tournament.retire_all();
                out.retired_pending += retired as u64;
                out.schedule.retired += retired;
                if budget.work_open(clock) {
                    return Err(error);
                }
                out.status = "work_deadline".into();
                break;
            }
        }
        checkpoint(Stage::Traversal).or_else(|error| {
            if budget.work_open(clock) {
                Err(error)
            } else {
                Ok(())
            }
        })?;
        if let Some(observer) = observer.as_deref_mut() {
            let retire = observer.retirement_requested();
            if let Some(snapshot) = &rounds.eligible {
                // Only full halving rounds, with every reservation retired. A
                // repeated single-candidate/remainder update is not a new round.
                if (snapshot.completed_rounds > published_round || retire)
                    && rounds.tournament.pending() == 0
                {
                    published_round = snapshot.completed_rounds;
                    let began = clock.now_ns();
                    let mut progress = out.clone();
                    progress.snapshot = Some(snapshot.clone());
                    progress.completed = rounds.tournament.completed;
                    progress.effective_k = rounds.tournament.effective_k;
                    progress.gumbels = rounds.tournament.gumbels.clone();
                    progress.eligible_plans.clear();
                    let info = &tree.get(root)?.info;
                    for candidate in
                        rounds.ranking(&tree.get(root)?.stats, actor, config.parameters())?
                    {
                        let plan = ledger.select(
                            &info.actions[candidate],
                            snapshot.support_watermark,
                            minimum_support,
                            actor,
                        );
                        if !plan.is_empty() {
                            progress.eligible_plans.push(plan);
                        }
                    }
                    let startup = progress
                        .quota
                        .as_ref()
                        .and_then(|q| q.estimate.as_ref())
                        .map_or(0, |e| e.nanoseconds);
                    let stop = observer.checkpoint(
                        game,
                        focus::Progress {
                            outcome: progress,
                            cost_ns: costs.conservative(startup),
                            at_ns: clock.now_ns(),
                            extraction_ns: clock.now_ns().saturating_sub(began),
                        },
                    )?;
                    if stop {
                        out.status = "policy_round_stop".into();
                        break;
                    }
                }
            }
        }
    }
    check(
        rounds.tournament.pending() == 0,
        "leaked round reservations",
    )?;
    out.schedule.nodes = tree.len();
    out.schedule.edges = tree.edges();
    Ok(())
}
