//! Versioned two-session ensemble. Pool root snapshots, never private nodes.
use super::super::math::Parameters;
use super::cooperative::SharedRoot;
use super::*;

pub const IDENTITY: &str = "brass-gumbel-is-mcts/coordinator-v1";
#[derive(Debug, Clone, Serialize)]
pub struct Pool {
    pub eligible: Vec<usize>,
    pub stats: Statistics,
    pub ranking: Vec<usize>,
    pub scores: Vec<f64>,
}
pub fn pool(
    root: &Statistics,
    gumbels: &[f64],
    actor: usize,
    parameters: Parameters,
    snapshots: [Option<&Snapshot>; 2],
) -> Result<Pool> {
    parameters.validate()?;
    check(actor < 4, "invalid pooled actor")?;
    check(
        gumbels.len() == root.logits.len() && gumbels.iter().all(|x| x.is_finite()),
        "common perturbation",
    )?;
    let mut stats = Statistics::new(root.logits.clone(), root.value)?;
    let mut eligible = Vec::new();
    let mut survivors = std::collections::BTreeSet::new();
    for (lane, snapshot) in snapshots.into_iter().enumerate() {
        let Some(s) = snapshot else { continue };
        let total = s
            .stats
            .count
            .iter()
            .try_fold(0u64, |sum, n| sum.checked_add(*n))
            .ok_or_else(|| invalid("snapshot visit overflow"))?;
        check(
            s.stats.logits == root.logits
                && s.stats.value == root.value
                && s.stats.count.len() == stats.count.len()
                && s.stats.returns.len() == stats.returns.len()
                && total == s.support_watermark
                && s.support_watermark > 0
                && s.stats.returns.iter().flatten().all(|v| v.is_finite())
                && !s.survivors.is_empty(),
            "incompatible snapshot",
        )?;
        for i in &s.survivors {
            check(*i < stats.logits.len(), "survivor bounds")?;
            survivors.insert(*i);
        }
        eligible.push(lane);
        for i in 0..stats.count.len() {
            stats.count[i] = stats.count[i]
                .checked_add(s.stats.count[i])
                .ok_or_else(|| invalid("pooled visit overflow"))?;
            for seat in 0..4 {
                stats.returns[i][seat] += s.stats.returns[i][seat];
                check(stats.returns[i][seat].is_finite(), "pooled return overflow")?;
            }
        }
    }
    let base = if eligible.is_empty() {
        stats.logits.clone()
    } else {
        stats.scores(actor, parameters)?
    };
    let scores: Vec<_> = base.iter().zip(gumbels).map(|(a, b)| a + b).collect();
    check(
        scores.iter().all(|s| s.is_finite()),
        "pooled score overflow",
    )?;
    let mut ranking: Vec<_> = if eligible.is_empty() {
        (0..stats.count.len()).collect()
    } else {
        survivors.into_iter().collect()
    };
    ranking.sort_by(|a, b| scores[*b].total_cmp(&scores[*a]).then(a.cmp(b)));
    Ok(Pool {
        eligible,
        stats,
        ranking,
        scores,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct Selection {
    pub pool: Option<Pool>,
    pub source_session: Option<usize>,
    pub actions: Option<Vec<SupportedAction>>,
    pub proposal: String,
    pub provenance: String,
}
pub fn select(
    shared: &SharedRoot,
    lanes: [&Outcome; 2],
    parameters: Parameters,
    minimum: u64,
) -> Result<Selection> {
    check(
        lanes[0].identity == lanes[1].identity
            && lanes[0].request_id == lanes[1].request_id
            && lanes[0].revision == lanes[1].revision
            && lanes[0].turn == lanes[1].turn
            && lanes[0].context == lanes[1].context,
        "mixed request snapshots",
    )?;
    let info = &shared.info;
    if lanes
        .iter()
        .any(|o| o.status == "cancelled" || o.status == "completion_deadline")
    {
        return Err(invalid("retired session result"));
    }
    let continuation = lanes
        .iter()
        .all(|o| o.provenance.starts_with("continuation/"));
    if continuation || info.actions.len() == 1 || plans::phase(info)? == Phase::Debt {
        check(
            !lanes[0].actions.is_empty() && lanes[0].action_ids == lanes[1].action_ids,
            "non-search lane disagreement",
        )?;
        return Ok(Selection {
            pool: None,
            source_session: Some(0),
            actions: Some(lanes[0].actions.clone()),
            proposal: lanes[0].action_ids[0][0].clone(),
            provenance: lanes[0].provenance.clone(),
        });
    }
    let Some(root) = &shared.stats else {
        return Ok(Selection {
            pool: None,
            source_session: None,
            actions: None,
            proposal: info.actions[0].clone(),
            provenance: "bounded_legal_fallback".into(),
        });
    };
    let gumbels = shared
        .uniforms
        .iter()
        .map(|u| math::gumbel(*u))
        .collect::<Result<Vec<_>>>()?;
    for lane in &lanes {
        if lane.snapshot.is_some() {
            check(lane.gumbels == gumbels, "different root draws")?;
        }
    }
    let pool = pool(
        root,
        &gumbels,
        info.actor,
        parameters,
        [lanes[0].snapshot.as_ref(), lanes[1].snapshot.as_ref()],
    )?;
    let proposal = info.actions[pool.ranking[0]].clone();
    for candidate in &pool.ranking {
        let mut options = Vec::new();
        for lane in &pool.eligible {
            let watermark = lanes[*lane]
                .snapshot
                .as_ref()
                .expect("eligible")
                .support_watermark;
            for plan in &lanes[*lane].eligible_plans {
                if plan.is_empty() {
                    continue;
                }
                check(
                    plan.len() <= 2
                        && plan
                            .iter()
                            .all(|p| p.support >= minimum && p.support <= watermark),
                    "plan support outside snapshot",
                )?;
                if plan[0].action.steps.first().is_some_and(|s| {
                    s.precondition == *info && s.choice == info.actions[*candidate]
                }) {
                    options.push((*lane, plan));
                }
            }
        }
        options.sort_by(|(a, pa), (b, pb)| {
            pb[0]
                .support
                .cmp(&pa[0].support)
                .then_with(|| {
                    (pb[0].returns[info.actor] / pb[0].support as f64)
                        .total_cmp(&(pa[0].returns[info.actor] / pa[0].support as f64))
                })
                .then_with(|| {
                    pa.iter()
                        .flat_map(|p| p.action.ids())
                        .cmp(pb.iter().flat_map(|p| p.action.ids()))
                })
                .then(a.cmp(b))
        });
        if let Some((lane, plan)) = options.first() {
            return Ok(Selection {
                pool: Some(pool),
                source_session: Some(*lane),
                actions: Some((*plan).clone()),
                proposal,
                provenance: "pooled_search_supported".into(),
            });
        }
    }
    let provenance = if pool.eligible.is_empty() {
        "common_gumbel_prior_fallback"
    } else {
        "pooled_bounded_completion"
    };
    Ok(Selection {
        pool: Some(pool),
        source_session: None,
        actions: None,
        proposal,
        provenance: provenance.into(),
    })
}
