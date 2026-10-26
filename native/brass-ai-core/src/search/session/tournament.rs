//! Recommendation snapshots contain only balanced, completed work.
use super::super::{
    check,
    halving::Tournament,
    math::{Parameters, Statistics},
    Result,
};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Snapshot {
    pub stats: Statistics,
    pub survivors: Vec<usize>,
    pub completed_rounds: usize,
    pub support_watermark: u64,
    pub remainder_simulations: u64,
    pub reason: &'static str,
}
#[derive(Clone)]
pub struct Rounds {
    pub tournament: Tournament,
    pub eligible: Option<Snapshot>,
    budget: u64,
}
impl Rounds {
    pub fn new(logits: &[f64], uniforms: &[f64], k: usize, budget: u64) -> Result<Self> {
        Ok(Self {
            tournament: Tournament::new(logits, uniforms, k, budget)?,
            eligible: None,
            budget,
        })
    }
    pub fn complete(
        &mut self,
        edge: usize,
        stats: &Statistics,
        actor: usize,
        p: Parameters,
    ) -> Result<()> {
        let before = self.tournament.rounds.len();
        check(
            stats.count.iter().sum::<u64>() == self.tournament.completed + 1,
            "root count/tournament mismatch",
        )?;
        self.tournament.completed_backup(edge, stats, actor, p)?;
        let t = &self.tournament;
        let reason = if t.completed == self.budget {
            Some("quota_complete")
        } else if t.effective_k == 1 {
            Some("single_candidate")
        } else if t.rounds.len() != before {
            Some("balanced_round")
        } else {
            None
        };
        if let Some(reason) = reason {
            self.eligible = Some(Snapshot {
                stats: stats.clone(),
                survivors: t.survivors().to_vec(),
                completed_rounds: t.rounds.len(),
                support_watermark: t.completed,
                remainder_simulations: t.remainder_simulations,
                reason,
            });
        }
        Ok(())
    }
    pub fn ranking(&self, root: &Statistics, actor: usize, p: Parameters) -> Result<Vec<usize>> {
        let (mut candidates, scores) = if let Some(s) = &self.eligible {
            (s.survivors.clone(), s.stats.scores(actor, p)?)
        } else {
            ((0..root.logits.len()).collect(), root.logits.clone())
        };
        let scores: Vec<_> = scores
            .iter()
            .zip(&self.tournament.gumbels)
            .map(|(s, g)| s + g)
            .collect();
        check(
            scores.iter().all(|s| s.is_finite()),
            "nonfinite recommendation",
        )?;
        candidates.sort_by(|a, b| {
            scores[*b]
                .partial_cmp(&scores[*a])
                .expect("finite")
                .then(a.cmp(b))
        });
        Ok(candidates)
    }
}
