//! Fixed-count, post-backup sequential halving. No timed scheduling here.
use super::{
    check, invalid,
    math::{gumbel, Parameters, Statistics},
    Result,
};
use serde::Serialize;

pub fn effective_k(requested: usize, legal: usize, budget: u64) -> Result<usize> {
    check(requested > 0 && legal > 0, "K/legal count must be positive")?;
    let mut k = 1usize << (usize::BITS - 1 - requested.min(legal).leading_zeros());
    while k > 1 && budget < (k as u64).saturating_mul(k.ilog2() as u64) {
        k /= 2;
    }
    Ok(k)
}
#[derive(Debug, Clone, Serialize)]
pub struct Round {
    pub candidates: Vec<usize>,
    pub quota_per_action: u64,
    pub scores: Vec<f64>,
    pub retained: Vec<usize>,
}
#[derive(Clone)]
pub struct Tournament {
    pub gumbels: Vec<f64>,
    pub effective_k: usize,
    pub rounds: Vec<Round>,
    pub completed: u64,
    pub remainder_simulations: u64,
    survivors: Vec<usize>,
    budget: u64,
    round_done: u64,
    issued: std::collections::VecDeque<usize>,
}
fn rank(candidates: &[usize], scores: &[f64], count: usize) -> Result<Vec<usize>> {
    check(
        scores.iter().all(|s| s.is_finite()),
        "nonfinite root scores",
    )?;
    let mut result = candidates.to_vec();
    result.sort_by(|a, b| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .expect("finite")
            .then(a.cmp(b))
    });
    result.truncate(count);
    result.sort_unstable();
    Ok(result)
}
impl Tournament {
    pub fn survivors(&self) -> &[usize] {
        &self.survivors
    }
    pub fn partial_round_completed(&self) -> u64 {
        self.round_done
    }
    /// Retire the scalar launch slot without publishing a visit or elimination.
    pub fn abandon(&mut self) -> bool {
        self.issued.pop_front().is_some()
    }
    pub fn new(logits: &[f64], uniforms: &[f64], requested: usize, budget: u64) -> Result<Self> {
        check(
            logits.len() == uniforms.len() && logits.iter().all(|v| v.is_finite()),
            "root draws/logits mismatch",
        )?;
        let effective_k = effective_k(requested, logits.len(), budget)?;
        let gumbels: Vec<_> = uniforms.iter().map(|u| gumbel(*u)).collect::<Result<_>>()?;
        let scores: Vec<_> = gumbels.iter().zip(logits).map(|(g, l)| g + l).collect();
        let survivors = rank(&(0..logits.len()).collect::<Vec<_>>(), &scores, effective_k)?;
        Ok(Self {
            gumbels,
            effective_k,
            rounds: vec![],
            completed: 0,
            remainder_simulations: 0,
            survivors,
            budget,
            round_done: 0,
            issued: std::collections::VecDeque::new(),
        })
    }
    pub fn next_edge(&mut self) -> Result<Option<usize>> {
        check(
            self.issued.is_empty(),
            "trajectory must complete before another launch",
        )?;
        self.reserve()
    }
    pub fn pending(&self) -> usize {
        self.issued.len()
    }
    pub fn retire_all(&mut self) -> usize {
        let n = self.issued.len();
        self.issued.clear();
        n
    }
    /// FIFO reservations, round-robin launches, barrier before elimination.
    pub fn reserve(&mut self) -> Result<Option<usize>> {
        let issued = self.issued.len() as u64;
        let remaining = if self.survivors.len() > 1 {
            let quota =
                self.budget / (self.survivors.len() as u64 * self.effective_k.ilog2() as u64);
            quota * self.survivors.len() as u64 - self.round_done
        } else {
            self.budget - self.completed
        };
        if issued >= remaining {
            return Ok(None);
        }
        let edge = self.survivors[(self.round_done + issued) as usize % self.survivors.len()];
        self.issued.push_back(edge);
        Ok(Some(edge))
    }
    /// Call only after a complete backup. A failed trajectory aborts the search.
    pub fn completed_backup(
        &mut self,
        edge: usize,
        stats: &Statistics,
        actor: usize,
        p: Parameters,
    ) -> Result<()> {
        check(
            self.issued.front() == Some(&edge) && self.completed < self.budget,
            "unissued/duplicate backup",
        )?;
        let quota = if self.survivors.len() > 1 {
            self.budget / (self.survivors.len() as u64 * self.effective_k.ilog2() as u64)
        } else {
            0
        };
        // Compute fallible scores before changing the tournament.
        let round = if self.survivors.len() > 1
            && self.round_done + 1 == quota * self.survivors.len() as u64
        {
            let scores: Vec<_> = self
                .gumbels
                .iter()
                .zip(stats.scores(actor, p)?)
                .map(|(g, s)| g + s)
                .collect();
            let retained = rank(&self.survivors, &scores, self.survivors.len() / 2)?;
            Some(Round {
                candidates: self.survivors.clone(),
                quota_per_action: quota,
                scores: self.survivors.iter().map(|i| scores[*i]).collect(),
                retained,
            })
        } else {
            None
        };
        self.issued.pop_front();
        self.completed += 1;
        self.round_done += 1;
        if self.survivors.len() == 1 {
            self.remainder_simulations += 1;
        }
        if let Some(round) = round {
            self.survivors = round.retained.clone();
            self.rounds.push(round);
            self.round_done = 0;
        }
        Ok(())
    }
    pub fn winner(&self) -> Result<usize> {
        check(
            self.completed == self.budget && self.issued.is_empty() && self.survivors.len() == 1,
            "incomplete tournament",
        )?;
        self.survivors
            .first()
            .copied()
            .ok_or_else(|| invalid("empty tournament"))
    }
}
