//! Paper Appendix B and Equation (3), with f64 completed statistics.
use super::{check, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parameters {
    pub c1: f64,
    pub c2: f64,
}
impl Default for Parameters {
    fn default() -> Self {
        Self { c1: 50., c2: 0.1 }
    }
}
impl Parameters {
    pub fn validate(self) -> Result<()> {
        check(
            self.c1.is_finite() && self.c1 > 0. && self.c2.is_finite() && self.c2 > 0.,
            "invalid Gumbel parameters",
        )
    }
}
pub fn softmax(values: &[f64]) -> Result<Vec<f64>> {
    check(
        !values.is_empty() && values.iter().all(|v| v.is_finite()),
        "nonfinite or empty logits",
    )?;
    let peak = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out: Vec<_> = values.iter().map(|v| (v - peak).exp()).collect();
    let total: f64 = out.iter().sum();
    for x in &mut out {
        *x /= total;
    }
    Ok(out)
}
pub fn gumbel(u: f64) -> Result<f64> {
    check(
        u.is_finite() && u > 0. && u < 1.,
        "uniform must be strictly inside (0, 1)",
    )?;
    Ok(-(-u.ln()).ln())
}
/// The caller supplies canonical action order; exact ties retain the first item.
pub fn argmax(values: &[f64]) -> Result<usize> {
    check(
        !values.is_empty() && values.iter().all(|v| v.is_finite()),
        "invalid selection scores",
    )?;
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    Ok(best)
}
#[derive(Debug, Clone, Serialize)]
pub struct Statistics {
    pub logits: Vec<f64>,
    pub value: [f64; 4],
    pub prior: Vec<f64>,
    pub count: Vec<u64>,
    pub returns: Vec<[f64; 4]>,
}
impl Statistics {
    pub fn new(logits: Vec<f64>, value: [f64; 4]) -> Result<Self> {
        check(value.iter().all(|v| v.is_finite()), "nonfinite leaf value")?;
        let prior = softmax(&logits)?;
        Ok(Self {
            count: vec![0; logits.len()],
            returns: vec![[0.; 4]; logits.len()],
            logits,
            value,
            prior,
        })
    }
    pub fn completed_q(&self, actor: usize) -> Result<(Vec<f64>, f64)> {
        check(actor < 4, "invalid node actor")?;
        let total = self
            .count
            .iter()
            .try_fold(0u64, |a, n| a.checked_add(*n))
            .ok_or_else(|| super::invalid("visit overflow"))?;
        let mut mixed = self.value[actor];
        let visited: Vec<_> = self
            .count
            .iter()
            .enumerate()
            .filter_map(|(i, n)| (*n > 0).then_some(i))
            .collect();
        if total > 0 {
            // Normalizing visited logits directly avoids a full-prior 0/0.
            let weights = softmax(&visited.iter().map(|i| self.logits[*i]).collect::<Vec<_>>())?;
            let mean: f64 = visited
                .iter()
                .zip(weights)
                .map(|(i, w)| w * self.returns[*i][actor] / self.count[*i] as f64)
                .sum();
            mixed = (mixed + total as f64 * mean) / (1. + total as f64);
        }
        let q: Vec<_> = self
            .count
            .iter()
            .enumerate()
            .map(|(i, n)| {
                if *n > 0 {
                    self.returns[i][actor] / *n as f64
                } else {
                    mixed
                }
            })
            .collect();
        check(
            mixed.is_finite() && q.iter().all(|v| v.is_finite()),
            "nonfinite completed Q",
        )?;
        Ok((q, mixed))
    }
    pub fn scores(&self, actor: usize, p: Parameters) -> Result<Vec<f64>> {
        p.validate()?;
        let scale = p.c2
            * (p.c1
                + *self
                    .count
                    .iter()
                    .max()
                    .ok_or_else(|| super::invalid("empty node"))? as f64);
        let scores: Vec<_> = self
            .logits
            .iter()
            .zip(self.completed_q(actor)?.0)
            .map(|(l, q)| l + scale * q)
            .collect();
        check(scores.iter().all(|v| v.is_finite()), "score overflow")?;
        Ok(scores)
    }
    pub fn policy(&self, actor: usize, p: Parameters) -> Result<Vec<f64>> {
        softmax(&self.scores(actor, p)?)
    }
    pub fn select(&self, actor: usize, p: Parameters) -> Result<usize> {
        let denominator = 1. + self.count.iter().map(|n| *n as f64).sum::<f64>();
        argmax(
            &self
                .policy(actor, p)?
                .iter()
                .zip(&self.count)
                .map(|(p, n)| p - *n as f64 / denominator)
                .collect::<Vec<_>>(),
        )
    }
}
