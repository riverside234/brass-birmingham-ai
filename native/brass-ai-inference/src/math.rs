//! Full-legal-set policy arithmetic for Gumbel. Accumulate in f64; never floor priors.
use crate::{require, Result};
pub fn log_softmax(logits: &[f32]) -> Result<(Vec<f64>, Vec<f64>)> {
    require(
        !logits.is_empty() && logits.iter().all(|x| x.is_finite()),
        "nonfinite/empty legal logits",
    )?;
    let maximum = logits
        .iter()
        .map(|x| *x as f64)
        .fold(f64::NEG_INFINITY, f64::max);
    let shifted: Vec<f64> = logits.iter().map(|x| *x as f64 - maximum).collect();
    let normalizer = shifted.iter().map(|x| x.exp()).sum::<f64>().ln();
    let logs: Vec<f64> = shifted.iter().map(|x| x - normalizer).collect();
    let probabilities = logs.iter().map(|x| x.exp()).collect();
    Ok((logs, probabilities))
}
/// Normalize only visited logits, even if every visited prior underflowed in
/// the full softmax. The caller supplies canonical unique edge indices.
pub fn visited_priors(logits: &[f32], visited: &[usize]) -> Result<Vec<f64>> {
    require(
        visited.iter().all(|i| *i < logits.len()) && visited.windows(2).all(|w| w[0] < w[1]),
        "invalid visited indices",
    )?;
    let selected: Vec<f32> = visited.iter().map(|i| logits[*i]).collect();
    Ok(log_softmax(&selected)?.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extremes_ties_singletons_and_visited_underflow() {
        assert_eq!(log_softmax(&[42.]).unwrap(), (vec![0.], vec![1.]));
        assert_eq!(log_softmax(&[0., 0.]).unwrap().1, vec![0.5, 0.5]);
        let (log, p) = log_softmax(&[f32::MAX, -f32::MAX, -f32::MAX]).unwrap();
        assert!(log.iter().all(|x| x.is_finite()));
        assert_eq!(p, vec![1., 0., 0.]);
        assert_eq!(
            visited_priors(&[f32::MAX, -f32::MAX, -f32::MAX], &[1, 2]).unwrap(),
            vec![0.5, 0.5]
        );
        assert!(log_softmax(&[f32::NAN]).is_err());
        assert!(log_softmax(&[]).is_err());
        assert!(visited_priors(&[0., 1.], &[1, 1]).is_err());
    }
}
