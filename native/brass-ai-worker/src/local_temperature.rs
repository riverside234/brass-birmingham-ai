//! Local-game policy temperature. Value heads and legal candidate order stay
//! unchanged. Each owner wraps its own evaluator before any search/cache use.
use brass_ai_core::evaluation::Evaluation;
use brass_ai_inference::{manifest::Manifest, BatchOutput, Control, Evaluator, Input, Result};
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Arc,
};

pub(crate) fn validate(temperature: f64) -> Result<()> {
    // A saved game may still request the former T1.3 opening.
    if [1.0, 1.3, 1.5, 2.0].contains(&temperature) {
        Ok(())
    } else {
        Err(brass_ai_inference::invalid(
            "unsupported local NN temperature",
        ))
    }
}

fn apply(row: &mut Evaluation, temperature: f32) -> Result<()> {
    if temperature == 1.0 {
        return Ok(());
    }
    for logit in &mut row.logits {
        *logit /= temperature;
    }
    (row.log_probabilities, row.probabilities) =
        brass_ai_inference::math::log_softmax(&row.logits)?;
    Ok(())
}

pub(crate) struct Tempered<E> {
    inner: E,
    temperature: Arc<AtomicU32>,
}
impl<E: Evaluator> Tempered<E> {
    pub(crate) fn new(inner: E, temperature: Arc<AtomicU32>) -> Self {
        Self { inner, temperature }
    }
}
impl<E: Evaluator> Evaluator for Tempered<E> {
    fn manifest(&self) -> &Manifest {
        self.inner.manifest()
    }
    fn evaluate(&mut self, inputs: &[Input<'_>], control: &Control) -> Result<BatchOutput> {
        let mut output = self.inner.evaluate(inputs, control)?;
        let temperature = f32::from_bits(self.temperature.load(Ordering::Relaxed));
        if temperature != 1.0 {
            let start = std::time::Instant::now();
            control.check()?;
            for row in &mut output.evaluations {
                apply(row, temperature)?;
            }
            output.stats.postprocess_ns += brass_ai_inference::nanos(start);
            control.check()?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row(logits: Vec<f32>) -> Evaluation {
        let (log_probabilities, probabilities) =
            brass_ai_inference::math::log_softmax(&logits).unwrap();
        Evaluation {
            id: brass_ai_core::evaluation::EvaluationId {
                request_id: "temperature-test".into(),
                generation: 3,
                model_id: "model".into(),
                export_sha256: "export".into(),
                reward_id: "reward".into(),
                profile_id: "profile".into(),
                revision: 7,
                observer: 2,
            },
            choice_ids: (0..logits.len()).map(|i| format!("choice-{i}")).collect(),
            logits,
            log_probabilities,
            probabilities,
            win_share: [0.1, 0.2, 0.3, 0.4],
            placement: [1., 2., 3., 4.],
            vp: [10., 20., 30., 40.],
            utility: [-0.2, 0.1, 0.4, 0.7],
        }
    }
    #[test]
    fn temperature_one_preserves_the_complete_evaluation() {
        let mut r = row(vec![3., 1.5, 0.]);
        let before = serde_json::to_value(&r).unwrap();
        apply(&mut r, 1.).unwrap();
        assert_eq!(serde_json::to_value(&r).unwrap(), before);
    }
    #[test]
    fn temperature_one_point_five_softens_policy_and_preserves_identity_values() {
        let mut r = row(vec![3., 1.5, 0.]);
        let before = serde_json::to_value(&r).unwrap();
        let old_peak = r.probabilities[0];
        apply(&mut r, 1.5).unwrap();
        assert_eq!(r.logits, vec![2., 1., 0.]);
        for (p, expected) in r.probabilities.iter().zip([
            0.6652409557748218,
            0.2447284710547976,
            0.09003057317038046,
        ]) {
            assert!((p - expected).abs() < 1e-14);
        }
        assert!(r.probabilities[0] < old_peak);
        let after = serde_json::to_value(&r).unwrap();
        for key in [
            "id",
            "choice_ids",
            "win_share",
            "placement",
            "vp",
            "utility",
        ] {
            assert_eq!(after[key], before[key]);
        }
    }
    #[test]
    fn temperature_one_point_three_softens_policy_between_one_and_one_point_five() {
        validate(1.3).unwrap();
        let mut r = row(vec![3., 1.5, 0.]);
        let peak_at_one = r.probabilities[0];
        apply(&mut r, 1.3).unwrap();
        assert_eq!(r.logits, vec![3. / 1.3, 1.5 / 1.3, 0.]);
        let mut hotter = row(vec![3., 1.5, 0.]);
        apply(&mut hotter, 1.5).unwrap();
        assert!(hotter.probabilities[0] < r.probabilities[0] && r.probabilities[0] < peak_at_one);
    }
    #[test]
    fn temperature_two_softens_policy_more_than_one_point_five() {
        validate(2.0).unwrap();
        let mut r = row(vec![3., 1.5, 0.]);
        let mut cooler = r.clone();
        apply(&mut cooler, 1.5).unwrap();
        apply(&mut r, 2.0).unwrap();
        assert_eq!(r.logits, vec![1.5, 0.75, 0.]);
        assert!(r.probabilities[0] < cooler.probabilities[0]);
        assert_eq!(r.utility, cooler.utility);
        assert_eq!(r.choice_ids, cooler.choice_ids);
    }
    #[test]
    fn singleton_ties_and_extreme_logits_remain_finite() {
        for logits in [vec![12.], vec![0., 0.], vec![f32::MAX, -f32::MAX]] {
            let mut r = row(logits);
            apply(&mut r, 1.5).unwrap();
            assert!(r.log_probabilities.iter().all(|x| x.is_finite()));
            assert!((r.probabilities.iter().sum::<f64>() - 1.).abs() < 1e-14);
        }
    }
    #[test]
    fn unsupported_or_nonfinite_temperatures_are_rejected() {
        for temperature in [0., -1., 1.1, 2.1, f64::NAN, f64::INFINITY] {
            assert!(validate(temperature).is_err());
        }
        let mut r = row(vec![0., 1.]);
        r.logits[0] = f32::NAN;
        assert!(apply(&mut r, 1.5).is_err());
    }
}
