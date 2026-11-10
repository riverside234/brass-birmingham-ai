//! Teacher-only root coverage. Priors, Gumbels, quotas and backup equations stay intact.
use super::super::{check, Result};
use crate::state::{
    dto::{Choice, Industry, Kind},
    observation::ViewContext,
};

pub struct Coverage {
    pub context: ViewContext,
    pub limit: usize,
}

fn cotton(choice: &Choice) -> bool {
    choice.kind == Kind::Build && choice.industry == Some(Industry::CottonMill)
}

/// Reserve up to two distinct building locations within the existing K slots.
/// Two linear scans select the highest-prior representatives; no feature work,
/// extra RNG draws, extra trajectories or sorting of the full legal set.
pub fn candidates(
    logits: &[f64],
    gumbels: &[f64],
    choices: &[Choice],
    selected: &[usize],
    limit: usize,
) -> Result<Vec<usize>> {
    check(
        logits.len() == choices.len()
            && logits.len() == gumbels.len()
            && selected.iter().all(|i| *i < choices.len())
            && (1..=2).contains(&limit),
        "cotton coverage shape",
    )?;
    if selected.len() < 2 {
        return Ok(selected.to_vec());
    }
    let mut required = Vec::with_capacity(3);
    // Retain an ordinary comparator, even if the original top K was all cotton.
    let ordinary = (0..choices.len())
        .filter(|i| !cotton(&choices[*i]))
        .max_by(|a, b| {
            (logits[*a] + gumbels[*a])
                .total_cmp(&(logits[*b] + gumbels[*b]))
                .then(b.cmp(a))
        });
    for _ in 0..limit.min(selected.len() - usize::from(ordinary.is_some())) {
        let best = (0..choices.len())
            .filter(|i| {
                cotton(&choices[*i])
                    && !required
                        .iter()
                        .any(|j: &usize| choices[*j].target == choices[*i].target)
            })
            .max_by(|a, b| logits[*a].total_cmp(&logits[*b]).then(b.cmp(a)));
        if let Some(i) = best {
            required.push(i);
        } else {
            break;
        }
    }
    if required.is_empty() {
        return Ok(selected.to_vec());
    }
    if let Some(i) = ordinary {
        required.push(i);
    }
    let mut result = selected.to_vec();
    for i in &required {
        if result.contains(i) {
            continue;
        }
        let slot = (0..result.len())
            .filter(|s| !required.contains(&result[*s]))
            .min_by(|a, b| {
                let (a, b) = (result[*a], result[*b]);
                (logits[a] + gumbels[a])
                    .total_cmp(&(logits[b] + gumbels[b]))
                    .then(b.cmp(&a))
            })
            .ok_or_else(|| super::super::invalid("cotton coverage has no replacement slot"))?;
        result[slot] = *i;
    }
    result.sort_unstable();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn choice(id: &str, target: &str, cotton_: bool) -> Choice {
        Choice {
            id: id.into(),
            kind: Kind::Build,
            card: None,
            target: Some(target.into()),
            industry: Some(if cotton_ {
                Industry::CottonMill
            } else {
                Industry::CoalMine
            }),
            merchant: None,
            cards: None,
        }
    }
    #[test]
    fn covers_distinct_locations_without_changing_k_or_priors() {
        let c = vec![
            choice("a", "A", false),
            choice("b", "B", false),
            choice("c", "C", true),
            choice("d", "C", true),
            choice("e", "E", true),
        ];
        let l = vec![5., 4., -8., -9., -10.];
        let g = vec![0.; 5];
        assert_eq!(
            candidates(&l, &g, &c, &[0, 1, 2, 3], 2).unwrap(),
            vec![0, 1, 2, 4]
        );
        assert_eq!(candidates(&l, &g, &c, &[0, 1], 2).unwrap(), vec![0, 2]);
        assert_eq!(candidates(&l, &g, &c, &[0], 2).unwrap(), vec![0]);
        assert_eq!(l, vec![5., 4., -8., -9., -10.]);
    }
    #[test]
    fn missing_cotton_is_a_noop_and_ordinary_comparison_is_retained() {
        let c = vec![
            choice("a", "A", false),
            choice("b", "B", true),
            choice("c", "C", true),
        ];
        assert_eq!(
            candidates(&[0., 9., 8.], &[0.; 3], &c, &[1, 2], 1).unwrap(),
            vec![0, 1]
        );
        assert_eq!(candidates(&[0.], &[0.], &c[..1], &[0], 1).unwrap(), vec![0]);
    }
}
