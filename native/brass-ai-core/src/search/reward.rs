//! Exact undiscounted public insolvency ledger and official four-seat terminal utility.
use super::*;
use crate::{
    state::{
        dto::{Choice, Kind, ResultDto},
        observation::Observation,
    },
    types::{Era, Phase},
};
use std::collections::BTreeSet;
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Insolvency {
    pub event_cost: f64,
    pub max_cost: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reward {
    pub version: String,
    pub win: f64,
    pub placement: f64,
    pub vp: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vp_cap: Option<u16>,
    pub insolvency: Insolvency,
}
impl Reward {
    pub fn identity(&self) -> Result<String> {
        use sha2::{Digest, Sha256};
        let value = serde_json::to_value(self).map_err(|e| invalid(e.to_string()))?;
        let bytes = serde_json::to_vec(&value).map_err(|e| invalid(e.to_string()))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
    pub fn validate(&self) -> Result<()> {
        check(
            (self.version == "brass-combined-return/2.0.0" && self.vp_cap == Some(180))
                || (self.version == "brass-combined-return/1.0.0" && self.vp_cap.is_none()),
            "reward contract",
        )?;
        check(
            [self.win, self.placement, self.vp]
                .iter()
                .all(|x| x.is_finite() && *x >= 0.)
                && self.win > 0.,
            "reward coefficients",
        )?;
        let c = &self.insolvency;
        check(
            c.event_cost.is_finite()
                && c.max_cost.is_finite()
                && c.event_cost > 0.
                && c.event_cost <= c.max_cost
                && c.max_cost <= 0.16,
            "insolvency specification",
        )
    }
    pub fn terminal(&self, result: &ResultDto) -> Result<[f64; 4]> {
        self.validate()?;
        let scores = result.seats.each_ref().map(|s| (s.vp, s.income, s.cash));
        let best = *scores.iter().max().expect("four seats");
        let winners = scores.iter().filter(|s| **s == best).count();
        let mut out = [0.; 4];
        for i in 0..4 {
            let share = if scores[i] == best {
                1. / winners as f64
            } else {
                0.
            };
            check(
                u8::from(result.seats[i].seat) as usize == i
                    && result.seats[i].rank as usize
                        == 1 + scores.iter().filter(|s| **s > scores[i]).count()
                    && (result.seats[i].utility - share).abs() < 1e-8,
                "terminal rank/win mismatch",
            )?;
            let placement = (0..4)
                .filter(|j| *j != i)
                .map(|j| {
                    if scores[i] > scores[j] {
                        1.
                    } else if scores[i] == scores[j] {
                        0.5
                    } else {
                        0.
                    }
                })
                .sum::<f64>()
                / 3.;
            let vp = self.vp_cap.map_or(scores[i].0 as f64, |cap| {
                (scores[i].0 as f64).min(f64::from(cap))
            });
            out[i] = self.win * share + self.placement * placement + self.vp * vp;
        }
        Ok(out)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LedgerState {
    pub spent: [f64; 4],
    pub settled: Vec<(Era, u8)>,
}
#[derive(Debug, Clone)]
pub struct Ledger {
    reward: Reward,
    pub spent: [f64; 4],
    settled: BTreeSet<(u8, u8)>,
    previous: Option<Observation>,
}
fn era(e: Era) -> u8 {
    if e == Era::CanalEra {
        0
    } else {
        1
    }
}
impl Ledger {
    pub fn new(reward: Reward, state: LedgerState) -> Result<Self> {
        reward.validate()?;
        check(
            state
                .spent
                .iter()
                .all(|x| x.is_finite() && *x >= 0. && *x <= reward.insolvency.max_cost + 1e-8),
            "invalid spent budget",
        )?;
        Ok(Self {
            reward,
            spent: state.spent,
            settled: state
                .settled
                .into_iter()
                .map(|(e, r)| (era(e), r))
                .collect(),
            previous: None,
        })
    }
    pub fn state(&self) -> LedgerState {
        LedgerState {
            spent: self.spent,
            settled: self
                .settled
                .iter()
                .map(|(e, r)| (if *e == 0 { Era::CanalEra } else { Era::RailEra }, *r))
                .collect(),
        }
    }
    pub fn remaining(&self) -> [f64; 4] {
        self.spent
            .map(|s| (self.reward.insolvency.max_cost - s).max(0.))
    }
    fn charge(&mut self, seat: usize) {
        self.spent[seat] += self
            .reward
            .insolvency
            .event_cost
            .min(self.remaining()[seat]);
    }
    pub fn chosen(&mut self, view: &Observation, choice: &Choice) {
        // Only continuation Finish can infer an automatic settlement boundary.
        self.previous = if view.context.phase == Phase::Continuation && choice.kind == Kind::Finish
        {
            Some(view.clone())
        } else {
            None
        };
    }
    pub fn observe(&mut self, view: &Observation) -> Result<()> {
        let c = &view.context;
        let key = (era(c.era), c.round);
        if c.phase == Phase::Debt && self.settled.insert(key) {
            for p in &view.players {
                if p.debt > 0 {
                    self.charge(u8::from(p.seat) as usize);
                }
            }
        }
        if let Some(before) = self.previous.take() {
            let old = &before.context;
            let oldkey = (era(old.era), old.round);
            if (key != oldkey || c.terminal)
                && !(c.terminal && old.era == Era::RailEra)
                && !self.settled.contains(&oldkey)
            {
                for p in &before.players {
                    if p.cash + i32::from(p.income) < 0 {
                        check(
                            !before
                                .slots
                                .iter()
                                .filter_map(|s| s.tile.as_ref())
                                .any(|t| t.seat == p.seat),
                            "missing simulated debt decisions",
                        )?;
                        self.charge(u8::from(p.seat) as usize);
                    }
                }
                self.settled.insert(oldkey);
            }
        }
        Ok(())
    }
}
