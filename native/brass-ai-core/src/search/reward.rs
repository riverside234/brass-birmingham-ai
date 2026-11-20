//! Exact undiscounted public insolvency ledger and official four-seat terminal utility.
use super::*;
use crate::{
    state::{
        dto::{Choice, Kind, ResultDto},
        observation::{Accounts, Observation},
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
    // Public accounts of a continuation Finish; never a full view copy.
    previous: Option<Accounts>,
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
            Some(Accounts::from_view(view))
        } else {
            None
        };
    }
    pub fn observe(&mut self, view: &Observation) -> Result<()> {
        self.observe_accounts(&Accounts::from_view(view))
    }
    /// The same accounting from public facts read off a view or the owned state.
    pub fn observe_accounts(&mut self, now: &Accounts) -> Result<()> {
        let key = (era(now.era), now.round);
        if now.phase == Phase::Debt && self.settled.insert(key) {
            for seat in 0..4 {
                if now.debt[seat] > 0 {
                    self.charge(seat);
                }
            }
        }
        if let Some(before) = self.previous.take() {
            let oldkey = (era(before.era), before.round);
            if (key != oldkey || now.terminal)
                && !(now.terminal && before.era == Era::RailEra)
                && !self.settled.contains(&oldkey)
            {
                for seat in 0..4 {
                    if before.cash[seat] + i32::from(before.income[seat]) < 0 {
                        check(!before.on_board[seat], "missing simulated debt decisions")?;
                        self.charge(seat);
                    }
                }
                self.settled.insert(oldkey);
            }
        }
        Ok(())
    }
    /// Exact ledger bytes for a search identity; `previous` is always consumed
    /// by `observe` before a decision is identified.
    pub(crate) fn identity(&self, out: &mut Vec<u8>) {
        for s in self.spent {
            out.extend_from_slice(&s.to_bits().to_le_bytes());
        }
        out.push(self.previous.is_some() as u8);
        out.extend_from_slice(&(self.settled.len() as u16).to_le_bytes());
        for (e, r) in &self.settled {
            out.extend_from_slice(&[*e, *r]);
        }
    }
}
