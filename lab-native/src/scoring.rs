//! Passive current-era targets from native transitions, outside actor inputs.
use crate::{check, Result};
use brass_ai_core::state::dto::StateDto;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const PRODUCER: &str = "rust-local-scoring/1.0.0";

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recorder {
    start: u32,
    initial: [i32; 4],
    previous: [i32; 4],
    initial_bonus: [i32; 4],
    eras: Vec<Value>,
}

impl Recorder {
    pub fn new(d: &StateDto) -> Self {
        let initial = std::array::from_fn(|s| d.players[s].vp);
        Self {
            start: d.context.revision,
            initial,
            previous: initial,
            initial_bonus: std::array::from_fn(|s| d.players[s].bonus_vp),
            eras: vec![],
        }
    }

    pub fn boundary(&mut self, before: &StateDto, after: &StateDto) -> Result<()> {
        if before.context.era == after.context.era && !after.context.terminal {
            return Ok(());
        }
        check(
            !before.context.terminal && before.context.round == 8,
            "invalid scoring boundary",
        )?;
        let mut industry = [0i32; 4];
        let mut network = [0i32; 4];
        let slots: Vec<Value> = before
            .slots
            .iter()
            .map(|s| {
                let t = s.tile.as_ref();
                let flipped = t.is_some_and(|t| t.flipped);
                let owner = if flipped {
                    u8::from(t.unwrap().seat) + 1
                } else {
                    0
                };
                let vp = if flipped { t.unwrap().vp } else { 0 };
                if owner > 0 {
                    industry[owner as usize - 1] += vp as i32;
                }
                json!({"id":s.id,"owner":owner,"vp":vp,"valid":true,"instance":t.map(|t| &t.id)})
            })
            .collect();
        let links: Vec<Value> = before
            .links
            .iter()
            .map(|l| {
                let valid = format!("{:?}Era", l.era) == format!("{:?}", before.context.era);
                let owner = l.owner.map(|s| u8::from(s) + 1).unwrap_or(0);
                let vp: i32 = if owner == 0 {
                    0
                } else {
                    l.regions
                        .iter()
                        .map(|region| {
                            if before
                                .merchants
                                .iter()
                                .any(|m| m.id.split(':').next() == Some(region))
                            {
                                2
                            } else {
                                before
                                    .slots
                                    .iter()
                                    .filter(|s| s.id.split(':').next() == Some(region))
                                    .filter_map(|s| s.tile.as_ref())
                                    .filter(|t| t.flipped)
                                    .map(|t| t.link_vp as i32)
                                    .sum()
                            }
                        })
                        .sum()
                };
                if owner > 0 {
                    network[owner as usize - 1] += vp;
                }
                json!({"id":l.id,"owner":owner,"vp":vp,"valid":valid})
            })
            .collect();
        let pre: [i32; 4] = std::array::from_fn(|s| {
            before.players[s].vp
                - if before.debt_pending {
                    before.players[s].vp.min(before.settlement_debts[s]).max(0)
                } else {
                    0
                }
        });
        let post: [i32; 4] = std::array::from_fn(|s| after.players[s].vp);
        check(
            (0..4).all(|s| pre[s] + industry[s] + network[s] == post[s]),
            "native scoring audit failed",
        )?;
        let bonus: [i32; 4] = std::array::from_fn(|s| before.players[s].bonus_vp);
        let since: [i32; 4] = std::array::from_fn(|s| bonus[s] - self.initial_bonus[s]);
        let outside: [i32; 4] = std::array::from_fn(|s| pre[s] - self.previous[s]);
        let adjustment: [i32; 4] = std::array::from_fn(|s| outside[s] - since[s]);
        self.eras.push(json!({"era":before.context.era,"slots":slots,"links":links,"before":pre,
            "audit":{"industry":industry,"network":network,"bonus":bonus,"bonus_since_recording":since,
            "outside_adjustment":adjustment,"outside_scoring":outside,"after":post}}));
        self.previous = post;
        self.initial_bonus = [0; 4];
        Ok(())
    }

    pub fn finish(&self, d: &StateDto) -> Result<Value> {
        check(
            d.context.terminal && self.eras.last().is_some_and(|v| v["era"] == "RailEra"),
            "incomplete scoring",
        )?;
        Ok(
            json!({"contract":"local-scoring-targets/1.0.0","rules":d.rules,"producer":PRODUCER,
            "start_revision":self.start,"initial_vp":self.initial,"eras":self.eras,"final_vp":self.previous}),
        )
    }
}
