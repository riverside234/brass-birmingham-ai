use super::*;
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Outcome {
    pub card: String,
    pub probability: f64,
}
#[derive(Debug, Serialize)]
pub struct ChanceOutcomes {
    pub kind: ChanceKind,
    pub remaining: usize,
    pub outcomes: Vec<Outcome>,
}

impl NativeState {
    pub fn chance_outcomes(&self) -> Option<ChanceOutcomes> {
        self.chance.as_ref().map(|c| {
            let mut counts = std::collections::BTreeMap::<CardId, usize>::new();
            for id in &c.pool {
                *counts.entry(*id).or_default() += 1;
            }
            ChanceOutcomes {
                kind: c.kind,
                remaining: usize::from(c.needed) - c.picked.len(),
                outcomes: counts
                    .into_iter()
                    .map(|(id, n)| Outcome {
                        card: CATALOG.card_names[id.0].clone(),
                        probability: n as f64 / c.pool.len() as f64,
                    })
                    .collect(),
            }
        })
    }
    pub fn resolve_chance(&mut self, revision: u32, card: &str) -> Result<(), CoreError> {
        *self = self.after_chance(revision, card)?;
        Ok(())
    }
    /// Prepare an owned successor so callers can finish fallible work before commit.
    pub fn after_chance(&self, revision: u32, card: &str) -> Result<Self, CoreError> {
        require(
            revision == self.context.revision && revision < u32::MAX,
            "stale revision or revision limit",
        )?;
        let c = self
            .chance
            .as_ref()
            .ok_or_else(|| CoreError::Invalid("no chance pending".into()))?;
        let id = CATALOG.card(card)?;
        let index = c
            .pool
            .iter()
            .position(|x| *x == id)
            .ok_or_else(|| CoreError::Invalid("illegal chance".into()))?;
        let mut next = self.clone();
        let c = next.chance.as_mut().unwrap();
        c.pool.remove(index);
        c.picked.push(id);
        if c.picked.len() == usize::from(c.needed) {
            let c = next.chance.take().unwrap();
            match c.kind {
                ChanceKind::Draw => {
                    next.cards.deck = c.picked.into_iter().chain(c.pool).collect();
                    next.end_turn()?;
                    next.advance()?;
                }
                ChanceKind::RailShuffle => {
                    next.end_round()?;
                    next.cards.discard_starts =
                        std::array::from_fn(|s| next.cards.discards[s].len());
                    for (i, s) in next.order.iter().enumerate() {
                        next.cards.hands[seat(*s)] = c.picked[i * 8..(i + 1) * 8].to_vec();
                    }
                    next.cards.deck = c.picked[32..].to_vec();
                    next.advance()?;
                }
            }
        }
        next.context.revision += 1;
        next.refresh_context();
        next.rebuild_indexes();
        Ok(next)
    }
}
