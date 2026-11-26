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
                        card: CATALOG.card_names[id.0].to_string(),
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
        let (index, id) = self.chance_pick(revision, card)?;
        let mut next = self.clone();
        next.commit_chance(index, id)?;
        Ok(next)
    }
    /// The search step's hot path: `after_chance` applied to this state itself, with
    /// no copy of it. On an error the state is partly updated, so the caller drops
    /// it (a search world never outlives a failed step).
    pub(crate) fn resolve_chance_in_place(
        &mut self,
        revision: u32,
        card: &str,
    ) -> Result<(), CoreError> {
        let (index, id) = self.chance_pick(revision, card)?;
        self.commit_chance(index, id)
    }
    /// The checks before a chance outcome: its pool position and card.
    fn chance_pick(&self, revision: u32, card: &str) -> Result<(usize, CardId), CoreError> {
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
        Ok((index, id))
    }
    /// The chance outcome itself, on the state it updates.
    fn commit_chance(&mut self, index: usize, id: CardId) -> Result<(), CoreError> {
        let c = self.chance.as_mut().unwrap();
        c.pool.remove(index);
        c.picked.push(id);
        if c.picked.len() == usize::from(c.needed) {
            let c = self.chance.take().unwrap();
            match c.kind {
                ChanceKind::Draw => {
                    self.cards.deck = c.picked.into_iter().chain(c.pool).collect();
                    self.end_turn()?;
                    self.advance()?;
                }
                ChanceKind::RailShuffle => {
                    self.end_round()?;
                    self.cards.discard_starts =
                        std::array::from_fn(|s| self.cards.discards[s].len());
                    for (i, s) in self.order.iter().enumerate() {
                        self.cards.hands[seat(*s)] = c.picked[i * 8..(i + 1) * 8].to_vec();
                    }
                    self.cards.deck = c.picked[32..].to_vec();
                    self.advance()?;
                }
            }
        }
        self.context.revision += 1;
        self.refresh_context();
        self.rebuild_indexes();
        Ok(())
    }
}
