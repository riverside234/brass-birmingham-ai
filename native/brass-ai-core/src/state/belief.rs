//! Broad reference beliefs: permitted current view only, never recall or live state.
use super::{
    observation::{validate_recall, Observation, ObservationMemory, Recall},
    *,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const SAMPLER: &str =
    "brass-permitted-sampler/1.0.0:rand_chacha-0.3.1:ChaCha12Rng:fisher-yates-u32";

/// Descending Fisher-Yates indices have explicit finite uniform support 0..=i.
/// Replay these draws across languages; Python and ChaCha seeds are not equivalent.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SamplingDraws {
    Recorded { indices: Vec<u32> },
    Chacha12 { seed: u64, stream: u64 },
}

#[derive(Debug, Clone)]
pub struct PreparedRoot {
    template: NativeState,
    public: Arc<Observation>,
    pool: Vec<CardId>,
    missing_hands: [usize; 4],
    hidden_discards: [usize; 4],
    root: Seat,
    recall: Recall,
}
impl PreparedRoot {
    /// No full-state overload. Preparation parses public history/topology once.
    pub fn new(view: &Observation, recall: Recall) -> Result<Self, CoreError> {
        require(
            view.protocol == "0.2.0" && view.rules == TARGET_RULES,
            "observation contract",
        )?;
        require(
            !view.context.terminal
                && view.context.phase != Phase::Chance
                && (0..4).contains(&view.context.actor),
            "sample requires a nonterminal observed decision",
        )?;
        require(
            view.own_hand.len() <= 8 && view.own_hand.is_sorted(),
            "own hand bounds/order",
        )?;
        require(
            view.deck_count <= 64 && view.wild_industry_count <= 4 && view.wild_location_count <= 4,
            "card pile bounds",
        )?;
        validate_recall(view, &recall)?;
        let own = seat(view.seat);
        let mut normal: Vec<_> = CATALOG
            .card_names
            .iter()
            .zip(&CATALOG.card_counts)
            .map(|(name, n)| {
                if name.starts_with("wild:") {
                    0i32
                } else {
                    i32::from(*n)
                }
            })
            .collect();
        let mut cards = Cards {
            hands: std::array::from_fn(|_| vec![]),
            discards: std::array::from_fn(|_| vec![]),
            discard_starts: [0; 4],
            deck: vec![],
            wild_industry: vec![CATALOG.card("wild:industry")?; view.wild_industry_count as usize],
            wild_location: vec![CATALOG.card("wild:location")?; view.wild_location_count as usize],
        };
        let mut missing_hands = [0; 4];
        let mut hidden_discards = [0; 4];
        for (s, p) in view.players.iter().enumerate() {
            require(seat(p.seat) == s, "players must be in stable seat order")?;
            require(
                p.hand_count <= 8 && p.discards.len() <= 128 && p.known_wild.len() <= 8,
                "public card counts",
            )?;
            require(
                p.hidden_discards == u8::from(view.context.era == Era::CanalEra),
                "hidden setup burn count",
            )?;
            require(
                p.known_wild
                    .iter()
                    .all(|c| matches!(c.as_str(), "wild:industry" | "wild:location")),
                "known_wild contains a normal card",
            )?;
            cards.hands[s] = if s == own {
                &view.own_hand
            } else {
                &p.known_wild
            }
            .iter()
            .map(|c| CATALOG.card(c))
            .collect::<Result<_, _>>()?;
            cards.discards[s] = p
                .discards
                .iter()
                .map(|c| CATALOG.card(c))
                .collect::<Result<_, _>>()?;
            for id in cards.discards[s].iter().chain(&cards.hands[s]) {
                if !CATALOG.card_names[id.0].starts_with("wild:") {
                    normal[id.0] -= 1;
                    require(
                        normal[id.0] >= 0,
                        "public card constraints are inconsistent",
                    )?;
                }
            }
            missing_hands[s] = (p.hand_count as usize)
                .checked_sub(cards.hands[s].len())
                .ok_or_else(|| CoreError::Invalid("hand count mismatch".into()))?;
            require(s != own || missing_hands[s] == 0, "own hand count mismatch")?;
            hidden_discards[s] = p.hidden_discards as usize;
        }
        let pool: Vec<_> = normal
            .iter()
            .enumerate()
            .flat_map(|(i, n)| std::iter::repeat_n(CardId(i), *n as usize))
            .collect();
        require(
            pool.len()
                == missing_hands.iter().sum::<usize>()
                    + hidden_discards.iter().sum::<usize>()
                    + view.deck_count as usize,
            "unseen card conservation failed",
        )?;
        let base_cards = cards.clone();
        deal(&mut cards, &pool, &missing_hands, &hidden_discards);
        let used: std::collections::BTreeSet<_> = view
            .players
            .iter()
            .flat_map(|p| p.mat.iter())
            .flat_map(|m| m.tiles.iter())
            .chain(view.slots.iter().filter_map(|s| s.tile.as_ref()))
            .map(|t| t.id.as_str())
            .collect();
        let mut removed: Vec<_> = CATALOG
            .tiles
            .iter()
            .filter(|t| !used.contains(t.id.as_str()))
            .map(|t| t.id.clone())
            .collect();
        removed.sort();
        let debt = view.context.phase == Phase::Debt;
        // Match C# ImportPosition: absent tiles are removed, old Rail discard
        // aliases are absent, and debt reconstructs the settlement flow.
        let dto = StateDto {
            format: STATE_FORMAT.into(),
            rules: TARGET_RULES.into(),
            context: view.context.state_context(),
            current_player: if debt {
                Seat::try_from(view.context.actor as u8).unwrap()
            } else {
                view.context.turn_owner
            },
            round_pending: debt,
            debt_pending: debt,
            settlement_debts: std::array::from_fn(|s| if debt { view.players[s].debt } else { 0 }),
            order: view.order,
            players: view.players.each_ref().map(|p| p.state_player()),
            slots: view.slots.clone(),
            links: view.links.clone(),
            merchants: view.merchants.clone(),
            removed,
            retired: vec![],
            cards: map_cards(&cards, |c| Ok(CATALOG.card_names[c.0].clone()))?,
            coal_market: view.coal_market,
            iron_market: view.iron_market,
            pending: view.pending.clone(),
            history: view.history.clone(),
            chance: None,
            result: None,
        };
        let mut template = NativeState::import(&dto)?;
        require(
            template.observation(view.seat)? == *view,
            "permitted view does not reconstruct exactly",
        )?;
        // Keep only the fixed permitted card zones in the prepared template.
        // No caller can observe this incomplete template; sample fills every zone.
        template.cards = base_cards;
        let mut public = view.clone();
        public.seat = Seat::Zero;
        public.own_hand.clear();
        Ok(Self {
            template,
            public: Arc::new(public),
            pool,
            missing_hands,
            hidden_discards,
            root: view.seat,
            recall,
        })
    }
    pub fn unseen_cards(&self) -> usize {
        self.pool.len()
    }
    pub fn draw_count(&self) -> usize {
        self.pool.len().saturating_sub(1)
    }
    pub fn sample(
        &self,
        random: SamplingDraws,
    ) -> Result<(NativeState, ObservationMemory, Vec<u32>), CoreError> {
        crate::work::checkpoint(crate::work::Stage::Sampling)?;
        let indices = match random {
            SamplingDraws::Recorded { indices } => indices,
            SamplingDraws::Chacha12 { seed, stream } => {
                let mut rng = ChaCha12Rng::seed_from_u64(seed);
                rng.set_stream(stream);
                (1..self.pool.len() as u32)
                    .rev()
                    .map(|i| rng.gen_range(0..=i))
                    .collect()
            }
        };
        let mut pool = self.pool.clone();
        shuffle(&mut pool, &indices)?;
        let copy_started = std::time::Instant::now();
        let mut state = self.template.clone();
        crate::work::state_copied(copy_started);
        deal(
            &mut state.cards,
            &pool,
            &self.missing_hands,
            &self.hidden_discards,
        );
        // Card replacement cannot change the board's indexes. No DTO export,
        // replay, parser or reference callback occurs per sampled world.
        let mut memory = ObservationMemory::with_public(&state, self.public.clone());
        memory.recall[seat(self.root)] = self.recall.clone();
        Ok((state, memory, indices))
    }
}

fn shuffle(pool: &mut [CardId], indices: &[u32]) -> Result<(), CoreError> {
    require(
        indices.len() == pool.len().saturating_sub(1),
        "shuffle draw count mismatch",
    )?;
    for (i, j) in (1..pool.len()).rev().zip(indices) {
        crate::work::checkpoint(crate::work::Stage::Sampling)?;
        require((*j as usize) <= i, "shuffle draw outside support")?;
        pool.swap(i, *j as usize);
    }
    Ok(())
}

fn deal(cards: &mut Cards<CardId>, pool: &[CardId], missing: &[usize; 4], hidden: &[usize; 4]) {
    let mut offset = 0;
    for (s, n) in missing.iter().enumerate() {
        cards.hands[s].extend_from_slice(&pool[offset..offset + n]);
        offset += n;
    }
    for (s, n) in hidden.iter().enumerate() {
        let mut discards = pool[offset..offset + n].to_vec();
        offset += n;
        discards.append(&mut cards.discards[s]);
        cards.discards[s] = discards;
    }
    cards.deck = pool[offset..].to_vec();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    #[test]
    fn recorded_shuffle_has_exact_uniform_support_and_duplicate_multiplicity() {
        let mut distinct = BTreeSet::new();
        let mut repeated = BTreeMap::new();
        for a in 0..3 {
            for b in 0..2 {
                let mut cards = [CardId(0), CardId(1), CardId(2)];
                shuffle(&mut cards, &[a, b]).unwrap();
                distinct.insert(cards);
                let mut cards = [CardId(0), CardId(0), CardId(1)];
                shuffle(&mut cards, &[a, b]).unwrap();
                *repeated.entry(cards).or_insert(0) += 1;
            }
        }
        assert_eq!(distinct.len(), 6);
        assert_eq!(repeated.len(), 3);
        assert!(repeated.values().all(|n| *n == 2));
        assert!(shuffle(&mut [], &[]).is_ok());
        assert!(shuffle(&mut [CardId(0)], &[]).is_ok());
    }

    #[test]
    fn native_samples_match_unseen_multiset_marginals_and_replay() {
        let dto =
            serde_json::from_str(include_str!("../../tests/data/initial-state.json")).unwrap();
        let view = NativeState::import(&dto)
            .unwrap()
            .observation(Seat::Zero)
            .unwrap();
        let root = PreparedRoot::new(&view, vec![]).unwrap();
        let mut count = vec![0usize; CATALOG.card_names.len()];
        let expected = counts(&root.pool);
        for stream in 0..4096 {
            let (state, _, draws) = root
                .sample(SamplingDraws::Chacha12 { seed: 890, stream })
                .unwrap();
            count[state.cards.hands[1][0].0] += 1;
            if stream < 8 {
                assert_eq!(
                    state.export(),
                    root.sample(SamplingDraws::Recorded { indices: draws })
                        .unwrap()
                        .0
                        .export()
                );
                assert_eq!(
                    counts(
                        state
                            .cards
                            .hands
                            .iter()
                            .flatten()
                            .chain(state.cards.discards.iter().flatten())
                            .chain(&state.cards.deck)
                            .chain(&state.cards.wild_industry)
                            .chain(&state.cards.wild_location)
                    ),
                    CATALOG
                        .card_counts
                        .iter()
                        .map(|n| *n as usize)
                        .collect::<Vec<_>>()
                );
            }
        }
        let mut chi_squared = 0.0;
        for (seen, n) in count.iter().zip(expected) {
            if n == 0 {
                assert_eq!(*seen, 0);
            } else {
                let mean = 4096.0 * n as f64 / root.pool.len() as f64;
                chi_squared += (*seen as f64 - mean).powi(2) / mean;
            }
        }
        // Fixed stream census, a deliberately generous regression threshold.
        // Exact distribution support is tested separately by exhaustive tapes.
        assert!(
            chi_squared < 100.0,
            "unseen-card marginal chi-square {chi_squared}"
        );
    }
}
