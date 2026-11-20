//! Explicit policy allowlist. Private zones never enter the public projection.
use super::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewContext {
    pub protocol: String,
    pub rules: String,
    pub revision: u32,
    pub actor: i8,
    pub turn_owner: Seat,
    pub phase: Phase,
    pub actions_remaining: u8,
    pub era: Era,
    pub round: u8,
    pub turn: u8,
    pub terminal: bool,
}
impl ViewContext {
    pub(super) fn state_context(&self) -> Context {
        Context {
            protocol: self.protocol.clone(),
            rules: self.rules.clone(),
            revision: self.revision,
            actor: self.actor,
            turn_owner: self.turn_owner,
            phase: self.phase,
            actions_remaining: self.actions_remaining,
            era: self.era,
            round: self.round,
            turn: self.turn,
            terminal: self.terminal,
            chance: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewPlayer {
    pub seat: Seat,
    pub cash: i32,
    pub income_marker: u8,
    pub income: i8,
    pub vp: i32,
    pub bonus_vp: i32,
    pub link_tiles: u8,
    pub hand_count: u8,
    pub spent: i32,
    pub discards: Vec<String>,
    pub hidden_discards: u8,
    pub known_wild: Vec<String>,
    pub mat: [Mat; 6],
    pub debt: i32,
}
impl ViewPlayer {
    pub(super) fn state_player(&self) -> Player {
        Player {
            seat: self.seat,
            cash: self.cash,
            income_marker: self.income_marker,
            income: self.income,
            vp: self.vp,
            bonus_vp: self.bonus_vp,
            link_tiles: self.link_tiles,
            spent: self.spent,
            mat: self.mat.clone(),
            debt: self.debt,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub protocol: String,
    pub rules: String,
    pub seat: Seat,
    pub context: ViewContext,
    pub own_hand: Vec<String>,
    pub players: [ViewPlayer; 4],
    pub order: [Seat; 4],
    pub slots: Vec<Slot>,
    pub links: Vec<Link>,
    pub merchants: Vec<Merchant>,
    pub coal_market: u8,
    pub iron_market: u8,
    pub deck_count: u8,
    pub wild_industry_count: u8,
    pub wild_location_count: u8,
    pub pending: Pending,
    pub history: Vec<History>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecallEntry {
    pub revision: u32,
    pub hand: Vec<String>,
}
pub type Recall = Vec<RecallEntry>;

pub fn validate_recall(view: &Observation, recall: &Recall) -> Result<(), CoreError> {
    require(recall.len() <= 8192, "recall too long")?;
    for entry in recall {
        require(entry.revision <= view.context.revision, "future recall")?;
        require(
            entry.hand.len() <= 8 && entry.hand.is_sorted(),
            "recall hand bounds/order",
        )?;
        for card in &entry.hand {
            CATALOG.card(card)?;
        }
    }
    require(
        recall.last().is_none_or(|r| r.hand == view.own_hand),
        "recall does not end at own hand",
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservationEnvelope {
    pub view: Option<Observation>,
    pub setup: bool,
    pub chance: Option<ChanceKind>,
}

/// Typed equality, not a board hash, defines the permitted information set.
/// The diagnostic digest is generated only at the transport boundary.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InformationState {
    pub observation: ObservationEnvelope,
    pub recall: Recall,
}
impl InformationState {
    pub fn observation_string(&self) -> String {
        // Match Python json.dumps(sort_keys=True)'s spaces. Values in this
        // allowlist contain only ASCII catalog strings, integers and booleans.
        // Serde owns all JSON escaping/encoding; only separators differ.
        struct PythonSeparators;
        impl serde_json::ser::Formatter for PythonSeparators {
            fn begin_array_value<W: ?Sized + std::io::Write>(
                &mut self,
                w: &mut W,
                first: bool,
            ) -> std::io::Result<()> {
                if first {
                    Ok(())
                } else {
                    w.write_all(b", ")
                }
            }
            fn begin_object_key<W: ?Sized + std::io::Write>(
                &mut self,
                w: &mut W,
                first: bool,
            ) -> std::io::Result<()> {
                self.begin_array_value(w, first)
            }
            fn begin_object_value<W: ?Sized + std::io::Write>(
                &mut self,
                w: &mut W,
            ) -> std::io::Result<()> {
                w.write_all(b": ")
            }
        }
        let sorted = serde_json::to_value(&self.observation).expect("typed observation");
        let mut bytes = Vec::new();
        sorted
            .serialize(&mut serde_json::Serializer::with_formatter(
                &mut bytes,
                PythonSeparators,
            ))
            .expect("typed observation");
        String::from_utf8(bytes).expect("JSON is UTF-8")
    }
    pub fn reference_digest(&self) -> String {
        let value =
            serde_json::json!({"observation": self.observation_string(), "recall": self.recall});
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&value).expect("typed identity"))
        )
    }
}

/// Public accounting facts read by the insolvency ledger. A permitted view and
/// the owned state it was projected from give identical facts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Accounts {
    pub era: Era,
    pub round: u8,
    pub phase: Phase,
    pub terminal: bool,
    pub debt: [i32; 4],
    pub cash: [i32; 4],
    pub income: [i8; 4],
    pub on_board: [bool; 4],
}
impl Accounts {
    pub fn from_view(view: &Observation) -> Self {
        let c = &view.context;
        let mut a = Self {
            era: c.era,
            round: c.round,
            phase: c.phase,
            terminal: c.terminal,
            debt: [0; 4],
            cash: [0; 4],
            income: [0; 4],
            on_board: [false; 4],
        };
        for p in &view.players {
            let s = seat(p.seat);
            a.debt[s] = p.debt;
            a.cash[s] = p.cash;
            a.income[s] = p.income;
        }
        for t in view.slots.iter().filter_map(|s| s.tile.as_ref()) {
            a.on_board[seat(t.seat)] = true;
        }
        a
    }
}

fn put_ids(out: &mut Vec<u8>, ids: impl ExactSizeIterator<Item = usize>) {
    out.extend_from_slice(&(ids.len() as u16).to_le_bytes());
    for id in ids {
        out.extend_from_slice(&(id as u16).to_le_bytes());
    }
}
fn put_target(out: &mut Vec<u8>, t: Option<Target>) {
    let (tag, index) = match t {
        None => (0u8, 0usize),
        Some(Target::Slot(s)) => (1, s.0),
        Some(Target::Link(l)) => (2, l.0),
        Some(Target::Merchant(m)) => (3, m.0),
        Some(Target::Market) => (4, 0),
    };
    out.push(tag);
    out.extend_from_slice(&(index as u16).to_le_bytes());
}

/// One immutable public projection, four small private hand/recall records.
/// Chance preserves the last permitted reveal; it never refreshes temporary hands.
#[derive(Debug, Clone, Default)]
pub struct ObservationMemory {
    public: Option<Arc<Observation>>,
    hands: [Vec<String>; 4],
    pub(super) recall: [Recall; 4],
    chance: Option<ChanceKind>,
}
impl ObservationMemory {
    /// Trusted checkpoint boundary only. Live policy roots use PreparedRoot.
    pub fn restore_checkpoint(
        state: &NativeState,
        information: [InformationState; 4],
    ) -> Result<Self, CoreError> {
        let chance = state.chance.as_ref().map(|c| c.kind);
        let mut result = Self::default();
        for (s, info) in information.into_iter().enumerate() {
            require(
                !info.observation.setup && info.observation.chance == chance,
                "checkpoint reveal kind",
            )?;
            if let Some(mut view) = info.observation.view {
                require(seat(view.seat) == s, "checkpoint observer order")?;
                validate_recall(&view, &info.recall)?;
                if chance.is_none() {
                    require(
                        view == state.observation(view.seat)?,
                        "checkpoint current view",
                    )?;
                } else {
                    require(
                        view.context.revision < state.context.revision,
                        "checkpoint prior reveal revision",
                    )?;
                    super::belief::PreparedRoot::new(&view, info.recall.clone())?;
                }
                result.hands[s] = std::mem::take(&mut view.own_hand);
                view.seat = Seat::Zero;
                if s == 0 {
                    result.public = Some(Arc::new(view));
                } else {
                    require(
                        result.public.as_deref() == Some(&view),
                        "checkpoint public views differ",
                    )?;
                }
            } else {
                require(
                    chance.is_some() && info.recall.is_empty() && result.public.is_none(),
                    "checkpoint missing view",
                )?;
            }
            result.recall[s] = info.recall;
        }
        // Missing reveals must be missing for every observer.
        require(
            result.public.is_none()
                || result.hands.iter().enumerate().all(|(s, h)| {
                    h.len() == result.public.as_ref().unwrap().players[s].hand_count as usize
                }),
            "checkpoint hand counts",
        )?;
        result.chance = chance;
        Ok(result)
    }
    pub(super) fn with_public(state: &NativeState, public: Arc<Observation>) -> Self {
        let hands: [Vec<String>; 4] = std::array::from_fn(|s| state.sorted_hand(s));
        let recall = std::array::from_fn(|s| {
            vec![RecallEntry {
                revision: state.context.revision,
                hand: hands[s].clone(),
            }]
        });
        Self {
            public: Some(public),
            hands,
            recall,
            chance: None,
        }
    }
    pub fn new(state: &NativeState) -> Self {
        let mut result = Self::default();
        result.refresh(state);
        result
    }
    pub fn refresh(&mut self, state: &NativeState) {
        self.refresh_checked(state).expect("uncontrolled reveal");
    }
    /// Search hot path: the same hands and recall as `refresh_checked`, but the
    /// public projection is dropped rather than rebuilt. `information_at` and
    /// `accounts_at` then read the owned state directly.
    pub fn refresh_recall(&mut self, state: &NativeState) -> Result<(), CoreError> {
        crate::work::checkpoint(crate::work::Stage::Observation)?;
        self.chance = state.chance.as_ref().map(|c| c.kind);
        if self.chance.is_some() {
            return Ok(());
        }
        self.public = None;
        for s in 0..4 {
            self.hands[s] = state.sorted_hand(s);
            if self.recall[s]
                .last()
                .is_none_or(|r| r.hand != self.hands[s])
            {
                self.recall[s].push(RecallEntry {
                    revision: state.context.revision,
                    hand: self.hands[s].clone(),
                });
            }
        }
        Ok(())
    }
    /// `information(observer)` for the reveal `state`; equal to the published
    /// projection, built directly when `refresh_recall` dropped it.
    pub fn information_at(
        &self,
        state: &NativeState,
        observer: Seat,
    ) -> Result<InformationState, CoreError> {
        if self.public.is_some() || self.chance.is_some() {
            return Ok(self.information(observer));
        }
        Ok(InformationState {
            observation: ObservationEnvelope {
                view: Some(state.observation(observer)?),
                setup: false,
                chance: None,
            },
            recall: self.recall[seat(observer)].clone(),
        })
    }
    /// Facts of `accounting_view`, read off the owned state when no projection is kept.
    pub fn accounts_at(&self, state: &NativeState) -> Option<Accounts> {
        if let Some(view) = self.accounting_view() {
            Some(Accounts::from_view(view))
        } else if self.chance.is_some() {
            None
        } else {
            Some(state.accounts())
        }
    }
    pub fn recall_len(&self, observer: Seat) -> usize {
        self.recall[seat(observer)].len()
    }
    /// Recall bytes after a caller-certified common prefix of `skip` entries.
    pub fn recall_identity(&self, observer: Seat, skip: usize, out: &mut Vec<u8>) {
        let recall = &self.recall[seat(observer)];
        out.extend_from_slice(&(recall.len() as u32).to_le_bytes());
        for entry in &recall[skip.min(recall.len())..] {
            out.extend_from_slice(&entry.revision.to_le_bytes());
            out.push(entry.hand.len() as u8);
            for card in &entry.hand {
                out.push(card.len() as u8);
                out.extend_from_slice(card.as_bytes());
            }
        }
    }
    pub fn refresh_checked(&mut self, state: &NativeState) -> Result<(), CoreError> {
        crate::work::checkpoint(crate::work::Stage::Observation)?;
        self.chance = state.chance.as_ref().map(|c| c.kind);
        if self.chance.is_some() {
            return Ok(());
        }
        // Build public fields once; only seat and own_hand vary by observer.
        let mut public = state.observation(Seat::Zero)?;
        public.own_hand.clear();
        for s in 0..4 {
            self.hands[s] = state.sorted_hand(s);
            if self.recall[s]
                .last()
                .is_none_or(|r| r.hand != self.hands[s])
            {
                self.recall[s].push(RecallEntry {
                    revision: state.context.revision,
                    hand: self.hands[s].clone(),
                });
            }
        }
        self.public = Some(Arc::new(public));
        Ok(())
    }
    pub fn attach_recall(&mut self, view: &Observation, recall: Recall) -> Result<(), CoreError> {
        validate_recall(view, &recall)?;
        require(
            self.view(view.seat).as_ref() == Some(view),
            "recall observation mismatch",
        )?;
        self.recall[seat(view.seat)] = recall;
        Ok(())
    }
    fn view(&self, observer: Seat) -> Option<Observation> {
        self.public.as_ref().map(|public| {
            let mut v = (**public).clone();
            v.seat = observer;
            v.own_hand = self.hands[seat(observer)].clone();
            v
        })
    }
    /// Borrow already-published public fields for causal debt accounting.
    /// Own hand is intentionally empty; callers must not use this as a policy
    /// observation. Chance states have no newly revealed accounting boundary.
    pub(crate) fn accounting_view(&self) -> Option<&Observation> {
        if self.chance.is_some() {
            None
        } else {
            self.public.as_deref()
        }
    }
    pub fn information(&self, observer: Seat) -> InformationState {
        InformationState {
            observation: ObservationEnvelope {
                view: self.view(observer),
                setup: false,
                chance: self.chance,
            },
            recall: self.recall[seat(observer)].clone(),
        }
    }
}

impl NativeState {
    pub fn accounts(&self) -> Accounts {
        let c = &self.context;
        let mut a = Accounts {
            era: c.era,
            round: c.round,
            phase: c.phase,
            terminal: c.terminal,
            debt: self.players.each_ref().map(|p| p.debt),
            cash: self.players.each_ref().map(|p| p.cash),
            income: self
                .players
                .each_ref()
                .map(|p| CATALOG.incomes[usize::from(p.income_marker)]),
            on_board: [false; 4],
        };
        for id in self.slots.iter().flatten() {
            a.on_board[seat(CATALOG.tiles[id.0].seat)] = true;
        }
        a
    }
    /// Compact refinement of `observation(observer)` and of the observer's legal
    /// set: every owned field except other seats' hands, the deck order and the
    /// hidden discard prefixes. Equal bytes imply an equal permitted view and
    /// equal legal choices; nothing here is read from a sampled hidden zone.
    pub fn visible_identity(&self, observer: Seat, out: &mut Vec<u8>) {
        let o = seat(observer);
        let c = &self.context;
        out.push(o as u8);
        out.extend_from_slice(&c.revision.to_le_bytes());
        out.extend_from_slice(&[
            c.actor as u8,
            u8::from(c.turn_owner),
            c.phase as u8,
            c.actions_remaining,
            c.era as u8,
            c.round,
            c.turn,
            c.terminal as u8,
            c.chance as u8,
            c.protocol.len() as u8,
        ]);
        out.extend_from_slice(c.protocol.as_bytes());
        out.push(c.rules.len() as u8);
        out.extend_from_slice(c.rules.as_bytes());
        out.extend_from_slice(&[
            u8::from(self.current_player),
            self.round_pending as u8,
            self.debt_pending as u8,
        ]);
        for d in self.settlement_debts {
            out.extend_from_slice(&d.to_le_bytes());
        }
        for s in self.order {
            out.push(u8::from(s));
        }
        for p in &self.players {
            for x in [p.cash, p.vp, p.bonus_vp, p.spent, p.debt] {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out.extend_from_slice(&[p.income_marker, p.link_tiles]);
            for m in &p.mat {
                put_ids(out, m.iter().map(|t| t.0));
            }
        }
        for t in &self.tiles {
            out.extend_from_slice(&[t.flipped as u8, t.resource]);
        }
        for s in &self.slots {
            out.extend_from_slice(&s.map_or(u16::MAX, |t| t.0 as u16).to_le_bytes());
        }
        for l in &self.links {
            out.push(l.map_or(u8::MAX, u8::from));
        }
        for m in &self.merchants {
            out.extend_from_slice(&[m.defined as u8, m.beer, m.types.len() as u8]);
            for t in &m.types {
                out.push(*t as u8);
            }
        }
        put_ids(out, self.removed.iter().map(|t| t.0));
        put_ids(out, self.retired.iter().map(|t| t.0));
        let cards = &self.cards;
        for s in 0..4 {
            out.push(cards.hands[s].len() as u8);
            let start = if c.era == Era::CanalEra {
                1
            } else {
                cards.discard_starts[s]
            };
            out.extend_from_slice(&(cards.discard_starts[s] as u16).to_le_bytes());
            let visible = &cards.discards[s][start.min(cards.discards[s].len())..];
            put_ids(out, visible.iter().map(|c| c.0));
        }
        let mut hand = [usize::MAX; 8];
        let held = &cards.hands[o];
        for (slot, card) in hand.iter_mut().zip(held) {
            *slot = card.0;
        }
        let hand = &mut hand[..held.len().min(8)];
        hand.sort_unstable();
        out.push(held.len() as u8);
        put_ids(out, hand.iter().copied());
        put_ids(out, cards.wild_industry.iter().map(|c| c.0));
        put_ids(out, cards.wild_location.iter().map(|c| c.0));
        out.extend_from_slice(&(cards.deck.len() as u16).to_le_bytes());
        out.extend_from_slice(&[self.coal_market, self.iron_market]);
        let p = &self.pending;
        out.push(p.kind.map_or(u8::MAX, |k| k as u8));
        put_target(out, p.target);
        out.push(p.merchant.map_or(u8::MAX, |m| m.0 as u8));
        out.extend_from_slice(&[p.parts, p.coal, p.iron, p.beer]);
        out.extend_from_slice(&(self.history.len() as u32).to_le_bytes());
        for h in &self.history {
            out.extend_from_slice(&[u8::from(h.seat), h.era as u8, h.kind as u8]);
            put_target(out, h.target);
            out.extend_from_slice(&h.card.map_or(u16::MAX, |c| c.0 as u16).to_le_bytes());
            out.push(h.industry.map_or(u8::MAX, |i| i as u8));
            out.push(h.merchant.map_or(u8::MAX, |m| m.0 as u8));
            match &h.cards {
                None => out.push(u8::MAX),
                Some(v) => {
                    out.push(v.len() as u8);
                    for card in v {
                        out.extend_from_slice(&(card.0 as u16).to_le_bytes());
                    }
                }
            }
        }
        out.push(self.chance.is_some() as u8);
    }
    fn sorted_hand(&self, s: usize) -> Vec<String> {
        let mut hand: Vec<_> = self.cards.hands[s]
            .iter()
            .map(|c| CATALOG.card_names[c.0].clone())
            .collect();
        hand.sort();
        hand
    }
    /// This raw projection is available only at a reveal boundary. At chance,
    /// callers use ObservationMemory's previous permitted observation instead.
    pub fn observation(&self, observer: Seat) -> Result<Observation, CoreError> {
        require(
            self.chance.is_none(),
            "temporary chance state has no new policy observation",
        )?;
        let mut known: [Vec<String>; 4] = std::array::from_fn(|_| vec![]);
        for h in &self.history {
            crate::work::checkpoint(crate::work::Stage::Observation)?;
            if h.era != self.context.era {
                continue;
            }
            let k = &mut known[seat(h.seat)];
            if let Some(card) = h.card {
                let name = &CATALOG.card_names[card.0];
                if name.starts_with("wild:") {
                    if let Some(i) = k.iter().position(|x| x == name) {
                        k.remove(i);
                    }
                }
            }
            if h.kind == Kind::Scout {
                k.extend(["wild:industry".into(), "wild:location".into()]);
            }
        }
        for k in &mut known {
            k.sort();
        }
        let c = &self.context;
        Ok(Observation {
            protocol: "0.2.0".into(),
            rules: TARGET_RULES.into(),
            seat: observer,
            context: ViewContext {
                protocol: c.protocol.clone(),
                rules: c.rules.clone(),
                revision: c.revision,
                actor: c.actor,
                turn_owner: c.turn_owner,
                phase: c.phase,
                actions_remaining: c.actions_remaining,
                era: c.era,
                round: c.round,
                turn: c.turn,
                terminal: c.terminal,
            },
            own_hand: self.sorted_hand(seat(observer)),
            players: std::array::from_fn(|s| {
                let p = &self.players[s];
                let hidden = usize::from(c.era == Era::CanalEra);
                let start = if hidden == 1 {
                    1
                } else {
                    self.cards.discard_starts[s]
                };
                ViewPlayer {
                    seat: Seat::try_from(s as u8).unwrap(),
                    cash: p.cash,
                    income_marker: p.income_marker,
                    income: CATALOG.incomes[p.income_marker as usize],
                    vp: p.vp,
                    bonus_vp: p.bonus_vp,
                    link_tiles: p.link_tiles,
                    hand_count: self.cards.hands[s].len() as u8,
                    spent: p.spent,
                    discards: self.cards.discards[s]
                        .iter()
                        .skip(start)
                        .map(|c| CATALOG.card_names[c.0].clone())
                        .collect(),
                    hidden_discards: hidden as u8,
                    known_wild: known[s].clone(),
                    debt: p.debt,
                    mat: std::array::from_fn(|i| Mat {
                        industry: Industry::ALL[i],
                        tiles: p.mat[i].iter().map(|id| self.tile(*id)).collect(),
                    }),
                }
            }),
            order: self.order,
            slots: CATALOG
                .slots
                .iter()
                .zip(&self.slots)
                .map(|(d, t)| Slot {
                    id: d.id.clone(),
                    types: d.types.clone(),
                    tile: t.map(|id| self.tile(id)),
                })
                .collect(),
            links: CATALOG
                .links
                .iter()
                .zip(&self.links)
                .map(|(d, o)| Link {
                    id: d.id.clone(),
                    regions: d.regions.clone(),
                    era: d.era,
                    owner: *o,
                })
                .collect(),
            merchants: CATALOG
                .merchants
                .iter()
                .zip(&self.merchants)
                .map(|(d, m)| Merchant {
                    id: d.id.clone(),
                    types: m.defined.then(|| m.types.clone()),
                    beer: m.beer,
                    bonus: d.bonus,
                })
                .collect(),
            coal_market: self.coal_market,
            iron_market: self.iron_market,
            deck_count: self.cards.deck.len() as u8,
            wild_industry_count: self.cards.wild_industry.len() as u8,
            wild_location_count: self.cards.wild_location.len() as u8,
            pending: self.pending.export(),
            history: self.history.iter().map(HistoryState::export).collect(),
        })
    }
}
