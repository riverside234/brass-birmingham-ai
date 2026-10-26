//! Owned semantic simulator state. Import validates before exposing a replacement.
pub mod belief;
pub(crate) mod catalog;
mod chance;
mod choices;
pub mod initial;
pub mod observation;
mod phases;
pub mod simulation;
pub use chance::ChanceOutcomes;
pub mod dto;
mod indexes;

use crate::{
    types::{Era, Phase, Seat},
    CoreError, TARGET_RULES,
};
use catalog::*;
use dto::*;
pub use indexes::IndexSnapshot;
use indexes::Indexes;

fn require(valid: bool, message: &str) -> Result<(), CoreError> {
    if valid {
        Ok(())
    } else {
        Err(CoreError::Invalid(message.to_owned()))
    }
}
fn seat(s: Seat) -> usize {
    u8::from(s) as usize
}

#[derive(Debug, Clone, Copy, Default)]
struct TileState {
    flipped: bool,
    resource: u8,
}
#[derive(Debug, Clone)]
struct PlayerState {
    cash: i32,
    income_marker: u8,
    vp: i32,
    bonus_vp: i32,
    link_tiles: u8,
    spent: i32,
    debt: i32,
    mat: [Vec<TileId>; 6],
}
#[derive(Debug, Clone)]
struct MerchantState {
    types: Vec<Industry>,
    defined: bool,
    beer: u8,
}
#[derive(Debug, Clone, Copy)]
enum Target {
    Slot(SlotId),
    Link(LinkId),
    Merchant(MerchantId),
    Market,
}
impl Target {
    fn parse(s: &str) -> Result<Self, CoreError> {
        if s == "market" {
            return Ok(Self::Market);
        }
        if let Some(m) = s.strip_prefix("merchant:") {
            return CATALOG.merchant(m).map(Self::Merchant);
        }
        if let Ok(id) = CATALOG.slot(s) {
            return Ok(Self::Slot(id));
        }
        CATALOG.link(s).map(Self::Link)
    }
    fn name(self) -> String {
        match self {
            Self::Slot(id) => CATALOG.slots[id.0].id.clone(),
            Self::Link(id) => CATALOG.links[id.0].id.clone(),
            Self::Merchant(id) => format!("merchant:{}", CATALOG.merchants[id.0].id),
            Self::Market => "market".into(),
        }
    }
}
#[derive(Debug, Clone)]
struct HistoryState {
    seat: Seat,
    era: Era,
    kind: Kind,
    target: Option<Target>,
    card: Option<CardId>,
    industry: Option<Industry>,
    merchant: Option<MerchantId>,
    cards: Option<Vec<CardId>>,
}
impl HistoryState {
    fn import(h: &History) -> Result<Self, CoreError> {
        let c = &h.choice;
        require(
            c.id == c.semantic_id(),
            "history semantic choice ID mismatch",
        )?;
        require(
            c.cards.as_ref().is_none_or(|x| x.len() <= 3),
            "history scout card count",
        )?;
        Ok(Self {
            seat: h.seat,
            era: h.era,
            kind: c.kind,
            target: c.target.as_deref().map(Target::parse).transpose()?,
            card: c.card.as_deref().map(|s| CATALOG.card(s)).transpose()?,
            industry: c.industry,
            merchant: c
                .merchant
                .as_deref()
                .map(|s| CATALOG.merchant(s))
                .transpose()?,
            cards: c
                .cards
                .as_ref()
                .map(|v| v.iter().map(|s| CATALOG.card(s)).collect())
                .transpose()?,
        })
    }
    fn export(&self) -> History {
        let mut choice = Choice {
            id: String::new(),
            kind: self.kind,
            target: self.target.map(Target::name),
            card: self.card.map(|id| CATALOG.card_names[id.0].clone()),
            industry: self.industry,
            merchant: self.merchant.map(|id| CATALOG.merchants[id.0].id.clone()),
            cards: self.cards.as_ref().map(|v| {
                v.iter()
                    .map(|id| CATALOG.card_names[id.0].clone())
                    .collect()
            }),
        };
        choice.id = choice.semantic_id();
        History {
            seat: self.seat,
            era: self.era,
            choice,
        }
    }
}
#[derive(Debug, Clone, Copy)]
struct PendingState {
    kind: Option<Kind>,
    target: Option<Target>,
    merchant: Option<MerchantId>,
    parts: u8,
    coal: u8,
    iron: u8,
    beer: u8,
}
impl PendingState {
    fn import(p: &Pending) -> Result<Self, CoreError> {
        let kind = match p.kind.as_str() {
            "" => None,
            "build" => Some(Kind::Build),
            "network" => Some(Kind::Network),
            "sell" => Some(Kind::Sell),
            "develop" => Some(Kind::Develop),
            "loan" => Some(Kind::Loan),
            "pass" => Some(Kind::Pass),
            "scout" => Some(Kind::Scout),
            _ => return Err(CoreError::Invalid("unknown pending action".into())),
        };
        let target = if p.target.is_empty() {
            None
        } else {
            Some(Target::parse(&p.target)?)
        };
        let merchant = if p.merchant.is_empty() {
            None
        } else {
            Some(CATALOG.merchant(&p.merchant)?)
        };
        require(
            p.parts <= 49 && p.coal <= 2 && p.iron <= 2 && p.beer <= 2,
            "pending counters out of bounds",
        )?;
        match kind {
            None => require(
                target.is_none()
                    && merchant.is_none()
                    && p.parts == 0
                    && p.coal == 0
                    && p.iron == 0
                    && p.beer == 0,
                "idle action has pending fields",
            )?,
            Some(Kind::Build) => require(
                matches!(target, Some(Target::Slot(_)))
                    && merchant.is_none()
                    && p.parts == 1
                    && p.beer == 0,
                "build pending fields",
            )?,
            Some(Kind::Network) => require(
                matches!(target, Some(Target::Link(_)))
                    && merchant.is_none()
                    && (1..=2).contains(&p.parts)
                    && p.iron == 0,
                "network pending fields",
            )?,
            Some(Kind::Sell) => require(
                matches!(target, Some(Target::Slot(_)))
                    && merchant.is_some()
                    && p.parts > 0
                    && p.coal == 0
                    && p.iron == 0,
                "sell pending fields",
            )?,
            Some(Kind::Develop) => require(
                target.is_none()
                    && merchant.is_none()
                    && (1..=2).contains(&p.parts)
                    && p.coal == 0
                    && p.beer == 0,
                "develop pending fields",
            )?,
            _ => require(
                target.is_none()
                    && merchant.is_none()
                    && p.parts == 1
                    && p.coal == 0
                    && p.iron == 0
                    && p.beer == 0,
                "simple action pending fields",
            )?,
        }
        Ok(Self {
            kind,
            target,
            merchant,
            parts: p.parts,
            coal: p.coal,
            iron: p.iron,
            beer: p.beer,
        })
    }
    fn export(self) -> Pending {
        Pending {
            kind: self.kind.map(|k| k.name().into()).unwrap_or_default(),
            target: self.target.map(Target::name).unwrap_or_default(),
            merchant: self
                .merchant
                .map(|id| CATALOG.merchants[id.0].id.clone())
                .unwrap_or_default(),
            parts: self.parts,
            coal: self.coal,
            iron: self.iron,
            beer: self.beer,
        }
    }
}

fn map_cards<A, B>(
    c: &Cards<A>,
    mut f: impl FnMut(&A) -> Result<B, CoreError>,
) -> Result<Cards<B>, CoreError> {
    let mut hands = std::array::from_fn(|_| vec![]);
    let mut discards = std::array::from_fn(|_| vec![]);
    for i in 0..4 {
        hands[i] = c.hands[i].iter().map(&mut f).collect::<Result<_, _>>()?;
        discards[i] = c.discards[i].iter().map(&mut f).collect::<Result<_, _>>()?;
    }
    Ok(Cards {
        hands,
        discards,
        discard_starts: c.discard_starts,
        deck: c.deck.iter().map(&mut f).collect::<Result<_, _>>()?,
        wild_industry: c
            .wild_industry
            .iter()
            .map(&mut f)
            .collect::<Result<_, _>>()?,
        wild_location: c
            .wild_location
            .iter()
            .map(&mut f)
            .collect::<Result<_, _>>()?,
    })
}
fn counts<'a>(v: impl IntoIterator<Item = &'a CardId>) -> Vec<usize> {
    let mut result = vec![0; CATALOG.card_names.len()];
    for id in v {
        result[id.0] += 1;
    }
    result
}

/// All mutable data and derived indexes are owned. Catalog sharing is read-only.
#[derive(Debug, Clone)]
pub struct NativeState {
    context: Context,
    current_player: Seat,
    round_pending: bool,
    debt_pending: bool,
    settlement_debts: [i32; 4],
    order: [Seat; 4],
    players: [PlayerState; 4],
    tiles: Vec<TileState>,
    slots: Vec<Option<TileId>>,
    links: Vec<Option<Seat>>,
    merchants: Vec<MerchantState>,
    removed: Vec<TileId>,
    retired: Vec<TileId>,
    cards: Cards<CardId>,
    coal_market: u8,
    iron_market: u8,
    pending: PendingState,
    history: Vec<HistoryState>,
    chance: Option<Chance<CardId>>,
    indexes: Indexes,
}
/// Opaque save point: callers cannot alias or mutate its internals.
#[derive(Debug, Clone)]
pub struct StateSnapshot(NativeState);

impl NativeState {
    pub fn context(&self) -> &Context {
        &self.context
    }
    pub fn import(d: &StateDto) -> Result<Self, CoreError> {
        if d.format != STATE_FORMAT
            || d.rules != TARGET_RULES
            || d.context.rules != TARGET_RULES
            || d.context.protocol != "0.2.0"
        {
            return Err(CoreError::UnsupportedContract(
                "state format/rules/protocol".into(),
            ));
        }
        let c = &d.context;
        require(
            (1..=8).contains(&c.round) && (1..=4).contains(&c.turn) && c.actions_remaining <= 2,
            "turn context bounds",
        )?;
        require(
            !(c.era == Era::CanalEra && c.round == 1 && c.actions_remaining > 1),
            "first Canal round has only one action",
        )?;
        require(
            c.phase != Phase::Chance,
            "chance is an explicit flag, not a host phase",
        )?;
        require(
            c.terminal == (c.phase == Phase::Terminal) && c.chance == d.chance.is_some(),
            "phase flags disagree",
        )?;
        require(
            if c.terminal || c.chance {
                c.actor == -1
            } else {
                (0..4).contains(&c.actor)
            },
            "acting seat disagrees with phase",
        )?;
        let mut order = d.order.map(u8::from);
        order.sort();
        require(
            order == [0, 1, 2, 3],
            "turn order must contain four distinct seats",
        )?;
        require(
            d.order[usize::from(c.turn) - 1] == c.turn_owner,
            "turn owner disagrees with order",
        )?;
        require(
            d.slots.len() == CATALOG.slots.len()
                && d.links.len() == CATALOG.links.len()
                && d.merchants.len() == CATALOG.merchants.len(),
            "board topology size",
        )?;
        require(d.history.len() <= 8192, "history too long")?;
        require(
            d.coal_market <= 14 && d.iron_market <= 10,
            "market capacity",
        )?;
        let mut tiles = vec![TileState::default(); CATALOG.tiles.len()];
        let mut used = vec![false; tiles.len()];
        let mut take_tile = |t: &Tile| -> Result<TileId, CoreError> {
            let id = CATALOG.validate_tile(t)?;
            require(!used[id.0], "duplicate tile identity")?;
            used[id.0] = true;
            tiles[id.0] = TileState {
                flipped: t.flipped,
                resource: t.resource,
            };
            Ok(id)
        };
        let mut players = Vec::with_capacity(4);
        for (i, p) in d.players.iter().enumerate() {
            require(seat(p.seat) == i, "players must be in stable seat order")?;
            require(p.income_marker <= 100, "income marker bounds")?;
            require(
                p.income == CATALOG.incomes[usize::from(p.income_marker)],
                "derived income mismatch",
            )?;
            require(
                p.cash >= 0 && p.spent >= 0 && p.bonus_vp >= 0 && p.link_tiles <= 14 && p.debt >= 0,
                "player counters out of bounds",
            )?;
            // Preserve the reference's signed VP field, including authored positions.
            let mut mat = std::array::from_fn(|_| vec![]);
            for (j, m) in p.mat.iter().enumerate() {
                require(m.industry == Industry::ALL[j], "mat industry order")?;
                let mut previous = None;
                for t in &m.tiles {
                    require(
                        t.seat == p.seat && t.industry == m.industry,
                        "mat tile owner/type",
                    )?;
                    let id = take_tile(t)?;
                    require(
                        !t.flipped && t.resource == CATALOG.tiles[id.0].resource,
                        "mat tile dynamic state",
                    )?;
                    require(previous.is_none_or(|x| x < id), "mat tile order")?;
                    previous = Some(id);
                    mat[j].push(id);
                }
            }
            players.push(PlayerState {
                cash: p.cash,
                income_marker: p.income_marker,
                vp: p.vp,
                bonus_vp: p.bonus_vp,
                link_tiles: p.link_tiles,
                spent: p.spent,
                debt: p.debt,
                mat,
            });
        }
        let mut slots = Vec::with_capacity(d.slots.len());
        for (s, def) in d.slots.iter().zip(&CATALOG.slots) {
            require(
                s.id == def.id && s.types == def.types,
                "slot topology differs",
            )?;
            if let Some(t) = &s.tile {
                require(s.types.contains(&t.industry), "tile cannot occupy slot")?;
            }
            slots.push(s.tile.as_ref().map(&mut take_tile).transpose()?);
        }
        let mut removed = Vec::with_capacity(d.removed.len());
        for name in &d.removed {
            let id = CATALOG.tile(name)?;
            require(!used[id.0], "removed tile duplicates live tile")?;
            used[id.0] = true;
            removed.push(id);
        }
        let mut retired = Vec::with_capacity(d.retired.len());
        for name in &d.retired {
            let id = CATALOG.tile(name)?;
            require(!used[id.0], "retired tile duplicates another zone")?;
            used[id.0] = true;
            retired.push(id);
        }
        require(
            d.retired.windows(2).all(|w| w[0] < w[1]),
            "retired tile order/duplicates",
        )?;
        require(used.iter().all(|x| *x), "tile conservation: missing tile")?;
        require(
            d.removed.windows(2).all(|w| w[0] < w[1]),
            "removed tile order/duplicates",
        )?;
        let mut links = Vec::with_capacity(d.links.len());
        let mut placed = [0u8; 4];
        for (l, def) in d.links.iter().zip(&CATALOG.links) {
            require(
                l.id == def.id && l.regions == def.regions && l.era == def.era,
                "link topology differs",
            )?;
            if let Some(owner) = l.owner {
                require(
                    matches!(
                        (c.era, l.era),
                        (Era::CanalEra, LinkEra::Canal) | (Era::RailEra, LinkEra::Rail)
                    ),
                    "link belongs to inactive era",
                )?;
                placed[seat(owner)] += 1;
            }
            links.push(l.owner);
        }
        for (i, p) in players.iter().enumerate() {
            require(
                u16::from(p.link_tiles) + u16::from(placed[i]) == 14,
                "link tile conservation",
            )?;
        }
        let mut merchants = Vec::with_capacity(d.merchants.len());
        let mut merchant_types = Vec::new();
        for (m, def) in d.merchants.iter().zip(&CATALOG.merchants) {
            require(
                m.id == def.id && m.bonus == def.bonus && m.beer <= 1,
                "merchant identity/bonus/beer",
            )?;
            require(
                m.types.as_ref().is_some_and(|types| !types.is_empty()) || m.beer == 0,
                "empty merchant has beer",
            )?;
            merchants.push(MerchantState {
                types: m.types.clone().unwrap_or_default(),
                defined: m.types.is_some(),
                beer: m.beer,
            });
            merchant_types.push(m.types.clone().unwrap_or_default());
        }
        merchant_types.sort();
        let mut expected = CATALOG.merchant_types.clone();
        expected.sort();
        require(merchant_types == expected, "merchant tile conservation")?;
        let cards = map_cards(&d.cards, |s| CATALOG.card(s))?;
        for i in 0..4 {
            require(
                cards.hands[i].len() <= 8
                    && cards.discards[i].len() <= 128
                    && cards.discard_starts[i] <= cards.discards[i].len(),
                "card zone bounds",
            )?;
            require(
                c.era != Era::CanalEra || cards.discard_starts[i] == 0,
                "Canal discard prefix",
            )?;
        }
        require(
            cards
                .deck
                .iter()
                .all(|id| !CATALOG.card_names[id.0].starts_with("wild:")),
            "wild card in deck",
        )?;
        require(
            cards
                .wild_industry
                .iter()
                .all(|id| CATALOG.card_names[id.0] == "wild:industry")
                && cards
                    .wild_location
                    .iter()
                    .all(|id| CATALOG.card_names[id.0] == "wild:location"),
            "wild pile type",
        )?;
        // Pre-Rail discard prefixes are public history aliases, not physical cards.
        let live = cards
            .hands
            .iter()
            .flatten()
            .chain(cards.deck.iter())
            .chain(cards.wild_industry.iter())
            .chain(cards.wild_location.iter())
            .chain((0..4).flat_map(|i| cards.discards[i][cards.discard_starts[i]..].iter()));
        require(
            counts(live)
                == CATALOG
                    .card_counts
                    .iter()
                    .map(|n| usize::from(*n))
                    .collect::<Vec<_>>(),
            "card conservation",
        )?;
        let pending = PendingState::import(&d.pending)?;
        let owing = players.iter().any(|p| p.debt > 0);
        let liquidation_actor = (0..4).find(|i| {
            d.settlement_debts[*i] > 0
                && slots
                    .iter()
                    .flatten()
                    .any(|id| seat(CATALOG.tiles[id.0].seat) == *i)
        });
        require(
            d.settlement_debts.iter().all(|n| *n >= 0),
            "negative settlement debt",
        )?;
        require(
            d.debt_pending || d.settlement_debts == [0; 4],
            "inactive settlement has debt",
        )?;
        for (i, p) in players.iter().enumerate() {
            require(
                p.debt
                    == if c.phase == Phase::Debt {
                        d.settlement_debts[i]
                    } else {
                        0
                    },
                "public debt differs from settlement",
            )?;
        }
        let payment = pending.coal > 0 || pending.iron > 0 || pending.beer > 0;
        require(
            owing == (c.phase == Phase::Debt),
            "debt amounts disagree with phase",
        )?;
        match c.phase {
            Phase::Action | Phase::Debt | Phase::Terminal => {
                require(pending.kind.is_none(), "idle phase has pending action")?
            }
            Phase::Payment => require(
                pending.kind.is_some() && payment,
                "payment phase has no unpaid resources",
            )?,
            Phase::Continuation => require(
                pending.kind.is_some() && !payment,
                "continuation resources/action mismatch",
            )?,
            Phase::Bonus => require(
                pending.kind == Some(Kind::Sell)
                    && !payment
                    && pending
                        .merchant
                        .is_some_and(|id| CATALOG.merchants[id.0].bonus == Bonus::Develop),
                "bonus state mismatch",
            )?,
            Phase::Chance => unreachable!(),
        }
        if c.phase == Phase::Debt {
            require(
                d.round_pending
                    && d.debt_pending
                    && c.actions_remaining == 0
                    && liquidation_actor == Some(c.actor as usize),
                "debt flow flags/actor",
            )?;
        } else if !c.terminal && !c.chance {
            require(
                c.actor == u8::from(c.turn_owner) as i8
                    && d.current_player == c.turn_owner
                    && c.actions_remaining > 0
                    && !d.round_pending
                    && !d.debt_pending,
                "ordinary flow flags/actor",
            )?;
        }
        if let Some(Target::Slot(id)) = pending.target {
            require(
                slots[id.0].is_some_and(|t| CATALOG.tiles[t.0].seat == c.turn_owner),
                "pending action target ownership",
            )?;
        }
        if let Some(Target::Link(id)) = pending.target {
            require(links[id.0] == Some(c.turn_owner), "pending link ownership")?;
        }
        let chance = d
            .chance
            .as_ref()
            .map(|ch| -> Result<_, CoreError> {
                require(
                    c.phase == Phase::Action
                        && c.actions_remaining == 0
                        && ch.needed > 0
                        && ch.picked.len() < usize::from(ch.needed)
                        && !ch.pool.is_empty(),
                    "chance progress bounds/phase",
                )?;
                let pool = ch
                    .pool
                    .iter()
                    .map(|s| CATALOG.card(s))
                    .collect::<Result<Vec<_>, _>>()?;
                let picked = ch
                    .picked
                    .iter()
                    .map(|s| CATALOG.card(s))
                    .collect::<Result<Vec<_>, _>>()?;
                require(
                    pool.windows(2).all(|w| w[0] <= w[1]),
                    "chance pool ordering",
                )?;
                let actual = counts(pool.iter().chain(&picked));
                match ch.kind {
                    ChanceKind::Draw => {
                        require(
                            ch.needed as usize == 8 - cards.hands[seat(d.current_player)].len()
                                && ch.needed as usize <= cards.deck.len()
                                && d.current_player == c.turn_owner
                                && !d.round_pending
                                && !d.debt_pending,
                            "draw continuation",
                        )?;
                        require(actual == counts(&cards.deck), "draw pool conservation")?;
                    }
                    ChanceKind::RailShuffle => {
                        require(
                            c.era == Era::CanalEra
                                && c.round == 8
                                && c.turn == 4
                                && d.round_pending
                                && liquidation_actor.is_none()
                                && ch.needed == 64,
                            "rail shuffle continuation",
                        )?;
                        let expected = CATALOG
                            .card_names
                            .iter()
                            .zip(&CATALOG.card_counts)
                            .map(|(name, n)| {
                                if name.starts_with("wild:") {
                                    0
                                } else {
                                    usize::from(*n)
                                }
                            })
                            .collect::<Vec<_>>();
                        require(actual == expected, "rail shuffle card conservation")?;
                        let available = counts(
                            cards
                                .deck
                                .iter()
                                .chain(cards.discards.iter().flatten())
                                .filter(|id| !CATALOG.card_names[id.0].starts_with("wild:")),
                        );
                        require(
                            actual == available,
                            "rail shuffle pool references held cards",
                        )?;
                    }
                }
                Ok(Chance {
                    kind: ch.kind,
                    needed: ch.needed,
                    pool,
                    picked,
                })
            })
            .transpose()?;
        if c.terminal {
            require(
                c.era == Era::RailEra
                    && c.round == 8
                    && c.turn == 4
                    && c.actions_remaining == 0
                    && !c.chance
                    && !d.round_pending
                    && !d.debt_pending,
                "terminal flow",
            )?;
        }
        let mut state = Self {
            context: c.clone(),
            current_player: d.current_player,
            round_pending: d.round_pending,
            debt_pending: d.debt_pending,
            settlement_debts: d.settlement_debts,
            order: d.order,
            players: players.try_into().expect("four validated players"),
            tiles,
            slots,
            links,
            merchants,
            removed,
            retired,
            cards,
            coal_market: d.coal_market,
            iron_market: d.iron_market,
            pending,
            history: d
                .history
                .iter()
                .map(HistoryState::import)
                .collect::<Result<_, _>>()?,
            chance,
            indexes: Indexes::empty(),
        };
        require(
            d.result == state.result(),
            "terminal result/ranking differs",
        )?;
        state.rebuild_indexes();
        Ok(state)
    }

    fn tile(&self, id: TileId) -> Tile {
        let mut t = CATALOG.tiles[id.0].clone();
        t.flipped = self.tiles[id.0].flipped;
        t.resource = self.tiles[id.0].resource;
        t
    }
    pub fn result(&self) -> Option<ResultDto> {
        if !self.context.terminal {
            return None;
        }
        let keys = self
            .players
            .each_ref()
            .map(|p| (p.vp, CATALOG.incomes[usize::from(p.income_marker)], p.cash));
        let best = *keys.iter().max().expect("four seats");
        let winners = keys.iter().filter(|k| **k == best).count();
        Some(ResultDto {
            objective: "win-share/1.0.0".into(),
            seats: std::array::from_fn(|i| ResultSeat {
                seat: Seat::try_from(i as u8).unwrap(),
                vp: keys[i].0,
                income: keys[i].1,
                cash: keys[i].2,
                rank: 1 + keys.iter().filter(|k| **k > keys[i]).count() as u8,
                utility: if keys[i] == best {
                    1.0 / winners as f64
                } else {
                    0.0
                },
            }),
        })
    }
    pub fn export(&self) -> StateDto {
        StateDto {
            format: STATE_FORMAT.into(),
            rules: TARGET_RULES.into(),
            context: self.context.clone(),
            current_player: self.current_player,
            round_pending: self.round_pending,
            debt_pending: self.debt_pending,
            settlement_debts: self.settlement_debts,
            order: self.order,
            players: std::array::from_fn(|i| {
                let p = &self.players[i];
                Player {
                    seat: Seat::try_from(i as u8).unwrap(),
                    cash: p.cash,
                    income_marker: p.income_marker,
                    income: CATALOG.incomes[usize::from(p.income_marker)],
                    vp: p.vp,
                    bonus_vp: p.bonus_vp,
                    link_tiles: p.link_tiles,
                    spent: p.spent,
                    debt: p.debt,
                    mat: std::array::from_fn(|j| Mat {
                        industry: Industry::ALL[j],
                        tiles: p.mat[j].iter().map(|id| self.tile(*id)).collect(),
                    }),
                }
            }),
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
                .map(|(d, owner)| Link {
                    id: d.id.clone(),
                    regions: d.regions.clone(),
                    era: d.era,
                    owner: *owner,
                })
                .collect(),
            merchants: CATALOG
                .merchants
                .iter()
                .zip(&self.merchants)
                .map(|(d, m)| Merchant {
                    id: d.id.clone(),
                    bonus: d.bonus,
                    types: m.defined.then(|| m.types.clone()),
                    beer: m.beer,
                })
                .collect(),
            removed: self
                .removed
                .iter()
                .map(|id| CATALOG.tiles[id.0].id.clone())
                .collect(),
            retired: self
                .retired
                .iter()
                .map(|id| CATALOG.tiles[id.0].id.clone())
                .collect(),
            cards: map_cards(&self.cards, |id| Ok(CATALOG.card_names[id.0].clone()))
                .expect("infallible export"),
            coal_market: self.coal_market,
            iron_market: self.iron_market,
            pending: self.pending.export(),
            history: self.history.iter().map(HistoryState::export).collect(),
            chance: self.chance.as_ref().map(|ch| Chance {
                kind: ch.kind,
                needed: ch.needed,
                pool: ch
                    .pool
                    .iter()
                    .map(|id| CATALOG.card_names[id.0].clone())
                    .collect(),
                picked: ch
                    .picked
                    .iter()
                    .map(|id| CATALOG.card_names[id.0].clone())
                    .collect(),
            }),
            result: self.result(),
        }
    }
    fn rebuild_indexes(&mut self) {
        self.indexes = Indexes::build(self);
    }
    pub fn indexes(&self) -> IndexSnapshot {
        self.indexes.export()
    }
    pub fn snapshot(&self) -> StateSnapshot {
        StateSnapshot(self.clone())
    }
    pub fn restore(&mut self, snapshot: &StateSnapshot) {
        self.clone_from(&snapshot.0);
        self.rebuild_indexes();
    }

    /// Exercise every mutable container on a private fork, then restore it.
    pub fn clone_isolated(&self) -> bool {
        let original = self.export();
        let original_indexes = self.indexes();
        let saved = self.snapshot();
        let mut fork = self.clone();
        fork.context.revision = fork.context.revision.wrapping_add(1);
        fork.current_player = Seat::Zero;
        fork.round_pending = !fork.round_pending;
        fork.debt_pending = !fork.debt_pending;
        fork.settlement_debts.fill(0);
        fork.order.reverse();
        for p in &mut fork.players {
            p.cash = p.cash.wrapping_add(1);
            for mat in &mut p.mat {
                mat.clear();
            }
        }
        for t in &mut fork.tiles {
            t.flipped = !t.flipped;
            t.resource = 0;
        }
        fork.slots.fill(None);
        fork.links.fill(None);
        fork.removed.clear();
        fork.retired.clear();
        fork.history.clear();
        for m in &mut fork.merchants {
            m.beer = 0;
            m.types.clear();
            m.defined = !m.defined;
        }
        for h in &mut fork.cards.hands {
            h.clear();
        }
        for d in &mut fork.cards.discards {
            d.clear();
        }
        fork.cards.deck.clear();
        fork.cards.wild_industry.clear();
        fork.cards.wild_location.clear();
        fork.cards.discard_starts.fill(0);
        fork.pending.parts = 0;
        fork.coal_market = 0;
        fork.iron_market = 0;
        if let Some(ch) = &mut fork.chance {
            ch.pool.clear();
            ch.picked.clear();
        }
        fork.rebuild_indexes();
        let independent = self.export() == original
            && self.indexes() == original_indexes
            && saved.0.export() == original;
        fork.restore(&saved);
        independent && fork.export() == original && fork.indexes() == original_indexes
    }
}
