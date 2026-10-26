//! Exact permitted Features-4, derived facts and v11.2 auxiliary inputs.
//! Runtime encoding consumes typed observations/choices; JSON is diagnostic only.
mod board;
mod derived;
mod indices;
pub mod layout;
mod measure;
pub mod session;
use crate::{
    state::{
        catalog::CATALOG,
        dto::{Bonus, Choice, Industry, Link, Merchant, Slot, Tile},
        observation::{Observation, ViewPlayer},
    },
    types::{Era, Phase, Seat},
    CoreError,
};
use board::{income, market_buy, settlement, Board};
use layout::{kind_index, target_index, LAYOUT, META};
use measure::Measure;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fmt::Write,
};

pub const FORMAT: &str = "brass-native-features/1.0.0";
pub const FEATURE_VERSION: &str = "permitted-features/4.0.0";
pub const STATE_DIM: usize = 3128;
pub const ACTION_DIM: usize = 327;
pub const MAX_CANDIDATES: usize = 8192;
use crate::state::dto::Kind;
fn invalid(message: &str) -> CoreError {
    CoreError::Invalid(message.into())
}
fn check(value: bool, message: &str) -> Result<(), CoreError> {
    if value {
        Ok(())
    } else {
        Err(invalid(message))
    }
}
fn seat(s: Seat) -> usize {
    u8::from(s) as usize
}
fn relative(s: usize, observer: usize) -> usize {
    (s + 4 - observer) % 4
}

/// Model/reward/profile identity is carried through, never inferred from equal tensors.
/// L09 binds these IDs and max_cost to an admitted model manifest before inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeatureContext {
    pub model_id: String,
    pub reward_id: String,
    pub profile_id: String,
    pub style: [f64; 4],
    pub intent: [f64; 8],
    /// Stable seat order, as consumed by UtilityNet; not observer-relative.
    pub remaining: [f64; 4],
    pub max_cost: f64,
    /// Explicit critic diagnostics only; deployment default is all unknown.
    pub stable_descriptors: Option<Vec<Vec<f32>>>,
}
impl FeatureContext {
    pub fn validate(&self) -> Result<(), CoreError> {
        for id in [&self.model_id, &self.reward_id, &self.profile_id] {
            check(
                !id.is_empty() && id.len() <= 512 && !id.chars().any(char::is_control),
                "invalid feature context identity",
            )?;
        }
        check(
            self.style
                .iter()
                .chain(&self.intent)
                .all(|v| v.is_finite() && (-1. ..=1.).contains(v)),
            "invalid personality feature context",
        )?;
        check(
            self.max_cost.is_finite()
                && (0. ..=1.).contains(&self.max_cost)
                && self
                    .remaining
                    .iter()
                    .all(|v| v.is_finite() && (0. ..=self.max_cost).contains(v)),
            "invalid remaining utility budget",
        )?;
        if let Some(d) = &self.stable_descriptors {
            check(
                d.len() == 4
                    && d.iter().all(|r| {
                        r.len() == 34
                            && r.iter().all(|v| v.is_finite() && (0. ..=1.).contains(v))
                            && r[..5].iter().all(|v| *v == 0. || *v == 1.)
                            && r[..5].iter().sum::<f32>() == 1.
                            && (r[4] != 1. || r[5..].iter().all(|v| *v == 0.))
                    }),
                "invalid seat descriptors",
            )?;
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct FeatureStats {
    pub effect_hits: usize,
    pub effect_misses: usize,
    pub state_capacity: usize,
    pub action_capacity: usize,
}
#[derive(Debug, Default)]
pub struct FeatureTensors {
    pub state: Vec<f32>,
    pub actions: Vec<f32>,
    pub legal_mask: Vec<bool>,
    pub stable_to_relative: [i64; 4],
    pub remaining: [f32; 4],
    pub descriptors: Vec<f32>,
    pub stats: FeatureStats,
}
/// Owned transport copy only; production inference borrows FeatureTensors instead.
#[derive(Debug, Serialize)]
pub struct FeatureReply {
    pub format: &'static str,
    pub feature_version: &'static str,
    pub schema_sha256: &'static str,
    pub group_layout: &'static str,
    pub entity_layout: &'static str,
    pub state_dim: usize,
    pub action_dim: usize,
    pub choice_ids: Vec<String>,
    pub candidate_offsets: [usize; 2],
    pub total_candidates: usize,
    pub state: Vec<f32>,
    pub actions: Vec<f32>,
    pub legal_mask: Vec<bool>,
    pub stable_to_relative: [i64; 4],
    pub remaining: [f32; 4],
    pub descriptors: Vec<f32>,
    pub context: FeatureContext,
    pub stats: FeatureStats,
}
impl FeatureReply {
    pub fn page(
        t: &FeatureTensors,
        choices: &[Choice],
        context: &FeatureContext,
        start: usize,
        limit: usize,
    ) -> Result<Self, CoreError> {
        check(
            (1..=64).contains(&limit) && start < choices.len(),
            "invalid feature page",
        )?;
        let end = (start + limit).min(choices.len());
        Ok(Self {
            format: FORMAT,
            feature_version: FEATURE_VERSION,
            schema_sha256: &META.feature_schema_sha256,
            group_layout: META.group_layout["identity"]
                .as_str()
                .expect("generated group identity"),
            entity_layout: META.entity_layout["identity"]
                .as_str()
                .expect("generated entity identity"),
            state_dim: STATE_DIM,
            action_dim: ACTION_DIM,
            choice_ids: choices[start..end].iter().map(|c| c.id.clone()).collect(),
            candidate_offsets: [start, end],
            total_candidates: choices.len(),
            state: t.state.clone(),
            actions: t.actions[start * ACTION_DIM..end * ACTION_DIM].to_vec(),
            legal_mask: t.legal_mask[start..end].to_vec(),
            stable_to_relative: t.stable_to_relative,
            remaining: t.remaining,
            descriptors: t.descriptors.clone(),
            context: context.clone(),
            stats: t.stats.clone(),
        })
    }
}
#[derive(Debug, Clone, Copy)]
struct Candidate {
    kind: Kind,
    target: Option<usize>,
    industry: Option<Industry>,
    merchant: Option<usize>,
    cards: [usize; 4],
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct EffectKey(u8, Option<usize>, Option<usize>, Option<usize>);
type OpportunityKey = (u8, Option<usize>, Option<usize>, [usize; 4]);
impl Candidate {
    fn effect_key(self) -> EffectKey {
        EffectKey(
            self.kind as u8,
            self.target,
            self.industry.map(Industry::index),
            self.merchant,
        )
    }
    fn slot(self) -> Result<usize, CoreError> {
        self.target
            .filter(|n| *n < 49)
            .ok_or_else(|| invalid("candidate requires slot"))
    }
    fn industry(self) -> Result<usize, CoreError> {
        self.industry
            .map(Industry::index)
            .ok_or_else(|| invalid("candidate requires industry"))
    }
    fn merchant(self) -> Result<usize, CoreError> {
        self.merchant
            .ok_or_else(|| invalid("candidate requires merchant"))
    }
}
struct SemanticCheck<'a>(&'a str);
impl Write for SemanticCheck<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0 = self.0.strip_prefix(s).ok_or(std::fmt::Error)?;
        Ok(())
    }
}
fn prepare(c: &Choice) -> Result<Candidate, CoreError> {
    let mut id = SemanticCheck(&c.id);
    write!(
        id,
        "{}|{}|",
        c.kind.name(),
        c.target.as_deref().unwrap_or("")
    )
    .map_err(|_| invalid("semantic choice ID mismatch"))?;
    if let Some(i) = c.industry {
        write!(id, "{i:?}").map_err(|_| invalid("semantic choice ID mismatch"))?;
    }
    write!(
        id,
        "|{}|{}|",
        c.card.as_deref().unwrap_or(""),
        c.merchant.as_deref().unwrap_or("")
    )
    .map_err(|_| invalid("semantic choice ID mismatch"))?;
    if let Some(cards) = &c.cards {
        for (i, card) in cards.iter().enumerate() {
            if i > 0 {
                id.write_char(',')
                    .map_err(|_| invalid("semantic choice ID mismatch"))?;
            }
            id.write_str(card)
                .map_err(|_| invalid("semantic choice ID mismatch"))?;
        }
    }
    check(
        id.0.is_empty() && c.id.len() <= 4096,
        "semantic choice ID mismatch",
    )?;
    let mut cards = [usize::MAX; 4];
    let mut n = 0;
    for card in c.card.iter().chain(c.cards.iter().flatten()) {
        check(n < 4, "too many candidate cards")?;
        cards[n] = *LAYOUT
            .cards
            .get(card.as_str())
            .ok_or_else(|| invalid("unknown candidate card"))?;
        n += 1;
    }
    let result = Candidate {
        kind: c.kind,
        target: target_index(c.target.as_deref().unwrap_or(""))?,
        industry: c.industry,
        merchant: c
            .merchant
            .as_ref()
            .map(|m| {
                LAYOUT
                    .merchants
                    .get(m.as_str())
                    .copied()
                    .ok_or_else(|| invalid("unknown candidate merchant"))
            })
            .transpose()?,
        cards,
    };
    match result.kind {
        Kind::Build => {
            result.slot()?;
            result.industry()?;
        }
        Kind::Develop | Kind::BonusDevelop => {
            result.industry()?;
        }
        Kind::Sell => {
            result.slot()?;
            result.merchant()?;
        }
        Kind::Liquidate => {
            result.slot()?;
        }
        Kind::Network => check(
            result.target.is_some_and(|t| (49..118).contains(&t)),
            "network requires link",
        )?,
        Kind::Coal | Kind::Iron | Kind::Beer => {
            check(result.target.is_some(), "payment requires source")?
        }
        Kind::Scout => check(n == 3, "scout requires three cards")?,
        _ => {}
    }
    Ok(result)
}

#[derive(Default)]
pub struct FeatureEncoder {
    tensors: FeatureTensors,
    valid: bool,
    candidates: Vec<Candidate>,
    effects: HashMap<EffectKey, usize>,
    dynamic: Vec<f32>,
    ids: Vec<usize>,
    opportunities: HashSet<OpportunityKey>,
}
impl FeatureEncoder {
    pub fn tensors(&self) -> Option<&FeatureTensors> {
        self.valid.then_some(&self.tensors)
    }
    pub fn encode(
        &mut self,
        view: &Observation,
        choices: &[Choice],
        context: &FeatureContext,
        checkpoint: &mut impl FnMut() -> Result<(), CoreError>,
    ) -> Result<&FeatureTensors, CoreError> {
        self.valid = false;
        crate::work::checkpoint(crate::work::Stage::Encoding)?;
        checkpoint()?;
        context.validate()?;
        check(
            !choices.is_empty() && choices.len() <= MAX_CANDIDATES,
            "feature candidate count",
        )?;
        let board = Board::new(view)?;
        self.candidates.clear();
        self.effects.clear();
        self.dynamic.clear();
        self.ids.clear();
        self.opportunities.clear();
        self.ids.extend(0..choices.len());
        self.ids
            .sort_unstable_by(|a, b| choices[*a].id.cmp(&choices[*b].id));
        check(
            !self
                .ids
                .windows(2)
                .any(|w| choices[w[0]].id == choices[w[1]].id),
            "duplicate feature choices",
        )?;
        for choice in choices {
            crate::work::checkpoint(crate::work::Stage::Encoding)?;
            checkpoint()?;
            self.candidates.push(prepare(choice)?);
        }
        let t = &mut self.tensors;
        t.state.clear();
        t.actions.clear();
        t.legal_mask.clear();
        t.descriptors.clear();
        t.stats = FeatureStats::default();
        base_state(&board, &mut t.state)?;
        let facts = derived::state(&board, &self.candidates, &mut self.opportunities)?;
        for s in &META.state_inputs {
            facts[s.index].emit(s, &mut t.state)?;
        }
        t.state.extend(
            context
                .style
                .iter()
                .chain(&context.intent)
                .map(|x| *x as f32),
        );
        let observer = seat(view.seat);
        t.stable_to_relative = std::array::from_fn(|s| relative(s, observer) as i64);
        t.remaining = context.remaining.map(|v| v as f32);
        for r in 0..4 {
            if let Some(d) = &context.stable_descriptors {
                t.descriptors.extend_from_slice(&d[(observer + r) % 4]);
            } else {
                t.descriptors
                    .extend((0..34).map(|j| if j == 4 { 1. } else { 0. }));
            }
        }
        for candidate in &self.candidates {
            crate::work::checkpoint(crate::work::Stage::Encoding)?;
            checkpoint()?;
            base_action(*candidate, &mut t.actions);
            let key = candidate.effect_key();
            let offset = if let Some(offset) = self.effects.get(&key) {
                t.stats.effect_hits += 1;
                *offset
            } else {
                let facts = derived::action(&board, *candidate)?;
                let offset = self.dynamic.len();
                for s in &META.action_inputs {
                    facts[s.index].emit(s, &mut self.dynamic)?;
                }
                self.effects.insert(key, offset);
                t.stats.effect_misses += 1;
                offset
            };
            t.actions
                .extend_from_slice(&self.dynamic[offset..offset + 143]);
            t.legal_mask.push(true);
        }
        check(
            t.state.len() == STATE_DIM
                && t.actions.len() == choices.len() * ACTION_DIM
                && t.descriptors.len() == 136,
            "native feature dimensions",
        )?;
        check(
            t.state.iter().chain(&t.actions).all(|v| v.is_finite()),
            "nonfinite native features",
        )?;
        crate::work::checkpoint(crate::work::Stage::Encoding)?;
        checkpoint()?;
        t.stats.state_capacity = t.state.capacity();
        t.stats.action_capacity = t.actions.capacity();
        self.valid = true;
        Ok(t)
    }
}
fn one(out: &mut Vec<f32>, index: Option<usize>, width: usize) {
    out.extend((0..width).map(|i| if index == Some(i) { 1. } else { 0. }));
}
fn cards(out: &mut Vec<f32>, items: &[String], scale: f64) -> Result<(), CoreError> {
    let mut counts = [0u32; 27];
    for card in items {
        counts[*LAYOUT
            .cards
            .get(card.as_str())
            .ok_or_else(|| invalid("unknown card count"))?] += 1;
    }
    out.extend(counts.map(|n| (f64::from(n) / scale) as f32));
    Ok(())
}
fn tile(out: &mut Vec<f32>, t: Option<&Tile>, board: bool) {
    let fields = if board {
        &LAYOUT.board_tile_fields
    } else {
        out.push(f32::from(t.is_some()));
        &LAYOUT.tile_fields
    };
    out.extend(
        fields
            .iter()
            .map(|(f, s)| t.map_or(0., |t| (f.value(t) / s) as f32)),
    );
}
fn base_action(c: Candidate, out: &mut Vec<f32>) {
    one(out, Some(kind_index(c.kind)), 14);
    one(out, c.target, 128);
    one(out, c.merchant, 9);
    one(out, c.industry.map(Industry::index), 6);
    let mut count = [0u8; 27];
    for card in c.cards {
        if card != usize::MAX {
            count[card] += 1;
        }
    }
    out.extend(count.map(|n| (f64::from(n) / 3.) as f32));
}
fn base_state(b: &Board<'_>, out: &mut Vec<f32>) -> Result<(), CoreError> {
    let v = b.view;
    let s = &META.schema;
    let o = seat(v.seat);
    let ctx = &v.context;
    one(out, Some(usize::from(ctx.era == Era::RailEra)), 2);
    one(out, s.phases.iter().position(|p| *p == ctx.phase), 6);
    out.extend_from_slice(&[
        f32::from(ctx.round) / 8.,
        f32::from(ctx.turn) / 4.,
        f32::from(ctx.actions_remaining) / 2.,
    ]);
    one(out, Some(relative(seat(ctx.turn_owner), o)), 4);
    for owner in v.order {
        one(out, Some(relative(seat(owner), o)), 4);
    }
    cards(out, &v.own_hand, 8.)?;
    for r in 0..4 {
        let player = (o + r) % 4;
        let p = &v.players[player];
        out.extend(
            LAYOUT
                .player_fields
                .iter()
                .map(|(f, s)| (f.value(p) / s) as f32),
        );
        cards(out, &p.discards, 8.)?;
        for name in ["wild:industry", "wild:location"] {
            out.push(p.known_wild.iter().filter(|c| c.as_str() == name).count() as f32);
        }
        check(
            p.known_wild
                .iter()
                .all(|s| s == "wild:industry" || s == "wild:location"),
            "unknown known-wild card",
        )?;
        for industry in 0..6 {
            let stack = b.mat(player, industry);
            let mut levels = [0u8; 8];
            for t in stack {
                levels[t.level as usize - 1] += 1;
            }
            out.extend(levels.map(|n| f32::from(n) / 4.));
            tile(out, stack.first(), false);
            tile(out, stack.get(1), false);
        }
    }
    for i in 0..49 {
        let t = b.slot(i).tile.as_ref();
        one(out, t.map(|t| relative(seat(t.seat), o)), 4);
        one(out, t.map(|t| t.industry.index()), 6);
        tile(out, t, true);
    }
    for i in 0..69 {
        one(out, b.link(i).owner.map(|p| relative(seat(p), o)), 4);
    }
    for i in 0..9 {
        let m = b.merchant(i);
        out.extend(Industry::ALL.map(|t| {
            m.types
                .as_ref()
                .map_or(0, |ts| ts.iter().filter(|x| **x == t).count()) as f32
        }));
        out.push(f32::from(m.beer));
    }
    out.extend_from_slice(&[
        (f64::from(v.coal_market) / 14.) as f32,
        (f64::from(v.iron_market) / 10.) as f32,
        (f64::from(v.deck_count) / 64.) as f32,
        (f64::from(v.wild_industry_count) / 4.) as f32,
        (f64::from(v.wild_location_count) / 4.) as f32,
    ]);
    let p = &v.pending;
    let kind = if p.kind.is_empty() {
        None
    } else {
        Some(
            s.kinds
                .iter()
                .position(|k| k.name() == p.kind)
                .ok_or_else(|| invalid("unknown pending kind"))?,
        )
    };
    one(out, kind, 14);
    one(out, target_index(&p.target)?, 128);
    one(
        out,
        if p.merchant.is_empty() {
            None
        } else {
            Some(
                *LAYOUT
                    .merchants
                    .get(p.merchant.as_str())
                    .ok_or_else(|| invalid("unknown pending merchant"))?,
            )
        },
        9,
    );
    out.extend([p.parts, p.coal, p.iron, p.beer].map(|n| f32::from(n) / 4.));
    let mut history = [0u32; 72];
    for h in &v.history {
        if let Some(k) = s.history_kinds.iter().position(|k| *k == h.choice.kind) {
            history[relative(seat(h.seat), o) * 18 + usize::from(h.era == Era::RailEra) * 9 + k] +=
                1;
        }
    }
    out.extend(history.map(|n| (f64::from(n) / 32.) as f32));
    check(out.len() == 2527, "base feature layout")
}
