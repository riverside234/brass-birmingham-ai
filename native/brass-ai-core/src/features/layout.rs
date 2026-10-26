//! One-time metadata admission. No JSON parsing or string-built fact keys in encoding.
use super::*;
use serde::Deserialize;
use std::{collections::BTreeMap, sync::LazyLock};

#[derive(Debug, Deserialize)]
pub struct Selection {
    pub index: usize,
    pub scale: f64,
    pub width: usize,
}
#[derive(Debug, Deserialize)]
pub struct Schema {
    pub state_dim: usize,
    pub action_dim: usize,
    pub slots: Vec<String>,
    pub links: Vec<String>,
    pub merchants: Vec<String>,
    pub card_types: Vec<String>,
    pub kinds: Vec<Kind>,
    pub industries: Vec<Industry>,
    pub phases: Vec<Phase>,
    pub history_kinds: Vec<Kind>,
    pub player_fields: BTreeMap<String, f64>,
    pub tile_fields: BTreeMap<String, f64>,
    pub board_tile_fields: BTreeMap<String, f64>,
}
#[derive(Debug, Deserialize)]
pub struct Metadata {
    pub schema: Schema,
    pub feature_schema_sha256: String,
    pub group_layout: serde_json::Value,
    pub entity_layout: serde_json::Value,
    pub state_inputs: Vec<Selection>,
    pub action_inputs: Vec<Selection>,
}
pub static META: LazyLock<Metadata> = LazyLock::new(|| {
    let m: Metadata = serde_json::from_str(include_str!("../../data/feature-layout.json"))
        .expect("generated, checked feature metadata");
    assert_eq!(
        (m.schema.state_dim, m.schema.action_dim),
        (STATE_DIM, ACTION_DIM)
    );
    assert_eq!(
        (
            m.schema.slots.len(),
            m.schema.links.len(),
            m.schema.merchants.len()
        ),
        (49, 69, 9)
    );
    assert_eq!(m.schema.card_types.len(), 27);
    assert_eq!(m.state_inputs.iter().map(|s| s.width).sum::<usize>(), 589);
    assert_eq!(m.action_inputs.iter().map(|s| s.width).sum::<usize>(), 143);
    assert!(m
        .state_inputs
        .iter()
        .all(|s| s.index < 326 && [3, 7].contains(&s.width) && s.scale > 0.));
    assert!(m
        .action_inputs
        .iter()
        .all(|s| s.index < 52 && [3, 7].contains(&s.width) && s.scale > 0.));
    m
});

#[derive(Debug, Clone, Copy)]
pub enum TileField {
    Beer,
    CanalOnly,
    Coal,
    Cost,
    Developable,
    Flipped,
    Income,
    Iron,
    Level,
    LinkVp,
    RailOnly,
    Resource,
    Vp,
}
impl TileField {
    fn parse(name: &str) -> Self {
        match name {
            "beer" => Self::Beer,
            "canal_only" => Self::CanalOnly,
            "coal" => Self::Coal,
            "cost" => Self::Cost,
            "developable" => Self::Developable,
            "flipped" => Self::Flipped,
            "income" => Self::Income,
            "iron" => Self::Iron,
            "level" => Self::Level,
            "link_vp" => Self::LinkVp,
            "rail_only" => Self::RailOnly,
            "resource" => Self::Resource,
            "vp" => Self::Vp,
            _ => panic!("unknown generated tile field"),
        }
    }
    pub fn value(self, t: &Tile) -> f64 {
        match self {
            Self::Beer => t.beer.into(),
            Self::CanalOnly => u8::from(t.canal_only).into(),
            Self::Coal => t.coal.into(),
            Self::Cost => t.cost.into(),
            Self::Developable => u8::from(t.developable).into(),
            Self::Flipped => u8::from(t.flipped).into(),
            Self::Income => t.income.into(),
            Self::Iron => t.iron.into(),
            Self::Level => t.level.into(),
            Self::LinkVp => t.link_vp.into(),
            Self::RailOnly => u8::from(t.rail_only).into(),
            Self::Resource => t.resource.into(),
            Self::Vp => t.vp.into(),
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub enum PlayerField {
    BonusVp,
    Cash,
    Debt,
    HandCount,
    Income,
    IncomeMarker,
    LinkTiles,
    Spent,
    Vp,
}
impl PlayerField {
    fn parse(name: &str) -> Self {
        match name {
            "bonus_vp" => Self::BonusVp,
            "cash" => Self::Cash,
            "debt" => Self::Debt,
            "hand_count" => Self::HandCount,
            "income" => Self::Income,
            "income_marker" => Self::IncomeMarker,
            "link_tiles" => Self::LinkTiles,
            "spent" => Self::Spent,
            "vp" => Self::Vp,
            _ => panic!("unknown generated player field"),
        }
    }
    pub fn value(self, p: &ViewPlayer) -> f64 {
        match self {
            Self::BonusVp => p.bonus_vp.into(),
            Self::Cash => p.cash.into(),
            Self::Debt => p.debt.into(),
            Self::HandCount => p.hand_count.into(),
            Self::Income => p.income.into(),
            Self::IncomeMarker => p.income_marker.into(),
            Self::LinkTiles => p.link_tiles.into(),
            Self::Spent => p.spent.into(),
            Self::Vp => p.vp.into(),
        }
    }
}
pub struct Layout {
    pub slots: BTreeMap<&'static str, usize>,
    pub links: BTreeMap<&'static str, usize>,
    pub merchants: BTreeMap<&'static str, usize>,
    pub cards: BTreeMap<&'static str, usize>,
    pub slot_catalog: [usize; 49],
    pub link_catalog: [usize; 69],
    pub merchant_catalog: [usize; 9],
    pub slot_regions: [usize; 49],
    pub merchant_regions: [usize; 9],
    pub link_regions: [u64; 69],
    pub tile_fields: Vec<(TileField, f64)>,
    pub board_tile_fields: Vec<(TileField, f64)>,
    pub player_fields: Vec<(PlayerField, f64)>,
}
pub static LAYOUT: LazyLock<Layout> = LazyLock::new(|| {
    let s = &META.schema;
    let map =
        |v: &'static Vec<String>| v.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
    let slot_catalog = std::array::from_fn(|i| {
        CATALOG
            .slots
            .iter()
            .position(|x| x.id == s.slots[i])
            .expect("slot metadata")
    });
    let link_catalog = std::array::from_fn(|i| {
        CATALOG
            .links
            .iter()
            .position(|x| x.id == s.links[i])
            .expect("link metadata")
    });
    let merchant_catalog = std::array::from_fn(|i| {
        CATALOG
            .merchants
            .iter()
            .position(|x| x.id == s.merchants[i])
            .expect("merchant metadata")
    });
    assert!(CATALOG.regions.len() <= 64);
    Layout {
        slots: map(&s.slots),
        links: map(&s.links),
        merchants: map(&s.merchants),
        cards: map(&s.card_types),
        slot_regions: std::array::from_fn(|i| CATALOG.slot_regions[slot_catalog[i]]),
        link_regions: std::array::from_fn(|i| {
            CATALOG.link_regions[link_catalog[i]]
                .iter()
                .fold(0, |m, r| m | (1u64 << r))
        }),
        merchant_regions: std::array::from_fn(|i| {
            CATALOG
                .regions
                .iter()
                .position(|r| r == s.merchants[i].split(':').next().expect("region"))
                .expect("merchant region")
        }),
        slot_catalog,
        link_catalog,
        merchant_catalog,
        tile_fields: s
            .tile_fields
            .iter()
            .map(|(n, s)| (TileField::parse(n), *s))
            .collect(),
        board_tile_fields: s
            .board_tile_fields
            .iter()
            .map(|(n, s)| (TileField::parse(n), *s))
            .collect(),
        player_fields: s
            .player_fields
            .iter()
            .map(|(n, s)| (PlayerField::parse(n), *s))
            .collect(),
    }
});

pub fn kind_index(kind: Kind) -> usize {
    META.schema
        .kinds
        .iter()
        .position(|k| *k == kind)
        .expect("complete kind vocabulary")
}
pub fn target_index(name: &str) -> Result<Option<usize>, CoreError> {
    if name.is_empty() {
        return Ok(None);
    }
    if name == "market" {
        return Ok(Some(127));
    }
    if let Some(m) = name.strip_prefix("merchant:") {
        return LAYOUT
            .merchants
            .get(m)
            .map(|i| Some(118 + i))
            .ok_or_else(|| invalid("unknown merchant target"));
    }
    if let Some(i) = LAYOUT.slots.get(name) {
        return Ok(Some(*i));
    }
    LAYOUT
        .links
        .get(name)
        .map(|i| Some(49 + i))
        .ok_or_else(|| invalid("unknown target"))
}
