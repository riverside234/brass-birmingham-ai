use super::{dto::*, require};
use crate::CoreError;
use brass_upstream_rules::core::player::{Player as UpstreamPlayer, PlayerId};
use serde::Deserialize;
use std::{collections::BTreeMap, sync::LazyLock};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TileId(pub(super) usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CardId(pub(super) usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotId(pub(super) usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkId(pub(super) usize);
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MerchantId(pub(super) usize);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct MerchantDef {
    pub id: String,
    pub bonus: Bonus,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Definitions {
    source_initial_sha256: String,
    tiles: Vec<Tile>,
    slots: Vec<Slot>,
    links: Vec<Link>,
    merchants: Vec<MerchantDef>,
    merchant_types: Vec<Vec<Industry>>,
    cards: BTreeMap<String, u8>,
}
pub(super) struct Catalog {
    pub tiles: Vec<Tile>,
    pub slots: Vec<Slot>,
    pub links: Vec<Link>,
    pub merchants: Vec<MerchantDef>,
    pub card_names: Vec<String>,
    pub card_counts: Vec<u8>,
    pub merchant_types: Vec<Vec<Industry>>,
    tile_ids: BTreeMap<String, TileId>,
    card_ids: BTreeMap<String, CardId>,
    pub regions: Vec<String>,
    pub slot_regions: Vec<usize>,
    pub link_regions: Vec<Vec<usize>>,
    pub incomes: [i8; 101],
}
pub(super) static CATALOG: LazyLock<Catalog> = LazyLock::new(|| {
    // Embedded, generated definitions are immutable and shared; no session state is shared.
    let d: Definitions = serde_json::from_str(include_str!("../../data/state-catalog.json"))
        .expect("embedded state catalog");
    assert_eq!(d.source_initial_sha256.len(), 64);
    let mut regions: Vec<_> = d
        .links
        .iter()
        .flat_map(|l| l.regions.iter().cloned())
        .chain(
            d.slots
                .iter()
                .map(|s| s.id.split(':').next().unwrap().to_string()),
        )
        .collect();
    regions.sort();
    regions.dedup();
    let slot_regions = d
        .slots
        .iter()
        .map(|s| {
            regions
                .binary_search(&s.id.split(':').next().unwrap().to_string())
                .unwrap()
        })
        .collect();
    let link_regions = d
        .links
        .iter()
        .map(|l| {
            l.regions
                .iter()
                .map(|r| regions.binary_search(r).unwrap())
                .collect()
        })
        .collect();
    let upstream = UpstreamPlayer::new(PlayerId::Coade, vec![]);
    Catalog {
        tile_ids: d
            .tiles
            .iter()
            .enumerate()
            .map(|(i, t)| (t.id.clone(), TileId(i)))
            .collect(),
        card_ids: d
            .cards
            .keys()
            .enumerate()
            .map(|(i, c)| (c.clone(), CardId(i)))
            .collect(),
        card_names: d.cards.keys().cloned().collect(),
        card_counts: d.cards.values().copied().collect(),
        tiles: d.tiles,
        slots: d.slots,
        links: d.links,
        merchants: d.merchants,
        merchant_types: d.merchant_types,
        regions,
        slot_regions,
        link_regions,
        incomes: std::array::from_fn(|i| upstream.get_income_amount(i as u8)),
    }
});
impl Catalog {
    pub fn tile(&self, s: &str) -> Result<TileId, CoreError> {
        self.tile_ids
            .get(s)
            .copied()
            .ok_or_else(|| CoreError::Invalid(format!("unknown tile {s}")))
    }
    pub fn card(&self, s: &str) -> Result<CardId, CoreError> {
        self.card_ids
            .get(s)
            .copied()
            .ok_or_else(|| CoreError::Invalid(format!("unknown card {s}")))
    }
    pub fn slot(&self, s: &str) -> Result<SlotId, CoreError> {
        self.slots
            .iter()
            .position(|x| x.id == s)
            .map(SlotId)
            .ok_or_else(|| CoreError::Invalid(format!("unknown slot {s}")))
    }
    pub fn link(&self, s: &str) -> Result<LinkId, CoreError> {
        self.links
            .iter()
            .position(|x| x.id == s)
            .map(LinkId)
            .ok_or_else(|| CoreError::Invalid(format!("unknown link {s}")))
    }
    pub fn merchant(&self, s: &str) -> Result<MerchantId, CoreError> {
        self.merchants
            .iter()
            .position(|x| x.id == s)
            .map(MerchantId)
            .ok_or_else(|| CoreError::Invalid(format!("unknown merchant {s}")))
    }
    pub fn validate_tile(&self, t: &Tile) -> Result<TileId, CoreError> {
        let id = self.tile(&t.id)?;
        let def = &self.tiles[id.0];
        let mut check = t.clone();
        check.flipped = def.flipped;
        check.resource = def.resource;
        require(check == *def, "immutable tile metadata differs")?;
        let maximum = if t.industry == Industry::Brewery {
            2
        } else {
            def.resource
        };
        require(t.resource <= maximum, "tile resource exceeds capacity")?;
        require(
            !t.flipped || t.resource == 0,
            "flipped tile retains resources",
        )?;
        Ok(id)
    }
}
