//! Public-only prepared board. Fixed indexed storage is reused by all candidates.
use super::*;

pub struct Board<'a> {
    pub view: &'a Observation,
    slots: [usize; 49],
    links: [usize; 69],
    merchants: [usize; 9],
    mats: [[usize; 6]; 4],
    connected: [u64; 64],
}
impl<'a> Board<'a> {
    pub fn new(view: &'a Observation) -> Result<Self, CoreError> {
        check(
            view.protocol == "0.2.0"
                && view.context.protocol == view.protocol
                && view.rules == crate::TARGET_RULES
                && view.context.rules == view.rules,
            "feature rules mismatch",
        )?;
        check(
            view.context.actor == u8::from(view.seat) as i8
                && !view.context.terminal
                && !matches!(view.context.phase, Phase::Chance | Phase::Terminal),
            "features require the actual nonterminal acting observer",
        )?;
        check(
            (1..=8).contains(&view.context.round)
                && (1..=4).contains(&view.context.turn)
                && view.context.actions_remaining <= 2,
            "feature turn bounds",
        )?;
        check(
            view.coal_market <= 14
                && view.iron_market <= 10
                && view.deck_count <= 64
                && view.wild_industry_count <= 4
                && view.wild_location_count <= 4
                && view.own_hand.len() <= 8
                && view.own_hand.len() == usize::from(view.players[seat(view.seat)].hand_count)
                && view.history.len() <= 8192,
            "feature observation bounds",
        )?;
        check(
            view.slots.len() == 49 && view.links.len() == 69 && view.merchants.len() == 9,
            "feature board vocabulary",
        )?;
        let mut b = Self {
            view,
            slots: [usize::MAX; 49],
            links: [usize::MAX; 69],
            merchants: [usize::MAX; 9],
            mats: [[usize::MAX; 6]; 4],
            connected: std::array::from_fn(|r| 1u64 << r),
        };
        let mut order = view.order.map(u8::from);
        order.sort();
        check(order == [0, 1, 2, 3], "feature order permutation")?;
        for (i, p) in view.players.iter().enumerate() {
            check(
                seat(p.seat) == i
                    && p.cash >= 0
                    && p.debt >= 0
                    && p.spent >= 0
                    && p.vp >= 0
                    && p.bonus_vp >= 0
                    && p.link_tiles <= 14
                    && p.known_wild.len() <= 8
                    && p.income_marker <= 100
                    && i32::from(p.income) == income(i32::from(p.income_marker))
                    && p.hidden_discards == u8::from(view.context.era == Era::CanalEra)
                    && p.hand_count <= 8
                    && p.discards.len() <= 64,
                "feature player facts",
            )?;
            for (j, m) in p.mat.iter().enumerate() {
                let k = m.industry.index();
                check(b.mats[i][k] == usize::MAX, "duplicate feature mat")?;
                b.mats[i][k] = j;
                check(m.tiles.len() <= 16, "feature mat size")?;
                for t in &m.tiles {
                    validate_tile(t)?;
                    check(
                        t.seat == p.seat && t.industry == m.industry,
                        "feature mat ownership",
                    )?;
                }
            }
        }
        for (i, s) in view.slots.iter().enumerate() {
            let n = *LAYOUT
                .slots
                .get(s.id.as_str())
                .ok_or_else(|| invalid("unknown feature slot"))?;
            check(
                b.slots[n] == usize::MAX && s.types == CATALOG.slots[LAYOUT.slot_catalog[n]].types,
                "feature slot profile",
            )?;
            b.slots[n] = i;
            if let Some(t) = &s.tile {
                validate_tile(t)?;
            }
        }
        for (i, l) in view.links.iter().enumerate() {
            let n = *LAYOUT
                .links
                .get(l.id.as_str())
                .ok_or_else(|| invalid("unknown feature link"))?;
            let def = &CATALOG.links[LAYOUT.link_catalog[n]];
            check(
                b.links[n] == usize::MAX && l.regions == def.regions && l.era == def.era,
                "feature link profile",
            )?;
            b.links[n] = i;
            if l.owner.is_some() {
                let regions = LAYOUT.link_regions[n];
                for r in 0..CATALOG.regions.len() {
                    if regions & (1u64 << r) != 0 {
                        b.connected[r] |= regions;
                    }
                }
            }
        }
        for k in 0..CATALOG.regions.len() {
            for r in 0..CATALOG.regions.len() {
                if b.connected[r] & (1u64 << k) != 0 {
                    b.connected[r] |= b.connected[k];
                }
            }
        }
        for (i, m) in view.merchants.iter().enumerate() {
            let n = *LAYOUT
                .merchants
                .get(m.id.as_str())
                .ok_or_else(|| invalid("unknown feature merchant"))?;
            check(
                b.merchants[n] == usize::MAX
                    && m.bonus == CATALOG.merchants[LAYOUT.merchant_catalog[n]].bonus
                    && m.beer <= 1,
                "feature merchant profile",
            )?;
            if let Some(types) = &m.types {
                check(
                    CATALOG.merchant_types.contains(types),
                    "feature merchant types",
                )?;
            }
            b.merchants[n] = i;
        }
        Ok(b)
    }
    pub fn slot(&self, i: usize) -> &Slot {
        &self.view.slots[self.slots[i]]
    }
    pub fn link(&self, i: usize) -> &Link {
        &self.view.links[self.links[i]]
    }
    pub fn merchant(&self, i: usize) -> &Merchant {
        &self.view.merchants[self.merchants[i]]
    }
    pub fn mat(&self, s: usize, k: usize) -> &[Tile] {
        &self.view.players[s].mat[self.mats[s][k]].tiles
    }
    pub fn reach(&self, starts: u64) -> u64 {
        (0..CATALOG.regions.len())
            .filter(|r| starts & (1u64 << r) != 0)
            .fold(0, |m, r| m | self.connected[r])
    }
    pub fn market_connected(&self, reach: u64) -> bool {
        LAYOUT
            .merchant_regions
            .iter()
            .any(|r| reach & (1u64 << r) != 0)
    }
    pub fn beer(&self, region: usize) -> i32 {
        let reach = self.reach(1u64 << region);
        (0..49)
            .filter_map(|i| self.slot(i).tile.as_ref().map(|t| (i, t)))
            .filter(|(i, t)| {
                t.industry == Industry::Brewery
                    && (t.seat == self.view.seat || reach & (1u64 << LAYOUT.slot_regions[*i]) != 0)
            })
            .map(|(_, t)| i32::from(t.resource))
            .sum()
    }
    pub fn resource_cost(&self, coal: i32, iron: i32, starts: u64) -> Measure {
        let reach = self.reach(starts);
        let mut free_coal = 0;
        let mut free_iron = 0;
        for i in 0..49 {
            if let Some(t) = &self.slot(i).tile {
                if t.industry == Industry::CoalMine && reach & (1u64 << LAYOUT.slot_regions[i]) != 0
                {
                    free_coal += i32::from(t.resource);
                }
                if t.industry == Industry::IronWorks {
                    free_iron += i32::from(t.resource);
                }
            }
        }
        let paid_coal = (coal - free_coal).max(0);
        let paid_iron = (iron - free_iron).max(0);
        if paid_coal > 0 && !self.market_connected(reach) {
            Measure::interval(Some(0.), None)
        } else {
            Measure::exact(
                market_buy(true, self.view.coal_market, paid_coal)
                    + market_buy(false, self.view.iron_market, paid_iron),
            )
        }
    }
    pub fn resale(&self, t: &Tile, region: usize) -> (i32, bool) {
        let coal = t.industry == Industry::CoalMine;
        if !coal && t.industry != Industry::IronWorks {
            return (0, false);
        }
        if coal && !self.market_connected(self.reach(1u64 << region)) {
            return (0, false);
        }
        let stock = if coal {
            self.view.coal_market
        } else {
            self.view.iron_market
        };
        let cap = if coal { 14 } else { 10 };
        let sold = i32::from(t.resource).min(cap - i32::from(stock));
        let proceeds = (0..sold)
            .map(|i| (cap - i32::from(stock) - 1 - i) / 2 + 1)
            .sum();
        (proceeds, sold == i32::from(t.resource) && sold > 0)
    }
}
fn validate_tile(t: &Tile) -> Result<(), CoreError> {
    let def = CATALOG
        .tile_definition(&t.id)
        .ok_or_else(|| invalid("unknown feature tile"))?;
    check(
        t.seat == def.seat
            && t.industry == def.industry
            && LAYOUT
                .tile_fields
                .iter()
                .filter(|(f, _)| {
                    !matches!(f, layout::TileField::Flipped | layout::TileField::Resource)
                })
                .all(|(f, _)| f.value(t) == f.value(def))
            && t.resource
                <= if t.industry == Industry::Brewery {
                    2
                } else {
                    def.resource
                }
            && (!t.flipped || t.resource == 0),
        "feature tile profile",
    )
}
pub fn income(marker: i32) -> i32 {
    if marker < 11 {
        marker - 10
    } else if marker < 31 {
        (marker - 9) / 2
    } else if marker < 61 {
        (marker + 2) / 3
    } else {
        30.min((marker + 23) / 4)
    }
}
pub fn settlement(v: &Observation) -> bool {
    !(v.context.terminal || v.context.era == Era::RailEra && v.context.round == 8)
}
pub fn market_buy(coal: bool, stock: u8, count: i32) -> i32 {
    let (cap, empty) = if coal { (14, 8) } else { (10, 6) };
    (0..count)
        .map(|i| {
            if i32::from(stock) - i <= 0 {
                empty
            } else {
                (cap - i32::from(stock) + i) / 2 + 1
            }
        })
        .sum()
}
