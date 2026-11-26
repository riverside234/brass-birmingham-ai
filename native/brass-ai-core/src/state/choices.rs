//! Direct staged enumeration: never constructs Cartesian products of complete plans.
use super::*;
use brass_upstream_rules::market::pricing::MarketPricing;
use std::collections::VecDeque;
use std::sync::LazyLock;

/// Each merchant's region, in catalog order (`merchant_region`).
static MERCHANT_REGIONS: LazyLock<Vec<usize>> = LazyLock::new(|| {
    CATALOG
        .merchants
        .iter()
        .map(|m| {
            CATALOG
                .regions
                .binary_search(&m.id.split(':').next().unwrap().to_owned())
                .unwrap()
        })
        .collect()
});
/// What a card permits a build on, read once from its catalog name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CardRule {
    /// `location:<region>`: that region only; `None` when the name matches no region.
    Location(Option<usize>),
    /// `wild:location`: every region except Central and SouthWest.
    WildLocation,
    /// `industry:<A+B>` and `wild:industry`: these industries (bit = `Industry::index`), in the network.
    Industries(u8),
}
static CARD_RULES: LazyLock<Vec<CardRule>> = LazyLock::new(|| {
    CATALOG
        .card_names
        .iter()
        .map(|name| {
            if let Some(location) = name.strip_prefix("location:") {
                CardRule::Location(CATALOG.regions.iter().position(|r| r == location))
            } else if name == "wild:location" {
                CardRule::WildLocation
            } else {
                let mut mask = 0u8;
                for industry in Industry::ALL {
                    if name == "wild:industry"
                        || name.strip_prefix("industry:").is_some_and(|types| {
                            types.split('+').any(|t| t == format!("{industry:?}"))
                        })
                    {
                        mask |= 1 << industry.index();
                    }
                }
                CardRule::Industries(mask)
            }
        })
        .collect()
});
/// Regions a `wild:location` card may build in.
static WILD_LOCATION_REGIONS: LazyLock<Vec<bool>> = LazyLock::new(|| {
    CATALOG
        .regions
        .iter()
        .map(|r| !matches!(r.as_str(), "Central" | "SouthWest"))
        .collect()
});

impl NativeState {
    pub(super) fn connected(&self, a: usize, b: usize) -> bool {
        self.indexes.connected(a, b)
    }
    pub(super) fn merchant_region(id: usize) -> usize {
        MERCHANT_REGIONS[id]
    }
    pub(super) fn market_connected(&self, regions: &[usize]) -> bool {
        (0..self.merchants.len()).any(|m| {
            regions
                .iter()
                .any(|r| self.connected(*r, Self::merchant_region(m)))
        })
    }
    pub(super) fn resource_slots(&self, industry: Industry) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(s, t)| {
                t.filter(|t| {
                    CATALOG.tiles[t.0].industry == industry && self.tiles[t.0].resource > 0
                })
                .map(|_| s)
            })
            .collect()
    }
    pub(super) fn coal_cost(&self, regions: &[usize], needed: u8) -> Option<i32> {
        let cubes: u16 = self
            .resource_slots(Industry::CoalMine)
            .iter()
            .filter(|s| {
                regions
                    .iter()
                    .any(|r| self.connected(*r, CATALOG.slot_regions[**s]))
            })
            .map(|s| u16::from(self.tiles[self.slots[*s].unwrap().0].resource))
            .sum();
        let missing = u16::from(needed).saturating_sub(cubes) as u8;
        if missing == 0 {
            Some(0)
        } else if self.market_connected(regions) {
            Some(i32::from(MarketPricing::coal_price(
                self.coal_market,
                missing,
            )))
        } else {
            None
        }
    }
    pub(super) fn coal_sources(&self, regions: &[usize]) -> Vec<Target> {
        // Multi-source BFS counts links, including every endpoint of the 3-way link.
        let mut distance = vec![usize::MAX; CATALOG.regions.len()];
        let mut queue = VecDeque::new();
        for r in regions {
            distance[*r] = 0;
            queue.push_back(*r);
        }
        while let Some(r) = queue.pop_front() {
            for (l, owner) in self.links.iter().enumerate() {
                if owner.is_none() || !CATALOG.link_regions[l].contains(&r) {
                    continue;
                }
                for next in &CATALOG.link_regions[l] {
                    if distance[*next] == usize::MAX {
                        distance[*next] = distance[r] + 1;
                        queue.push_back(*next);
                    }
                }
            }
        }
        let mines = self.resource_slots(Industry::CoalMine);
        let nearest = mines
            .iter()
            .map(|s| distance[CATALOG.slot_regions[*s]])
            .min()
            .unwrap_or(usize::MAX);
        if nearest != usize::MAX {
            mines
                .into_iter()
                .filter(|s| distance[CATALOG.slot_regions[*s]] == nearest)
                .map(|s| Target::Slot(SlotId(s)))
                .collect()
        } else if self.market_connected(regions)
            && self.players[seat(self.current_player)].cash
                >= i32::from(MarketPricing::coal_price(self.coal_market, 1))
        {
            vec![Target::Market]
        } else {
            vec![]
        }
    }
    pub(super) fn iron_sources(&self) -> Vec<Target> {
        let slots = self.resource_slots(Industry::IronWorks);
        if !slots.is_empty() {
            slots.into_iter().map(|s| Target::Slot(SlotId(s))).collect()
        } else if self.players[seat(self.current_player)].cash
            >= i32::from(MarketPricing::iron_price(self.iron_market, 1))
        {
            vec![Target::Market]
        } else {
            vec![]
        }
    }
    pub(super) fn beer_sources(&self, regions: &[usize]) -> Vec<usize> {
        self.resource_slots(Industry::Brewery)
            .into_iter()
            .filter(|s| {
                CATALOG.tiles[self.slots[*s].unwrap().0].seat == self.current_player
                    || regions
                        .iter()
                        .any(|r| self.connected(*r, CATALOG.slot_regions[*s]))
            })
            .collect()
    }
    pub(super) fn target_regions(&self) -> Vec<usize> {
        match self.pending.target {
            Some(Target::Slot(s)) => vec![CATALOG.slot_regions[s.0]],
            Some(Target::Link(l)) => CATALOG.link_regions[l.0].clone(),
            _ => vec![],
        }
    }
    fn developable(&self) -> Vec<Industry> {
        Industry::ALL
            .into_iter()
            .filter(|t| {
                self.players[seat(self.current_player)].mat[t.index()]
                    .first()
                    .is_some_and(|id| CATALOG.tiles[id.0].developable)
            })
            .collect()
    }
    /// Whether `card` permits a build in `region` with `industry`: the only card-dependent rule of a build.
    fn card_allows(&self, card: CardId, region: usize, industry: Industry) -> bool {
        match CARD_RULES[card.0] {
            CardRule::Location(r) => r == Some(region),
            CardRule::WildLocation => WILD_LOCATION_REGIONS[region],
            CardRule::Industries(mask) => {
                mask & (1 << industry.index()) != 0
                    && self.indexes.in_network(self.current_player, region)
            }
        }
    }
    /// Every catalog slot and type the current player could build on with a permitting card, in catalog
    /// order: the tile, era, overbuild, slot-type, Canal one-tile-per-region, coal and cash rules, which
    /// read no card. A build is legal exactly when its card also permits it (`card_allows`).
    fn buildable(&self) -> Result<Vec<(usize, Industry)>, CoreError> {
        let me = self.current_player;
        let p = &self.players[seat(me)];
        let canal = self.context.era == Era::CanalEra;
        let rail = self.context.era == Era::RailEra;
        let regions = CATALOG.regions.len();
        // Own tiles per region, and per region the industries of its empty single-type slots.
        let mut own = vec![0u16; regions];
        let mut single_empty = vec![0u8; regions];
        let mut mines: Vec<(usize, u16)> = vec![];
        let (mut iron_cubes, mut coal_on_board, mut iron_on_board) = (0u16, false, false);
        for (s, t) in self.slots.iter().enumerate() {
            let r = CATALOG.slot_regions[s];
            match t {
                Some(id) => {
                    let def = &CATALOG.tiles[id.0];
                    if def.seat == me {
                        own[r] += 1;
                    }
                    let resource = self.tiles[id.0].resource;
                    if resource > 0 {
                        match def.industry {
                            Industry::CoalMine => {
                                mines.push((r, u16::from(resource)));
                                coal_on_board = true;
                            }
                            Industry::IronWorks => {
                                iron_cubes += u16::from(resource);
                                iron_on_board = true;
                            }
                            _ => {}
                        }
                    }
                }
                None => {
                    if let [only] = &*CATALOG.slots[s].types {
                        single_empty[r] |= 1 << only.index();
                    }
                }
            }
        }
        // `coal_cost(&[region], n)` per region, computed on first use.
        let mut cubes: Vec<Option<u16>> = vec![None; regions];
        let mut market: Vec<Option<bool>> = vec![None; regions];
        let mut coal_for = |region: usize, needed: u8| -> Option<i32> {
            let available = *cubes[region].get_or_insert_with(|| {
                mines
                    .iter()
                    .filter(|(r, _)| self.connected(region, *r))
                    .map(|(_, n)| *n)
                    .sum()
            });
            let missing = u16::from(needed).saturating_sub(available) as u8;
            if missing == 0 {
                Some(0)
            } else if *market[region].get_or_insert_with(|| self.market_connected(&[region])) {
                Some(i32::from(MarketPricing::coal_price(
                    self.coal_market,
                    missing,
                )))
            } else {
                None
            }
        };
        let mut result = vec![];
        for (s, slot) in CATALOG.slots.iter().enumerate() {
            crate::work::checkpoint(crate::work::Stage::Generation)?;
            let region = CATALOG.slot_regions[s];
            for &industry in slot.types.iter() {
                let Some(id) = p.mat[industry.index()].first() else {
                    continue;
                };
                let tile = &CATALOG.tiles[id.0];
                if (canal && tile.rail_only) || (rail && tile.canal_only) {
                    continue;
                }
                let mine = if let Some(old) = self.slots[s] {
                    let old = &CATALOG.tiles[old.0];
                    if old.industry != industry || old.level >= tile.level {
                        continue;
                    }
                    if old.seat != me {
                        let available = match industry {
                            Industry::CoalMine => self.coal_market > 0 || coal_on_board,
                            Industry::IronWorks => self.iron_market > 0 || iron_on_board,
                            _ => continue,
                        };
                        if available {
                            continue;
                        }
                    }
                    old.seat == me
                } else {
                    if slot.types.len() > 1 && single_empty[region] & (1 << industry.index()) != 0 {
                        continue;
                    }
                    false
                };
                // Canal: no other own tile in the region.
                if canal && own[region] > u16::from(mine) {
                    continue;
                }
                let Some(coal) = coal_for(region, tile.coal) else {
                    continue;
                };
                if p.cash
                    >= i32::from(tile.cost)
                        + coal
                        + i32::from(MarketPricing::iron_price(
                            self.iron_market,
                            u16::from(tile.iron).saturating_sub(iron_cubes) as u8,
                        ))
                {
                    result.push((s, industry));
                }
            }
        }
        Ok(result)
    }
    fn network_links(&self) -> Vec<usize> {
        let p = &self.players[seat(self.current_player)];
        if p.link_tiles == 0 {
            return vec![];
        }
        let second = self.pending.kind == Some(Kind::Network) && self.pending.parts > 0;
        (0..self.links.len())
            .filter(|l| {
                let def = &CATALOG.links[*l];
                let regions = &CATALOG.link_regions[*l];
                if self.links[*l].is_some()
                    || (def.era == LinkEra::Canal) != (self.context.era == Era::CanalEra)
                    || !regions
                        .iter()
                        .any(|r| self.indexes.in_network(self.current_player, *r))
                {
                    return false;
                }
                if self.context.era == Era::CanalEra {
                    p.cash >= 3
                } else {
                    self.coal_cost(regions, 1)
                        .is_some_and(|cost| p.cash >= cost + if second { 10 } else { 5 })
                        && (!second || !self.beer_sources(regions).is_empty())
                }
            })
            .collect()
    }
    fn choice(&self, kind: Kind) -> HistoryState {
        HistoryState {
            seat: Seat::try_from(self.context.actor as u8).unwrap(),
            era: self.context.era,
            kind,
            target: None,
            card: None,
            industry: None,
            merchant: None,
            cards: None,
        }
    }
    /// The (slot, merchant) sales open to the current player, in slot then merchant order; card independent.
    fn sale_targets(&self) -> Result<Vec<(usize, usize)>, CoreError> {
        let mut targets = vec![];
        for (s, tile) in self.slots.iter().enumerate() {
            crate::work::checkpoint(crate::work::Stage::Generation)?;
            let Some(id) = tile else {
                continue;
            };
            let tile = &CATALOG.tiles[id.0];
            if tile.seat != self.current_player
                || self.tiles[id.0].flipped
                || !matches!(
                    tile.industry,
                    Industry::CottonMill | Industry::Manufacturer | Industry::Pottery
                )
            {
                continue;
            }
            let r = CATALOG.slot_regions[s];
            let beer: u16 = self
                .beer_sources(&[r])
                .iter()
                .map(|s| u16::from(self.tiles[self.slots[*s].unwrap().0].resource))
                .sum();
            for (m, merchant) in self.merchants.iter().enumerate() {
                if merchant.types.contains(&tile.industry)
                    && self.connected(r, Self::merchant_region(m))
                    && beer + u16::from(merchant.beer) >= u16::from(tile.beer)
                {
                    targets.push((s, m));
                }
            }
        }
        Ok(targets)
    }
    fn sales(&self, card: Option<CardId>, targets: &[(usize, usize)], choices: &mut Vec<HistoryState>) {
        for &(s, m) in targets {
            let mut c = self.choice(Kind::Sell);
            c.card = card;
            c.target = Some(Target::Slot(SlotId(s)));
            c.merchant = Some(MerchantId(m));
            choices.push(c);
        }
    }
    fn staged_choices(&self) -> Result<Vec<HistoryState>, CoreError> {
        if self.context.terminal || self.chance.is_some() {
            return Ok(vec![]);
        }
        let mut choices = vec![];
        match self.context.phase {
            Phase::Debt => {
                for (s, t) in self.slots.iter().enumerate() {
                    if t.is_some_and(|id| {
                        seat(CATALOG.tiles[id.0].seat) == self.context.actor as usize
                    }) {
                        let mut c = self.choice(Kind::Liquidate);
                        c.target = Some(Target::Slot(SlotId(s)));
                        choices.push(c);
                    }
                }
            }
            Phase::Bonus => {
                choices.push(self.choice(Kind::BonusSkip));
                for t in self.developable() {
                    let mut c = self.choice(Kind::BonusDevelop);
                    c.industry = Some(t);
                    choices.push(c);
                }
            }
            Phase::Payment => {
                let regions = self.target_regions();
                if self.pending.coal > 0 {
                    for target in self.coal_sources(&regions) {
                        let mut c = self.choice(Kind::Coal);
                        c.target = Some(target);
                        choices.push(c);
                    }
                }
                if self.pending.iron > 0 {
                    for target in self.iron_sources() {
                        let mut c = self.choice(Kind::Iron);
                        c.target = Some(target);
                        choices.push(c);
                    }
                }
                if self.pending.beer > 0 {
                    for s in self.beer_sources(&regions) {
                        let mut c = self.choice(Kind::Beer);
                        c.target = Some(Target::Slot(SlotId(s)));
                        choices.push(c);
                    }
                    if self.pending.kind == Some(Kind::Sell) {
                        if let Some(m) = self.pending.merchant {
                            if self.merchants[m.0].beer > 0 {
                                let mut c = self.choice(Kind::Beer);
                                c.target = Some(Target::Merchant(m));
                                choices.push(c);
                            }
                        }
                    }
                }
            }
            Phase::Continuation => {
                choices.push(self.choice(Kind::Finish));
                match self.pending.kind {
                    Some(Kind::Develop)
                        if self.pending.parts < 2 && !self.iron_sources().is_empty() =>
                    {
                        for t in self.developable() {
                            let mut c = self.choice(Kind::Develop);
                            c.industry = Some(t);
                            choices.push(c);
                        }
                    }
                    Some(Kind::Network)
                        if self.pending.parts < 2 && self.context.era == Era::RailEra =>
                    {
                        for l in self.network_links() {
                            let mut c = self.choice(Kind::Network);
                            c.target = Some(Target::Link(LinkId(l)));
                            choices.push(c);
                        }
                    }
                    Some(Kind::Sell) => {
                        let targets = self.sale_targets()?;
                        self.sales(None, &targets, &mut choices);
                    }
                    _ => {}
                }
            }
            Phase::Action => {
                let mut cards = self.cards.hands[seat(self.current_player)].clone();
                cards.sort();
                cards.dedup();
                let developable = if self.iron_sources().is_empty() {
                    vec![]
                } else {
                    self.developable()
                };
                let links = self.network_links();
                // Every rule of a build or sale except the card's own permission is the same for all cards.
                let builds = self.buildable()?;
                let sales = self.sale_targets()?;
                let loan = CATALOG.incomes
                    [usize::from(self.players[seat(self.current_player)].income_marker)]
                    >= -7;
                for card in cards {
                    crate::work::checkpoint(crate::work::Stage::Generation)?;
                    let mut pass = self.choice(Kind::Pass);
                    pass.card = Some(card);
                    choices.push(pass);
                    if loan {
                        let mut c = self.choice(Kind::Loan);
                        c.card = Some(card);
                        choices.push(c);
                    }
                    for &(s, t) in &builds {
                        if self.card_allows(card, CATALOG.slot_regions[s], t) {
                            let mut c = self.choice(Kind::Build);
                            c.card = Some(card);
                            c.target = Some(Target::Slot(SlotId(s)));
                            c.industry = Some(t);
                            choices.push(c);
                        }
                    }
                    for t in &developable {
                        let mut c = self.choice(Kind::Develop);
                        c.card = Some(card);
                        c.industry = Some(*t);
                        choices.push(c);
                    }
                    for l in &links {
                        let mut c = self.choice(Kind::Network);
                        c.card = Some(card);
                        c.target = Some(Target::Link(LinkId(*l)));
                        choices.push(c);
                    }
                    self.sales(Some(card), &sales, &mut choices);
                }
                let mut hand = self.cards.hands[seat(self.current_player)].clone();
                hand.sort();
                if hand.len() >= 3
                    && !hand
                        .iter()
                        .any(|c| CATALOG.card_names[c.0].starts_with("wild:"))
                    && !self.cards.wild_industry.is_empty()
                    && !self.cards.wild_location.is_empty()
                {
                    let mut seen = std::collections::BTreeSet::new();
                    for a in 0..hand.len() {
                        for b in a + 1..hand.len() {
                            crate::work::checkpoint(crate::work::Stage::Generation)?;
                            for c in b + 1..hand.len() {
                                if seen.insert([hand[a], hand[b], hand[c]]) {
                                    let mut choice = self.choice(Kind::Scout);
                                    choice.card = Some(hand[a]);
                                    choice.cards = Some(vec![hand[b], hand[c]]);
                                    choices.push(choice);
                                }
                            }
                        }
                    }
                }
            }
            Phase::Chance | Phase::Terminal => {}
        }
        Ok(choices)
    }
    pub(super) fn legal_native(&self) -> Result<Vec<(String, HistoryState)>, CoreError> {
        self.sorted_choices(|c| (c.export_choice().id, c), |c| &c.0)
    }
    pub fn legal_choices(&self) -> Result<Vec<Choice>, CoreError> {
        self.sorted_choices(|c| c.export_choice(), |c| &c.id)
    }
    fn sorted_choices<T>(
        &self,
        export: impl FnMut(HistoryState) -> T,
        id: impl Fn(&T) -> &str,
    ) -> Result<Vec<T>, CoreError> {
        crate::work::checkpoint(crate::work::Stage::Generation)?;
        let mut choices: Vec<_> = self.staged_choices()?.into_iter().map(export).collect();
        // Unique IDs give one order, so an unstable sort equals the stable one; duplicates are refused below.
        choices.sort_unstable_by(|a, b| id(a).cmp(id(b)));
        require(
            !choices.windows(2).any(|w| id(&w[0]) == id(&w[1])),
            "duplicate semantic choice",
        )?;
        require(
            self.context.terminal || self.chance.is_some() || !choices.is_empty(),
            "decision dead end",
        )?;
        Ok(choices)
    }
}

#[cfg(test)]
mod reference_tests {
    //! The restructured generation against the original per-card enumeration, on every state of played games.
    use super::*;

    impl NativeState {
        fn reference_can_build(&self, slot: usize, industry: Industry, card: CardId) -> bool {
            let p = &self.players[seat(self.current_player)];
            let Some(id) = p.mat[industry.index()].first() else {
                return false;
            };
            let tile = &CATALOG.tiles[id.0];
            let region = CATALOG.slot_regions[slot];
            if (self.context.era == Era::CanalEra && tile.rail_only)
                || (self.context.era == Era::RailEra && tile.canal_only)
            {
                return false;
            }
            if let Some(old) = self.slots[slot] {
                let old = &CATALOG.tiles[old.0];
                if old.industry != industry || old.level >= tile.level {
                    return false;
                }
                if old.seat != self.current_player {
                    let available = match industry {
                        Industry::CoalMine => u16::from(self.coal_market),
                        Industry::IronWorks => u16::from(self.iron_market),
                        _ => return false,
                    };
                    if available > 0 || !self.resource_slots(industry).is_empty() {
                        return false;
                    }
                }
            } else if CATALOG.slots[slot].types.len() > 1
                && self.slots.iter().enumerate().any(|(s, t)| {
                    t.is_none()
                        && CATALOG.slot_regions[s] == region
                        && *CATALOG.slots[s].types == [industry]
                })
            {
                return false;
            }
            if self.context.era == Era::CanalEra
                && self.slots.iter().enumerate().any(|(s, t)| {
                    s != slot
                        && CATALOG.slot_regions[s] == region
                        && t.is_some_and(|t| CATALOG.tiles[t.0].seat == self.current_player)
                })
            {
                return false;
            }
            let name = &CATALOG.card_names[card.0];
            let region_name = &CATALOG.regions[region];
            let valid_card = if let Some(location) = name.strip_prefix("location:") {
                location == region_name
            } else if name == "wild:location" {
                !matches!(region_name.as_str(), "Central" | "SouthWest")
            } else {
                (name == "wild:industry"
                    || name
                        .strip_prefix("industry:")
                        .is_some_and(|types| types.split('+').any(|t| t == format!("{industry:?}"))))
                    && self.indexes.in_network(self.current_player, region)
            };
            if !valid_card {
                return false;
            }
            let Some(coal) = self.coal_cost(&[region], tile.coal) else {
                return false;
            };
            let iron: u16 = self
                .resource_slots(Industry::IronWorks)
                .iter()
                .map(|s| u16::from(self.tiles[self.slots[*s].unwrap().0].resource))
                .sum();
            p.cash
                >= i32::from(tile.cost)
                    + coal
                    + i32::from(MarketPricing::iron_price(
                        self.iron_market,
                        u16::from(tile.iron).saturating_sub(iron) as u8,
                    ))
        }
        fn reference_merchant_region(id: usize) -> usize {
            CATALOG
                .regions
                .binary_search(&CATALOG.merchants[id].id.split(':').next().unwrap().to_owned())
                .unwrap()
        }
        fn reference_sales(&self, card: Option<CardId>) -> Vec<HistoryState> {
            let mut choices = vec![];
            for (s, tile) in self.slots.iter().enumerate() {
                let Some(id) = tile else {
                    continue;
                };
                let tile = &CATALOG.tiles[id.0];
                if tile.seat != self.current_player
                    || self.tiles[id.0].flipped
                    || !matches!(
                        tile.industry,
                        Industry::CottonMill | Industry::Manufacturer | Industry::Pottery
                    )
                {
                    continue;
                }
                let r = CATALOG.slot_regions[s];
                let beer: u16 = self
                    .beer_sources(&[r])
                    .iter()
                    .map(|s| u16::from(self.tiles[self.slots[*s].unwrap().0].resource))
                    .sum();
                for (m, merchant) in self.merchants.iter().enumerate() {
                    if merchant.types.contains(&tile.industry)
                        && self.connected(r, Self::reference_merchant_region(m))
                        && beer + u16::from(merchant.beer) >= u16::from(tile.beer)
                    {
                        let mut c = self.choice(Kind::Sell);
                        c.card = card;
                        c.target = Some(Target::Slot(SlotId(s)));
                        c.merchant = Some(MerchantId(m));
                        choices.push(c);
                    }
                }
            }
            choices
        }
        /// The original Action-phase enumeration (per card: every slot and type through `can_build`).
        fn reference_action_choices(&self) -> Vec<HistoryState> {
            let mut choices = vec![];
            let mut cards = self.cards.hands[seat(self.current_player)].clone();
            cards.sort();
            cards.dedup();
            let developable = if self.iron_sources().is_empty() {
                vec![]
            } else {
                self.developable()
            };
            let links = self.network_links();
            for card in cards {
                let mut pass = self.choice(Kind::Pass);
                pass.card = Some(card);
                choices.push(pass);
                if CATALOG.incomes[usize::from(self.players[seat(self.current_player)].income_marker)] >= -7 {
                    let mut c = self.choice(Kind::Loan);
                    c.card = Some(card);
                    choices.push(c);
                }
                for (s, slot) in CATALOG.slots.iter().enumerate() {
                    for t in slot.types.iter() {
                        if self.reference_can_build(s, *t, card) {
                            let mut c = self.choice(Kind::Build);
                            c.card = Some(card);
                            c.target = Some(Target::Slot(SlotId(s)));
                            c.industry = Some(*t);
                            choices.push(c);
                        }
                    }
                }
                for t in &developable {
                    let mut c = self.choice(Kind::Develop);
                    c.card = Some(card);
                    c.industry = Some(*t);
                    choices.push(c);
                }
                for l in &links {
                    let mut c = self.choice(Kind::Network);
                    c.card = Some(card);
                    c.target = Some(Target::Link(LinkId(*l)));
                    choices.push(c);
                }
                choices.extend(self.reference_sales(Some(card)));
            }
            choices
        }
    }

    fn mix(text: &str, salt: u64) -> u64 {
        text.bytes().fold(0xcbf2_9ce4_8422_2325 ^ salt, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        })
    }
    fn ids(choices: &[HistoryState]) -> Vec<String> {
        let mut ids: Vec<_> = choices.iter().map(|c| c.export_choice().semantic_id()).collect();
        ids.sort();
        ids
    }

    #[test]
    fn restructured_generation_equals_the_per_card_enumeration() {
        let dto: StateDto =
            serde_json::from_str(include_str!("../../tests/data/initial-state.json")).unwrap();
        let (mut states, mut action_states, mut sell_states, mut merchants_checked) = (0, 0, 0, 0);
        for m in 0..CATALOG.merchants.len() {
            assert_eq!(NativeState::merchant_region(m), NativeState::reference_merchant_region(m));
            merchants_checked += 1;
        }
        for game in 0..6u64 {
            let mut state = NativeState::import(&dto).unwrap();
            let mut step = 0u64;
            while !state.context().terminal {
                step += 1;
                let revision = state.context().revision;
                if let Some(chance) = state.chance_outcomes() {
                    let pick = (mix("chance", step ^ (game << 32)) % chance.outcomes.len() as u64) as usize;
                    state = state.after_chance(revision, &chance.outcomes[pick].card).unwrap();
                    continue;
                }
                states += 1;
                let legal = state.legal_choices().unwrap();
                // Every exported choice carries its semantic ID.
                assert!(legal.iter().all(|c| c.id == c.semantic_id()));
                let staged = state.staged_choices().unwrap();
                if state.context.phase == Phase::Action {
                    action_states += 1;
                    // Scouts follow the card loop unchanged; the reference covers the per-card choices.
                    let per_card: Vec<_> = staged.iter().filter(|c| c.kind != Kind::Scout).cloned().collect();
                    assert_eq!(ids(&per_card), ids(&state.reference_action_choices()), "game {game} step {step}");
                }
                if state.context.phase == Phase::Continuation && state.pending.kind == Some(Kind::Sell) {
                    sell_states += 1;
                    let mut new = vec![];
                    state.sales(None, &state.sale_targets().unwrap(), &mut new);
                    assert_eq!(ids(&new), ids(&state.reference_sales(None)));
                }
                // Mostly the most productive kinds, so boards fill as in real games; sometimes anything.
                let rank = |k: Kind| match k {
                    Kind::Build => 6,
                    Kind::Network => 5,
                    Kind::Sell => 4,
                    Kind::Develop => 3,
                    Kind::Loan => 2,
                    Kind::Pass => 0,
                    _ => 1,
                };
                let pick = if mix("explore", step ^ (game << 40)) % 4 == 0 {
                    legal[(mix("any", step ^ game) % legal.len() as u64) as usize].clone()
                } else {
                    legal
                        .iter()
                        .max_by_key(|c| (rank(c.kind), mix(&c.id, step ^ (game << 16))))
                        .unwrap()
                        .clone()
                };
                state = state.after_choice(revision, &pick.id).unwrap();
            }
            // History exports carry their semantic IDs too.
            for h in &state.history {
                let c = h.export_choice();
                assert_eq!(c.id, c.semantic_id());
            }
        }
        println!("states {states} action {action_states} continued sales {sell_states}");
        assert!(states > 1000 && action_states > 400 && merchants_checked > 0, "{states} {action_states}");
    }
}
