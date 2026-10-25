//! Direct staged enumeration: never constructs Cartesian products of complete plans.
use super::*;
use brass_upstream_rules::market::pricing::MarketPricing;
use std::collections::VecDeque;

impl NativeState {
    pub(super) fn connected(&self, a: usize, b: usize) -> bool {
        self.indexes.connected(a, b)
    }
    pub(super) fn merchant_region(id: usize) -> usize {
        CATALOG
            .regions
            .binary_search(
                &CATALOG.merchants[id]
                    .id
                    .split(':')
                    .next()
                    .unwrap()
                    .to_owned(),
            )
            .unwrap()
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
    fn can_build(&self, slot: usize, industry: Industry, card: CardId) -> bool {
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
                    && CATALOG.slots[s].types == [industry]
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
    fn sales(&self, card: Option<CardId>) -> Vec<HistoryState> {
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
                    && self.connected(r, Self::merchant_region(m))
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
    fn staged_choices(&self) -> Vec<HistoryState> {
        if self.context.terminal || self.chance.is_some() {
            return vec![];
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
                    Some(Kind::Sell) => choices.extend(self.sales(None)),
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
                for card in cards {
                    let mut pass = self.choice(Kind::Pass);
                    pass.card = Some(card);
                    choices.push(pass);
                    if CATALOG.incomes
                        [usize::from(self.players[seat(self.current_player)].income_marker)]
                        >= -7
                    {
                        let mut c = self.choice(Kind::Loan);
                        c.card = Some(card);
                        choices.push(c);
                    }
                    for (s, slot) in CATALOG.slots.iter().enumerate() {
                        for t in &slot.types {
                            if self.can_build(s, *t, card) {
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
                    choices.extend(self.sales(Some(card)));
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
        choices
    }
    pub(super) fn legal_native(&self) -> Result<Vec<(String, HistoryState)>, CoreError> {
        let mut choices: Vec<_> = self
            .staged_choices()
            .into_iter()
            .map(|c| (c.export().choice.id, c))
            .collect();
        choices.sort_by(|a, b| a.0.cmp(&b.0));
        require(
            !choices.windows(2).any(|w| w[0].0 == w[1].0),
            "duplicate semantic choice",
        )?;
        require(
            self.context.terminal || self.chance.is_some() || !choices.is_empty(),
            "decision dead end",
        )?;
        Ok(choices)
    }
    pub fn legal_choices(&self) -> Result<Vec<Choice>, CoreError> {
        Ok(self
            .legal_native()?
            .into_iter()
            .map(|(_, c)| c.export().choice)
            .collect())
    }
}
