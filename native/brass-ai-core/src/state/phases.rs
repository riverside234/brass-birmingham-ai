//! Target-rules adapter over the owned state. A rejected step never mutates it.
use super::*;
use brass_upstream_rules::market::pricing::MarketPricing;

impl NativeState {
    pub fn apply_choice(&mut self, revision: u32, id: &str) -> Result<(), CoreError> {
        *self = self.after_choice(revision, id)?;
        Ok(())
    }
    /// Prepare an owned successor so callers can finish fallible work before commit.
    pub fn after_choice(&self, revision: u32, id: &str) -> Result<Self, CoreError> {
        require(revision == self.context.revision, "stale revision")?;
        let choice = self
            .legal_native()?
            .into_iter()
            .find(|(key, _)| key == id)
            .map(|(_, c)| c)
            .ok_or_else(|| CoreError::Invalid("illegal choice".into()))?;
        self.after_admitted(revision, choice)
    }
    /// Private hot path: Brass selected this exact choice from this revision's legal set.
    pub(crate) fn after_legal_choice(
        &self,
        revision: u32,
        choice: &Choice,
    ) -> Result<Self, CoreError> {
        require(revision == self.context.revision, "stale revision")?;
        self.after_admitted(
            revision,
            HistoryState::import(&History {
                seat: Seat::try_from(self.context.actor as u8)
                    .map_err(|_| CoreError::Invalid("actor".into()))?,
                era: self.context.era,
                choice: choice.clone(),
            })?,
        )
    }
    fn after_admitted(&self, revision: u32, choice: HistoryState) -> Result<Self, CoreError> {
        crate::work::checkpoint(crate::work::Stage::Application)?;
        require(
            self.history.len() < 8192 && revision < u32::MAX,
            "state history/revision limit",
        )?;
        // One owned transactional fork; errors and arithmetic limits leave the original intact.
        let copy_started = std::time::Instant::now();
        let mut next = self.clone();
        crate::work::state_copied(copy_started);
        next.execute(&choice)?;
        next.history.push(choice);
        next.context.revision += 1;
        next.rebuild_indexes();
        crate::work::checkpoint(crate::work::Stage::Application)?;
        Ok(next)
    }
    fn add(value: &mut i32, amount: i32) -> Result<(), CoreError> {
        *value = value
            .checked_add(amount)
            .ok_or_else(|| CoreError::Invalid("numeric state limit".into()))?;
        Ok(())
    }
    fn spend(&mut self, amount: i32) -> Result<(), CoreError> {
        let p = &mut self.players[seat(self.current_player)];
        require(amount >= 0 && p.cash >= amount, "insufficient cash")?;
        p.cash -= amount;
        Self::add(&mut p.spent, amount)
    }
    fn income(&mut self, player: Seat, amount: u8) {
        let p = &mut self.players[seat(player)];
        p.income_marker = p.income_marker.saturating_add(amount).min(99);
    }
    fn flip(&mut self, tile: TileId) {
        if !self.tiles[tile.0].flipped {
            self.tiles[tile.0].flipped = true;
            self.income(CATALOG.tiles[tile.0].seat, CATALOG.tiles[tile.0].income);
        }
    }
    fn discard(&mut self, card: CardId) -> Result<(), CoreError> {
        let s = seat(self.current_player);
        let index = self.cards.hands[s]
            .iter()
            .position(|c| *c == card)
            .ok_or_else(|| CoreError::Invalid("card missing".into()))?;
        self.cards.hands[s].remove(index);
        match CATALOG.card_names[card.0].as_str() {
            "wild:industry" => self.cards.wild_industry.push(card),
            "wild:location" => self.cards.wild_location.push(card),
            _ => self.cards.discards[s].push(card),
        }
        Ok(())
    }
    fn develop(&mut self, industry: Industry) {
        let id = self.players[seat(self.current_player)].mat[industry.index()].remove(0);
        self.retired.push(id);
        self.sort_offboard();
    }
    fn sort_offboard(&mut self) {
        self.removed
            .sort_by(|a, b| CATALOG.tiles[a.0].id.cmp(&CATALOG.tiles[b.0].id));
        self.retired
            .sort_by(|a, b| CATALOG.tiles[a.0].id.cmp(&CATALOG.tiles[b.0].id));
    }
    fn finish_build(&mut self) -> Result<(), CoreError> {
        let Some(Target::Slot(s)) = self.pending.target else {
            return Err(CoreError::Invalid("build target missing".into()));
        };
        let id = self.slots[s.0].unwrap();
        let tile = &CATALOG.tiles[id.0];
        let (amount, maximum) = match tile.industry {
            Industry::IronWorks => (self.iron_market, 10),
            Industry::CoalMine if self.market_connected(&[CATALOG.slot_regions[s.0]]) => {
                (self.coal_market, 14)
            }
            _ => return Ok(()),
        };
        let sold = self.tiles[id.0].resource.min(maximum - amount);
        let money: i32 = (0..sold)
            .map(|i| i32::from((maximum - amount - 1 - i) / 2 + 1))
            .sum();
        Self::add(&mut self.players[seat(self.current_player)].cash, money)?;
        if tile.industry == Industry::CoalMine {
            self.coal_market += sold;
        } else {
            self.iron_market += sold;
        }
        self.tiles[id.0].resource -= sold;
        if sold > 0 && self.tiles[id.0].resource == 0 {
            self.flip(id);
        }
        Ok(())
    }
    fn set_phase(&mut self) {
        if self.context.phase == Phase::Bonus {
            return;
        }
        self.context.phase = if self.pending.coal + self.pending.iron + self.pending.beer > 0 {
            Phase::Payment
        } else {
            Phase::Continuation
        };
    }
    fn execute(&mut self, c: &HistoryState) -> Result<(), CoreError> {
        if c.kind == Kind::Liquidate {
            let Some(Target::Slot(s)) = c.target else {
                return Err(CoreError::Invalid("liquidation target".into()));
            };
            let id = self.slots[s.0].take().unwrap();
            let money = i32::from(CATALOG.tiles[id.0].cost / 2);
            let owner = seat(c.seat);
            Self::add(&mut self.players[owner].cash, money)?;
            self.settlement_debts[owner] = (self.settlement_debts[owner] - money).max(0);
            self.retired.push(id);
            self.sort_offboard();
            self.focus_debtor();
            self.advance()?;
        } else if matches!(c.kind, Kind::BonusSkip | Kind::BonusDevelop) {
            if let Some(t) = c.industry {
                self.develop(t);
            }
            self.context.phase = Phase::Continuation;
            self.set_phase();
        } else if c.kind == Kind::Finish {
            require(self.context.actions_remaining > 0, "no actions remaining")?;
            self.context.actions_remaining -= 1;
            self.pending = PendingState {
                kind: None,
                target: None,
                merchant: None,
                parts: 0,
                coal: 0,
                iron: 0,
                beer: 0,
            };
            self.context.phase = Phase::Action;
            self.advance()?;
        } else if matches!(c.kind, Kind::Coal | Kind::Iron | Kind::Beer) {
            match c.target.unwrap() {
                Target::Market => {
                    let price = if c.kind == Kind::Coal {
                        MarketPricing::coal_price(self.coal_market, 1)
                    } else {
                        MarketPricing::iron_price(self.iron_market, 1)
                    };
                    self.spend(i32::from(price))?;
                    if c.kind == Kind::Coal {
                        self.coal_market = self.coal_market.saturating_sub(1);
                    } else {
                        self.iron_market = self.iron_market.saturating_sub(1);
                    }
                }
                Target::Slot(s) => {
                    let id = self.slots[s.0].unwrap();
                    self.tiles[id.0].resource -= 1;
                    if self.tiles[id.0].resource == 0 {
                        self.flip(id);
                    }
                }
                Target::Merchant(m) => {
                    self.merchants[m.0].beer -= 1;
                    match CATALOG.merchants[m.0].bonus {
                        Bonus::Develop => self.context.phase = Phase::Bonus,
                        Bonus::Income => self.income(self.current_player, 2),
                        Bonus::Money => {
                            Self::add(&mut self.players[seat(self.current_player)].cash, 5)?
                        }
                        Bonus::VP => {
                            let points = if CATALOG.merchants[m.0].id.starts_with("Shrewsbury:") {
                                4
                            } else {
                                3
                            };
                            let p = &mut self.players[seat(self.current_player)];
                            Self::add(&mut p.vp, points)?;
                            Self::add(&mut p.bonus_vp, points)?;
                        }
                    }
                }
                Target::Link(_) => return Err(CoreError::Invalid("resource target".into())),
            }
            match c.kind {
                Kind::Coal => self.pending.coal -= 1,
                Kind::Iron => self.pending.iron -= 1,
                Kind::Beer => self.pending.beer -= 1,
                _ => unreachable!(),
            }
            if self.pending.kind == Some(Kind::Build) && self.pending.coal + self.pending.iron == 0
            {
                self.finish_build()?;
            }
            if self.pending.kind == Some(Kind::Sell) && self.pending.beer == 0 {
                if let Some(Target::Slot(s)) = self.pending.target {
                    self.flip(self.slots[s.0].unwrap());
                }
            }
            self.set_phase();
        } else {
            if self.context.phase == Phase::Action {
                self.pending.kind = Some(c.kind);
                self.pending.parts = 0;
                self.discard(c.card.unwrap())?;
            }
            if c.target.is_some() {
                self.pending.target = c.target;
            }
            if c.merchant.is_some() {
                self.pending.merchant = c.merchant;
            }
            match c.kind {
                Kind::Build => {
                    let Some(Target::Slot(s)) = c.target else {
                        return Err(CoreError::Invalid("build target".into()));
                    };
                    let t = c.industry.unwrap();
                    let id = self.players[seat(self.current_player)].mat[t.index()].remove(0);
                    let def = &CATALOG.tiles[id.0];
                    if let Some(old) = self.slots[s.0].replace(id) {
                        self.removed.push(old);
                        self.sort_offboard();
                    }
                    if t == Industry::Brewery {
                        self.tiles[id.0].resource = if self.context.era == Era::RailEra {
                            2
                        } else {
                            1
                        };
                    }
                    self.spend(i32::from(def.cost))?;
                    self.pending.coal = def.coal;
                    self.pending.iron = def.iron;
                    self.rebuild_indexes();
                    if def.coal + def.iron == 0 {
                        self.finish_build()?;
                    }
                }
                Kind::Network => {
                    let Some(Target::Link(l)) = c.target else {
                        return Err(CoreError::Invalid("network target".into()));
                    };
                    self.links[l.0] = Some(self.current_player);
                    self.players[seat(self.current_player)].link_tiles -= 1;
                    if self.context.era == Era::CanalEra {
                        self.spend(3)?;
                    } else {
                        self.pending.coal = 1;
                        let second = self.pending.parts > 0;
                        self.pending.beer = u8::from(second);
                        self.spend(if second { 10 } else { 5 })?;
                    }
                }
                Kind::Develop => {
                    self.develop(c.industry.unwrap());
                    self.pending.iron = 1;
                }
                Kind::Sell => {
                    let Some(Target::Slot(s)) = c.target else {
                        return Err(CoreError::Invalid("sell target".into()));
                    };
                    let id = self.slots[s.0].unwrap();
                    self.pending.beer = CATALOG.tiles[id.0].beer;
                    if self.pending.beer == 0 {
                        self.flip(id);
                    }
                }
                Kind::Loan => {
                    let p = &mut self.players[seat(self.current_player)];
                    let target = CATALOG.incomes[usize::from(p.income_marker)] - 3;
                    p.income_marker = (0..=99)
                        .rev()
                        .find(|i| CATALOG.incomes[*i] == target)
                        .unwrap() as u8;
                    Self::add(&mut p.cash, 30)?;
                }
                Kind::Scout => {
                    let s = seat(self.current_player);
                    let traded = c.cards.as_ref().unwrap();
                    let first = self.cards.hands[s]
                        .iter()
                        .position(|id| *id == traded[0])
                        .unwrap();
                    let second = self.cards.hands[s]
                        .iter()
                        .enumerate()
                        .find(|(i, id)| *i != first && **id == traded[1])
                        .unwrap()
                        .0;
                    self.cards.discards[s].extend(traded);
                    self.cards.hands[s][first] = self.cards.wild_industry.remove(0);
                    self.cards.hands[s][second] = self.cards.wild_location.remove(0);
                }
                Kind::Pass => {}
                _ => return Err(CoreError::Invalid("unexpected staged choice".into())),
            }
            self.pending.parts += 1;
            self.set_phase();
        }
        self.refresh_context();
        Ok(())
    }
    fn debtor(&self, s: Seat) -> bool {
        self.settlement_debts[seat(s)] > 0
            && self
                .slots
                .iter()
                .flatten()
                .any(|t| CATALOG.tiles[t.0].seat == s)
    }
    fn focus_debtor(&mut self) {
        if let Some(s) = self.order.iter().copied().find(|s| self.debtor(*s)) {
            self.current_player = s;
        }
    }
    pub(super) fn refresh_context(&mut self) {
        self.context.chance = self.chance.is_some();
        self.context.actor = if self.context.terminal || self.context.chance {
            -1
        } else if self.context.phase == Phase::Debt {
            (0..4)
                .find(|s| self.debtor(Seat::try_from(*s as u8).unwrap()))
                .unwrap() as i8
        } else {
            u8::from(self.current_player) as i8
        };
        for i in 0..4 {
            self.players[i].debt = if self.context.phase == Phase::Debt {
                self.settlement_debts[i]
            } else {
                0
            };
        }
    }
    pub(super) fn advance(&mut self) -> Result<(), CoreError> {
        crate::work::checkpoint(crate::work::Stage::Application)?;
        if self.chance.is_some() {
            return Ok(());
        }
        if self.context.terminal {
            self.context.phase = Phase::Terminal;
            return Ok(());
        }
        if self.pending.kind.is_some() {
            return Ok(());
        }
        if self.round_pending {
            if self.debt_pending && (0..4).any(|s| self.debtor(Seat::try_from(s).unwrap())) {
                self.context.phase = Phase::Debt;
                return Ok(());
            }
            if self.context.era == Era::CanalEra && self.context.round == 8 {
                let mut pool: Vec<_> = self
                    .cards
                    .deck
                    .iter()
                    .chain(self.cards.discards.iter().flatten())
                    .copied()
                    .filter(|c| !CATALOG.card_names[c.0].starts_with("wild:"))
                    .collect();
                pool.sort();
                require(pool.len() == 64, "rail card conservation")?;
                self.context.phase = Phase::Action;
                self.chance = Some(Chance {
                    kind: ChanceKind::RailShuffle,
                    needed: 64,
                    pool,
                    picked: vec![],
                });
                return Ok(());
            }
            self.end_round()?;
        } else if self.context.actions_remaining == 0 {
            let needed = 8usize.saturating_sub(self.cards.hands[seat(self.current_player)].len());
            if needed > 0 && self.cards.deck.len() >= needed {
                let mut pool = self.cards.deck.clone();
                pool.sort();
                self.chance = Some(Chance {
                    kind: ChanceKind::Draw,
                    needed: needed as u8,
                    pool,
                    picked: vec![],
                });
                return Ok(());
            }
            self.end_turn()?;
            self.advance()?;
            return Ok(());
        }
        self.context.phase = if self.context.terminal {
            Phase::Terminal
        } else {
            Phase::Action
        };
        if !self.context.terminal {
            self.context.turn_owner = self.current_player;
        }
        Ok(())
    }
    pub(super) fn end_turn(&mut self) -> Result<(), CoreError> {
        let s = seat(self.current_player);
        let n = (8usize.saturating_sub(self.cards.hands[s].len())).min(self.cards.deck.len());
        self.cards.hands[s].extend(self.cards.deck.drain(..n));
        let index = self
            .order
            .iter()
            .position(|s| *s == self.current_player)
            .unwrap();
        self.current_player = self.order[(index + 1) % 4];
        if self.context.turn == 4 {
            self.round_pending = true;
            if !(self.context.era == Era::RailEra && self.context.round == 8) {
                self.settlement_debts = std::array::from_fn(|s| {
                    (-i64::from(CATALOG.incomes[usize::from(self.players[s].income_marker)])
                        - i64::from(self.players[s].cash))
                    .max(0) as i32
                });
                self.debt_pending = self.settlement_debts.iter().any(|v| *v > 0);
                if self.debt_pending {
                    self.focus_debtor();
                }
            }
        } else {
            self.context.turn += 1;
            self.start_turn();
        }
        Ok(())
    }
    fn start_turn(&mut self) {
        self.context.actions_remaining =
            if self.context.era == Era::CanalEra && self.context.round == 1 {
                1
            } else {
                2
            };
    }
    fn score_era(&mut self) -> Result<(), CoreError> {
        let mut region_points = vec![0i32; CATALOG.regions.len()];
        for (s, tile) in self.slots.iter().enumerate() {
            if let Some(id) = tile {
                if self.tiles[id.0].flipped {
                    let def = &CATALOG.tiles[id.0];
                    Self::add(&mut self.players[seat(def.seat)].vp, i32::from(def.vp))?;
                    region_points[CATALOG.slot_regions[s]] += i32::from(def.link_vp);
                }
            }
        }
        for m in 0..self.merchants.len() {
            region_points[Self::merchant_region(m)] = 2;
        }
        for (l, owner) in self.links.iter().enumerate() {
            if let Some(s) = owner {
                let points = CATALOG.link_regions[l]
                    .iter()
                    .map(|r| region_points[*r])
                    .sum();
                Self::add(&mut self.players[seat(*s)].vp, points)?;
            }
        }
        Ok(())
    }
    pub(super) fn end_round(&mut self) -> Result<(), CoreError> {
        if self.debt_pending {
            for s in 0..4 {
                let loss = self.players[s].vp.min(self.settlement_debts[s]);
                if loss > 0 {
                    self.players[s].vp -= loss;
                }
            }
        }
        self.round_pending = false;
        self.debt_pending = false;
        self.settlement_debts = [0; 4];
        let end_era = self.context.round == 8;
        if end_era {
            self.score_era()?;
            for p in &mut self.players {
                p.bonus_vp = 0;
            }
        }
        if end_era && self.context.era == Era::RailEra {
            self.context.terminal = true;
            self.context.phase = Phase::Terminal;
            return Ok(());
        }
        self.order.sort_by_key(|s| self.players[seat(*s)].spent);
        for p in &mut self.players {
            p.spent = 0;
            let amount = i32::from(CATALOG.incomes[usize::from(p.income_marker)]);
            Self::add(&mut p.cash, amount)?;
            p.cash = p.cash.max(0);
        }
        if end_era {
            for slot in &mut self.slots {
                if slot.is_some_and(|id| CATALOG.tiles[id.0].level == 1) {
                    self.removed.push(slot.take().unwrap());
                }
            }
            self.links.fill(None);
            for p in &mut self.players {
                p.link_tiles = 14;
                p.bonus_vp = 0;
                // Unbuilt Canal-only mat tiles remain and must be developed in Rail.
            }
            for m in &mut self.merchants {
                m.beer = u8::from(!m.types.is_empty());
            }
            self.sort_offboard();
            self.context.era = Era::RailEra;
            self.context.round = 1;
        } else {
            self.context.round += 1;
        }
        self.context.turn = 1;
        self.current_player = self.order[0];
        self.start_turn();
        Ok(())
    }
}
