//! Port of the frozen FE1/FE2/FE3 public accounting oracle. No look-ahead rules calls.
use super::{indices::*, *};
type M = Measure;
fn eligible(t: &Tile, era: Era) -> bool {
    if era == Era::CanalEra {
        !t.rail_only
    } else {
        !t.canal_only
    }
}
fn properties(tile: Option<&Tile>) -> [M; 7] {
    tile.map_or([M::absent(); 7], |t| {
        [
            M::exact(t.level),
            M::exact(t.cost),
            M::exact(t.vp),
            M::exact(t.beer),
            M::boolean(!t.rail_only),
            M::boolean(!t.canal_only),
            M::boolean(t.level > 1),
        ]
    })
}
fn target(stack: &[Tile], era: Era) -> [M; 17] {
    let mut out = [M::absent(); 17];
    out[..4].fill(M::exact(0));
    if let Some(head) = stack.first() {
        if let Some((index, tile)) = stack
            .iter()
            .enumerate()
            .find(|(_, t)| t.level > head.level && eligible(t, era))
        {
            let reachable = stack[..index].iter().all(|t| t.developable);
            out[..5].copy_from_slice(&[
                M::exact(1),
                M::boolean(reachable),
                M::boolean(!reachable),
                M::boolean(index == 0),
                if reachable {
                    M::exact(index as f64)
                } else {
                    M::unknown()
                },
            ]);
            let props = properties(Some(tile));
            let previous = properties(Some(head));
            out[5..12].copy_from_slice(&props);
            for (i, p) in [1, 2, 3, 4, 5].into_iter().enumerate() {
                out[12 + i] = M::exact(props[p].value - previous[p].value);
            }
        }
    }
    out
}
fn finances(v: &Observation, cash: &[f64; 4], markers: &[i32; 4], debts: &[i32; 4]) -> [M; 20] {
    let mut result = [M::absent(); 20];
    let applies = settlement(v);
    for r in 0..4 {
        let s = (seat(v.seat) + r) % 4;
        let inc = income(markers[s]);
        let start = r * 5;
        result[start] = M::boolean(applies);
        result[start + 4] = M::exact(debts[s]);
        if applies {
            result[start + 1] = M::exact((-inc).max(0));
            if v.context.phase == Phase::Debt && debts[s] > 0 {
                result[start + 2] = M::unknown();
                result[start + 3] = M::unknown();
            } else {
                let balance = cash[s] + f64::from(inc);
                result[start + 2] = M::exact(balance);
                result[start + 3] = M::exact((-balance).max(0.));
            }
        }
    }
    result
}
fn registers(v: &Observation) -> ([f64; 4], [i32; 4], [i32; 4]) {
    (
        // Exact for the admitted i32 money domain, including a one-action quote
        // above i32::MAX. Feature arithmetic must not overflow an engine register.
        std::array::from_fn(|i| f64::from(v.players[i].cash)),
        std::array::from_fn(|i| i32::from(v.players[i].income_marker)),
        std::array::from_fn(|i| v.players[i].debt),
    )
}
pub(super) fn state(
    b: &Board<'_>,
    choices: &[Candidate],
    opportunities: &mut HashSet<OpportunityKey>,
) -> Result<[M; 326], CoreError> {
    let v = b.view;
    let observer = seat(v.seat);
    let ctx = &v.context;
    let mut out = [M::absent(); 326];
    let (cash, markers, debts) = registers(v);
    out[..20].copy_from_slice(&finances(v, &cash, &markers, &debts));
    let mut order = v.order;
    order.sort_by_key(|s| v.players[seat(*s)].spent);
    for r in 0..4 {
        let s = (observer + r) % 4;
        out[S_ORDER_0_POSITION + r * 2] = if settlement(v) {
            M::exact(
                order
                    .iter()
                    .position(|p| seat(*p) == s)
                    .expect("permutation") as f64,
            )
        } else {
            M::absent()
        };
        out[S_ORDER_0_SPENDING_DIFFERENCE + r * 2] =
            M::exact(v.players[observer].spent - v.players[s].spent);
    }
    let mut counts = [0u32; 14];
    for c in choices {
        let mut target = None;
        let mut detail = None;
        let mut cards = [usize::MAX; 4];
        match c.kind {
            Kind::Scout => {
                cards = c.cards;
                cards.sort_unstable();
            }
            Kind::Build => {
                target = c.target;
                detail = c.industry.map(Industry::index);
            }
            Kind::Develop | Kind::BonusDevelop => detail = c.industry.map(Industry::index),
            Kind::Sell => {
                target = c.target;
                detail = c.merchant;
            }
            Kind::Coal | Kind::Iron | Kind::Beer | Kind::Network | Kind::Liquidate => {
                target = c.target
            }
            _ => {}
        }
        if opportunities.insert((c.kind as u8, target, detail, cards)) {
            counts[kind_index(c.kind)] += 1;
        }
    }
    out[S_OPPORTUNITIES_MAIN_ACTION_AVAILABLE] = M::boolean(ctx.phase == Phase::Action);
    for (i, kind) in META.schema.kinds.iter().enumerate() {
        let relevant = match ctx.phase {
            Phase::Action => matches!(
                kind,
                Kind::Build
                    | Kind::Develop
                    | Kind::Sell
                    | Kind::Loan
                    | Kind::Pass
                    | Kind::Scout
                    | Kind::Network
            ),
            Phase::Payment => matches!(kind, Kind::Coal | Kind::Iron | Kind::Beer),
            Phase::Bonus => matches!(kind, Kind::BonusSkip | Kind::BonusDevelop),
            Phase::Continuation => *kind == Kind::Finish || kind.name() == v.pending.kind,
            Phase::Debt => *kind == Kind::Liquidate,
            _ => false,
        };
        if relevant {
            out[29 + i * 2] = M::boolean(counts[i] > 0);
            out[30 + i * 2] = M::exact(counts[i]);
        }
    }
    let mut values = [0i32; 11];
    for i in 0..49 {
        if let Some(t) = &b.slot(i).tile {
            if t.seat != v.seat {
                continue;
            }
            if t.flipped {
                values[7] += i32::from(t.vp);
            }
            if t.level == 1 && !t.flipped {
                values[5] += 1;
                values[6] += i32::from(t.vp);
            }
            if !t.flipped
                && matches!(
                    t.industry,
                    Industry::CottonMill | Industry::Manufacturer | Industry::Pottery
                )
            {
                let in_payment = ctx.phase == Phase::Payment
                    && v.pending.kind == "sell"
                    && v.pending.target == b.slot(i).id;
                let needed = if in_payment { v.pending.beer } else { t.beer };
                values[0] += 1;
                values[1] += i32::from(t.vp);
                values[2] += i32::from(needed);
                let reach = b.reach(1u64 << LAYOUT.slot_regions[i]);
                let feasible = (0..9).any(|m| {
                    let merchant = b.merchant(m);
                    merchant
                        .types
                        .as_ref()
                        .is_some_and(|ts| ts.contains(&t.industry))
                        && reach & (1u64 << LAYOUT.merchant_regions[m]) != 0
                        && (!in_payment || merchant.id == v.pending.merchant)
                        && b.beer(LAYOUT.slot_regions[i]) + i32::from(merchant.beer)
                            >= i32::from(needed)
                });
                if feasible {
                    values[3] += 1;
                    values[4] += i32::from(t.vp);
                }
            }
        }
    }
    let active = matches!(
        ctx.phase,
        Phase::Payment | Phase::Continuation | Phase::Bonus
    ) && ctx.turn_owner == v.seat;
    let current = if ctx.phase != Phase::Debt && ctx.turn_owner == v.seat {
        i32::from(ctx.actions_remaining)
    } else {
        0
    };
    let slots = current + 2 * (8 - i32::from(ctx.round));
    let cards = v.own_hand.len() as i32 + i32::from(active);
    values[8] = slots;
    values[10] = i32::from(active);
    for (i, value) in values.into_iter().enumerate() {
        out[S_REALIZATION_SALEABLE_COUNT + i] = M::exact(value);
    }
    out[S_REALIZATION_CARD_SUPPORTED_ACTIONS] = M::interval(
        Some(f64::from(slots.min(cards))),
        Some(f64::from(slots.min(cards + i32::from(v.deck_count)))),
    );
    if ctx.era == Era::RailEra {
        out[S_REALIZATION_UNFLIPPED_LEVEL_ONE_COUNT] = M::absent();
        out[S_REALIZATION_UNFLIPPED_LEVEL_ONE_VP] = M::absent();
    }
    for k in 0..6 {
        let stack = b.mat(observer, k);
        let head = stack.first();
        let start = S_DEVELOPMENT_BREWERY_CURRENT_PRESENT + k * 43;
        out[start] = M::boolean(head.is_some());
        out[start + 1] = M::boolean(head.is_some_and(|t| t.level > 1 && eligible(t, ctx.era)));
        out[start + 2..start + 9].copy_from_slice(&properties(head));
        out[start + 9..start + 26].copy_from_slice(&target(stack, ctx.era));
        if ctx.era == Era::CanalEra {
            out[start + 26..start + 43].copy_from_slice(&target(stack, Era::RailEra));
        }
    }
    Ok(out)
}
fn development_step(b: &Board<'_>, c: Candidate) -> Result<[M; 14], CoreError> {
    let mut out = [M::absent(); 14];
    if !matches!(c.kind, Kind::Develop | Kind::BonusDevelop) {
        return Ok(out);
    }
    let stack = b.mat(seat(b.view.seat), c.industry()?);
    check(
        stack.first().is_some_and(|t| t.developable),
        "illegal development feature candidate",
    )?;
    let after = &stack[1..];
    let era = b.view.context.era;
    for (i, e) in [era, Era::RailEra].into_iter().enumerate() {
        if i == 1 && era != Era::CanalEra {
            continue;
        }
        let before = target(stack, e)[4];
        out[i * 2] = before;
        out[i * 2 + 1] = if before.known {
            M::exact((before.value - 1.).max(0.))
        } else {
            before
        };
    }
    out[4] = M::exact(1);
    out[5] = M::boolean(
        after
            .first()
            .is_some_and(|t| t.level > stack[0].level && eligible(t, era)),
    );
    out[6] = M::boolean(c.kind == Kind::BonusDevelop);
    out[7..].copy_from_slice(&properties(after.first()));
    Ok(out)
}
fn flip(t: &Tile, markers: &mut [i32; 4], flips: &mut [i32; 4], secured: &mut [i32; 4]) {
    if !t.flipped {
        let s = seat(t.seat);
        flips[s] += 1;
        secured[s] += i32::from(t.vp);
        markers[s] = (markers[s] + i32::from(t.income)).min(99);
    }
}
fn placed<'a>(b: &'a Board<'_>, index: usize) -> Result<&'a Tile, CoreError> {
    b.slot(index)
        .tile
        .as_ref()
        .ok_or_else(|| invalid("candidate references empty slot"))
}
fn balance(cash: i32, cost: M, proceeds: M) -> M {
    M::interval(
        cost.upper
            .zip(proceeds.lower)
            .map(|(c, p)| f64::from(cash) - c + p),
        cost.lower
            .zip(proceeds.upper)
            .map(|(c, p)| f64::from(cash) - c + p),
    )
}
pub(super) fn action(b: &Board<'_>, c: Candidate) -> Result<[M; 52], CoreError> {
    let v = b.view;
    let observer = seat(v.seat);
    let pending = &v.pending;
    let ctx = &v.context;
    let actor = &v.players[observer];
    let (mut cash, mut markers, mut debts) = registers(v);
    let mut flips = [0; 4];
    let mut secured = [0; 4];
    let mut vp = [0; 4];
    let mut out = [M::exact(0); 52];
    let mut cost = 0;
    let mut proceeds = 0;
    let mut used = [0; 3];
    let mut part = [0; 3];
    let mut part_base = 0;
    let mut part_proceeds = M::exact(0);
    let mut regions = 0u64;
    let mut future_marker = 0;
    match c.kind {
        Kind::Build => {
            let tile = b
                .mat(observer, c.industry()?)
                .first()
                .ok_or_else(|| invalid("build from empty mat"))?;
            let slot = c.slot()?;
            let region = LAYOUT.slot_regions[slot];
            regions = 1u64 << region;
            cost = i32::from(tile.cost);
            part_base = cost;
            part[0] = i32::from(tile.coal);
            part[1] = i32::from(tile.iron);
            if let Some(old) = &b.slot(slot).tile {
                if old.flipped {
                    secured[seat(old.seat)] -= i32::from(old.vp);
                }
            }
            let (sale, will_flip) = b.resale(tile, region);
            part_proceeds = M::exact(sale);
            if part[0] + part[1] == 0 {
                proceeds += sale;
                if will_flip {
                    flip(tile, &mut markers, &mut flips, &mut secured);
                }
            } else if will_flip {
                future_marker = i32::from(tile.income);
            }
        }
        Kind::Network => {
            let rail = ctx.era == Era::RailEra;
            let second = pending.kind == "network" && pending.parts > 0;
            cost = if rail {
                if second {
                    10
                } else {
                    5
                }
            } else {
                3
            };
            part_base = cost;
            part[0] = i32::from(rail);
            part[2] = i32::from(rail && second);
            regions = LAYOUT.link_regions[c.target.expect("validated network") - 49];
        }
        Kind::Develop | Kind::BonusDevelop => part[1] = i32::from(c.kind == Kind::Develop),
        Kind::Loan => {
            proceeds = 30;
            part_proceeds = M::exact(30);
            let target = income(markers[observer]) - 3;
            while markers[observer] > 0 && income(markers[observer]) != target {
                markers[observer] -= 1;
            }
        }
        Kind::Sell => {
            let slot = c.slot()?;
            let tile = placed(b, slot)?;
            part[2] = i32::from(tile.beer);
            regions = 1u64 << LAYOUT.slot_regions[slot];
            if part[2] == 0 {
                flip(tile, &mut markers, &mut flips, &mut secured);
            } else {
                future_marker = i32::from(tile.income);
            }
        }
        Kind::Coal | Kind::Iron | Kind::Beer => {
            used = [
                i32::from(c.kind == Kind::Coal),
                i32::from(c.kind == Kind::Iron),
                i32::from(c.kind == Kind::Beer),
            ];
            part = [
                i32::from(pending.coal),
                i32::from(pending.iron),
                i32::from(pending.beer),
            ];
            let source = c.target.expect("validated resource");
            match source {
                127 => {
                    check(c.kind != Kind::Beer, "beer market is unsupported")?;
                    cost = market_buy(
                        c.kind == Kind::Coal,
                        if c.kind == Kind::Coal {
                            v.coal_market
                        } else {
                            v.iron_market
                        },
                        1,
                    );
                }
                118..=126 => {
                    check(c.kind == Kind::Beer, "merchant only supplies beer")?;
                    let m = b.merchant(source - 118);
                    match m.bonus {
                        Bonus::Money => proceeds = 5,
                        Bonus::Income => markers[observer] += 2,
                        Bonus::VP => {
                            vp[observer] += if m.id.starts_with("Shrewsbury:") {
                                4
                            } else {
                                3
                            }
                        }
                        Bonus::Develop => out[A_MERCHANT_FREE_DEVELOP] = M::exact(1),
                    }
                }
                0..=48 => {
                    let tile = placed(b, source)?;
                    if tile.resource == 1 {
                        flip(tile, &mut markers, &mut flips, &mut secured);
                    }
                }
                _ => return Err(invalid("resource source is not a slot/market/merchant")),
            }
            if pending.kind == "build" || pending.kind == "sell" {
                let slot = target_index(&pending.target)?
                    .filter(|i| *i < 49)
                    .ok_or_else(|| invalid("pending industry target"))?;
                let tile = placed(b, slot)?;
                let region = LAYOUT.slot_regions[slot];
                regions = 1u64 << region;
                if pending.kind == "build" {
                    let (sale, will_flip) = b.resale(tile, region);
                    part_proceeds = M::exact(sale);
                    if part[0] + part[1] == 1 {
                        proceeds += sale;
                        if will_flip {
                            flip(tile, &mut markers, &mut flips, &mut secured);
                        }
                    } else if will_flip {
                        future_marker = i32::from(tile.income);
                    }
                } else if part[2] == 1 {
                    flip(tile, &mut markers, &mut flips, &mut secured);
                } else {
                    future_marker = i32::from(tile.income);
                }
            } else if pending.kind == "network" {
                let target = target_index(&pending.target)?
                    .filter(|i| (49..118).contains(i))
                    .ok_or_else(|| invalid("pending link target"))?;
                regions = LAYOUT.link_regions[target - 49];
            }
        }
        Kind::Liquidate => {
            let tile = placed(b, c.slot()?)?;
            proceeds = i32::from(tile.cost) / 2;
            part_proceeds = M::exact(proceeds);
            if tile.flipped {
                secured[seat(tile.seat)] -= i32::from(tile.vp);
            }
        }
        _ => {}
    }
    let sale_merchant = if c.kind == Kind::Sell {
        c.merchant
    } else if pending.kind == "sell" && c.kind == Kind::Beer {
        Some(
            *LAYOUT
                .merchants
                .get(pending.merchant.as_str())
                .ok_or_else(|| invalid("pending sale merchant"))?,
        )
    } else {
        None
    };
    if let Some(m) = sale_merchant.filter(|_| part[2] > 0) {
        let merchant = b.merchant(m);
        let bonus = if merchant.bonus == Bonus::Money && merchant.beer > 0 {
            5
        } else {
            0
        };
        let selected = c.kind == Kind::Beer && c.target.is_some_and(|i| (118..127).contains(&i));
        let region = regions.trailing_zeros() as usize;
        check(region < 64, "sale region required")?;
        let forced = b.beer(region) < part[2];
        part_proceeds = if selected {
            M::exact(proceeds)
        } else if forced {
            M::exact(proceeds + bonus)
        } else {
            M::interval(Some(proceeds.into()), Some(proceeds.max(bonus).into()))
        };
    } else if matches!(c.kind, Kind::Coal | Kind::Iron | Kind::Beer) && pending.kind != "build" {
        part_proceeds = M::exact(proceeds);
    }
    let quote = if part[0] + part[1] > 0 {
        b.resource_cost(part[0], part[1], regions)
    } else {
        M::exact(0)
    };
    let mut part_cost = M::interval(
        Some(f64::from(part_base) + quote.lower.unwrap_or(0.)),
        quote.upper.map(|v| f64::from(part_base) + v),
    );
    if matches!(
        c.kind,
        Kind::Loan
            | Kind::Liquidate
            | Kind::Pass
            | Kind::Scout
            | Kind::Finish
            | Kind::BonusDevelop
            | Kind::BonusSkip
    ) {
        part_cost = M::exact(cost);
    }
    out[A_CASH_SPENT] = M::exact(cost);
    out[A_CASH_PROCEEDS] = M::exact(proceeds);
    for i in 0..3 {
        out[A_COAL_USED + i * 2] = M::exact(used[i]);
        out[A_PORTION_COAL + i * 2] = M::exact(part[i]);
    }
    out[A_PORTION_CASH_SPENT] = part_cost;
    out[A_PORTION_CASH_PROCEEDS] = part_proceeds;
    cash[observer] += f64::from(proceeds - cost);
    out[A_CASH_AFTER] = M::exact(cash[observer]);
    if c.kind == Kind::Liquidate {
        debts[observer] = (debts[observer] - proceeds).max(0);
    }
    let after = finances(v, &cash, &markers, &debts);
    out[A_PROJECTED_CASH] = after[2];
    out[A_SHORTFALL] = after[3];
    out[A_DEBT_AFTER] = if ctx.phase == Phase::Debt {
        M::exact((actor.debt - proceeds).max(0))
    } else {
        M::absent()
    };
    let portion_balance = balance(actor.cash, part_cost, part_proceeds);
    out[A_PORTION_CASH_AFTER] = portion_balance;
    let projection = if !settlement(v) {
        M::absent()
    } else if ctx.phase == Phase::Debt && debts[observer] > 0 {
        M::unknown()
    } else {
        let minimum = if future_marker != 0 {
            income((markers[observer] + future_marker).min(99))
        } else {
            income(markers[observer])
        };
        let remaining = std::array::from_fn::<_, 3, _>(|i| part[i] - used[i]);
        let reach = b.reach(regions);
        let mut possible = 0;
        for (k, amount) in [Industry::CoalMine, Industry::IronWorks, Industry::Brewery]
            .into_iter()
            .zip(remaining)
        {
            for i in 0..49 {
                if let Some(tile) = &b.slot(i).tile {
                    if tile.industry != k || tile.resource == 0 {
                        continue;
                    }
                    let left = i32::from(tile.resource)
                        - i32::from(
                            c.target == Some(i)
                                && matches!(c.kind, Kind::Coal | Kind::Iron | Kind::Beer),
                        );
                    let connected =
                        k != Industry::CoalMine || reach & (1u64 << LAYOUT.slot_regions[i]) != 0;
                    if tile.seat == v.seat
                        && !tile.flipped
                        && connected
                        && left > 0
                        && left <= amount
                    {
                        possible += i32::from(tile.income);
                    }
                }
            }
        }
        if let Some(m) = sale_merchant.filter(|_| remaining[2] > 0) {
            let merchant = b.merchant(m);
            if merchant.bonus == Bonus::Income
                && merchant.beer > 0
                && !(c.kind == Kind::Beer && c.target.is_some_and(|i| (118..127).contains(&i)))
            {
                possible += 2;
            }
        }
        let maximum = if future_marker != 0 || possible != 0 {
            income((markers[observer] + future_marker + possible).min(99))
        } else {
            minimum
        };
        M::interval(
            portion_balance.lower.map(|b| b + f64::from(minimum)),
            portion_balance.upper.map(|b| b + f64::from(maximum)),
        )
    };
    out[A_PORTION_PROJECTED_CASH] = projection;
    out[A_PORTION_SHORTFALL] = if !projection.applicable {
        M::absent()
    } else {
        M::interval(
            Some(projection.upper.map_or(0., |u| (-u).max(0.))),
            projection.lower.map(|l| (-l).max(0.)),
        )
    };
    for r in 0..4 {
        let s = (observer + r) % 4;
        out[A_EFFECTS_0_FLIPS + r * 5..A_EFFECTS_0_FLIPS + (r + 1) * 5].copy_from_slice(&[
            M::exact(flips[s]),
            M::exact(markers[s] - i32::from(v.players[s].income_marker)),
            M::exact(income(markers[s]) - i32::from(v.players[s].income)),
            M::exact(vp[s]),
            M::exact(secured[s]),
        ]);
    }
    out[A_DEVELOP_REMOVALS_BEFORE..].copy_from_slice(&development_step(b, c)?);
    Ok(out)
}
