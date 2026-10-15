"""FE3 incremental choice accounting, separate from its determined action part.

No host, simulator, legal-action enumeration or completion tree is used here.
Automatic turn/round advancement is excluded from both accounting bases.
"""
from .contracts import Measure,ACTION_FIELDS
from .finance import financial
from .industry import development_step
from .rules import income_level,loan_marker,market_buy,market_sell,has_settlement


def balance_after(cash,cost,proceeds):
    lower=cash-cost.upper+proceeds.lower if cost.upper_known and proceeds.lower_known else None
    upper=cash-cost.lower+proceeds.upper if cost.lower_known and proceeds.upper_known else None
    return Measure.interval(lower,upper)


def action_effects(board,choice):
    view=board.view;observer=board.seat;pending=view['pending'];kind=choice['kind'];ctx=view['context']
    actor=view['players'][observer];markers=[p['income_marker'] for p in view['players']]
    cash=[p['cash'] for p in view['players']];flips=[0]*4;secured=[0]*4;vp=[0]*4
    out={f.name:Measure.exact(0) for f in ACTION_FIELDS if not f.name.startswith('develop.')}
    cost=proceeds=0;coal=iron=beer=0;part_coal=part_iron=part_beer=0
    part_base=0;part_proceeds=Measure.exact(0);target_tile=None;target_regions=()
    future_actor_marker=0
    def flip(tile):
        if tile is None or tile['flipped']:return
        seat=tile['seat'];flips[seat]+=1;secured[seat]+=tile['vp'];markers[seat]=min(99,markers[seat]+tile['income'])
    def resale(tile,region):
        if tile['type'] not in ('CoalMine','IronWorks'):return 0,False
        resource='coal' if tile['type']=='CoalMine' else 'iron'
        if resource=='coal' and not any(board.region(m) in board.reach((region,)) for m in board.merchants):return 0,False
        stock=view[resource+'_market'];capacity=14 if resource=='coal' else 10
        # A resource industry's own-type input is zero in this pinned tile set.
        sold=min(tile['resource'],capacity-stock)
        return market_sell(resource,stock,sold),sold==tile['resource'] and sold>0
    if kind=='build':
        target_tile=board.mats[choice['industry']][0];target_regions=(board.region(choice['target']),)
        cost=part_base=target_tile['cost'];part_coal=target_tile['coal'];part_iron=target_tile['iron']
        previous=board.slots[choice['target']].get('tile')
        if previous and previous['flipped']:secured[previous['seat']]-=previous['vp']
        sale,will_flip=resale(target_tile,target_regions[0]);part_proceeds=Measure.exact(sale)
        if part_coal+part_iron==0:
            proceeds+=sale
            if will_flip:flip(target_tile)
        elif will_flip:future_actor_marker=target_tile['income']
    elif kind=='network':
        rail=ctx['era']=='RailEra';second=pending['kind']=='network' and pending['parts']>0
        cost=part_base=(10 if second else 5) if rail else 3
        part_coal=int(rail);part_beer=int(rail and second);target_regions=tuple(board.links[choice['target']]['regions'])
    elif kind in ('develop','bonus_develop'):
        part_iron=int(kind=='develop')
    elif kind=='loan':
        proceeds=30;part_proceeds=Measure.exact(30);markers[observer]=loan_marker(markers[observer])
    elif kind=='sell':
        target_tile=board.slots[choice['target']]['tile'];part_beer=target_tile['beer']
        if part_beer==0:flip(target_tile)
        else:future_actor_marker=target_tile['income']
        target_regions=(board.region(choice['target']),)
    elif kind in ('coal','iron','beer'):
        coal=int(kind=='coal');iron=int(kind=='iron');beer=int(kind=='beer')
        part_coal=pending['coal'];part_iron=pending['iron'];part_beer=pending['beer']
        target=choice['target']
        if target=='market':cost=market_buy(kind,view[kind+'_market'])
        elif target.startswith('merchant:'):
            merchant=board.merchants[target.removeprefix('merchant:')];bonus=merchant['bonus']
            if bonus=='Money':proceeds=5
            elif bonus=='Income':markers[observer]+=2  # Merchant event is not the clamped industry increment.
            elif bonus=='VP':vp[observer]+=4 if board.region(merchant['id'])=='Shrewsbury' else 3
            elif bonus=='Develop':out['merchant_free_develop']=Measure.exact(1)
        else:
            source=board.slots[target]['tile']
            if source['resource']==1:flip(source)
        if pending['kind']=='build':
            target_tile=board.slots[pending['target']]['tile'];region=board.region(pending['target']);target_regions=(region,)
            sale,will_flip=resale(target_tile,region);part_proceeds=Measure.exact(sale)
            if part_coal+part_iron==1:
                proceeds+=sale
                if will_flip:flip(target_tile)
            elif will_flip:future_actor_marker=target_tile['income']
        elif pending['kind']=='network':target_regions=tuple(board.links[pending['target']]['regions'])
        elif pending['kind']=='sell':
            target_tile=board.slots[pending['target']]['tile'];target_regions=(board.region(pending['target']),)
            if part_beer==1:flip(target_tile)
            else:future_actor_marker=target_tile['income']
    elif kind=='liquidate':
        tile=board.slots[choice['target']]['tile'];proceeds=tile['cost']//2;part_proceeds=Measure.exact(proceeds)
        if tile['flipped']:secured[tile['seat']]-=tile['vp']

    # Merchant source remains a real choice. Do not assume its bonus at a sale root.
    sale_merchant=choice.get('merchant') if kind=='sell' else pending['merchant'] if pending['kind']=='sell' and kind=='beer' else None
    if sale_merchant and part_beer>0:
        merchant=board.merchants[sale_merchant]
        bonus_cash=5 if merchant['bonus']=='Money' and merchant['beer'] else 0
        selected_merchant=kind=='beer' and choice['target'].startswith('merchant:')
        forced_merchant=board.beer(target_regions[0])<part_beer
        part_proceeds=Measure.exact(proceeds) if selected_merchant else Measure.exact(proceeds+bonus_cash) if forced_merchant else Measure.interval(proceeds,max(proceeds,bonus_cash))
    elif kind in ('coal','iron','beer') and pending['kind']!='build':part_proceeds=Measure.exact(proceeds)

    # Costs of the chosen determined part include only its still-outstanding inputs.
    quote=board.resource_cost(part_coal,part_iron,target_regions) if part_coal or part_iron else Measure.exact(0)
    part_cost=Measure.interval(part_base+quote.lower if quote.lower_known else part_base,
                               part_base+quote.upper if quote.upper_known else None)
    if kind in ('loan','liquidate','pass','scout','finish','bonus_develop','bonus_skip'):part_cost=Measure.exact(cost)
    out.update(cash_spent=Measure.exact(cost),cash_proceeds=Measure.exact(proceeds),
        coal_used=Measure.exact(coal),iron_used=Measure.exact(iron),beer_used=Measure.exact(beer),
        portion_coal=Measure.exact(part_coal),portion_iron=Measure.exact(part_iron),portion_beer=Measure.exact(part_beer),
        portion_cash_spent=part_cost,portion_cash_proceeds=part_proceeds)
    cash[observer]+=proceeds-cost;out['cash_after']=Measure.exact(cash[observer])
    debts=[p['debt'] for p in view['players']]
    if kind=='liquidate':debts[observer]=max(0,debts[observer]-proceeds)
    after=financial(view,cash,markers,debts)
    out.update(projected_cash=after['finance.0.projected_cash'],shortfall=after['finance.0.shortfall'])
    out['debt_after']=Measure.exact(max(0,actor['debt']-proceeds)) if ctx['phase']=='debt' else Measure.absent()
    portion_balance=balance_after(actor['cash'],part_cost,part_proceeds);out['portion_cash_after']=portion_balance
    if not has_settlement(ctx):projection=Measure.absent()
    elif ctx['phase']=='debt' and debts[observer]>0:projection=Measure()
    else:
        minimum_income=income_level(min(99,markers[observer]+future_actor_marker)) if future_actor_marker else income_level(markers[observer])
        remaining=dict(CoalMine=part_coal-coal,IronWorks=part_iron-iron,Brewery=part_beer-beer)
        reach=board.reach(target_regions);possible_marker=0
        for resource,amount in remaining.items():
            for source in board.resources[resource]:
                tile=source['tile'];left=tile['resource']-int(choice.get('target')==source['id'] and kind in ('coal','iron','beer'))
                connected=resource!='CoalMine' or board.region(source['id']) in reach
                if tile['seat']==observer and not tile['flipped'] and connected and 0<left<=amount:
                    possible_marker+=tile['income']
        if sale_merchant and remaining['Brewery']>0:
            merchant=board.merchants[sale_merchant]
            if merchant['bonus']=='Income' and merchant['beer'] and not(kind=='beer' and choice['target'].startswith('merchant:')):possible_marker+=2
        maximum_income=income_level(min(99,markers[observer]+future_actor_marker+possible_marker)) if future_actor_marker or possible_marker else minimum_income
        projection=Measure.interval(portion_balance.lower+minimum_income if portion_balance.lower_known else None,
            portion_balance.upper+maximum_income if portion_balance.upper_known else None)
    out['portion_projected_cash']=projection
    out['portion_shortfall']=Measure.absent() if not projection.applicable else Measure.interval(
        max(0,-projection.upper) if projection.upper_known else 0,max(0,-projection.lower) if projection.lower_known else None)
    for relative in range(4):
        seat=(observer+relative)%4;p=view['players'][seat]
        for name,value in dict(flips=flips[seat],income_marker_delta=markers[seat]-p['income_marker'],
            income_delta=income_level(markers[seat])-p['income'],immediate_vp=vp[seat],secured_era_vp_delta=secured[seat]).items():
            out[f'effects.{relative}.{name}']=Measure.exact(value)
    out.update(development_step(board,choice))
    return out
