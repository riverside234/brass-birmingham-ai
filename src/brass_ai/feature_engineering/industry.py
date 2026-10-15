"""FE2: individual realization facts and deterministic development-only targets."""
from .contracts import Measure,INDUSTRIES,SALEABLE,TILE_FIELDS,TARGET_FIELDS
from .rules import eligible
from brass_ai.learning.contracts import require


def tile_properties(tile):
    if tile is None:return {k:Measure.absent() for k in TILE_FIELDS}
    values=dict(level=tile['level'],cash_cost=tile['cost'],vp=tile['vp'],beer=tile['beer'],
        canal_eligible=not tile['rail_only'],rail_eligible=not tile['canal_only'],survives_canal=tile['level']>1)
    return {k:Measure.exact(v) for k,v in values.items()}


def target(stack,era):
    reference=stack[0] if stack else None
    item=next(((i,t) for i,t in enumerate(stack) if t['level']>reference['level'] and eligible(t,era)),None) if reference else None
    values={k:Measure.absent() for k in TARGET_FIELDS}
    if item is None:
        values.update(present=Measure.exact(False),reachable=Measure.exact(False),blocked=Measure.exact(False),already_unlocked=Measure.exact(False))
        return values,None
    index,tile=item;reachable=all(t['developable'] for t in stack[:index])
    values.update(tile_properties(tile),present=Measure.exact(True),reachable=Measure.exact(reachable),
        blocked=Measure.exact(not reachable),already_unlocked=Measure.exact(index==0),
        removals=Measure.exact(index) if reachable else Measure())
    ref=tile_properties(reference)
    for name,prop in (('delta_cost','cash_cost'),('delta_vp','vp'),('delta_beer','beer'),('delta_canal_eligible','canal_eligible'),('delta_rail_eligible','rail_eligible')):
        values[name]=Measure.exact(values[prop].value-ref[prop].value)
    return values,item


def development(board):
    out={};era=board.view['context']['era']
    for industry in INDUSTRIES:
        stack=board.mats[industry];current=stack[0] if stack else None
        values=tile_properties(current)
        values.update(present=Measure.exact(current is not None),
            already_unlocked=Measure.exact(bool(current and current['level']>1 and eligible(current,era))))
        out.update({f'development.{industry}.current.{k}':v for k,v in values.items()})
        current_target,_=target(stack,era)
        next_target,_=target(stack,'RailEra') if era=='CanalEra' else ({k:Measure.absent() for k in TARGET_FIELDS},None)
        for label,values in (('current',current_target),('next_era',next_target)):
            out.update({f'development.{industry}.{label}_target.{k}':v for k,v in values.items()})
    return out


def development_step(board,choice):
    out={k:Measure.absent() for k in ('removals_before','removals_after','next_era_removals_before','next_era_removals_after','removed_tiles','immediately_unlocks','free')}
    out.update({'result.'+k:Measure.absent() for k in TILE_FIELDS})
    if choice['kind'] not in ('develop','bonus_develop'):return {'develop.'+k:v for k,v in out.items()}
    stack=board.mats[choice['industry']];require(stack and stack[0]['developable'],'Illegal development feature candidate')
    after=stack[1:];era=board.view['context']['era']
    for label,which in (('',era),('next_era_','RailEra')):
        if label and era!='CanalEra':continue
        facts,item=target(stack,which)
        before=facts['removals']
        out[label+'removals_before']=before
        out[label+'removals_after']=Measure.exact(max(0,before.value-1)) if before.known else before
    out.update(removed_tiles=Measure.exact(1),free=Measure.exact(choice['kind']=='bonus_develop'),
        immediately_unlocks=Measure.exact(bool(after and after[0]['level']>stack[0]['level'] and eligible(after[0],era))))
    out.update({'result.'+k:v for k,v in tile_properties(after[0] if after else None).items()})
    return {'develop.'+k:v for k,v in out.items()}


def own_action_slots(view):
    ctx=view['context'];in_progress=ctx['phase'] in ('payment','continuation','bonus') and ctx['turn_owner']==view['seat']
    current=ctx['actions_remaining'] if ctx['phase']!='debt' and ctx['turn_owner']==view['seat'] else 0
    slots=current+2*(8-ctx['round'])
    cards=len(view['own_hand'])+int(in_progress)
    return slots,Measure.interval(min(slots,cards),min(slots,cards+view['deck_count'])),in_progress


def realization(board):
    view=board.view;owned=[s for s in view['slots'] if s.get('tile',{}).get('seat')==board.seat]
    sales=[s for s in owned if s['tile']['type'] in SALEABLE and not s['tile']['flipped']]
    def needed(slot):
        pending=view['pending']
        return pending['beer'] if pending['kind']=='sell' and pending['target']==slot['id'] and view['context']['phase']=='payment' else slot['tile']['beer']
    def merchants(slot):
        choices=board.merchants_for(slot)
        if view['context']['phase']=='payment' and view['pending']['kind']=='sell' and view['pending']['target']==slot['id']:
            choices=tuple(m for m in choices if m['id']==view['pending']['merchant'])
        return choices
    feasible=[s for s in sales if any(board.beer(board.region(s['id']))+m['beer']>=needed(s) for m in merchants(s))]
    obsolete=[s['tile'] for s in owned if s['tile']['level']==1 and not s['tile']['flipped']]
    slots,supported,active=own_action_slots(view)
    values=dict(saleable_count=len(sales),potential_sale_vp=sum(s['tile']['vp'] for s in sales),beer_needed=sum(needed(s) for s in sales),
        individually_saleable_count=len(feasible),individually_saleable_vp=sum(s['tile']['vp'] for s in feasible),
        secured_industry_vp=sum(s['tile']['vp'] for s in owned if s['tile']['flipped']),
        own_action_slots=slots,current_action_in_progress=active)
    out={k:Measure.exact(v) for k,v in values.items()};out['card_supported_actions']=supported
    out['unflipped_level_one_count']=Measure.exact(len(obsolete)) if view['context']['era']=='CanalEra' else Measure.absent()
    out['unflipped_level_one_vp']=Measure.exact(sum(t['vp'] for t in obsolete)) if view['context']['era']=='CanalEra' else Measure.absent()
    return {'realization.'+k:v for k,v in out.items()}
