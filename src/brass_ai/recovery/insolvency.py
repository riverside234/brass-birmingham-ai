"""Training-only, capped insolvency costs derived from public complete replays.

Each event happens BEFORE its first debt choice. Its cost-to-go therefore trains
earlier decisions, not the mandatory choice of which asset to remove. Official
win/rank/VP labels, observations and serialized replay formats remain unchanged.
"""
from collections import Counter

from brass_ai.learning.contracts import require


def events(records):
    result=[];settled=set()
    for i,row in enumerate(records):
        view=row['view'];context=view['context'];key=(context['era'],context['round'])
        if context['phase']=='debt' and key not in settled:
            settled.add(key)
            # The first debtor's public view includes ALL seats' initial debts,
            # including a debtor with no remaining assets and thus no choice.
            for player in view['players']:
                if player['debt']>0:
                    result.append(dict(before=i,seat=player['seat'],era=key[0],round=key[1],
                                       shortfall=player['debt']))
        after=records[i+1]['view']['context'] if i+1<len(records) else None
        boundary=after is None or (after['era'],after['round'])!=key
        if not boundary or key in settled:continue
        # Declared B01 rule: final Rail has NO income/debt settlement.
        if after is None and context['era']=='RailEra':continue
        # With no assets anywhere requiring disposal, the host settles debt
        # automatically. Confirming the last completed action changes neither
        # cash nor income before that settlement. Do not infer debt mid-action.
        choice=row['choices'][row['chosen']]
        if context['phase']!='continuation' or choice['kind']!='finish':continue
        settled.add(key)
        owners={s['tile']['seat'] for s in view['slots'] if s.get('tile')}
        for player in view['players']:
            shortfall=max(0,-player['cash']-player['income'])
            if shortfall:
                require(player['seat'] not in owners,'Missing liquidation decisions at round boundary')
                result.append(dict(before=i+1,seat=player['seat'],era=key[0],round=key[1],
                                   shortfall=shortfall))
    return result


def returns(records,config,initial_spent=None):
    """Undiscounted future costs, capped once per seat/game in forward time."""
    found=events(records);charges={};totals=list(initial_spent or [0.0]*4);counts=[0]*4;shortfalls=[0]*4
    phases=[Counter() for _ in range(4)]
    for event in found:
        seat=event['seat'];cost=min(config['event_cost'],max(0.,config['max_cost']-totals[seat]))
        totals[seat]+=cost;counts[seat]+=1;shortfalls[seat]+=event['shortfall'];phases[seat][event['era']]+=1
        charges.setdefault(event['before'],[]).append((seat,cost))
    future=[0.0]*4;costs=[0.0]*len(records)
    for i in range(len(records)-1,-1,-1):
        for seat,cost in charges.get(i+1,[]):future[seat]+=cost
        costs[i]=future[records[i]['seat']]
    summaries=[dict(events=counts[s],shortfall=shortfalls[s],cost=totals[s],by_era=dict(phases[s])) for s in range(4)]
    return costs,summaries


def advantage_costs(data,config):
    """Only trusted, locally derived labels reach PPO; never stored in replays."""
    if config is None:return [0.0]*len(data),dict(enabled=False)
    costs=[];summaries={}
    for row,_,trajectory,_ in data:
        require('_insolvency_cost_to_go' in row,'Missing derived insolvency return')
        value=row['_insolvency_cost_to_go']
        require(0<=value<=config['max_cost']+1e-8,'Invalid insolvency return')
        costs.append(value)
        if '_insolvency_trajectory' in row:
            require(trajectory not in summaries,'Duplicate insolvency trajectory')
            summaries[trajectory]=row['_insolvency_trajectory']
    require(set(summaries)=={d[2] for d in data},'Missing insolvency trajectory summary')
    branch_summaries=[v for k,v in summaries.items() if '/branch/' in k[0]]
    summaries={k:v for k,v in summaries.items() if '/branch/' not in k[0]}
    phases=Counter()
    for summary in summaries.values():phases.update(summary['by_era'])
    total=sum(s['cost'] for s in summaries.values())
    return costs,dict(enabled=True,**config,events=sum(s['events'] for s in summaries.values()),
        shortfall=sum(s['shortfall'] for s in summaries.values()),by_era=dict(phases),
        learner_trajectories=len(summaries),affected_trajectories=sum(s['events']>0 for s in summaries.values()),
        branch_trajectories=len(branch_summaries),branch_events=sum(s['events'] for s in branch_summaries),
        scope='Ordinary trajectory statistics; branch continuation events separate, costs applied to every eligible row',
        total_cost=total,mean_cost_per_trajectory=total/max(1,len(summaries)),
        decisions_with_future_cost=sum(c>0 for c in costs),
        objective='terminal win share minus future capped insolvency costs; official critic targets unchanged')
