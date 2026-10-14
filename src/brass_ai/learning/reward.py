"""Versioned, undiscounted training utility; official scoring stays separate."""
import math
import numpy as np

from .contracts import digest, require
from .replay import terminal_targets

VERSION = 'brass-combined-return/1.0.0'
COEFFICIENTS = dict(win=1.0, placement=.10, vp=.001)


def specification(coefficients=None, insolvency=None):
    result=dict(version=VERSION, **(COEFFICIENTS if coefficients is None else coefficients),
                insolvency=dict(event_cost=.005,max_cost=.075) if insolvency is None else dict(insolvency))
    validate(result)
    return result


def validate(spec):
    require(type(spec) is dict and set(spec)=={'version','win','placement','vp','insolvency'} and spec['version']==VERSION,
            'Unknown combined reward contract')
    require(all(type(spec[k]) in (int,float) and math.isfinite(spec[k]) and spec[k]>=0
                for k in COEFFICIENTS) and spec['win']>0,'Invalid reward coefficients')
    costs=spec['insolvency']
    require(set(costs)=={'event_cost','max_cost'} and all(type(v) in (int,float) and math.isfinite(v) for v in costs.values())
            and 0<costs['event_cost']<=costs['max_cost']<=.1,'Invalid reward insolvency budget')
    return spec


def terminal_components(result):
    targets=terminal_targets(result)
    scores=[(s['vp'],s['income'],s['cash']) for s in result['seats']]
    # Exact ties after BOTH official tiebreakers receive half pairwise credit.
    placement=[sum(float(a>b)+.5*float(a==b) for j,b in enumerate(scores) if i!=j)/3
               for i,a in enumerate(scores)]
    return dict(win=targets['win_share'],placement=placement,vp=targets['vp'])


def terminal_utility(result,spec):
    parts=terminal_components(result)
    return np.asarray([sum(spec[k]*parts[k][s] for k in COEFFICIENTS) for s in range(4)],dtype=np.float64)


def replay_returns(records,result,spec):
    """Four-seat future costs and budgets at each predecision observation."""
    from brass_ai.recovery.insolvency import events
    charges={};spent=np.zeros(4);cap=spec['insolvency']['max_cost']
    for event in events(records):
        s=event['seat'];cost=min(spec['insolvency']['event_cost'],max(0.,cap-spent[s]))
        spent[s]+=cost;charges.setdefault(event['before'],np.zeros(4))[s]+=cost
    total=spent.copy();past=np.zeros(4);terminal=terminal_utility(result,spec)
    output=[]
    for i in range(len(records)):
        past+=charges.get(i,0.)
        output.append(dict(remaining=(cap-past).clip(0).tolist(),
                           utility=(terminal-(total-past)).tolist()))
    return output,dict(components=terminal_components(result),cost=total.tolist(),
                       utility=(terminal-total).tolist(),reward=spec,reward_id=digest(spec))


class CostLedger:
    """Incremental public event accounting for live/simulated paths, never per tile."""
    def __init__(self,spec,spent=None,settled=()):
        self.spec=validate(spec);self.spent=np.asarray([0.]*4 if spent is None else spent,dtype=float).copy()
        self.settled=set(map(tuple,settled));self.previous=None
        require(self.spent.shape==(4,) and np.isfinite(self.spent).all() and
                (self.spent>=0).all() and (self.spent<=spec['insolvency']['max_cost']+1e-8).all(),
                'Invalid spent insolvency budget')

    @property
    def remaining(self):return np.maximum(0.,self.spec['insolvency']['max_cost']-self.spent).tolist()

    def snapshot(self):return dict(spent=self.spent.tolist(),settled=sorted(self.settled))

    def charge(self,seat):
        self.spent[seat]+=min(self.spec['insolvency']['event_cost'],self.remaining[seat])

    def observe(self,view):
        context=view['context'];key=(context['era'],context['round'])
        # The first debt view exposes all initial shortfalls. Process it before
        # attempting automatic settlement inference from the preceding finish.
        if context['phase']=='debt' and key not in self.settled:
            self.settled.add(key)
            for p in view['players']:
                if p['debt']>0:self.charge(p['seat'])
        if self.previous is not None:
            before,choice=self.previous;old=before['context'];oldkey=(old['era'],old['round'])
            boundary=key!=oldkey or context.get('terminal',False)
            final_rail=context.get('terminal',False) and old['era']=='RailEra'
            if boundary and not final_rail and oldkey not in self.settled and old['phase']=='continuation' and choice['kind']=='finish':
                self.settled.add(oldkey)
                owners={s['tile']['seat'] for s in before['slots'] if s.get('tile')}
                for p in before['players']:
                    if p['cash']+p['income']<0:
                        require(p['seat'] not in owners,'Missing simulated debt decisions')
                        self.charge(p['seat'])
        self.previous=None

    def chosen(self,view,choice):self.previous=(view,choice)
