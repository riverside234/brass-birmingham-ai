"""Pure public rule calculations for the pinned four-player rules profile.

These are deterministic accounting/eligibility facts, not strategic scores.
Native engine transitions are the test oracle; no engine is accepted here.
"""
import math
from .contracts import INDUSTRIES,Measure


def income_level(marker):
    if marker<11:return marker-10
    if marker<31:return (marker-10+1)//2
    if marker<61:return (marker+2)//3
    return min(30,(marker+20+3)//4)


def loan_marker(marker):
    target=income_level(marker)-3
    while income_level(marker)!=target and marker>0:marker-=1
    return marker


def market_buy(kind,stock,count=1):
    capacity,empty=(14,8) if kind=='coal' else (10,6)
    return sum(empty if stock-i<=0 else (capacity-stock+i)//2+1 for i in range(count))


def market_sell(kind,stock,count):
    capacity=14 if kind=='coal' else 10
    return sum((capacity-stock-1-i)//2+1 for i in range(min(count,capacity-stock)))


def eligible(tile,era):
    return not (tile['rail_only'] if era=='CanalEra' else tile['canal_only'])


def has_settlement(context):
    return not context['terminal'] and not(context['era']=='RailEra' and context['round']==8)


class BoardFacts:
    """One connected-component and resource inventory pass per permitted view."""
    def __init__(self,view):
        self.view=view;self.seat=view['seat'];self.slots={s['id']:s for s in view['slots']}
        self.links={l['id']:l for l in view['links']};self.merchants={m['id']:m for m in view['merchants']}
        self.mats={m['type']:m['tiles'] for m in view['players'][self.seat]['mat']}
        self.graph={}
        for link in view['links']:
            if link.get('owner') is None:continue
            for region in link['regions']:self.graph.setdefault(region,set()).update(link['regions'])
        self.components={}
        regions=set(self.graph)|{self.region(s) for s in self.slots}|{self.region(m) for m in self.merchants}
        for region in sorted(regions):
            if region in self.components:continue
            seen={region};todo=[region]
            while todo:
                for other in self.graph.get(todo.pop(),()):
                    if other not in seen:seen.add(other);todo.append(other)
            frozen=frozenset(seen)
            for other in seen:self.components[other]=frozen
        self.resources={k:tuple(s for s in view['slots'] if s.get('tile',{}).get('type')==k and s['tile']['resource']>0)
                        for k in ('CoalMine','IronWorks','Brewery')}

    @staticmethod
    def region(identity):return identity.split(':')[0]
    def reach(self,starts):return frozenset().union(*(self.components.get(r,frozenset((r,))) for r in starts))
    def beer(self,region):
        reach=self.reach((region,))
        return sum(s['tile']['resource'] for s in self.resources['Brewery']
                   if s['tile']['seat']==self.seat or self.region(s['id']) in reach)
    def merchants_for(self,slot):
        tile=slot['tile'];reach=self.reach((self.region(slot['id']),))
        return tuple(m for m in self.merchants.values() if tile['type'] in (m.get('types') or ()) and self.region(m['id']) in reach)
    def resource_cost(self,coal,iron,starts):
        reach=self.reach(starts)
        free_coal=sum(s['tile']['resource'] for s in self.resources['CoalMine'] if self.region(s['id']) in reach)
        free_iron=sum(s['tile']['resource'] for s in self.resources['IronWorks'])
        paid_coal=max(0,coal-free_coal);paid_iron=max(0,iron-free_iron)
        if paid_coal and not any(self.region(m) in reach for m in self.merchants):return Measure.interval(0,None)
        return Measure.exact(market_buy('coal',self.view['coal_market'],paid_coal)+market_buy('iron',self.view['iron_market'],paid_iron))
