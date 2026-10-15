"""Authored permitted positions reconstructed by the pinned native test host."""
import copy
from brass_ai.beliefs import sample_world_setup
from brass_ai.host import RulesHost
from brass_ai.learning.contracts import SOURCE_ROOT,read_json

COMPOSITION=read_json(SOURCE_ROOT/'configs/contracts/v0.2.0/card-composition.json')


class Factory:
    def __init__(self):self.host=RulesHost();self.reset()
    def close(self):self.host.close()
    def reset(self):self.context=self.host.reset(1201);return self.view()
    def view(self):return self.host.observe(self.context['actor'])
    def position(self):return self.view(),self.host.choices()
    def load(self,view):self.context=self.host.reset(99,sample_world_setup(view,COMPOSITION,99));return self.position()
    def choose(self,kind,**fields):
        choice=next(c for c in self.host.choices() if c['kind']==kind and all(c.get(k)==v for k,v in fields.items()))
        self.context=self.host.apply(self.context,choice);return choice
    def put(self,view,slot,kind,level=1,seat=None,resource=None,flipped=False):
        seat=view['seat'] if seat is None else seat
        mat=next(m for m in view['players'][seat]['mat'] if m['type']==kind)
        index=next(i for i,t in enumerate(mat['tiles']) if t['level']==level)
        tile=mat['tiles'][index];del mat['tiles'][:index+1];tile['flipped']=flipped
        if resource is not None:tile['resource']=resource
        next(s for s in view['slots'] if s['id']==slot)['tile']=tile
        return tile
    def connect(self,view,link,seat=None):
        next(l for l in view['links'] if l['id']==link)['owner']=view['seat'] if seat is None else seat
    def rail(self,view):
        view['context'].update(era='RailEra',round=2,actions_remaining=2)
        for p in view['players']:view['deck_count']+=p['hidden_discards'];p['hidden_discards']=0
    def hand(self,view,cards):
        view['deck_count']+=len(view['own_hand'])-len(cards)
        view['players'][view['seat']]['hand_count']=len(cards);view['own_hand']=sorted(cards)


def benchmark_positions():
    f=Factory();out=[]
    def save(name):
        view,choices=f.position();out.append(dict(name=name,view=view,choices=choices))
    try:
        save('canal-opening')
        f.choose('develop',industry='CottonMill');save('develop-payment')
        f.choose('iron');save('develop-continuation')
        v=f.reset();f.put(v,'Worcester:0','CottonMill');f.put(v,'Worcester:1','CottonMill')
        f.connect(v,'CanalWorcesterGloucester')
        for m in v['merchants']:
            if m['id'].startswith('Gloucester'):m.update(types=['CottonMill'],beer=1)
        f.load(v);save('shared-merchant-beer')
        f.choose('sell',target='Worcester:0',merchant='Gloucester:0');save('sale-payment')
        f.choose('beer');save('free-development-bonus')
        f.choose('bonus_develop',industry='Brewery');save('sale-continuation')
        v=f.reset();f.rail(v);v['players'][v['seat']]['cash']=40
        f.put(v,'Birmingham:1','Manufacturer',level=2,flipped=True)
        f.put(v,'Coventry:1','CoalMine',level=2,seat=(v['seat']+1)%4,resource=1)
        f.put(v,'Dudley:0','CoalMine',level=3,seat=(v['seat']+1)%4,resource=3)
        f.put(v,'Central:0','Brewery',level=2,resource=2)
        f.load(v);save('rail-action')
        f.choose('network',target='RailBirminghamCoventry');save('rail-first-coal')
        f.choose('coal',target='Coventry:1');save('rail-second-link')
        f.choose('network',target='RailBirminghamDudley');save('rail-second-payment')
        v=f.reset();debtor=(v['seat']+1)%4;v['context'].update(turn=4,actions_remaining=1)
        v['players'][debtor].update(cash=0,income_marker=0,vp=12)
        f.put(v,'Dudley:0','CoalMine',seat=debtor,resource=2)
        f.put(v,'Dudley:1','IronWorks',seat=debtor,resource=4)
        f.load(v);f.choose('pass');f.choose('finish');save('pending-debt')
        f.choose('liquidate',target='Dudley:0');save('partial-debt')
        f.choose('liquidate',target='Dudley:1');save('settled-next-round')
        return out
    finally:f.close()
