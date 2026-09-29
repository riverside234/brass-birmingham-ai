"""R2 permitted-input adapter. State is private to one game/seat RNG context."""
from __future__ import annotations
from collections import deque
import time
import weakref
from brass_ai.bank import FrozenOpponent
from .contracts import require


class LegacyPolicy:
    kind='external'
    def __init__(self,root,policy_id,*,purpose='training',mono=None):
        self.opponent=FrozenOpponent(root,policy_id,purpose=purpose,mono=mono)
        self.policy_id=policy_id;self.contexts=weakref.WeakKeyDictionary()
    def select_batch(self,inputs):
        output=[]
        for view,choices,rng in inputs:
            c=view['context'];seat=view['seat'];revision=c['revision']
            key=(c['era'],c['round'],c['turn'],seat)
            state=self.contexts.setdefault(rng,dict(revision=-1,turn=None,intent=None,queue=deque(),deadline=0,work=0,reply_work=0))
            if revision<=state['revision']:
                state.update(revision=-1,turn=None,intent=None,queue=deque())
            if state['turn']!=key:
                state['turn']=key;state['queue'].clear()
                state['work']=state['reply_work']=0
                state['deadline']=time.monotonic()+self.opponent.profile['milliseconds']/1000
            elif revision!=state['revision']+1:state['queue'].clear()
            ids=[x['id'] for x in choices]
            if state['queue'] and state['queue'][0] not in ids:state['queue'].clear()
            if not state['queue']:
                profile=dict(self.opponent.profile,milliseconds=max(0,1000*(state['deadline']-time.monotonic())),
                    work_limit=max(1,self.opponent.profile['work_limit']-state['work']),
                    reply_work=max(1,self.opponent.profile.get('reply_work',2048)-state['reply_work']))
                result=self.opponent.worker.call('plan',observation=view,choices=choices,profile=profile,
                    fixed_work=self.opponent.fixed_work,sampling_seed=1,intent=state['intent'])
                state['queue'].extend(x for action in result['selected']['actions'] for x in action)
                state['intent']=result['selected'].get('intent')
                state['work']+=result['nodes'];state['reply_work']+=result.get('replies',{}).get('choices',0)
            chosen=state['queue'].popleft();require(chosen in ids,'Frozen opponent returned an illegal continuation')
            state['revision']=revision
            output.append((ids.index(chosen),1.0,None))
        return output
    def context_features(self,view,rng):
        state=self.contexts[rng];intent=state['intent'] or {};style=self.opponent.profile.get('tactic','balanced')
        group={'cotton':0,'goods-beer':0,'pottery-flex':0,'rail-tempo':1,'supplier':2,'balanced':3}[style]
        styles=[float(i==group) for i in range(4)]
        industries=['CottonMill','Manufacturer','Pottery','Brewery','IronWorks','CoalMine']
        vector=[float(intent.get('focus')==x) for x in industries]+[float(bool(intent.get('target'))),min(1,intent.get('age',0)/4)]
        return styles,vector
    def close(self):
        self.contexts.clear();self.opponent.close()
