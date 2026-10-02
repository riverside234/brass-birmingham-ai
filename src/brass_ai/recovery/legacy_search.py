"""Direct one-second v3 Masked ISMCTS bank and private CPU root forests.

The strength receipt is a small Balanced repair screen, not per-style ranking.
Historical banks, reports and their original clocks are preserved unchanged.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import weakref
from brass_ai.bank import FrozenOpponent,admission,load_bank,read,sha
from brass_ai.learning.contracts import digest,require
from brass_ai.learning.legacy_policy import LegacyPolicy
from brass_ai.search.common import turn_key
from brass_ai.step5.league import write_json

STYLES=('balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex')

CONTRACT = 'v3-s1-training/1.0.0'  # Archived heuristic BC provenance only.
SELECTED_BANK = 'bank-981744ddae4aaf07f1cf'


def prepare(root, *, publish=True):
    """Retain the existing qualified Masked bank after the requested S1 screen.

    The quick 16x2 comparison selects the training default; it does not create
    another engine identity or reinterpret the original strength qualification.
    """
    from brass_ai.bank import validate_admission
    root=Path(root).resolve()
    manifest=load_bank(root,SELECTED_BANK)
    pool=read(root/'training/opponents/archive/pre-s1-training-pool.json')
    require(pool['backend']=='masked_ismcts','Selected Masked pool changed')
    for purpose in ('training','evaluation'):
        for identity,entry in pool[purpose].items():
            require(identity.split('/')[0]==SELECTED_BANK,'Unexpected selected bank')
            validate_admission(root,identity,entry,purpose)
    profiles=list(manifest['payload']['profiles'])
    receipt=root/'training/opponents/evaluations/masked-strength-1.1.0/report.json'
    record=dict(format='masked-v3-bank/1.1.0',bank=SELECTED_BANK,
        manifest_sha256=sha(root/f'training/opponents/bank/{SELECTED_BANK}/manifest.json'),
        seconds=1.,backend='masked_ismcts',profiles=profiles,
        strength='screened-balanced-repair',qualification_required=True,
        qualification=receipt.relative_to(root).as_posix(),qualification_sha256=sha(receipt))
    if publish:
        selection=root/'training/opponents/evaluations/s1-vs-masked-smoke-1.0.0/report.json'
        screen=read(selection)
        require(screen['passed'] and screen['games']==8 and screen['selected_backend']=='masked_ismcts'
                and screen['masked_bank']==SELECTED_BANK,'Selected bank must match the requested completed comparison')
        record.update(selection_receipt=selection.relative_to(root).as_posix(),selection_sha256=sha(selection),
                      selection_scope='eight-game smoke at one second/two workers; no human-strength qualification')
        archive=root/'training/opponents/archive/s1-training-smoke-pool.json'
        current=read(root/'training/opponents/pool.json')
        if current.get('backend')=='S1' and not archive.exists():write_json(archive,current)
        write_json(root/'training/opponents/pool.json',pool)
        write_json(root/'training/opponents/current.json',record)
        write_json(root/'training/opponents/lab/current.json',record)
    return record

def legacy_spec(root,parent,budget=None,purpose='training',*,search_free=False):
    admission(root,parent,purpose)
    profile=load_bank(root,parent.split('/')[0])['payload']['profiles'][parent.split('/')[1]]
    require(profile['mode']=='masked_ismcts','Historical search is not an active opponent')
    if not budget:return dict(kind='legacy',policy_id=parent,purpose=purpose)
    from .config import legacy_budget_config
    legacy_budget_config(budget)
    identity=('masked-v3-fast/' if search_free else 'masked-v3-forest/')+digest(dict(parent=parent,budget=budget,search_free=search_free,version=3))
    return dict(kind='lab-legacy',parent=parent,policy_id=identity,budget=budget,purpose=purpose,search_free=search_free)

def teachers(root,budget=None):
    return {legacy_spec(root,p,budget)['policy_id'] for p in read(Path(root)/'training/opponents/pool.json')['training']}

class LabLegacyPolicy(LegacyPolicy):
    def __init__(self,root,parent,budget,*,mono=None,purpose='training',diagnostic=False,search_free=False):
        self.root,self.parent,self.budget=Path(root),parent,budget
        self.policy_id=legacy_spec(root,parent,budget,purpose,search_free=search_free)['policy_id']
        self.search_free=search_free
        self.native_forest='src/BrassMaskedSearch/RootForest.cs' in load_bank(root,parent.split('/')[0])['payload']['runtime_source_paths']
        self.workers=[];self.contexts=weakref.WeakKeyDictionary()
        try:
            for _ in range(1 if self.native_forest or search_free else budget['search_workers']):self.workers.append(FrozenOpponent(root,parent,purpose=purpose,mono=mono))
        except BaseException:
            for w in self.workers:w.close()
            raise
        self.opponent=self.workers[0];self.pool=ThreadPoolExecutor(max_workers=len(self.workers))
        self.cpu_workers=1 if search_free else budget['search_workers']
        self.telemetry=dict(turns=0,branches=0,nodes=0,unique_plans=0,failures=0,over_budget_turns=0,max_turn_seconds=0.,seconds=budget['seconds'],search_workers=self.cpu_workers,search_free=search_free,completed=0,cpu_batching={})

    def decision(self,view,choices,state):
        if self.native_forest or self.search_free:
            profile=dict(self.opponent.profile,milliseconds=max(0,1000*(state['deadline']-time.monotonic())),
                incumbent_only=self.search_free,work_limit=max(1,self.opponent.profile['work_limit']-state['work']))
            d=self.opponent.worker.call('plan_forest' if self.native_forest else 'plan',observation=view,choices=choices,
                profile=profile,fixed_work=False,sampling_seed=state['seed'],intent=state['intent'],search_workers=self.cpu_workers)
            require(d['mode']=='masked_ismcts' and d['failures']==0,'Legacy search failed; reject incomplete game')
            state['work']+=d['nodes'];stats=d['search']
            require(stats['opponent_nodes']==0,'Opponent nodes cannot be merged')
            self.telemetry['branches']+=self.cpu_workers;self.telemetry['nodes']+=d['nodes']
            self.telemetry['completed']+=stats['completed'];self.telemetry['unique_plans']+=len(d.get('alternatives',[]))+1
            self.telemetry['cpu_batching']=stats.get('cpu_batching',{})
            if self.search_free:require(stats['completed']==0,'Fast opponent performed MCTS')
            self.last_decision=dict(d,root=stats['root'],plans=stats['plans'],completed=stats['completed'],opponent_nodes=0)
            return d['selected']
        def branch(index):
            profile=dict(self.opponent.profile,milliseconds=max(0,1000*(state['deadline']-time.monotonic())),work_limit=max(1,self.opponent.profile['work_limit']-state['work']),partition_count=len(self.workers),partition_index=index)
            return self.workers[index].worker.call('plan',observation=view,choices=choices,profile=profile,fixed_work=False,sampling_seed=state['seed'],intent=state['intent'])
        # Drain actual work before returning/closing: a deadline is not completion.
        results=list(self.pool.map(branch,range(len(self.workers))))
        root={};completed=0
        for d in results:
            require(d['mode']=='masked_ismcts' and d['failures']==0,'Legacy search failed; reject incomplete game')
            stats=d['search'];require(stats['opponent_nodes']==0,'Opponent nodes cannot be merged');completed+=stats['completed']
            for identity,edge in stats['root'].items():
                target=root.setdefault(identity,dict(visits=0,returns=[0.]*4,available=0));target['visits']+=edge['visits'];target['available']+=edge['available']
                for seat in range(4):target['returns'][seat]+=edge['returns'][seat]
        plans={p['signature']:p for d in results for p in [d['selected']]+d.get('alternatives',[])}
        evidence={}
        for d in results:
            for signature,e in d['search'].get('plans',{}).items():
                require(e['visits']>0 and len(e['returns'])==4 and e['plan']['signature']==signature,'Invalid full-turn evidence')
                plans[signature]=e['plan']
                merged=evidence.setdefault(signature,dict(visits=0,returns=[0.]*4,scenarios={}))
                merged['visits']+=e['visits']
                for seat in range(4):merged['returns'][seat]+=e['returns'][seat]
                for scenario,value in e['scenarios'].items():merged['scenarios'].setdefault(scenario,[]).append(value)
        require(plans,'No legal complete incumbent')
        baselines=[plans[d['search'].get('incumbent_signature',d['selected']['signature'])] for d in results]
        baseline=max(baselines,key=lambda p:(-p['cash']['shortfall'],not p['cash']['future_credit_blocked'],p['score'],p['preference'],p['signature']))
        # Evidence belongs to this WHOLE signature, not every plan with its first edge.
        control=evidence.get(baseline['signature'],dict(visits=0,scenarios={}))
        minimum=self.opponent.profile.get('minimum_plan_visits',2)
        candidates=[]
        for signature,e in evidence.items():
            p=plans[signature];shared=set(e['scenarios']) & set(control['scenarios'])
            if e['visits']<minimum or control['visits']<minimum or len(shared)<minimum:continue
            if p['cash']['shortfall']>baseline['cash']['shortfall']:continue
            if not baseline['cash']['future_credit_blocked'] and p['cash']['future_credit_blocked']:continue
            floors=[d['search'].get('conversion_floor') for d in results if d['search'].get('conversion_floor') is not None]
            if floors and p['score']<max(floors):continue
            if baseline['ledger'].get('win_share') is not None:
                if p['ledger'].get('win_share',-1)<baseline['ledger']['win_share'] or p['ledger']['rank']>baseline['ledger']['rank']:continue
            mean=lambda values:sum(values)/len(values)
            advantage=sum(mean(e['scenarios'][k])-mean(control['scenarios'][k]) for k in shared)/len(shared)
            if advantage>=0:candidates.append((advantage,p['score'],p['preference'],signature,p))
        selected=max(candidates,key=lambda x:x[:4])[-1] if candidates else baseline
        state['work']+=sum(d['nodes'] for d in results)
        self.telemetry['branches']+=len(results);self.telemetry['nodes']+=sum(d['nodes'] for d in results);self.telemetry['unique_plans']+=len(plans);self.telemetry['completed']+=completed
        self.last_decision=dict(selected=selected,root=root,plans=evidence,completed=completed,opponent_nodes=0,failures=0,fallback=selected['signature']==baseline['signature'])
        return selected

    def select_batch(self,inputs):
        output=[]
        for view,choices,rng in inputs:
            c=view['context'];key=turn_key(c);state=self.contexts.setdefault(rng,dict(turn=None,revision=-1,intent=None,queue=deque(),work=0))
            if state['turn']!=key or c['revision']<=state['revision']:
                state.update(turn=key,revision=-1,queue=deque(),work=0,overrun=False,deadline=time.monotonic()+self.budget['seconds'],began=time.monotonic(),seed=rng.randrange(1,2**32));self.telemetry['turns']+=1
            elif c['revision']!=state['revision']+1:state['queue'].clear()
            ids=[p['id'] for p in choices]
            if state['queue'] and state['queue'][0] not in ids:state['queue'].clear()
            if not state['queue']:
                plan=self.decision(view,choices,state);state['queue'].extend(x for a in plan['actions'] for x in a);state['intent']=plan.get('intent')
            selected=state['queue'].popleft();require(selected in ids,'Illegal legacy continuation')
            elapsed=time.monotonic()-state['began'];self.telemetry['max_turn_seconds']=max(self.telemetry['max_turn_seconds'],elapsed)
            if elapsed>self.budget['seconds'] and not state['overrun']:self.telemetry['over_budget_turns']+=1;state['overrun']=True
            state['revision']=c['revision'];output.append((ids.index(selected),1.,None))
        return output

    def close(self):
        self.pool.shutdown(wait=True);self.contexts.clear()
        for worker in self.workers:worker.close()
