"""Direct one-second v3 Masked ISMCTS bank and private CPU root forests.

The strength receipt is a small Balanced repair screen, not per-style ranking.
Historical banks, reports and their original clocks are preserved unchanged.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import weakref
from brass_ai.bank import FrozenOpponent,admission,freeze,load_bank,read,sha
from brass_ai.learning.contracts import digest,require
from brass_ai.learning.legacy_policy import LegacyPolicy
from brass_ai.search.common import turn_key
from brass_ai.step5.league import write_json

STYLES=('balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex')

def prepare(root, *, publish=True):
    root=Path(root).resolve();runtime=read(root/'artifacts/legacy-ai/runtime/runtime.json')
    for item in runtime['sources']:require(sha(root/item['path'])==item['sha256'],'Rebuild Masked ISMCTS v3 first')
    archive=root/'training/opponents/archive/pre-masked-pool.json'
    if not archive.exists():write_json(archive,read(root/'training/opponents/pool.json'))
    names=[]
    for style in STYLES:
        name=style+'-v3-masked-strength';profile=dict(read(root/f'configs/legacy-ai/{style}-v3.json'),
            id='legacy-hard-'+name,mode='masked_ismcts',milliseconds=1000,work_limit=1000000,
            simulations=100000,root_top_k=4,interior_top_k=2,max_choices=512,exploration=.6,
            search_contract="masked-ismcts/1.1.0",evaluator_kind="heuristic",q_transform="regularized_sibling",leaf_policy="settled_reply_prefix",opponent_turn_limit=0,
            candidate_policy="supported_progressive",selection_policy="puct",minimum_plan_visits=2,reopen_support=4,fpu_reduction=.02,q_min_span=.05,tail_admission=.05,
            cash_policy=1,conversion_policy=1,partition_index=0,partition_count=1)
        path=root/f'configs/legacy-ai/{name}.json'
        if path.exists():require(read(path)==profile,'Refusing to relabel an existing Masked profile')
        else:write_json(path,profile)
        names.append(name)
    bank=freeze(root,names);manifest=root/f'training/opponents/bank/{bank}/manifest.json'
    pool=dict(format='legacy-pool/2.0.0',backend='masked_ismcts',training={},evaluation={},
        note='Direct user-selected 1s clocks; no calibration or strength qualification.')
    for name in names:
        purpose='evaluation' if name.startswith('pottery-flex') else 'training'
        pool[purpose][bank+'/'+name]=dict(admission='user-directed-masked-ismcts/1.0.0',manifest_sha256=sha(manifest),profile=name)
    if publish:
        receipt=root/'training/opponents/evaluations/masked-strength-1.1.0/report.json'
        report=read(receipt)
        require(report['bank']==bank and report['qualified'] and report['screen']['games']==20,"Complete this bank's 20-game strength screen before publication")
        for purpose in ('training','evaluation'):
            for entry in pool[purpose].values():entry.update(admission='masked-strength-screen/1.1.0',qualification=receipt.relative_to(root).as_posix(),sha256=sha(receipt))
        pool['note']='One-second clocks; Balanced repair screened against pre-repair v3. Other styles share the qualified engine and native correctness checks; no per-style strength claim.'
        previous=root/'training/opponents/archive/pre-strength-pool.json'
        if not previous.exists():write_json(previous,read(root/'training/opponents/pool.json'))
        write_json(root/'training/opponents/pool.json',pool)
    record=dict(format='masked-v3-bank/1.1.0',bank=bank,manifest_sha256=sha(manifest),seconds=1.,backend='masked_ismcts',profiles=names,strength='unmeasured',qualification_required=False)
    if publish:
        record.update(strength='screened-balanced-repair',qualification_required=True,qualification=receipt.relative_to(root).as_posix(),qualification_sha256=sha(receipt))
        write_json(root/'training/opponents/current.json',record);write_json(root/'training/opponents/lab/current.json',record)
    return record

def legacy_spec(root,parent,budget=None,purpose='training'):
    admission(root,parent,purpose)
    profile=load_bank(root,parent.split('/')[0])['payload']['profiles'][parent.split('/')[1]]
    require(profile['mode']=='masked_ismcts','Historical search is not an active opponent')
    if not budget:return dict(kind='legacy',policy_id=parent,purpose=purpose)
    from .config import legacy_budget_config
    legacy_budget_config(budget)
    identity='masked-v3-forest/'+digest(dict(parent=parent,budget=budget,version=2))
    return dict(kind='lab-legacy',parent=parent,policy_id=identity,budget=budget,purpose=purpose)

def teachers(root,budget=None):
    return {legacy_spec(root,p,budget)['policy_id'] for p in read(Path(root)/'training/opponents/pool.json')['training']}

class LabLegacyPolicy(LegacyPolicy):
    def __init__(self,root,parent,budget,*,mono=None,purpose='training',diagnostic=False):
        self.root,self.parent,self.budget=Path(root),parent,budget
        self.policy_id=legacy_spec(root,parent,budget,purpose)['policy_id']
        self.workers=[];self.contexts=weakref.WeakKeyDictionary()
        try:
            for _ in range(budget['search_workers']):self.workers.append(FrozenOpponent(root,parent,purpose=purpose,mono=mono))
        except BaseException:
            for w in self.workers:w.close()
            raise
        self.opponent=self.workers[0];self.pool=ThreadPoolExecutor(max_workers=len(self.workers))
        self.telemetry=dict(turns=0,branches=0,nodes=0,unique_plans=0,failures=0,over_budget_turns=0,max_turn_seconds=0.,seconds=budget['seconds'],search_workers=len(self.workers),completed=0)

    def decision(self,view,choices,state):
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
