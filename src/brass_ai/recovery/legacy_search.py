"""Direct one-second v3 Masked ISMCTS bank and private CPU root forests.

The strength receipt is a small Balanced repair screen, not per-style ranking.
Historical banks, reports and their original clocks are preserved unchanged.
"""
from collections import deque
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import weakref
from brass_ai.bank import FrozenOpponent,admission,load_bank,read,sha
from brass_ai.learning.contracts import digest,require
from brass_ai.learning.legacy_policy import LegacyPolicy
from brass_ai.search.common import turn_key
from brass_ai.search.legacy_runtime import reference as shared_reference, validate_reference, runtime_directory
from brass_ai.legacy import LegacyWorker
from brass_ai.step5.league import write_json
from .config import time_budget

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

def legacy_spec(root,parent,budget=None,purpose='training',*,search_free=False,work_budget=None,search_runtime=None):
    admission(root,parent,purpose)
    payload=load_bank(root,parent.split('/')[0])['payload']
    profile=payload['profiles'][parent.split('/')[1]]
    require(profile['mode']=='masked_ismcts','Historical search is not an active opponent')
    if budget:
        from .config import legacy_budget_config
        legacy_budget_config(budget)
    if work_budget is not None:
        from .config import legacy_opponent_work_budget_config
        legacy_opponent_work_budget_config(work_budget)
        require(purpose in ('training','evaluation') and budget and
                (not fixed_simulations(work_budget) or budget['role'] in ('lab','reference')) and
                (not search_free or fixed_simulations(work_budget) or time_budget(work_budget)), 'Work-budget v4 requires a lab policy')
        require('src/BrassMaskedSearch/RootForest.cs' in payload['runtime_source_paths'],
                'Work-budget v4 requires the frozen native root forest')
        if not time_budget(work_budget):
            workers=budget['search_workers'];total=work_budget['simulations_per_turn']
            require(total%workers==0 and workers*4<=total<=workers*profile['simulations'],
                    'Legacy work must divide across workers and retain matched four-attempt blocks')
    if not budget:return dict(kind='legacy',policy_id=parent,purpose=purpose)
    identity=('masked-v3-fast/' if search_free else 'masked-v3-forest/')+digest(dict(parent=parent,budget=budget,search_free=search_free,version=3))
    result=dict(kind='lab-legacy',parent=parent,policy_id=identity,budget=budget,purpose=purpose,search_free=search_free)
    if work_budget is not None:
        result['work_budget']=copy.deepcopy(work_budget)
        identity=dict(parent=parent,budget=budget,work_budget=work_budget,version=1)
        if fixed_simulations(work_budget) or time_budget(work_budget):identity.update(version=2,search_free=search_free)
        result['policy_id']=('masked-v4-work-fast/' if search_free else 'masked-v4-work/')+digest(identity)
    if work_budget is not None and search_runtime is not False:
        ref=shared_reference(root,parent) if search_runtime is None else validate_reference(search_runtime)
        if ref is not None:
            require(ref['recipe']['bank']==parent.split('/')[0],'Shared search belongs to another frozen strategy')
            result['search_runtime']=ref
            result['policy_id']='shared-legacy-search/'+digest(dict(parent=result['policy_id'],runtime=ref['id']))
    return result

def teachers(root,budget=None,work_budget=None):
    return {legacy_spec(root,p,budget,work_budget=work_budget)['policy_id'] for p in read(Path(root)/'training/opponents/pool.json')['training']}


def fixed_simulations(work_budget):
    return work_budget is not None and work_budget['contract']=='legacy-opponent-turn-work/1.1.0'


def native_node_ceiling(profile,simulations):
    # The frozen FastIncumbents has at most 26 FastPlans, each at most 96
    # choices plus one world setup. 4096 keeps every one of those candidate
    # families and leaves headroom; each simulation adds <= max_choices+1.
    # This is a deterministic safety bound, not a replacement for simulations.
    return min(profile['work_limit'],4096+simulations*(profile.get('max_choices',512)+1))

class LabLegacyPolicy(LegacyPolicy):
    def __init__(self,root,parent,budget,*,mono=None,purpose='training',diagnostic=False,search_free=False,work_budget=None,search_runtime=None):
        self.root,self.parent,self.budget=Path(root),parent,budget
        spec=legacy_spec(root,parent,budget,purpose,search_free=search_free,work_budget=work_budget,search_runtime=search_runtime)
        self.policy_id=spec['policy_id']
        self.work_budget=copy.deepcopy(work_budget)
        if search_free and time_budget(work_budget):
            # Search-free planning retains its prior node cap and exceptional
            # watchdog. The one-second clock applies only to searched v4.
            from .config import LEGACY_OPPONENT_WORK_BUDGET
            self.work_budget=copy.deepcopy(LEGACY_OPPONENT_WORK_BUDGET)
            work_budget=self.work_budget
        self.search_free=search_free
        self.native_forest='src/BrassMaskedSearch/RootForest.cs' in load_bank(root,parent.split('/')[0])['payload']['runtime_source_paths']
        self.workers=[];self.contexts=weakref.WeakKeyDictionary()
        try:
            for _ in range(1 if self.native_forest or search_free else budget['search_workers']):
                if spec.get('search_runtime'):
                    from types import SimpleNamespace
                    ref=spec['search_runtime']
                    profile=read(Path(root)/'configs/legacy-ai'/(parent.split('/')[1]+'.json'))
                    worker=LegacyWorker(root,mono=mono,timeout=600,runtime_directory=runtime_directory(root,ref))
                    self.workers.append(SimpleNamespace(worker=worker,profile=profile,close=worker.close))
                else:self.workers.append(FrozenOpponent(root,parent,purpose=purpose,mono=mono))
        except BaseException:
            for w in self.workers:w.close()
            raise
        self.opponent=self.workers[0];self.pool=ThreadPoolExecutor(max_workers=len(self.workers))
        self.cpu_workers=1 if search_free else budget['search_workers']
        self.telemetry=dict(turns=0,branches=0,nodes=0,unique_plans=0,failures=0,over_budget_turns=0,max_turn_seconds=0.,seconds=budget['seconds'],search_workers=self.cpu_workers,search_free=search_free,completed=0,cpu_batching={})
        self.telemetry.update(search_elapsed_seconds=0.,time_budget_exhausted_turns=0,deadline_fallback_choices=0)
        self.telemetry.update(search_runtime=spec.get('search_runtime',{}).get('id'),deadline_admission_stops=0,
            deadline_unused_worker_seconds=0.,interrupted=0)
        if work_budget is not None:
            self.telemetry.update(work_budget=copy.deepcopy(work_budget),allocated_simulations=0,attempted_simulations=0,
                work_exhausted_turns=0,deadline_limited_turns=0,incumbent_only_decisions=0,forced_choices=0)
            if fixed_simulations(work_budget):
                self.telemetry.update(timing_mode='simulation-count-with-watchdog',reference_seconds=budget['seconds'],
                    seconds=work_budget['watchdog_seconds'],watchdog_failures=0,incomplete_work_failures=0,
                    candidate_node_ceiling=4096)
            elif time_budget(work_budget):self.telemetry['timing_mode']='shared-whole-turn-time'

    def admit_legacy_turn(self,context,rng):
        # The actor calls this after its whole worker group is admitted and
        # before observation/encoding. Forced choices never reset the deadline.
        if self.search_free or not time_budget(self.work_budget):return
        self._start_turn(context,rng)

    def _start_turn(self,context,rng):
        key=turn_key(context)
        state=self.contexts.setdefault(rng,dict(turn=None,revision=-1,intent=None,queue=deque(),work=0))
        if state['turn']!=key:
            began=time.monotonic()
            state.update(turn=key,revision=-1,queue=deque(),work=0,work_allocated=0,overrun=False,
                deadline=began+self.budget['seconds'],began=began,seed=rng.randrange(1,2**32))
            self.telemetry['turns']+=1
        return state

    def decision(self,view,choices,state):
        if self.native_forest or self.search_free:
            timed=time_budget(getattr(self,'work_budget',None)) and not self.search_free
            profile=dict(self.opponent.profile,milliseconds=max(0,1000*(state['deadline']-time.monotonic()-(.02 if timed else 0.))),
                incumbent_only=self.search_free,work_limit=max(1,self.opponent.profile['work_limit']-state['work']))
            if timed:profile['simulations']=100000 # emergency guard, not a target/quota
            # A fixed-count budget keeps its exceptional watchdog. Tail
            # admission must not voluntarily withhold required attempts.
            profile['deadline_admission']=not fixed_simulations(getattr(self,'work_budget',None))
            work_budget=getattr(self,'work_budget',None)
            allocated=0
            if work_budget is not None and not time_budget(work_budget):
                # The frozen native forest copies this cap to every worker.
                # Issue it ONCE for the whole turn. Node-work limits are separate:
                # nodes include setup/incumbent work, not simulation attempts.
                if not self.search_free and not state['work_allocated'] and view['context']['phase']!='debt' and len(choices)>1:
                    allocated=work_budget['simulations_per_turn']
                    state['work_allocated']=allocated;self.telemetry['allocated_simulations']+=allocated
                    profile['simulations']=allocated//self.cpu_workers
                else:
                    profile['incumbent_only']=True;self.telemetry['incumbent_only_decisions']+=1
                if fixed_simulations(work_budget):
                    cap=native_node_ceiling(self.opponent.profile,0 if self.search_free else work_budget['simulations_per_turn']//self.cpu_workers)
                    # Native reports SUM every private owner's node work. Keep
                    # the remaining allowance in forest units before assigning
                    # a per-owner cap; subtracting the sum from one owner's cap
                    # could incorrectly discard fallback candidate families.
                    profile['work_limit']=max(1,(cap*self.cpu_workers-state['work'])//self.cpu_workers)
            began=time.monotonic()
            d=self.opponent.worker.call('plan_forest' if self.native_forest else 'plan',observation=view,choices=choices,
                profile=profile,fixed_work=False,sampling_seed=state['seed'],intent=state['intent'],search_workers=self.cpu_workers)
            self.telemetry['search_elapsed_seconds']+=time.monotonic()-began
            require(d['mode']=='masked_ismcts' and d['failures']==0,'Legacy search failed; reject incomplete game')
            state['work']+=d['nodes'];stats=d['search']
            require(stats['opponent_nodes']==0,'Opponent nodes cannot be merged')
            self.telemetry['branches']+=self.cpu_workers;self.telemetry['nodes']+=d['nodes']
            self.telemetry['completed']+=stats['completed'];self.telemetry['unique_plans']+=len(d.get('alternatives',[]))+1
            self.telemetry['interrupted']+=stats['interrupted']
            self.telemetry['deadline_admission_stops']+=stats.get('deadline_stops',0)
            self.telemetry['deadline_unused_worker_seconds']+=stats.get('deadline_remaining_ms',0.)/1000
            self.telemetry['cpu_batching']=stats.get('cpu_batching',{})
            if self.search_free:require(stats['completed']==0,'Fast opponent performed MCTS')
            if time_budget(work_budget):
                self.telemetry['attempted_simulations']+=stats['completed']+stats['interrupted']
                self.telemetry['time_budget_exhausted_turns']+=int(d.get('overrun',False) or time.monotonic()>=state['deadline']-.02)
            elif work_budget is not None:
                # The frozen loop records exactly one completion/interruption
                # per launched trajectory. There are no retries past its cap.
                attempted=stats['completed']+stats['interrupted']
                require(0<=stats['completed']<=attempted<=allocated,'Legacy forest exceeded its shared turn allowance')
                self.telemetry['attempted_simulations']+=attempted
                if allocated:
                    self.telemetry['work_exhausted_turns']+=int(attempted==allocated)
                    self.telemetry['deadline_limited_turns']+=int(d.get('overrun',False) or time.monotonic()>=state['deadline'])
                if fixed_simulations(work_budget):
                    timed_out=d.get('overrun',False) or time.monotonic()>=state['deadline']
                    if timed_out or attempted!=allocated:
                        self.telemetry['watchdog_failures']+=int(timed_out)
                        self.telemetry['incomplete_work_failures']+=int(attempted!=allocated)
                        self.telemetry['failures']+=1
                        raise TimeoutError('Legacy fixed-work search incomplete: '
                            f'allocated={allocated}, attempted={attempted}, completed={stats["completed"]}, '
                            f'interrupted={stats["interrupted"]}, nodes={d["nodes"]}, watchdog_expired={timed_out}')
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
            work_budget=getattr(self,'work_budget',None)
            timed=time_budget(work_budget) and not self.search_free
            seconds=work_budget['watchdog_seconds'] if fixed_simulations(work_budget) else self.budget['seconds']
            if timed:state=self._start_turn(c,rng)
            elif state['turn']!=key or (c['revision']<=state['revision'] and work_budget is None):
                state.update(turn=key,revision=-1,queue=deque(),work=0,work_allocated=0,overrun=False,deadline=time.monotonic()+seconds,began=time.monotonic(),seed=rng.randrange(1,2**32));self.telemetry['turns']+=1
            elif c['revision']!=state['revision']+1:state['queue'].clear()
            if timed and c['revision']!=state['revision']+1:state['queue'].clear()
            ids=[p['id'] for p in choices]
            if state['queue'] and state['queue'][0] not in ids:state['queue'].clear()
            if work_budget is not None and len(choices)==1:
                if state['queue']:state['queue'].popleft()
                self.telemetry['forced_choices']+=1;selected=ids[0]
            else:
                if not state['queue']:
                    if timed and time.monotonic()>=state['deadline']-.02:
                        # An unexpected observation invalidated the completed
                        # plan. Use current legal choices; never launch a new
                        # incumbent RPC after expiry or invent another second.
                        index=max(range(len(choices)),key=lambda i:(choices[i]['kind']=='finish',-i))
                        state['queue'].append(ids[index]);self.telemetry['deadline_fallback_choices']+=1
                    else:
                        plan=self.decision(view,choices,state);state['queue'].extend(x for a in plan['actions'] for x in a);state['intent']=plan.get('intent')
                selected=state['queue'].popleft()
            require(selected in ids,'Illegal legacy continuation')
            elapsed=time.monotonic()-state['began'];self.telemetry['max_turn_seconds']=max(self.telemetry['max_turn_seconds'],elapsed)
            if elapsed>seconds and not state['overrun']:self.telemetry['over_budget_turns']+=1;state['overrun']=True
            state['revision']=c['revision'];output.append((ids.index(selected),1.,None))
        return output

    def close(self):
        self.pool.shutdown(wait=True);self.contexts.clear()
        for worker in self.workers:worker.close()
