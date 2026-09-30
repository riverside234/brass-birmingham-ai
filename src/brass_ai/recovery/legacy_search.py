"""Separately identified, hardware-qualified lab v3 search. Frozen v3 stays intact.

Every replica has a private engine process and a deterministic root partition.
The coordinator shares one turn deadline and work allowance across both actions.
No qualification of the three-second parent is reused as admission of this child.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
import os
import platform
import time
import weakref

from brass_ai.bank import FrozenOpponent,admission,freeze,load_bank,read,sha
from brass_ai.learning.contracts import digest,project_path,require
from brass_ai.learning.legacy_policy import LegacyPolicy
from brass_ai.step5.league import write_json,implementation_id


def hardware():
    model=platform.processor()
    if Path('/proc/cpuinfo').exists():
        model=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),model)
    return dict(system=platform.system(),cpu=model,logical_cpus=os.cpu_count(),
        affinity=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count())


def prepare(root):
    """Run after build-legacy.ps1; produces authored child profiles/source snapshot."""
    root=Path(root).resolve();parents=sorted(read(root/'training/opponents/pool.json')['training'])
    source=read(root/'artifacts/legacy-ai/runtime/runtime.json')
    for item in source['sources']: require(sha(root/item['path'])==item['sha256'],'Rebuild lab legacy runtime first')
    old=root/'training/opponents/lab/current.json'
    signature=digest(dict(sources=source['sources'],parents=parents,version=1))
    if old.exists() and read(old)['signature']==signature:
        record=read(old);load_bank(root,record['bank']);return record
    names=[]
    for parent in parents:
        admission(root,parent,'training');bank,name=parent.split('/')
        profile=dict(load_bank(root,bank)['payload']['profiles'][name])
        require(profile['mode']=='S1','Lab partition calibration currently requires complete-turn S1 parents')
        child=name+'-lab-1s';names.append(child)
        profile.update(id=profile['id']+'-lab-1s',milliseconds=1000,partition_count=1,partition_index=0)
        write_json(root/'configs/legacy-ai'/(child+'.json'),profile)
    bank=freeze(root,names,fixed_work=False)
    record=dict(format='lab-v3-bank/1.0.0',signature=signature,bank=bank,
        parents={p:bank+'/'+n for p,n in zip(parents,names)},seconds=1.0,
        limitation='Not admitted until lab-calibrate passes on the target hardware; original bank unchanged')
    write_json(old,record);return record


def budget_identity(root,budget):
    record=read(Path(root)/'training/opponents/lab/current.json');load_bank(root,record['bank'])
    return dict(bank=record,seconds=budget['seconds'],workers=budget['search_workers'],work_multiplier=budget['work_multiplier'],
        implementation=implementation_id(),hardware=hardware(),fixed_work=False)


def require_calibration(root,budget):
    report=read(project_path(root,budget['calibration']))
    require(report['format']=='lab-v3-calibration/1.0.0' and report['identity']==budget_identity(root,budget),
        'Lab search budget/source/hardware changed; run lab-calibrate on this machine')
    require(report['training_eligible'] and report['qualification_games']>=25,
        'One-second lab teacher not admitted; complete calibration or explicitly use the reference budget')
    for parent in report['identity']['bank']['parents']: admission(root,parent,'training')
    for entry in report['evidence']: require(sha(project_path(root,entry['path']))==entry['sha256'],'Lab qualification evidence changed')
    return report


def legacy_spec(root,parent,budget=None,purpose='training'):
    admission(root,parent,purpose)
    if not budget or budget['role']!='lab': return dict(kind='legacy',policy_id=parent,purpose=purpose)
    require(purpose=='training','Held-out teachers cannot enter the lab training bank')
    report=require_calibration(root,budget);child=report['identity']['bank']['parents'][parent]
    identity='lab-v3/'+digest(dict(child=child,identity=report['identity']))
    return dict(kind='lab-legacy',parent=parent,budget=budget,policy_id=identity,
        calibration_sha256=sha(project_path(root,budget['calibration'])))


def teachers(root,budget=None):
    parents=sorted(read(Path(root)/'training/opponents/pool.json')['training'])
    return {legacy_spec(root,p,budget)['policy_id']:p for p in parents}


class LabLegacyPolicy(LegacyPolicy):
    def __init__(self,root,parent,budget,*,mono=None,diagnostic=False):
        self.root=Path(root);self.parent=parent;self.budget=budget
        identity=budget_identity(root,budget)
        if not diagnostic: require_calibration(root,budget)
        admission(root,parent,'training');child=identity['bank']['parents'][parent]
        self.policy_id='lab-v3/'+digest(dict(child=child,identity=identity))
        self.workers=[];self.contexts=weakref.WeakKeyDictionary()
        try:
            for _ in range(budget['search_workers']): self.workers.append(FrozenOpponent(root,child,purpose='diagnostic',mono=mono))
        except BaseException:
            for worker in self.workers: worker.close()
            raise
        self.opponent=self.workers[0];self.pool=ThreadPoolExecutor(max_workers=len(self.workers))
        self.telemetry=dict(turns=0,branches=0,nodes=0,unique_plans=0,failures=0,over_budget_turns=0,
            max_turn_seconds=0.,seconds=budget['seconds'],search_workers=len(self.workers))

    def decision(self,view,choices,state):
        remaining=max(0.,state['deadline']-time.monotonic())
        count=len(self.workers) if view['context']['phase']=='action' else 1
        allowance=max(1,self.opponent.profile['work_limit']*self.budget['work_multiplier']-state['work'])
        def branch(index):
            profile=dict(self.opponent.profile,milliseconds=max(0,1000*(remaining-.05)),
                work_limit=max(1,allowance//count+(index<allowance%count)),
                partition_count=count,partition_index=index)
            worker=self.workers[index].worker;worker.timeout=max(2.,remaining+1.)
            return worker.call('plan',observation=view,choices=choices,profile=profile,
                fixed_work=False,sampling_seed=1,intent=state['intent'])
        results=list(self.pool.map(branch,range(count)))
        plans={p['signature']:p for d in results for p in [d['selected']]+d.get('alternatives',[])+list((d.get('family_best') or {}).values())}
        require(plans and all(d['failures']==0 for d in results),'Lab search failed; incomplete teacher game rejected')
        selected=results[0]['selected'] if count==1 else self.opponent.worker.call('select_plans',
            plans=list(plans.values()),profile=self.opponent.profile)
        state['work']+=sum(d['nodes'] for d in results)
        self.telemetry['branches']+=count;self.telemetry['nodes']+=sum(d['nodes'] for d in results)
        self.telemetry['unique_plans']+=len(plans)
        self.last_decision=dict(selected=selected,nodes=sum(d['nodes'] for d in results),unique_plans=len(plans),
            failures=sum(d['failures'] for d in results),work_allowance=allowance)
        return selected

    def select_batch(self,inputs):
        output=[]
        for view,choices,rng in inputs:
            c=view['context'];key=(c['era'],c['round'],c['turn'],view['seat'])
            state=self.contexts.setdefault(rng,dict(turn=None,revision=-1,intent=None,queue=deque(),work=0))
            if state['turn']!=key or c['revision']<=state['revision']:
                state.update(turn=key,revision=-1,queue=deque(),work=0,overrun=False,deadline=time.monotonic()+self.budget['seconds'],began=time.monotonic())
                self.telemetry['turns']+=1
            elif c['revision']!=state['revision']+1: state['queue'].clear()
            ids=[p['id'] for p in choices]
            if state['queue'] and state['queue'][0] not in ids: state['queue'].clear()
            if not state['queue']:
                plan=self.decision(view,choices,state)
                state['queue'].extend(x for a in plan['actions'] for x in a);state['intent']=plan.get('intent')
            chosen=state['queue'].popleft();require(chosen in ids,'Lab teacher returned an illegal continuation')
            elapsed=time.monotonic()-state['began'];self.telemetry['max_turn_seconds']=max(self.telemetry['max_turn_seconds'],elapsed)
            if elapsed>self.budget['seconds'] and not state.get('overrun'):
                self.telemetry['over_budget_turns']+=1;state['overrun']=True
            state['revision']=c['revision'];output.append((ids.index(chosen),1.,None))
        return output

    def close(self):
        self.pool.shutdown(wait=True);self.contexts.clear()
        for worker in self.workers: worker.close()


def calibrate(root,budget,*,games=25,mono=None,output=None):
    """Compact stage probes plus whole-game legality/economy qualification.

    Measures work/quality against the original 3s planner on this hardware.
    Passing certifies the declared engineering margins, not playing-strength
    equivalence; reference bank remains the gameplay evaluation control.
    """
    from brass_ai.host import RulesHost
    from brass_ai.recovery.execution import run_game
    from brass_ai.recovery.telemetry import Timings
    from brass_ai.recovery.diagnostics import economy
    import random
    root=Path(root);require(games>=25,'Lab qualification requires at least 25 complete four-player games')
    identity=budget_identity(root,budget);parents=list(identity['bank']['parents'])
    rows=[];complete=[];probes=[]
    evidence=[];target=project_path(root,output or budget['calibration'])
    directory=target.parent/(target.stem+'-evidence')
    require(not target.exists() and not directory.exists(),'Use a new calibration output; existing evidence is immutable')
    with ExitStack() as stack:
        labs=[stack.enter_context(_closing(LabLegacyPolicy(root,p,budget,mono=mono,diagnostic=True))) for p in parents]
        refs=[stack.enter_context(_closing(LegacyPolicy(root,p,mono=mono))) for p in parents]
        host=stack.enter_context(RulesHost(root/'artifacts/r1-host/guarded',mono=mono))
        for i in range(games):
            job=dict(seed=2817000+i,rotation=i%4,sampling_seed=718000+i)
            game=run_game(host,job,[labs[(i*4+s)%len(labs)] for s in range(4)],Timings(),2000)
            from .replay import write_shard
            path=directory/f'qualification-{i:04d}.jsonl.gz'
            write_shard(path,[game],dict(purpose='diagnostic',seed=job['seed'],sampling_seed=job['sampling_seed'],identity=digest(identity)),cache=False)
            evidence.append(dict(path=path.relative_to(root).as_posix(),sha256=sha(path)))
            complete.append(game)
            if i<2:
                for era,round_number in [('CanalEra',1),('CanalEra',7),('RailEra',1),('RailEra',7)]:
                    probe=next((r for r in game['records'] if r['view']['context']['era']==era and
                        r['view']['context']['round']>=round_number and r['view']['context']['phase']=='action'),None)
                    if probe: probes.append(probe)
            print(f'Lab v3 qualification: {i+1}/{games} complete',flush=True)
        for profile,(lab,ref) in enumerate(zip(labs,refs)):
            for row in probes:
                view,choices=row['view'],row['choices'];start=time.monotonic()
                baseline=ref.opponent.worker.call('plan',observation=view,choices=choices,profile=ref.opponent.profile,
                    fixed_work=False,sampling_seed=1)
                baseline_seconds=time.monotonic()-start;start=time.monotonic()
                result=lab.decision(view,choices,dict(deadline=start+budget['seconds'],work=0,intent=None))
                rows.append(dict(parent=parents[profile],era=view['context']['era'],round=view['context']['round'],
                    reference_seconds=baseline_seconds,lab_seconds=time.monotonic()-start,
                    reference_nodes=baseline['nodes'],lab_nodes=lab.last_decision['nodes'],unique_plans=lab.last_decision['unique_plans'],
                    reference_score=baseline['selected']['score'],lab_score=result['score'],
                    reference_shortfall=baseline['selected']['cash']['shortfall'],lab_shortfall=result['cash']['shortfall']))
    economic=economy(complete);work=sum(r['lab_nodes'] for r in rows)/max(1,sum(r['reference_nodes'] for r in rows))
    # Predeclared margins: at least 80% reference nodes and <=1 heuristic VP loss
    # per probe, no worse projected payment deficit, no debt forced sales.
    passed=bool(rows) and work>=.8 and all(r['lab_seconds']<=budget['seconds'] and
        r['lab_score']>=r['reference_score']-1. and r['lab_shortfall']<=r['reference_shortfall'] for r in rows) and economic['debt_decisions']==0
    report=dict(format='lab-v3-calibration/1.0.0',identity=identity,qualification_games=len(complete),
        training_eligible=passed,probes=rows,work_ratio=work,economy=economic,evidence=evidence,
        margins=dict(work_ratio=.8,score_loss=1.,whole_turn_seconds=budget['seconds'],debt_decisions=0),
        limitation='Engineering calibration only; equal playing strength still requires paired gameplay. Failure does not admit the shorter teacher.')
    write_json(target,report);return report


class _closing:
    def __init__(self,policy): self.policy=policy
    def __enter__(self): return self.policy
    def __exit__(self,*_): self.policy.close()
