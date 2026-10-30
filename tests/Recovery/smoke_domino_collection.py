"""Bounded matched CPU collection: 28 ordinary tables total, no production state."""
import argparse
import copy
from collections import Counter
from pathlib import Path
import queue
import threading
import time
import torch

from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,digest,require
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.execution import neural_spec
from brass_ai.recovery.legacy_search import legacy_spec
from brass_ai.recovery.utility import migrate,reward_spec
from brass_ai.recovery.telemetry import memory
from brass_ai.recovery import domino_sampling as A,domino_features as D
from brass_ai.rust_backend.lab_collection import collect_lane
from brass_ai.rust_backend.lab_b02_smoke import reference_part
from brass_ai.rust_backend.lab_transport import Pipe
from brass_ai.rust_backend.lab_families import FamilyStore,collect_families
from brass_ai.step5.league import write_json,seeds_for


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--resume-fixture',action='store_true')
    args=ap.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and (not out.exists() or args.resume_fixture),'Use a new isolated output directory')
    out.mkdir(parents=True,exist_ok=args.resume_fixture);torch.set_num_threads(2);torch.manual_seed(17343)
    c=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g1326-20261029T212054859818Z/recorded-config.json')
    executable=SOURCE_ROOT/'artifacts/b02-dev-target/x86_64-pc-windows-gnu/debug/brass-lab-collector.exe'
    bootstrap=SOURCE_ROOT/'artifacts/domino-bootstrap-fixed/bootstrap/BrassLabBootstrap.exe'
    require(executable.is_file() and bootstrap.is_file(),'Build the isolated native development collectors first')
    model=FourSeatNet(ModelConfig.preset(c['model_size']));optimizer=torch.optim.AdamW(model.parameters())
    migrate(model,optimizer,reward_spec(c));model.eval();other=copy.deepcopy(model)
    with torch.no_grad():next(other.parameters()).add_(.001)
    ex=copy.deepcopy(c['execution']);ex.update(processes=2,hosts_per_process=2,heavy_processes=1,
        worker_threads=1,cpu_budget=8,cpu_reserve=2,model_cache_size=max(2,ex.get('model_cache_size',2)))
    profiles=sorted(read(SOURCE_ROOT/'training/opponents/pool.json')['training'])
    bots=[legacy_spec(SOURCE_ROOT,p,c['legacy_budget'],search_free=True,work_budget=c['legacy_opponent_work_budget']) for p in profiles[:2]]
    started=time.perf_counter();reports=[];timings=[];seeds=seeds_for(763141,14)[0]
    with InferenceService(ex,'cpu') as service:
        mid=service.register(model);older=service.register(other)
        leases=service.channels(4);owned=[leases.acquire() for _ in range(4)];endpoints=queue.Queue()
        for e in owned:e.timeout=120.;endpoints.put(e)
        learner=neural_spec(mid,c['exploration'],'mixed',version=3,opening=c['opening'])
        historical=neural_spec(older,c['exploration'],'neutral',version=3,opening=c['opening'])
        def jobs(arm):
            planned=[];context_counter=0
            for i,seed in enumerate(seeds):
                kind='selfplay' if i<7 else 'historical' if i<13 else 'legacy'
                policies=[learner]*4 if kind=='selfplay' else [learner]*2+([historical]*2 if kind=='historical' else bots)
                rotation=i%4;focuses={}
                for s in range(4):
                    if policies[(s-rotation)%4]==learner:
                        point=context_counter%12;context_counter+=1
                        focuses[str(s)]='neutral' if point%2==0 else A.NAMES[1+point//2]
                proof=dict(producer='recovery/2.0.0',purpose='training',kind=kind,ordinary_game_index=i,
                    learner_behavior=dict(model_id=mid,policy_id=learner['policy_id']))
                if arm=='domino':
                    proof['diversity_assignment']=A.assignments(policies,rotation,seed=c['seed'],index=i,
                        learner_id=learner['policy_id'],current=mid,checkpoints={mid:'1'*64,older:'2'*64})
                    focuses={str(s):a['context'] for s,a in enumerate(proof['diversity_assignment']['seats']) if a is not None}
                job=dict(index=i,seed=seed,rotation=rotation,sampling_seed=c['seed']*7+i,stable_game_id='domino-smoke-'+str(i),
                    policies=policies,focuses=focuses,provenance=proof,path=str(out/arm/f'game-{i}.jsonl.gz'))
                if kind=='selfplay':job['branching']=c['branching']
                planned.append(job)
            return planned
        stop=threading.Event();lock=threading.Lock()
        def native(job,continuation,submit):
            e=endpoints.get(timeout=120)
            try:
                with Pipe([str(bootstrap)],binary=False) as reference:
                    p=collect_lane(SOURCE_ROOT,executable,reference,job,e,ex,continuation=continuation,submit_branch=submit,cancel=stop)
                    if continuation is not None:
                        ev=p['evidence'][0]['metadata'];meta=p['game']['branch']
                        mismatch={k:dict(native=v,original=meta.get(k)) for k,v in ev.items() if v!=meta.get(k)}
                        if mismatch:write_json(out/f'branch-roundtrip-{job["index"]}.json',mismatch)
                    with lock:timings.append(dict(job=job['index'],branch=continuation is not None,**p['timings']))
                    return p
            finally:endpoints.put(e)
        def fallback(job,continuation,submit):
            require(continuation is None,'Unexpected reference branch');e=endpoints.get(timeout=120)
            try:return reference_part(SOURCE_ROOT,job,e,ex,None)
            finally:endpoints.put(e)
        try:
            for arm in ('control','domino'):
                planned=jobs(arm);store=FamilyStore(out/arm,dict(scope='isolated-domino-smoke'))
                before=service.report();t=time.perf_counter()
                families,scheduler=collect_families(planned,store,native,fallback,lanes=4,cancel=stop,
                    model_cohorts=ex.get('model_cohorts',False))
                elapsed=time.perf_counter()-t;completed=Counter();events=0;branches=0
                for job,family in zip(planned,families):
                    game=family['parent']['game'];branches+=int(family['child'] is not None)
                    if arm=='domino':
                        from brass_ai.recovery.replay import CheckedGame
                        checked=CheckedGame(game);A.bind(checked,job['provenance'])
                        if family['child'] is not None:A.bind(CheckedGame(family['child']['game']),job['provenance'])
                        for a in job['provenance']['diversity_assignment']['seats']:
                            if a is not None:completed[f'{job["provenance"]["kind"]}/{a["training_role"]}/{a["physical_seat"]}/{a["strategy_id"]}']+=1
                        for r in game['records']:
                            a=job['provenance']['diversity_assignment']['seats'][r['seat']]
                            if a is not None and a['training_role']=='learner':events+=int(D.event(r) is not None)
                _,cached=collect_families(planned,store,lambda *a:require(False,'Retry recollected'),fallback,lanes=4)
                require(cached['counts']['cached']==14,'Incomplete collection retry')
                reports.append(dict(arm=arm,seconds=elapsed,games_per_second=14/elapsed,ordinary_games=14,branches=branches,
                    completed=dict(completed),descriptor_events=events,scheduler=scheduler,inference_before=before,
                    inference_after=service.report(),memory=memory('cpu'),cached_retry=True))
                write_json(out/'progress.json',dict(arms=reports));print(arm,elapsed,flush=True)
        finally:
            for e in owned:e.close()
    result=dict(passed=True,seconds=time.perf_counter()-started,parameters=model.parameter_count,arms=reports,timings=timings,
        scope='28 ordinary CPU tables total: 14 self-play, 12 history, 2 search-free v4, matched seeds/rotations and random v11.2 weights; native neural tables and complete reference v4 tables, real branch continuations. No training checkpoint, strength evaluation or production throughput claim.')
    write_json(out/'report.json',result);print(dict(passed=True,seconds=result['seconds']),flush=True)


if __name__=='__main__':main()
