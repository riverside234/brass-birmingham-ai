"""Bounded native A/B: identical frozen policies/seeds, serial vs queued forks."""
import argparse
import copy
import random
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.host import RulesHost
from brass_ai.learning.contracts import SOURCE_ROOT,digest,require
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.recovery import exploration as E
from brass_ai.recovery.execution import ActorPool,neural_spec
from brass_ai.recovery.host_runtime import training_host
from brass_ai.recovery.inference import InferenceService
from brass_ai.recovery.replay import read_shards
from brass_ai.recovery.utility import migrate,reward_spec
from brass_ai.step5.league import seeds_for,write_json
from test_exploration_revision import config


def portable_fork():
    path=training_host(SOURCE_ROOT)
    with RulesHost(path) as parent,RulesHost(path) as other:
        rng=random.Random(73043);context=parent.reset(73043);actions=0
        while actions<12 or context['phase']!='action':
            actions+=int(context['phase']=='action')
            context=parent.apply(context,rng.choice(parent.choices()))
        before=[parent.observe(s) for s in range(4)];choices=parent.choices()
        checkpoint=parent.call('fork_save',export=True)['checkpoint']
        restored=other.call('fork_restore',checkpoint=checkpoint,shuffle_seed=734)
        require(restored==context and other.choices()==choices,'Portable fork context/choices differ')
        require([other.observe(s) for s in range(4)]==before,'Portable shuffle changed seat information')
        parent.call('fork_resume',shuffle_seed=734)
        require(parent.call('canonical_hash')==other.call('canonical_hash'),'Portable shuffle differs from local shuffle')
        rows=0
        while not context['terminal']:
            require(rows<2000,'Smoke decision cap')
            choice=rng.choice(parent.choices())
            context=parent.apply(context,choice)
            require(other.apply(restored,choice)==context,'Cross-host continuation diverged')
            restored=context;rows+=1
        require(parent.call('result')==other.call('result'),'Portable terminal result differs')
    return dict(passed=True,continuation_decisions=rows,all_seat_observations_exact=True)


def signature(game):
    return dict(result=game['result'],opponents=game['opponents'],
        decisions=[(r['seat'],r['view']['context']['revision'],[c['id'] for c in r['choices']],r['chosen'],
                    r['behavior'],r['style'],r['intent']) for r in game['records']])


def compare(a,b):
    require(a.keys()==b.keys(),'Collection identities differ')
    largest=0.
    for k in a:
        left=copy.deepcopy(a[k]);right=copy.deepcopy(b[k])
        for x,y in zip(left['decisions'],right['decisions']):
            largest=max(largest,abs(x[4]['probability']-y[4]['probability']))
            x[4].pop('probability');y[4].pop('probability')
        require(left==right,'Frozen replay choices/result/context/provenance changed')
    require(largest<3e-6,'Frozen probabilities changed beyond floating tolerance')
    return largest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--games',type=int,default=8)
    ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    require(2<=args.games<=16 and 2<=args.workers<=8,'Bounded smoke only')
    out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new artifact directory')
    out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(73043)
    started=time.monotonic();native=portable_fork();cfg=config()
    model=FourSeatNet(ModelConfig.preset('medium'));opt=torch.optim.AdamW(model.parameters())
    migrate(model,opt,reward_spec(cfg));model.eval()
    ex=copy.deepcopy(cfg['execution'])
    ex.update(processes=2,hosts_per_process=max(1,args.workers//2),heavy_processes=1,
        worker_threads=1,cpu_budget=max(8,args.workers+4),cpu_reserve=2)
    actual_workers=ex['processes']*ex['hosts_per_process']
    seeds=seeds_for(73043,args.games)[0];arms=[];signatures={}
    with InferenceService(ex,'cpu') as service,ActorPool(SOURCE_ROOT,cfg['host'],service,ex) as pool:
        identity=service.register(model)
        def jobs(mode,n=None):
            result=[]
            for i,seed in enumerate(seeds[:n]):
                settings=dict(version=3,opening=E.OPENING,branching=E.BRANCH if mode!='plain' else None)
                job=dict(index=i,seed=seed,sampling_seed=seed*7,stable_game_id='branch-scheduling-'+str(i),
                    policies=[neural_spec(identity,.03,'mixed',version=3,opening=E.OPENING)]*4,
                    focuses={str(s):'neutral' for s in range(4)},path=str(out/mode/f'game-{i}.jsonl.gz'),
                    provenance=dict(producer='recovery/2.0.0',purpose='training',kind='selfplay',exploration=settings))
                if mode not in ('plain','warm'):job['branching']=E.BRANCH
                result.append(job)
            return result
        # Warm the same persistent lanes/inference service before all three arms.
        pool.collect(jobs('warm',actual_workers))
        for mode in ('plain','serial','queued'):
            began=time.monotonic()
            with patch.object(pool,'_branch_scheduling',return_value=mode=='queued'):
                records=pool.collect(jobs(mode))
            elapsed=time.monotonic()-began
            stats=copy.deepcopy(pool.last_collection_report);parents={};children={};branch_seconds=0.
            for path in dict.fromkeys(r['path'] for r in records):
                receipt=next(r for r in records if r['path']==path).get('features')
                for g in read_shards([path],receipts={path:receipt}):
                    parents[g['game_id']]=signature(g)
                    for child in g.get('branches',[]):
                        children[child['game_id']]=signature(child)
                        branch_seconds+=child['branch']['elapsed_seconds']
            signatures[mode]=(parents,children)
            arm=dict(mode=mode,collection_seconds=elapsed,games=len(parents),branches=len(children),
                branch_worker_seconds=branch_seconds,branch_worker_seconds_per_worker=branch_seconds/actual_workers,
                worker_seconds=sum(r['worker_seconds']+r.get('branch_execution',{}).get('worker_seconds',0) for r in records),
                scheduling=stats,shards=len({r['path'] for r in records}))
            arms.append(arm);write_json(out/'progress.json',dict(arms=arms))
            print(f'{mode}: {elapsed:.2f}s, {len(children)} branches, branch work/W={branch_seconds/actual_workers:.2f}s',flush=True)
        before=time.monotonic();cached=pool.collect(jobs('queued'))
        require(all(r['cached'] for r in cached),'Complete family cache not reused')
        require(pool.last_collection_report['branch_games']==0,'Cached branch ran again')
        cache=dict(seconds=time.monotonic()-before,all_cached=True,scheduling=pool.last_collection_report)
    parent_error=max(compare(signatures['plain'][0],signatures[m][0]) for m in ('serial','queued'))
    branch_error=compare(signatures['serial'][1],signatures['queued'][1])
    require(signatures['queued'][1],'Smoke seeds produced no branches')
    baseline,serial,queued=[a['collection_seconds'] for a in arms]
    write_json(out/'report.json',dict(passed=True,seconds=time.monotonic()-started,native=native,arms=arms,
        parameters=model.parameter_count,workers=actual_workers,cache=cache,
        parent_probability_max_error=parent_error,branch_probability_max_error=branch_error,
        serial_over_plain=serial/baseline-1,queued_over_plain=queued/baseline-1,
        queued_vs_serial=queued/serial-1,
        scope='Small isolated native CPU collection, random current v11.2 utility model, fixed seeds and production .005 fork rate; all tables self-play. Persistent warmed workers. Exact parent/child actions and results compared. This is not the 48-lane 50/45/5 lab workload, a total-iteration overhead measurement, or evidence of playing strength.'))


if __name__=='__main__':main()
