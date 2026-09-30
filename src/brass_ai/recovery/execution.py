"""Persistent spawned CPU actors, private host/planner lanes, asynchronous replay writes."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import gzip
import json
import multiprocessing
from pathlib import Path
import queue
import random
import threading
import time
import traceback
import uuid

import torch

from brass_ai.bank import sha
from brass_ai.host import RulesHost
from brass_ai.learning.contracts import CONFIG_DIR, digest, read_json, require
from brass_ai.learning.replay import split_for, terminal_targets, write_shard, read_shards
from brass_ai.learning.legacy_policy import LegacyPolicy
from brass_ai.step5.policy import NeuralPolicy, context_for
from brass_ai.step5.search import SearchPolicy, SearchConfig
from .inference import RemoteEvaluator
from .telemetry import Timings, instrument_host


def neural_spec(identity,exploration=.05,focus='neutral'):
    policy_id='step5-neural/'+digest(dict(model=identity,exploration=exploration,temperature=1.0,focus=focus,version=2))
    return dict(kind='neural',model_id=identity,exploration=exploration,focus=focus,policy_id=policy_id)


def make_policy(spec,root,endpoint,config,mono):
    if spec['kind']=='legacy':
        return LegacyPolicy(root,spec['policy_id'],purpose=spec.get('purpose','training'),mono=mono)
    evaluator=RemoteEvaluator(endpoint,spec['model_id'],config['batch_size'],config['max_candidate_cells'])
    if spec['kind']=='neural':
        policy=NeuralPolicy(None,exploration=spec['exploration'],focus=spec['focus'],evaluator=evaluator)
        require(policy.policy_id==spec['policy_id'],'Neural policy identity changed')
        return policy
    opponents=[NeuralPolicy(None,exploration=.05,focus='neutral',
        evaluator=RemoteEvaluator(endpoint,p,config['batch_size'],config['max_candidate_cells']))
        for p in spec.get('opponents',[spec['model_id']])]
    common=dict(config=SearchConfig(**spec['search']),mono=mono,focus=spec.get('focus','neutral'),
        opponents=opponents,evaluator=evaluator)
    if spec['kind']=='s2': policy=SearchPolicy(None,root/'artifacts/r1-host/guarded',**common)
    elif spec['kind']=='s3':
        from brass_ai.searchlab.s3 import S3Policy, TreeConfig
        policy=S3Policy(None,root/'artifacts/r1-host/guarded',tree=TreeConfig(**spec['tree']),**common)
    else: raise ValueError('Unsupported actor policy')
    instrument_host(policy.host)
    return policy


def run_game(host,job,policies,timings,max_decisions):
    pins=read_json(CONFIG_DIR/'host-1.0.0.json')
    with timings.phase('host.reset'): context=host.reset(job['seed'])
    assigned=[policies[(seat-job.get('rotation',0))%4] for seat in range(4)]
    group=digest(dict(rules=pins['rules'],setup_seed=job['seed']))
    game=dict(game_id=uuid.uuid4().hex,setup_group=group,players=4,split=split_for(group),
        opponents=[p.policy_id for p in assigned],result=None,records=[])
    rngs=[random.Random(digest(dict(sampling_seed=job['sampling_seed'],game=job.get('rng_index',0),seat=s))) for s in range(4)]
    for seat,p in enumerate(assigned):
        if isinstance(p,NeuralPolicy) and p.focus=='mixed':
            p.contexts[rngs[seat]]=context_for(job.get('focuses',{}).get(str(seat),'neutral'))
    for step in range(max_decisions):
        if context['terminal']: break
        seat=context['actor'];p=assigned[seat]
        with timings.phase('host.observe'): view=host.observe(seat)
        with timings.phase('host.choices'): choices=host.choices()
        require(view['seat']==seat and view['context']['revision']==context['revision'],'Stale actor observation')
        began=time.monotonic()
        with timings.phase('policy.'+p.kind):
            chosen,probability,teacher=p.select_batch([(view,choices,rngs[seat])])[0]
        require(type(chosen) is int and 0<=chosen<len(choices),'Actor returned an illegal choice')
        style,intent=p.context_features(view,rngs[seat]) if hasattr(p,'context_features') else ([0.0]*4,[0.0]*8)
        game['records'].append(dict(step=step,seat=seat,view=view,choices=choices,chosen=chosen,
            teacher=teacher,style=style,intent=intent,seconds=time.monotonic()-began,
            behavior=dict(kind=p.kind,policy_id=p.policy_id,probability=probability,temperature=getattr(p,'temperature',1.0))))
        with timings.phase('host.apply'): context=host.apply(context,choices[chosen])
    require(context['terminal'],'Decision cap reached; incomplete game excluded')
    with timings.phase('host.result'): game['result']=host.call('result')
    terminal_targets(game['result'])
    return game


def _actor(root,host_path,jobs,results,lease,config,mono,lanes):
    root=Path(root);torch.set_num_threads(config['worker_threads'])
    stop=threading.Event();slots=threading.Semaphore(2*lanes)
    with ThreadPoolExecutor(max_workers=1) as writer:
        def lane():
            endpoint=lease.acquire();timings=Timings()
            try:
                with ExitStack() as stack:
                    with timings.phase('host.startup'):
                        host=stack.enter_context(RulesHost(root/host_path,mono=mono))
                    instrument_host(host,timings)
                    cache={}
                    while not stop.is_set():
                        job=jobs.get()
                        if job is None: break
                        policies=[]
                        for spec in job['policies']:
                            key=digest(spec)
                            if key not in cache:
                                cache[key]=make_policy(spec,root,endpoint,config,mono)
                                if hasattr(cache[key],'close'): stack.callback(cache[key].close)
                            policies.append(cache[key])
                        game=run_game(host,job,policies,timings,config['max_decisions'])
                        telemetry={}
                        for i,p in enumerate(policies):
                            entry={}
                            if hasattr(p,'telemetry'): entry['search']=dict(p.telemetry)
                            evaluator=getattr(p,'evaluator',getattr(getattr(p,'base',None),'evaluator',None))
                            if evaluator is not None: entry['inference']=dict(evaluator.telemetry)
                            if hasattr(getattr(p,'host',None),'recovery_metrics'):
                                entry['host']=p.host.recovery_metrics.report()
                            if entry: telemetry[str(i)]=entry
                        slots.acquire()
                        def publish(game=game,job=job,metrics=timings.report(),telemetry=telemetry):
                            try:
                                began=time.monotonic()
                                path=Path(job['path']);write_shard(path,[game],job['provenance'])
                                results.put(dict(job=job['index'],path=str(path),sha256=sha(path),cached=False,
                                    metrics=metrics,replay_write_seconds=time.monotonic()-began,search=telemetry))
                            except BaseException:
                                stop.set();results.put(dict(error=traceback.format_exc()))
                            finally: slots.release()
                        writer.submit(publish)
                        timings=Timings()
                        instrument_host(host,timings)
            except BaseException:
                stop.set();results.put(dict(error=traceback.format_exc()))
        with ThreadPoolExecutor(max_workers=lanes) as actors:
            futures=[actors.submit(lane) for _ in range(lanes)]
            for future in futures: future.result()


class ActorPool:
    def __init__(self,root,host,service,config,*,mono=None,heavy=False):
        self.root,self.host,self.service,self.config=root,host,service,config
        self.mono,self.heavy=mono,heavy
        self.processes=[];self.mp=multiprocessing.get_context('spawn')
        self.jobs=self.mp.Queue();self.results=self.mp.Queue()
        self.width=config['heavy_processes'] if heavy else config['processes']
        self.lanes=1 if heavy else config['hosts_per_process']

    def start(self):
        if self.processes: return
        lease=self.service.channels(self.width*self.lanes)
        for _ in range(self.width):
            p=self.mp.Process(target=_actor,args=(str(self.root),str(self.host),self.jobs,self.results,
                lease,self.config,self.mono,self.lanes))
            p.start();self.processes.append(p)

    def collect(self,jobs):
        require(jobs and len({j['index'] for j in jobs})==len(jobs),'Missing/duplicate actor jobs')
        records=[];pending=[]
        for job in jobs:
            path=Path(job['path'])
            if path.exists():
                with gzip.open(path,'rt',encoding='utf-8') as stream: header=json.loads(next(stream))
                require(header['provenance']==job['provenance'],'Cached actor game differs')
                games=read_shards([path]);require(len(games)==1,'Expected one complete actor game')
                require(games[0]['setup_group']==digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed'])),'Cached setup differs')
                records.append(dict(job=job['index'],path=str(path),sha256=sha(path),cached=True))
            else: pending.append(job)
        if pending:
            self.start()
            for job in pending: self.jobs.put(job)
            outstanding={j['index'] for j in pending}
            while outstanding:
                try: record=self.results.get(timeout=1)
                except queue.Empty:
                    require(all(p.is_alive() for p in self.processes),'Actor process exited before completing its games')
                    require(self.service.error is None,'Inference service failed')
                    continue
                require('error' not in record,record.get('error','Actor failed'))
                require(record['job'] in outstanding,'Duplicate/misrouted actor result')
                outstanding.remove(record['job']);records.append(record)
                print(json.dumps(dict(phase='collection',complete=len(records),of=len(jobs),heavy=self.heavy)),flush=True)
        return sorted(records,key=lambda r:r['job'])

    def close(self,failed=False):
        if failed:
            for p in self.processes:
                if p.is_alive(): p.terminate()
        else:
            for _ in range(len(self.processes)*self.lanes): self.jobs.put(None)
        for p in self.processes:
            p.join(timeout=30)
            if p.is_alive(): p.terminate();p.join(timeout=5)
        # No pending input messages should be flushed into a failed consumer.
        self.jobs.cancel_join_thread();self.results.cancel_join_thread()
        self.jobs.close();self.results.close()

    def __enter__(self): return self
    def __exit__(self,kind,*_): self.close(failed=kind is not None)
