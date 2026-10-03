"""Persistent spawned CPU actors, private host/planner lanes, asynchronous replay writes."""
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict, defaultdict
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
from brass_ai.learning.replay import split_for, terminal_targets
from brass_ai.learning.legacy_policy import LegacyPolicy
from brass_ai.step5.policy import NeuralPolicy, context_for
from brass_ai.search.ismcts import MaskedSearchPolicy
from brass_ai.search.common import SearchConfig
from .inference import RemoteEvaluator
from .telemetry import Timings, instrument_host
from .inputs import BOUNDARY
from .encoding import PreparedEncoder
from .replay import CheckedGame,write_shard,read_shards,cache_receipt,write_cache


def neural_spec(identity,exploration=.05,focus='neutral'):
    policy_id='step5-neural/'+digest(dict(model=identity,exploration=exploration,temperature=1.0,focus=focus,version=2))
    return dict(kind='neural',model_id=identity,exploration=exploration,focus=focus,policy_id=policy_id)


def make_policy(spec,root,endpoint,config,mono):
    if spec['kind']=='lab-legacy':
        from .legacy_search import LabLegacyPolicy
        policy=LabLegacyPolicy(root,spec['parent'],spec['budget'],mono=mono,purpose=spec.get('purpose','training'),search_free=spec.get('search_free',False))
        require(policy.policy_id==spec['policy_id'],'Lab teacher identity changed')
        return policy
    if spec['kind']=='legacy':
        return LegacyPolicy(root,spec['policy_id'],purpose=spec.get('purpose','training'),mono=mono)
    endpoints=endpoint if isinstance(endpoint,list) else [endpoint]
    evaluator=RemoteEvaluator(endpoints[0],spec['model_id'],config['batch_size'],config['max_candidate_cells'])
    if spec['kind']=='neural':
        policy=NeuralPolicy(None,exploration=spec['exploration'],focus=spec['focus'],evaluator=evaluator)
        require(policy.policy_id==spec['policy_id'],'Neural policy identity changed')
        return policy
    opponents=[NeuralPolicy(None,exploration=.05,focus='neutral',
        evaluator=RemoteEvaluator(endpoints[0],p,config['batch_size'],config['max_candidate_cells']))
        for p in spec.get('opponents',[spec['model_id']])]
    from dataclasses import replace
    common=dict(config=replace(SearchConfig(**spec['search']),cpu_role=spec.get('cpu_role','reference')),mono=mono,focus=spec.get('focus','neutral'),
        opponents=opponents,evaluator=evaluator)
    require(spec['kind']=='masked_ismcts','Only Masked ISMCTS is an active search policy')
    policy=MaskedSearchPolicy(None,root,**common)
    if len(endpoints)>1:
        from .parallel import parallelize
        policy=parallelize(policy,len(endpoints)-1,spec.get('cpu_role','lab'),
            lambda i:make_policy(spec,root,endpoints[i+1],config,mono))
    instrument_host(policy.game.host)
    for game in policy.simulation_games:instrument_host(game.host)
    return policy


def run_game(host,job,policies,timings,max_decisions,admission=None):
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
    encoded=[];encoder=PreparedEncoder()
    from .resources import TurnLease
    lease=TurnLease(admission,timings)
    with ExitStack() as cleanup:
        cleanup.callback(lease.close)
        return _run_turns(host,job,assigned,timings,max_decisions,context,game,rngs,encoded,encoder,lease)


def _run_turns(host,job,assigned,timings,max_decisions,context,game,rngs,encoded,encoder,lease):
    for step in range(max_decisions):
        if context['terminal']: break
        seat=context['actor'];p=assigned[seat]
        from brass_ai.search.common import turn_key
        workers=p.config.workers if isinstance(p,MaskedSearchPolicy) else getattr(p,'cpu_workers',1)
        lease.update(turn_key(context)+(seat,),workers)
        with timings.phase('host.observe'): view=host.observe(seat)
        with timings.phase('host.choices'): choices=host.choices()
        with timings.phase('inputs.validate'): BOUNDARY.check(view,choices)
        require(view['seat']==seat and view['context']['revision']==context['revision'],'Stale actor observation')
        began=time.monotonic()
        with timings.phase('policy.'+p.kind):
            if type(p) is NeuralPolicy and len(choices)==1:
                p.context_features(view,rngs[seat])
                # Match rng.choices' draw so future nonforced sampling is unchanged.
                rngs[seat].choices([0],weights=[1.0],k=1)
                chosen,probability,teacher=0,1.0,None
            else: chosen,probability,teacher=p.select_batch([(view,choices,rngs[seat])])[0]
        require(type(chosen) is int and 0<=chosen<len(choices),'Actor returned an illegal choice')
        style,intent=p.context_features(view,rngs[seat]) if hasattr(p,'context_features') else ([0.0]*4,[0.0]*8)
        game['records'].append(dict(step=step,seat=seat,view=view,choices=choices,chosen=chosen,
            teacher=teacher,style=style,intent=intent,seconds=time.monotonic()-began,
            behavior=dict(kind=p.kind,policy_id=p.policy_id,probability=probability,temperature=getattr(p,'temperature',1.0))))
        e=getattr(getattr(p,'evaluator',None),'encoder',encoder)
        with timings.phase('inputs.encode'): encoded.append(e.encode(view,choices,style,intent,validated=True))
        with timings.phase('host.apply'): context=host.apply(context,choices[chosen])
    require(context['terminal'],'Decision cap reached; incomplete game excluded')
    with timings.phase('host.result'): game['result']=host.call('result')
    terminal_targets(game['result'])
    return CheckedGame(game).bind(encoded)


def _actor(root,host_path,jobs,results,lease,config,mono,lanes,search_channels,admission):
    root=Path(root);torch.set_num_threads(config['worker_threads'])
    stop=threading.Event();slots=threading.Semaphore(2*lanes)
    with ThreadPoolExecutor(max_workers=1) as writer:
        def lane():
            endpoint=[lease.acquire() for _ in range(search_channels)];timings=Timings()
            cache=OrderedDict()
            try:
                with ExitStack() as stack:
                    with timings.phase('host.startup'):
                        host=stack.enter_context(RulesHost(root/host_path,mono=mono))
                    instrument_host(host,timings)
                    while not stop.is_set():
                        packet=jobs.get()
                        if packet is None: break
                        completed=[]
                        for job in packet['jobs']:
                            policies=[];needed={digest(s) for s in job['policies']}
                            for key in list(cache):
                                # Search owns persistent private simulator lanes.
                                # Retire an unused model generation promptly;
                                # the light NN cache can retain its normal bound.
                                if key not in needed and (isinstance(cache[key],MaskedSearchPolicy) or len(cache)>=config.get('policy_cache',16)):
                                    old=cache.pop(key)
                                    if hasattr(old,'close'): old.close()
                            for spec in job['policies']:
                                key=digest(spec)
                                if key not in cache: cache[key]=make_policy(spec,root,endpoint,config,mono)
                                cache.move_to_end(key);policies.append(cache[key])
                            game=run_game(host,job,policies,timings,config['max_decisions'],admission)
                            telemetry={}
                            for i,p in enumerate(policies):
                                entry={}
                                if hasattr(p,'telemetry'): entry['search']=dict(p.telemetry)
                                evaluator=getattr(p,'evaluator',getattr(getattr(p,'base',None),'evaluator',None))
                                if evaluator is not None: entry['inference']=dict(evaluator.telemetry)
                                if hasattr(getattr(p,'host',None),'recovery_metrics'):
                                    entry['host']=p.host.recovery_metrics.report()
                                if entry: telemetry[str(i)]=entry
                            completed.append((game,job,timings.report(),telemetry))
                            timings=Timings();instrument_host(host,timings)
                        slots.acquire()
                        def publish(completed=completed,packet=packet):
                            try:
                                began=time.monotonic()
                                parts=[];part=[];size=0
                                for item in completed:
                                    weight=len(json.dumps(item[0],separators=(',',':')))
                                    if part and size+weight>config.get('shard_bytes',268435456): parts.append(part);part=[];size=0
                                    part.append(item);size+=weight
                                if part: parts.append(part)
                                records=[]
                                for number,part in enumerate(parts,packet['part_start']):
                                    path=Path(packet['prefix']+f'-part-{number:04d}.jsonl.gz')
                                    provenance=dict(producer='recovery/2.1.0',purpose=part[0][1]['provenance']['purpose'],
                                        collection=packet['identity'],jobs=[dict(index=j['index'],seed=j['seed'],provenance=j['provenance']) for _,j,_,_ in part])
                                    write_shard(path,[g for g,_,_,_ in part],provenance,cache=config.get('feature_cache',False))
                                    receipt=cache_receipt(path);replay_hash=receipt['replay_sha256'] if receipt else sha(path)
                                    for g,j,metrics,telemetry in part:
                                        records.append(dict(job=j['index'],game_id=g['game_id'],path=str(path),sha256=replay_hash,cached=False,
                                            features=receipt,metrics=metrics,search=telemetry))
                                elapsed=time.monotonic()-began
                                for record in records:
                                    record.update(replay_write_seconds=elapsed/len(completed),replay_write_scope='amortized packet')
                                results.put(dict(records=records))
                            except BaseException:
                                stop.set();results.put(dict(error=traceback.format_exc()))
                            finally: slots.release()
                        writer.submit(publish)
            except BaseException:
                stop.set();results.put(dict(error=traceback.format_exc()))
            finally:
                for p in cache.values():
                    if hasattr(p,'close'): p.close()
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

    def start(self,target=None):
        target=min(self.width,target or self.width)
        if len(self.processes)>=target: return
        search_channels=(1+self.config.get('search_workers_per_turn',1)) if self.heavy and self.config.get('search_workers_per_turn',1)>1 else 1
        lease=self.service.channels((target-len(self.processes))*self.lanes*search_channels)
        for _ in range(target-len(self.processes)):
            p=self.mp.Process(target=_actor,args=(str(self.root),str(self.host),self.jobs,self.results,
                lease,self.config,self.mono,self.lanes,search_channels,self.service.cpu))
            p.start();self.processes.append(p)

    def collect(self,jobs):
        require(jobs and len({j['index'] for j in jobs})==len(jobs),'Missing/duplicate actor jobs')
        records=[];pending=[]
        buckets=defaultdict(list)
        for job in jobs:
            group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed']))
            buckets[(str(Path(job['path']).parent),split_for(group))].append(job)
        for bucket in buckets.values():
            size=min(self.config.get('shard_games',1),max(1,len(bucket)//(self.width*self.lanes)))
            for start in range(0,len(bucket),size):
                group=bucket[start:start+size]
                identity=digest([j['provenance'] for j in group])
                prefix=str(Path(group[0]['path']).parent/('games-'+identity[:16]))
                expected={j['index']:j for j in group};existing=sorted(Path(prefix).parent.glob(Path(prefix).name+'-part-*.jsonl.gz'))
                seen=set()
                for path in existing:
                    with gzip.open(path,'rt',encoding='utf-8') as stream: header=json.loads(next(stream))
                    proof=header['provenance'];require(proof.get('collection')==identity,'Cached collection differs')
                    games=read_shards([path]);require(len(games)==len(proof['jobs']),'Cached shard membership differs')
                    # A crash may leave an uncommitted derived cache. Recreate it
                    # from authoritative validated records before issuing a receipt.
                    if self.config.get('feature_cache'): write_cache(path,games)
                    for item,game in zip(proof['jobs'],games):
                        index=item['index'];require(index in expected and index not in seen,'Duplicate/foreign cached job')
                        require(item['provenance']==expected[index]['provenance'],'Cached actor game differs')
                        require(game['setup_group']==digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=expected[index]['seed'])),'Cached setup differs')
                        seen.add(index);records.append(dict(job=index,game_id=game['game_id'],path=str(path),sha256=sha(path),
                            cached=True,features=cache_receipt(path)))
                missing=[j for j in group if j['index'] not in seen]
                if missing: pending.append(dict(jobs=missing,prefix=prefix,identity=identity,part_start=len(existing)))
        if pending:
            self.start(len(pending))
            for packet in pending: self.jobs.put(packet)
            outstanding={j['index'] for packet in pending for j in packet['jobs']}
            while outstanding:
                try: record=self.results.get(timeout=1)
                except queue.Empty:
                    require(all(p.is_alive() for p in self.processes),'Actor process exited before completing its games')
                    require(self.service.error is None,'Inference service failed')
                    continue
                require('error' not in record,record.get('error','Actor failed'))
                for item in record['records']:
                    require(item['job'] in outstanding,'Duplicate/misrouted actor result')
                    outstanding.remove(item['job']);records.append(item)
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
