"""Bounded initialized-model deployment search/transport comparison.

Reuses the capacity benchmark's model, parity and whole-turn checks. The legacy
host override is test-only; production architecture/checkpoint admission is intact.
Timings below are overlapping aggregate worker time, not additive turn wall time.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import functools
import json
from pathlib import Path
import shutil
import threading
import time
from unittest.mock import patch

import torch

from benchmark_deploy_capacity import (BenchmarkService, Profile, SearchSession,
    TrackedChannels, build_models, export_model, run_turn, search_config, sha,
    specification, write)
from brass_ai.host import RulesHost
from brass_ai.learning.contracts import SOURCE_ROOT, require
from brass_ai.recovery.encoding import PreparedEncoder
from brass_ai.searchlab.spiel import BrassSpielGame, BrassSpielState
from fe_fixtures import benchmark_positions


class Probe:
    def __init__(self):
        self.lock=threading.Lock();self.totals=defaultdict(lambda:dict(calls=0,seconds=0.))

    def add(self,key,seconds):
        with self.lock:
            row=self.totals[key];row['calls']+=1;row['seconds']+=seconds

    def wrap(self,fn,key):
        @functools.wraps(fn)
        def measured(*args,**kwargs):
            began=time.perf_counter()
            try:return fn(*args,**kwargs)
            finally:self.add(key,time.perf_counter()-began)
        return measured

    @contextmanager
    def active(self):
        call=RulesHost.call
        def measured_call(host,op,**kwargs):
            began=time.perf_counter()
            try:return call(host,op,**kwargs)
            finally:
                self.add('rpc/'+op,time.perf_counter()-began)
                timing=host.last_native_timing or {}
                for name in ('elapsed_seconds','cpu_seconds','materialize_serialize_seconds'):
                    if name in timing:self.add('native/'+name,float(timing[name]))
        with ExitStack() as stack:
            stack.enter_context(patch.object(RulesHost,'call',measured_call))
            for cls,method,label in ((BrassSpielGame,'from_observation','state_reconstruction'),
                    (BrassSpielState,'_apply_action','state_advance'),
                    (PreparedEncoder,'encode','feature_encoding')):
                stack.enter_context(patch.object(cls,method,self.wrap(getattr(cls,method),label)))
            yield self


def prepare(out):
    require(not out.exists(),'Use a fresh isolated directory')
    (out/'models').mkdir(parents=True)
    rows=benchmark_positions();write(out/'inputs.json',rows)
    metadata={}
    for name,(core,actor) in build_models().items():
        metadata[name]={}
        for head,model in (('utility',core),('actor',actor)):
            meta,graph=export_model(model,out/'models'/f'{name}-{head}.onnx',head=='actor')
            metadata[name][head]=meta
        print(f'Prepared {name}: {core.parameter_count} utility parameters; ONNX/batch parity passed',flush=True)
    write(out/'prepared.json',dict(models=metadata,inputs_sha256=sha(out/'inputs.json'),
        scope='Initialized weights. Search throughput and coverage only, not playing strength.',
        tensor_threads=2,seconds_per_turn=12.))


@contextmanager
def active_session(out,service,metadata,name,workers,batch,fast):
    channels=TrackedChannels(service.channels(1+2*workers));session=None
    config=replace(search_config(specification(),seconds=12.,workers=workers),
        cpu_role='laptop',pending_paths=batch,batch_size=batch).validate()
    def game(*args,**kwargs):
        kwargs['fast']=fast
        return BrassSpielGame(*args,**kwargs)
    try:
        # Select transport only; identical production tree policy, neural math,
        # shared cache, legal choices, seeds, deadlines and continuation logic.
        with patch('brass_ai.search.ismcts.BrassSpielGame',side_effect=game):
            session=SearchSession(SOURCE_ROOT,service,channels,metadata[name]['actor']['model_id'],
                [metadata[name]['utility']['model_id']],config,[Profile(alpha=0.)]*4,
                dict(bias_ready=False,temperature=1.,scope='Initialized timing only'))
        require(all(g.host.fast==fast for w in session.workers for g in [w.game,*w.simulation_games]),
            'Wrong native runtime capability')
        yield session
    finally:
        if session is not None:session.close()
        channels.close()


def search(out,comparison_only=False):
    require(not (out/'search.json').exists(),'Do not overwrite completed evidence')
    prepared=json.loads((out/'prepared.json').read_text(encoding='utf-8'))
    require(sha(out/'inputs.json')==prepared['inputs_sha256'],'Fixture corpus changed')
    rows=json.loads((out/'inputs.json').read_text(encoding='utf-8'))
    cases=[next(r for r in rows if r['name']==n) for n in ('settled-next-round','rail-action')]
    result=dict(format='deploy-transport-benchmark/1',scope=prepared['scope'],
        timing_scope='Overlapping aggregate worker time; RPC includes native compute and JSON/pipe wait.',
        optimized_native_sha256=sha(SOURCE_ROOT/'training/runtime/throughput/search/BrassHost.exe'),
        fast_host_source_sha256=sha(SOURCE_ROOT/'src/BrassTrainingHost/FastHost.cs'),
        comparison_only=comparison_only,
        baseline=[],pilot=[],comparison=[],selected={})
    def record(kind,session,service,game,name,workers,batch,fast,row,seed):
        service.config.update(batch_size=workers*batch,flush_ms=1.)
        probe=Probe()
        with probe.active():value=run_turn(session,service,game,row,seed)
        value.update(model=name,workers=workers,leaf_batch=batch,fast=fast,profile=dict(probe.totals))
        result[kind].append(value);write(out/'search-partial.json',result)
        print(f"{kind}: {name} W={workers} B={batch} fast={fast} {row['name']} "
            f"complete={value['metrics']['completed']} interrupted={value['metrics']['interrupted']} "
            f"qualified={value['qualified_labels']} elapsed={value['elapsed_seconds']:.3f}s",flush=True)
    for workers in (2,3):
        with BenchmarkService(out,prepared['models'],workers) as service:
            game=BrassSpielGame(root=SOURCE_ROOT,fast=True)
            try:
                if workers==2 and not comparison_only:
                    with active_session(out,service,prepared['models'],'v9',workers,1,False) as session:
                        for i,row in enumerate(cases):record('baseline',session,service,game,'v9',workers,1,False,row,893000+i)
                # Small fresh pilot after transport changes. Each condition
                # starts with empty full-feature LRUs; no cached fixture advantage.
                for batch in (() if comparison_only else ((1,2) if workers==2 else (2,1))):
                    with active_session(out,service,prepared['models'],'v9',workers,batch,True) as session:
                        for i,row in enumerate(cases):record('pilot',session,service,game,'v9',workers,batch,True,row,893000+i)
                scores={b:(sum(r['metrics']['completed'] for r in result['pilot'] if r['workers']==workers and r['leaf_batch']==b),
                    sum(r['qualified_labels'] for r in result['pilot'] if r['workers']==workers and r['leaf_batch']==b),-b) for b in (1,2)}
                # The follow-up reuses the measured winner from the transport
                # pilot (one pending path per worker); it does not retune on the
                # same comparison roots or silently expand pending work.
                batch=1 if comparison_only else max(scores,key=scores.get)
                result['selected'][str(workers)]=dict(leaf_batch=batch,service_batch_cap=workers*batch,
                    flush_ms=1.,scores=None if comparison_only else {str(k):list(v) for k,v in scores.items()},
                    selection='Previous transport pilot winner' if comparison_only else 'Current pilot winner')
                with ExitStack() as stack:
                    sessions={name:stack.enter_context(active_session(out,service,prepared['models'],name,workers,batch,True))
                        for name in ('v9','w352')}
                    for i,row in enumerate(cases):
                        for name in (('v9','w352') if i==0 else ('w352','v9')):
                            record('comparison',sessions[name],service,game,name,workers,batch,True,row,894000+i)
            finally:game.close()
            require(service.error is None,'Inference service failed')
    write(out/'search.json',result)


def cleanup(out):
    require((out/'search.json').is_file(),'Review failed experiment before cleanup')
    deleted=[]
    for path in sorted((out/'models').iterdir()):
        require(path.is_file() and path.suffix=='.onnx' and path.resolve().parent==out/'models','Unexpected model output')
        deleted.append(dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size));path.unlink()
    write(out/'model-cleanup.json',dict(deleted=deleted,remaining=list(p.name for p in (out/'models').iterdir())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('prepare','search','all'),default='all')
    parser.add_argument('--delete-models',action='store_true')
    parser.add_argument('--comparison-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts'),'Use an isolated artifacts directory')
    torch.set_num_threads(2)
    if args.phase in ('prepare','all'):prepare(out)
    if args.phase in ('search','all'):search(out,args.comparison_only)
    shutil.copyfile(__file__,out/'executed-benchmark.py')
    if args.delete_models:cleanup(out)


if __name__=='__main__':main()
