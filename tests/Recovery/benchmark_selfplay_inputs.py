"""Bounded before/after input-cache and inference-queue benchmark.

Uses fresh random weights and authored tensors, never an existing run. The
baseline is loaded directly from the requested Git revision. CUDA runs include
synchronization; production end-to-end collection gains need a lab measurement.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
import time
import types

import numpy as np
import torch

from brass_ai.learning.contracts import SOURCE_ROOT, require
from brass_ai.learning.model import FourSeatNet, ModelConfig
from brass_ai.learning.reward import specification
from brass_ai.learning.utility_model import UtilityNet, utility_config
from brass_ai.recovery.config import EXECUTION, UPDATE_V21
from brass_ai.recovery import inference, updates
from brass_ai.recovery.review_metrics import CategoricalProbe
from brass_ai.recovery.telemetry import Timings
from brass_ai.step5.league import write_json
from smoke_categorical_kl_guard import examples


def baseline_module(name, ref):
    relative='src/brass_ai/recovery/'+name+'.py'
    result=subprocess.run(['git','show',ref+':'+relative],cwd=SOURCE_ROOT,check=True,capture_output=True,text=True)
    module=types.ModuleType('brass_ai.recovery._speed_baseline_'+name)
    module.__package__='brass_ai.recovery';sys.modules[module.__name__]=module
    exec(compile(result.stdout,ref+':'+relative,'exec'),module.__dict__)
    return module


def timed(fn, repeats, device):
    samples=[];result=None
    for _ in range(repeats):
        if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
        start=time.perf_counter();result=fn()
        if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
        samples.append(time.perf_counter()-start)
    return dict(median_seconds=median(samples),samples_seconds=samples),result


def collection(service_type, models, data, device, *, lanes=16, rounds=24):
    config=dict(EXECUTION,processes=lanes,hosts_per_process=1,shared_slot_bytes=0,flush_ms=1.)
    service=service_type(config,device)
    identities=[service.register(model) for model in models]
    # Warm frozen-model copies outside the timed request stream.
    for identity in identities:service._model(identity)
    endpoints=[service.channels(1).acquire() for _ in range(lanes)]
    def play(lane):
        total=0.
        for step in range(rounds):
            index=(lane*rounds+step)%len(data);row,_,_,encoded=data[index]
            owner=0 if lane%2==0 else 1+(lane+step)%(len(identities)-1)
            token=str(lane)+'-'+str(step);submitted=time.monotonic()
            frame=dict(model=identities[owner],request_id=token,revision=step,seat=row['seat'],digest=token,
                deadline=submitted+60,submitted=submitted,cancelled=False,encoded=encoded)
            service.requests.put(dict(lane=endpoints[lane].lane,token=token,frames=[frame]))
            require(endpoints[lane].connection.poll(60),'Benchmark inference timeout')
            reply=endpoints[lane].connection.recv()
            require(reply['token']==token and 'error' not in reply,'Benchmark inference failure: '+str(reply.get('error')))
            response=reply['responses'][0]
            require(response['status']=='ok' and response['model_id']==identities[owner], 'Misrouted benchmark result')
            probabilities=response['probabilities']
            require(len(probabilities)==len(encoded.choice_ids) and abs(sum(probabilities)-1)<1e-5,
                    'Invalid benchmark probabilities')
            total+=sum((i+1)*value for i,value in enumerate(probabilities))
        return total
    try:
        with service,ThreadPoolExecutor(max_workers=lanes) as pool:
            start=time.perf_counter();checksums=list(pool.map(play,range(lanes)))
            seconds=time.perf_counter()-start
        metrics=service.report();count=lanes*rounds
        return dict(seconds=seconds,positions=count,positions_per_second=count/seconds,
            mean_batch=metrics['positions']/max(1,metrics['batches']),batches=metrics['batches'],
            padding_fraction=1-metrics['valid_candidate_cells']/max(1,metrics['padded_candidate_cells']),
            checksum=sum(checksums),batching=metrics.get('batching'),
            scope='16 synthetic concurrent lanes, 24 decisions each, four frozen models; inference service only, inline transport')
    finally:
        for endpoint in endpoints:endpoint.close()
        service.__exit__(None,None,None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-ref',default='HEAD')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--probe-rows',type=int,default=2048)
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts directory')
    require(1<=args.repeats<=10 and 32<=args.probe_rows<=2048,'Keep the benchmark bounded')
    out.mkdir(parents=True);torch.set_num_threads(2);torch.manual_seed(481)
    ref=subprocess.run(['git','rev-parse',args.baseline_ref],cwd=SOURCE_ROOT,check=True,capture_output=True,text=True).stdout.strip()
    legacy_updates=baseline_module('updates',ref)
    legacy_metrics=baseline_module('review_metrics',ref)
    legacy_inference=baseline_module('inference',ref)
    model=UtilityNet(utility_config(ModelConfig.preset('small'),specification())).eval()
    behavior,data=examples(model,args.probe_rows)
    model=model.to(args.device)
    before=legacy_updates.Dataset(data,args.device);after=updates.Dataset(data,args.device)
    config=dict(UPDATE_V21,microbatch_size=128,max_candidate_cells=8192)
    ids=list(range(len(data)))
    old_probe=legacy_metrics.CategoricalProbe(model,before,ids,behavior,config)
    new_probe=CategoricalProbe(model,after,ids,behavior,config)
    # Move the policy so nonzero KL parity is tested as well as entropy.
    with torch.no_grad():
        weight=model.policy_scorer[-1].weight
        weight.add_(torch.linspace(-.1,.1,weight.numel(),device=weight.device).reshape_as(weight))
    expected=old_probe.measure_step(model);actual=new_probe.measure_step(model)
    require(all(abs(actual[k]-expected[k])<1e-7 for k in ('categorical_kl','sampled_kl')),'Probe parity failed')
    require(expected['categorical_kl']>1e-6,'Benchmark policy perturbation was ineffective')
    old_time,_=timed(lambda:old_probe.measure_step(model),args.repeats,args.device)
    new_time,_=timed(lambda:new_probe.measure_step(model),args.repeats,args.device)
    old_scan=lambda:legacy_updates.policy_measure(model,before,ids,behavior,config,Timings())
    new_scan=lambda:updates.policy_measure(model,after,ids,behavior,config,Timings())
    require(abs(old_scan()-new_scan())<1e-7,'Full scan parity failed')
    old_scan_time,_=timed(old_scan,args.repeats,args.device)
    new_scan_time,_=timed(new_scan,args.repeats,args.device)
    models=[copy.deepcopy(model) for _ in range(4)]
    with torch.no_grad():
        for i,net in enumerate(models):net.policy_scorer[-1].weight[0,i].add_(i*.05)
    collection_trials=[]
    for trial in range(args.repeats):
        order=(('before',legacy_inference.InferenceService),('after',inference.InferenceService))
        if trial%2:order=order[::-1]
        pair={name:collection(kind,models,data,args.device) for name,kind in order}
        require(abs(pair['before']['checksum']-pair['after']['checksum'])<1e-4,'Collection output parity failed')
        collection_trials.append(pair)
    old_collect=min((trial['before'] for trial in collection_trials),
                    key=lambda r:abs(r['seconds']-median(t['before']['seconds'] for t in collection_trials)))
    new_collect=min((trial['after'] for trial in collection_trials),
                    key=lambda r:abs(r['seconds']-median(t['after']['seconds'] for t in collection_trials)))
    result=dict(passed=True,baseline_commit=ref,device=args.device,torch=str(torch.__version__),
        cuda=torch.version.cuda,threads=torch.get_num_threads(),positions=args.probe_rows,
        parameters=sum(p.numel() for p in model.parameters()),
        probe=dict(before=old_time,after=new_time,ratio=old_time['median_seconds']/new_time['median_seconds'],
                   parity=actual,cache=new_probe.cache_report()),
        scans=dict(before=old_scan_time,after=new_scan_time,
                   ratio=old_scan_time['median_seconds']/new_scan_time['median_seconds'],cache=after.scan_cache.report()),
        collection=dict(before=old_collect,after=new_collect,
                        ratio=old_collect['seconds']/new_collect['seconds'],trials=collection_trials,
                        flush_ms=1.),
        scope='Fresh random small UtilityNet, authored 2-32 legal-choice tensors, no gameplay or playing-strength measurement')
    write_json(out/'report.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
