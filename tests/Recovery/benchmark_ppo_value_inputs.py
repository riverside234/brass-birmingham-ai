"""Bounded cold-Dataset and repeated value-input benchmark against a Git revision.

No production checkpoints, games or optimizer changes. Candidate counts include
forced rows and wide lists; unlike a warm policy-probe benchmark this exercises
the packing path used repeatedly by head-only utility calibration.
"""
import argparse
import gc
import json
from pathlib import Path
from statistics import median
import time
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.learning.contracts import SOURCE_ROOT, require
from brass_ai.learning.features import Encoded, feature_schema
from brass_ai.learning.local_entities import entity_layout
from brass_ai.recovery import batch_cache, updates
from brass_ai.step5.league import write_json
from benchmark_selfplay_inputs import baseline_module


def examples(count):
    rng=np.random.default_rng(412);schema=feature_schema();data=[]
    sizes=(1,1,1,1,1,1,8,32,128,256,512,772)
    target=dict(win_share=[1.,0.,0.,0.],placement=[1.,2.,3.,4.],vp=[100.,90.,80.,70.])
    for i in range(count):
        n=sizes[i%len(sizes)]
        actions=rng.normal(0,.2,(n,schema['action_dim'])).astype(np.float32)
        for block in entity_layout()['action_fields'].values():
            actions[:,block['start']:block['stop']]=0
            rows=np.arange(n);selected=rows[rows%5!=0]
            actions[selected,block['start']+selected%(block['stop']-block['start'])]=1
        encoded=Encoded(rng.normal(0,.2,schema['state_dim']).astype(np.float32),actions,
            np.array([(s-i)%4 for s in range(4)],dtype=np.int64),tuple(str(j) for j in range(n)))
        row=dict(chosen=0,seat=i%4,intent=[0.]*8,choices=[dict(kind='loan') for _ in range(n)],
                 behavior=dict(probability=1/n))
        data.append((row,target,(f'bench-{i//4}',i%4),encoded))
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline-ref',default='b33c37be5031e8a522eb56ab3b066b9d8cee4296')
    parser.add_argument('--rows',type=int,default=512)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();out=args.output.resolve()
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts directory')
    require(32<=args.rows<=2048 and 1<=args.repeats<=5,'Keep the benchmark bounded')
    out.mkdir(parents=True);torch.set_num_threads(2)
    data=examples(args.rows);before=baseline_module('updates',args.baseline_ref)
    trials=[];reference=None
    for repeat in range(args.repeats):
        trial={};order=[('before',before.Dataset),('after',updates.Dataset)]
        if repeat%2:order.reverse()
        for name,kind in order:
            gc.collect();start=time.perf_counter();ds=kind(data,'cpu',policy=False)
            prepare=time.perf_counter()-start
            parts=list(ds.microbatches(range(len(data)),64,131072))
            packed=[ds.batch_cpu(part,value_only=True) for part in parts]
            # Exact tensor/mask/seat agreement outside the timing interval.
            if reference is None:reference=packed
            else:
                for (a,inputs,q),(b,other,r) in zip(reference,packed):
                    for x,y in zip((a,*inputs,q),(b,*other,r)):torch.testing.assert_close(x,y,rtol=0,atol=0)
            hashed=[0,0];original=batch_cache.array_digest
            def counted(array):
                hashed[0]+=array.nbytes;hashed[1]+=1
                return original(array)
            with patch.object(batch_cache,'array_digest',counted):
                start=time.perf_counter()
                for _ in range(5):
                    for part in parts:ds.batch_cpu(part,value_only=True)
                elapsed=time.perf_counter()-start
            trial[name]=dict(dataset_seconds=prepare,value_input_seconds=elapsed,
                             hashed_bytes=hashed[0],hash_calls=hashed[1])
            del ds,packed
        trials.append(trial)
    summaries={name:{key:median(t[name][key] for t in trials) for key in trials[0][name]}
               for name in ('before','after')}
    result=dict(passed=True,baseline_ref=args.baseline_ref,device='cpu',torch=str(torch.__version__),
        rows=len(data),passes=5,min_candidates=1,max_candidates=772,threads=torch.get_num_threads(),
        exact_input_parity=True,trials=trials,median=summaries,
        value_input_speedup=summaries['before']['value_input_seconds']/summaries['after']['value_input_seconds'],
        scope='CPU Dataset construction and five value-only packing passes, authored tensors; no model compute, production CUDA timing or strength measurement')
    write_json(out/'report.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
