"""Bounded same-weight scorer comparison; pre-encoded inputs, no games/training run."""
import argparse
import copy
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from types import MethodType

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))

import numpy as np
import torch

from brass_ai.learning.batching import collate,model_id
from brass_ai.learning.contracts import compatibility,digest,require
from brass_ai.learning.features import Encoder
from brass_ai.learning.model import ModelConfig,dense_multiply_adds
from brass_ai.learning.utility_model import UtilityNet,utility_config
from brass_ai.learning.reward import specification
from brass_ai.personality.laptop.export import export,Graph,BASE_INPUTS
from brass_ai.step5.league import implementation_id
from fe_fixtures import benchmark_positions
from test_factored_scorer import inputs


def reference_model(model):
    reference=copy.deepcopy(model)
    def context(self,x):return x
    def score(self,candidates,context):
        return self.policy_scorer(torch.cat((context[:,None].expand(-1,candidates.shape[1],-1),candidates),-1)).squeeze(-1)
    # Real bound methods: mocking/dispatch instrumentation is outside timings.
    reference.project_policy_context=MethodType(context,reference)
    reference.score_candidates=MethodType(score,reference)
    return reference


def compare(functions,rounds,calls):
    for f in functions.values():
        for _ in range(8):f()
    samples={k:[] for k in functions};blocks={k:[] for k in functions}
    names=list(functions)
    for cycle in range(rounds):
        for name in names if cycle%2==0 else names[::-1]:
            block=[]
            for _ in range(calls):
                start=time.perf_counter();functions[name]();block.append((time.perf_counter()-start)*1000)
            samples[name].extend(block);blocks[name].append(statistics.median(block))
    result={k:dict(median_ms=statistics.median(v),p95_ms=float(np.percentile(v,95)),
                   calls=len(v),round_median_ms=blocks[k]) for k,v in samples.items()}
    result['median_reduction_percent']=100*(1-result['factored']['median_ms']/result['unfactored']['median_ms'])
    result['paired_round_median_delta_ms']=statistics.median([a-b for a,b in zip(blocks['factored'],blocks['unfactored'])])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=9)
    parser.add_argument('--calls',type=int,default=20)
    args=parser.parse_args();require(args.rounds>=3 and args.calls>=3,'Need repeated alternating comparisons')
    require(not args.output.exists(),'Use a new isolated benchmark directory');args.output.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(725)
    model=UtilityNet(utility_config(ModelConfig.preset('medium'),specification())).eval()
    reference=reference_model(model);identity=model_id(model)
    require(model_id(reference)==identity,'Comparison weights differ')
    positions=benchmark_positions();encoder=Encoder()
    rows=[encoder.encode(p['view'],p['choices']) for p in positions]
    corpus=[collate([row]) for row in rows]
    (args.output/'positions.json').write_text(json.dumps(positions,indent=2)+'\n')
    report=dict(implementation=implementation_id(),compatibility=digest(compatibility()),model_id=identity,
        parameter_count=model.parameter_count,macs=dense_multiply_adds(model),
        runtime=dict(platform=platform.platform(),processor=platform.processor(),torch=torch.__version__,
                     threads=torch.get_num_threads(),cuda_available=torch.cuda.is_available()),
        scope='Same untrained width-304 weights. Synthetic shape grid plus 14 authored native-board positions. Inputs pre-encoded; excludes encoding, transport, search and full iteration work. No GPU measurements.',
        rounds=args.rounds,calls_per_round=args.calls,warmup_calls_per_mode=8,
        corpus=[dict(name=p['name'],candidates=len(p['choices'])) for p in positions],
        forward=[],onnx=[],forward_backward=[],maximum_logit_error=0.)
    cases=[(1,1),(1,8),(1,32),(1,128),(1,512),(8,32),(8,128),(32,128)]
    with torch.inference_mode():
        for batch,count in cases:
            data=inputs(model.config,batch,count)
            a,b=reference(*data),model(*data)
            torch.testing.assert_close(a[0],b[0],rtol=2e-5,atol=2e-6)
            for x,y in zip(a[1:],b[1:]):torch.testing.assert_close(x,y,rtol=0,atol=0)
            error=float((a[0][data[2]]-b[0][data[2]]).abs().max())
            report['maximum_logit_error']=max(report['maximum_logit_error'],error)
            row=dict(batch=batch,candidates=count,**compare(dict(unfactored=lambda:reference(*data),factored=lambda:model(*data)),args.rounds,args.calls))
            report['forward'].append(row);print('forward',batch,count,round(row['median_reduction_percent'],2),flush=True)
        report['native_corpus']=compare(dict(unfactored=lambda:[reference(*data) for data in corpus],
            factored=lambda:[model(*data) for data in corpus]),args.rounds,args.calls)
        for name,net in (('unfactored',reference),('factored',model)):
            meta=export(net,args.output/(name+'.onnx'))
            report[name+'_export']=dict(sha256=meta['sha256'],parity=meta['parity'])
        engines={name:Graph(args.output/(name+'.onnx')) for name in ('unfactored','factored')}
        report['runtime']['onnxruntime']=engines['factored'].version
        for batch,count in ((1,1),(1,32),(1,128),(1,512),(8,128)):
            data=inputs(model.config,batch,count)
            values=dict(zip(BASE_INPUTS,[t.numpy() for t in data]+[np.full((batch,4),.025,dtype='float32')]))
            a,b=[engine.run(values) for engine in engines.values()]
            for key in a:np.testing.assert_allclose(a[key],b[key],rtol=2e-5,atol=2e-6)
            row=dict(batch=batch,candidates=count,**compare({name:(lambda engine=engine:engine.run(values)) for name,engine in engines.items()},args.rounds,args.calls))
            report['onnx'].append(row);print('onnx',batch,count,round(row['median_reduction_percent'],2),flush=True)
    for batch,count in ((8,32),(8,128)):
        data=inputs(model.config,batch,count)
        def training(net):
            net.zero_grad(set_to_none=True);out=net(*data)
            loss=-out[0].log_softmax(-1)[:,0].mean()+.01*sum(x.square().mean() for x in out[1:])
            loss.backward()
        row=dict(batch=batch,candidates=count,**compare(dict(unfactored=lambda:training(reference),factored=lambda:training(model)),args.rounds,args.calls))
        report['forward_backward'].append(row);print('forward_backward',batch,count,round(row['median_reduction_percent'],2),flush=True)
    require(model_id(model)==model_id(reference)==identity,'Benchmark changed weights')
    report['weights_unchanged']=True
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output/'report.json'),maximum_logit_error=report['maximum_logit_error'],native_corpus=report['native_corpus'])))


if __name__=='__main__':main()
