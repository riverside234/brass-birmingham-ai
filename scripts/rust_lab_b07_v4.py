"""24-stage C# V4 decision parity: Rust vs C# authoritative progression.

Uses the existing frozen V4 spec and worker unchanged. No neural checkpoint or
full game is needed; this is separate from the actual g1250 mixed-table gate.
"""
import argparse
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import random
import time

from brass_ai.learning.contracts import SOURCE_ROOT,require,digest
from brass_ai.recovery.execution import make_policy
from brass_ai.rust_backend.lab_transport import Collector,Pipe,VERSION
from brass_ai.step5.league import write_json


def run(binary,bootstrap,frozen,output,mono='mono'):
    spec=next(p for j in frozen['jobs'] for p in j['policies'] if p['kind']=='lab-legacy')
    require(spec['search_free'] is True,'V4 ordinary fixture must retain frozen search-free work budget')
    job=next(j for j in frozen['jobs'] if spec in j['policies'])
    command=[bootstrap] if os.name=='nt' else [mono,bootstrap]
    with ExitStack() as cleanup:
        host=cleanup.enter_context(Pipe(command,binary=False))
        lane=cleanup.enter_context(Collector(binary,batch_forced=False))
        a=make_policy(spec,SOURCE_ROOT,None,frozen['execution'],mono);cleanup.callback(a.close)
        b=make_policy(spec,SOURCE_ROOT,None,frozen['execution'],mono);cleanup.callback(b.close)
        r1=[random.Random(821+s) for s in range(4)];r2=[random.Random(821+s) for s in range(4)]
        origin=host.call('init',seed=job['seed']);lane.begin('v4-parity',origin,5000)
        current=host.call('decision');counts=dict(decisions=0,forced=0,queued_continuations=0)
        phases=dict(csharp_step_seconds=0.,rust_step_seconds=0.,v4_reference_seconds=0.,v4_native_seconds=0.)
        trace=[]
        for step in range(24):
            d=lane.current['decision'];view,choices=d['view'],d['choices'];seat=view['seat']
            require(view==current['view'] and choices==current['choices'],'V4 progression observation/candidate order differs')
            t=time.perf_counter();ca=a.select_batch([(current['view'],current['choices'],r1[seat])])[0]
            phases['v4_reference_seconds']+=time.perf_counter()-t
            t=time.perf_counter();cb=b.select_batch([(view,choices,r2[seat])])[0]
            phases['v4_native_seconds']+=time.perf_counter()-t
            require(ca==cb,'V4 selected continuation differs')
            style,intent=b.context_features(view,r2[seat])
            require((style,intent)==a.context_features(current['view'],r1[seat]),'V4 intent differs')
            sa,sb=a.contexts[r1[seat]],b.contexts[r2[seat]]
            for key in ('turn','revision','intent','queue','work','work_allocated','seed'):
                require(sa.get(key)==sb.get(key),'V4 plan/turn budget differs: '+key)
            require(r1[seat].getstate()==r2[seat].getstate(),'V4 RNG differs')
            counts['decisions']+=1;counts['forced']+=len(choices)==1
            counts['queued_continuations']+=bool(sb['queue'])
            chosen=cb[0]
            row=dict(step=step,seat=seat,view=view,choices=choices,chosen=chosen,teacher=None,
                style=style,intent=intent,seconds=0.,behavior=dict(kind=b.kind,policy_id=b.policy_id,probability=1.,temperature=1.))
            evidence=dict(kind='ordinary',model_id=None,policy_id=b.policy_id,forced=len(choices)==1,
                raw_probabilities=None,feature_version=VERSION,semantic_id=choices[chosen]['id'])
            t=time.perf_counter();lane.append(row,evidence,allow_forced=False)
            phases['rust_step_seconds']+=time.perf_counter()-t
            t=time.perf_counter();current=host.call('step',choice=choices[chosen]['id'])
            phases['csharp_step_seconds']+=time.perf_counter()-t
            trace.append(dict(seat=seat,revision=view['context']['revision'],choice=choices[chosen]['id'],style=style,intent=intent))
        require(counts['queued_continuations']>0 and counts['forced']>0,'V4 smoke did not cover queued plans and forced stages')
        report=dict(passed=True,counts=counts,phases=phases,trace_sha256=digest(trace),policy_id=spec['policy_id'],
            seed=job['seed'],spec_sha256=digest(spec),rng_equal=True,plans_intent_budgets_equal=True,
            scope='24 stages; original C# V4 worker against Rust and C# progression; no full game/NN or throughput claim')
        write_json(output,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','bootstrap','frozen','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--mono',default='mono');args=p.parse_args()
    print(json.dumps(run(args.binary,args.bootstrap,json.loads(Path(args.frozen).read_text()),Path(args.out),args.mono),indent=2))
