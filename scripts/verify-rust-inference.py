"""L09 real-position PyTorch / Python ORT / native encoder + Rust ORT parity."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
FEATURES=runpy.run_path(str(ROOT/'scripts/verify-rust-features.py'))


def unique_cases():
    seen=set()
    for row in FEATURES['cases']():
        name,partition,view,choices,state=row
        key=hashlib.sha256(json.dumps([view,choices],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            yield row


def verify(a):
    import numpy as np
    import torch
    from brass_ai.learning.features import Encoder
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.learning.seat_descriptors import unknown_numpy, descriptor, relative
    from brass_ai.native.protocol import NativeWorker
    from brass_ai.native.inference import NativeEvaluator, ReferenceEvaluator
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model,raw=load_checkpoint(next((ROOT/'test_nn').glob('*/checkpoint.pt')));model.eval()
    assert model.utility_calibration is None
    manifest=json.loads(a.manifest.read_text());assert raw['model_id']==manifest['model_id']
    external=ReferenceEvaluator(a.manifest.parent/'model.onnx')
    names=external.graph.outputs
    encoder=Encoder();counts=Counter();phases=Counter();seats=Counter();partitions=Counter();batch_sizes=Counter()
    error={pair:{k:0. for k in names} for pair in ('torch_python_ort','python_rust_ort','torch_rust_ort','mixed_singleton')}
    probs_max=0.;candidate_count=0;owners=set();matrix=set();cases=[]
    for row in unique_cases():
        if a.limit and len(cases)>=a.limit:break
        cases.append(row)
    def close(pair,k,x,y):
        err=float(np.max(np.abs(x.astype(np.float64)-y)))
        error[pair][k]=max(error[pair][k],err)
        np.testing.assert_allclose(x,y,atol=2e-4,rtol=2e-4,err_msg=pair+'/'+k)
    with NativeWorker(a.worker,timeout=60) as w:
        native=[NativeEvaluator(w,s,a.manifest) for s in (0,1)]
        loaded=[n.loaded for n in native]
        for base in range(0,len(cases),4):
            chunk=cases[base:base+4];prepared=[];requests=[]
            session=(base//4)%2;native_eval=native[session]
            for j,(name,partition,view,choices,state) in enumerate(chunk):
                i=base+j;ctx=FEATURES['context'](i);ctx['reward_id']=manifest['reward_id']
                known=(i%2==1)
                stable=[descriptor(role,s*43,[((f+s)%9)/8 for f in range(28)]) for s,role in enumerate(('learner','snapshot','il','rule_bot'))] if known else None
                ctx['stable_descriptors']=stable
                e=encoder.encode(view,choices,ctx['style'],ctx['intent'])
                d=relative(stable,e.stable_to_relative) if known else unknown_numpy()
                prepared.append((e,ctx,d))
                requests.append(native_eval.request(view,choices,ctx,request_id=f'{i}/{name}'))
                phases[view['context']['phase']]+=1;seats[view['seat']]+=1;partitions[partition]+=1
                matrix.add((view['context']['era'],view['context']['phase'],view['seat']))
                counts['known_descriptors' if known else 'unknown_descriptors']+=1;candidate_count+=len(choices)
            width=max(len(e.choice_ids) for e,_,_ in prepared);b=len(prepared)
            inputs=dict(states=np.stack([e.state for e,_,_ in prepared]),actions=np.zeros((b,width,327),np.float32),
                        legal_mask=np.zeros((b,width),bool),stable_to_relative=np.stack([e.stable_to_relative for e,_,_ in prepared]),
                        remaining=np.asarray([c['remaining'] for _,c,_ in prepared],np.float32),descriptors=np.stack([d for _,_,d in prepared]))
            for j,(e,_,_) in enumerate(prepared):
                inputs['actions'][j,:len(e.choice_ids)]=e.actions;inputs['legal_mask'][j,:len(e.choice_ids)]=True
            with torch.inference_mode():
                expected=model(*(torch.from_numpy(inputs[k]) for k in ('states','actions','legal_mask','stable_to_relative','remaining','descriptors')))
            expected={k:t.numpy() for k,t in zip(names,expected)}
            python_ort=external.evaluate(inputs)
            for k in names:
                mask=inputs['legal_mask'] if k=='policy_logits' else np.ones(expected[k].shape,bool)
                assert np.isfinite(expected[k][mask]).all() and np.isfinite(python_ort[k][mask]).all()
                close('torch_python_ort',k,expected[k][mask],python_ort[k][mask])
            # Round-robin caps test the real cooperative producer. A 2ms cap may
            # flush early on debug builds; record actual batch sizes separately.
            cap=(1,2,4)[(base//4)%3]
            native_eval.submit(requests,budget_ms=30000,cap=cap,pending_limit=4,flush_us=2000 if cap>1 else 0)
            result=native_eval.wait();assert result['state']=='complete',result
            owners.add(result['owner']);batch_sizes[result['stats']['max_batch']]+=1
            for j,row in enumerate(result['evaluations']):
                n=len(prepared[j][0].choice_ids)
                actual=dict(policy_logits=np.asarray(row['logits'],np.float32),win_share_4=np.asarray(row['win_share'],np.float32),
                            placement_4=np.asarray(row['placement'],np.float32),vp_4=np.asarray(row['vp'],np.float32),remaining_utility_4=np.asarray(row['utility'],np.float32))
                for k in names:
                    x=python_ort[k][j,:n] if k=='policy_logits' else python_ort[k][j]
                    y=expected[k][j,:n] if k=='policy_logits' else expected[k][j]
                    close('python_rust_ort',k,actual[k],x);close('torch_rust_ort',k,actual[k],y)
                logits=python_ort['policy_logits'][j,:n].astype(np.float64);z=logits-logits.max();logs=z-np.log(np.exp(z).sum())
                probs_max=max(probs_max,float(np.max(np.abs(np.exp(logs)-row['probabilities']))))
                np.testing.assert_allclose(row['log_probabilities'],logs,atol=2e-4,rtol=2e-4)
                np.testing.assert_allclose(row['probabilities'],np.exp(logs),atol=2e-5,rtol=2e-4)
                # Explicit singleton vs mixed parity for each actor/era/phase,
                # plus regularly spaced roots/futures, using the same tensors.
                key=(chunk[j][2]['context']['era'],chunk[j][2]['context']['phase'],chunk[j][2]['seat'])
                repeat=key not in getattr(verify,'singletons',set()) or (base+j)%101==0
                if repeat:
                    if not hasattr(verify,'singletons'):verify.singletons=set()
                    verify.singletons.add(key)
                    native_eval.submit([requests[j]],budget_ms=30000)
                    single=native_eval.wait();assert single['state']=='complete'
                    for k,field in zip(names,('logits','win_share','placement','vp','utility')):
                        close('mixed_singleton',k,np.asarray(single['evaluations'][0][field]),actual[k])
                    counts['mixed_singleton_pairs']+=1
            counts['positions']+=b
            if base%100==0:print(json.dumps(dict(positions=counts['positions'],last=chunk[-1][0])),flush=True)
    report=dict(status='passed',counts=counts,candidate_rows=candidate_count,phases=phases,actors=seats,partitions=partitions,
                matrix=sorted(matrix),owners=sorted(owners),actual_max_batch_histogram=batch_sizes,errors=error,probability_max_error=probs_max,
                atol=2e-4,rtol=2e-4,loaded=loaded,mechanical_only=True,live_ready=False,external_evaluator_used_only_as_oracle=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('reference','worker','manifest','report'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    sys.path[:0]=[str(a.reference.resolve()/'src')]+[str(ROOT/d) for d in ('artifacts/r2-pythondeps','artifacts/r4-pythondeps','artifacts/r0-pythonlibs','artifacts/r0-validation-deps')]
    verify(a)
