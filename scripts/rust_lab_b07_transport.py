"""Short deterministic wire microbenchmark; no NN or lab throughput claim.

Both arms use the same binary, frozen origins and Python sampler. The baseline
uses the previous full-row/page/individual-row protocol. The lab ordinary gate
separately compares real frozen g1250 checkpoint jobs with reference collection.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace
import numpy as np

from brass_ai.learning.contracts import require,digest
from brass_ai.learning.features import Encoded
from brass_ai.recovery.branching import eligible
from brass_ai.recovery.exploration import REVISED_OPENING
from brass_ai.rust_backend.lab_transport import Collector,DecisionEncoder,feature_context,VERSION
from brass_ai.step5.policy import NeuralPolicy


class LegacyCollector(Collector):
    def append(self,row,evidence):
        self.current=self.call('append',token=self.token,revision=row['view']['context']['revision'],row=row,evidence=evidence)
    def records(self,receipt):
        for i in range(receipt['rows']):
            row=self.call('row',index=i)
            yield row['row_json'].encode(),row['evidence_json'].encode()


class LegacyEncoder:
    def __init__(self,lane,model):self.lane,self.model=lane,model
    def encode(self,view,choices,style,intent):
        pages=[self.lane.call('features',token=self.lane.token,revision=view['context']['revision'],
            context=feature_context(self.model,style,intent),start=i) for i in range(0,len(choices),64)]
        return Encoded(np.array(pages[0]['state'],np.float32),
            np.concatenate([np.array(p['actions'],np.float32).reshape(-1,327) for p in pages]),
            np.array(pages[0]['stable_to_relative'],np.int64),tuple(c['id'] for c in choices))


def walk(binary,origin,*,legacy=False,limit=64,batch_forced=True):
    began=time.perf_counter();metrics=Counter();trace=[];features=[];branches=[];complete=None
    cls=LegacyCollector if legacy else Collector
    with cls(binary,batch_forced=batch_forced) as lane:
        lane.begin('frozen-wire-fixture',origin,5000)
        metrics['startup_seconds']=time.perf_counter()-began
        rngs=[random.Random(317+s) for s in range(4)]
        policies=[NeuralPolicy(None,evaluator=SimpleNamespace(model_id='frozen-wire-fixture'),
            focus='neutral',exploration=.05,exploration_version=3,opening=REVISED_OPENING) for _ in range(4)]
        encoder=(LegacyEncoder if legacy else DecisionEncoder)(lane,'frozen-wire-fixture')
        began=time.perf_counter()
        while not lane.current.get('terminal') and len(trace)<limit:
            d=lane.current['decision'];view,choices=d['view'],d['choices'];seat=view['seat'];p=policies[seat];rng=rngs[seat]
            style,intent=p.context_features(view,rng)
            t=time.perf_counter()
            if legacy or len(choices)>1:
                encoded=encoder.encode(view,choices,style,intent)
                if len(choices)>1:features.append(hashlib.sha256(encoded.state.tobytes()+encoded.actions.tobytes()+encoded.stable_to_relative.tobytes()).hexdigest())
            metrics['features_seconds']+=time.perf_counter()-t
            raw=np.zeros(len(choices));chosen=next((i for i,c in enumerate(choices) if c['kind']=='pass'),0);raw[chosen]=1.
            t=time.perf_counter();chosen,probability,teacher=p.sample_training(view,choices,rng,raw,intent)
            behavior=dict(kind=p.kind,policy_id=p.policy_id,probability=probability,temperature=p.temperature)
            behavior.update(p.last_behavior[rng])
            metrics['sampling_seconds']+=time.perf_counter()-t
            row=dict(step=lane.current['step'],seat=seat,view=view,choices=choices,chosen=chosen,teacher=teacher,
                style=style,intent=intent,seconds=0.,behavior=behavior)
            if eligible(view,choices,behavior.get('action_decision',0),0):
                branches.append((row['step'],lane.checkpoint(view['context']['revision'])['sha256']))
            ev=dict(kind='ordinary',model_id=p.evaluator.model_id,policy_id=p.policy_id,forced=len(choices)==1,
                raw_probabilities=raw.tolist(),feature_version=VERSION,semantic_id=choices[chosen]['id'])
            trace.append((row,ev))
            t=time.perf_counter();lane.append(row,ev);metrics['append_seconds']+=time.perf_counter()-t
        metrics['collection_seconds']=time.perf_counter()-began
        if lane.current.get('terminal'):
            t=time.perf_counter();receipt=lane.call('finish');raw=list(lane.records(receipt))
            metrics['export_seconds']=time.perf_counter()-t
            for index,key in ((0,'records_sha256'),(1,'evidence_sha256')):
                require(hashlib.sha256(b'['+b','.join(bytes(x[index]) for x in raw)+b']').hexdigest()==receipt[key],'Export seal differs')
            require([(json.loads(bytes(r)),json.loads(bytes(e))) for r,e in raw]==trace,'Export training records differ')
            complete={k:receipt[k] for k in ('rows','records_sha256','evidence_sha256','scoring','result')}
        metrics.update({'ipc_'+k:v for k,v in lane.metrics.items()})
        return dict(metrics=dict(metrics),trace=trace,features=features,branches=branches,
                    rng=[r.getstate() for r in rngs],complete=complete,forced=sum(len(r['choices'])==1 for r,e in trace))


def main():
    p=argparse.ArgumentParser();p.add_argument('--binary',required=True);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();roots=json.loads(args.fixtures.read_text())['roots']
    selected=[max(roots,key=lambda r:len(r['expected']['choices'])),roots[-1]]
    runs=[]
    for repeat in range(2):
        for root in selected:
            arms={}
            for legacy in ((True,False) if repeat==0 else (False,True)):
                value=walk(args.binary,dict(state=root['expected']['state'],rail_keys=list(range(64))),legacy=legacy,limit=24 if root==selected[0] else 96)
                arms['baseline' if legacy else 'optimized']=value
            for key in ('trace','features','branches','rng','complete','forced'):require(arms['baseline'][key]==arms['optimized'][key],'Wire parity differs: '+key)
            runs.append(dict(root=root['id'],repeat=repeat,decisions=len(value['trace']),forced=value['forced'],
                completed=value['complete'] is not None,**{k:v['metrics'] for k,v in arms.items()}))
    report=dict(passed=True,scope='Two alternating matched wire-only fixture pairs; no NN/inference or accepted-work speed claim',
        frozen_origins=digest(selected),runs=runs)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
