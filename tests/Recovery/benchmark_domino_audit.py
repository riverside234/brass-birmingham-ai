"""Isolated CPU timings for exact DOMiNO kernels, not training throughput."""
import argparse
import copy
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from brass_ai.bank import read
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.learning.model import ModelConfig,FourSeatNet
from brass_ai.recovery import updates
from brass_ai.recovery.domino_auxiliary import Diversity,PPGCompanions,row_key
from brass_ai.recovery.telemetry import memory
from brass_ai.recovery.utility import migrate,reward_spec
from brass_ai.step5.league import write_json
from test_domino_audit import Audit,old_combine,ready


def measure(fn,repeats=5):
    fn();times=[]
    for _ in range(repeats):
        began=time.perf_counter();fn();times.append(time.perf_counter()-began)
    return dict(median_seconds=statistics.median(times),seconds=times)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    out=project_path(SOURCE_ROOT,args.output)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use new isolated output')
    out.mkdir(parents=True);torch.set_num_threads(2)
    m,_,c,p,d,ds,_=Audit().prepared(out/'cache')
    # Unique authored metadata rows, without allocating 32k copies of features.
    class Metadata:pass
    large=Metadata();large.rows=[];large.data=[];large.roles=[];n=32768
    sources=list(range(len(ds.data)));targets={};receipts={};training=set()
    for i in range(n):
        source=sources[i%len(sources)];r=copy.copy(ds.rows[source]);g=ds.data[source][2][0]+f'/fixture-{i}'
        key=(g,r['seat'],r['view']['context']['revision'])
        large.rows.append(r);large.data.append((r,None,(g,r['seat'])));large.roles.append('learner')
        if '_domino_target' in r and not r.get('_branch'):
            targets[key]=r['_domino_target'];receipts[key]=r['_domino']['receipt'];training.add(key)
    saved=(d.targets,d.target_receipts,d.train_keys)
    d.targets,d.target_receipts,d.train_keys=targets,receipts,training
    task=torch.linspace(-2,3,n);frozen=torch.linspace(-.3,.4,n);ids=list(range(n))
    reference=old_combine(d,task,large,ids,frozen);actual=d.combine(task,large,ids,frozen)
    require(torch.equal(reference,actual),'Vectorized objective differs')
    old=measure(lambda:old_combine(d,task,large,ids,frozen),3)
    new=measure(lambda:d.combine(task,large,ids,frozen),5)
    d.targets,d.target_receipts,d.train_keys=saved;d._indices.clear()
    actor=[i for i in range(len(ds.data)) if not ds.forced_cpu[i]]
    p.buffer.stage(p.critic_dataset,p.train_ids,p.heldout_ids,actor,p.collection_source,m,p.split['selection'])
    with p.buffer.datasets('cpu',policy=False) as (disk,_,_):
        query=np.random.default_rng(987).integers(len(disk.data),size=4096).tolist()
        legacy=lambda:[row_key(disk,i) in d.train_keys for i in query]
        require(legacy()==d.eligible(disk,query),'PPG eligibility differs')
        old_meta=measure(legacy);new_meta=measure(lambda:d.eligible(disk,query))
    # Current 512-wide architecture, same value inputs and true descriptors.
    actual_config=read(SOURCE_ROOT/'results/joint7-v11-2-grouped-lr-critic-g1326-20261029T212054859818Z/recorded-config.json')
    model=FourSeatNet(ModelConfig.preset(actual_config['model_size']));opt=torch.optim.AdamW(model.parameters())
    migrate(model,opt,reward_spec(actual_config));model.eval()
    intrinsic=Diversity(model,c['diversity'],regime='benchmark');ready(intrinsic)
    intrinsic.targets=d.targets;intrinsic.target_receipts=d.target_receipts;intrinsic.train_keys=d.train_keys
    chosen=[i for i in p.train_ids if row_key(p.critic_dataset,i) in d.train_keys][:128]
    ix,inputs,_=p.critic_dataset.value_batch(chosen,model)
    wrapper=PPGCompanions(None,intrinsic,opt,.5);wrapper.batch(p.critic_dataset,chosen)
    def old_forward():
        with torch.no_grad():
            core=updates.value_output(model,inputs,p.critic_dataset,ix)
            return core,intrinsic.prediction(model,inputs,p.critic_dataset,ix)
    def new_forward():
        with torch.no_grad():return wrapper.forward(model,inputs,p.critic_dataset,ix)
    expected=old_forward()[0];actual=new_forward()[0]
    for a,b in zip(expected[1:],actual[1:]):require(torch.equal(a,b),'Reused encoder changed task heads')
    old_encoder=measure(old_forward);new_encoder=measure(new_forward)
    def pair(a,b):return dict(before=a,after=b,speedup=a['median_seconds']/b['median_seconds'])
    report=dict(passed=True,device='cpu',torch_threads=2,cuda_available=torch.cuda.is_available(),
        advantage=dict(rows=n,bit_exact=True,**pair(old,new)),
        ppg_metadata=dict(rows=4096,identical_masks=True,scope='Warm iteration-local lookup, sealed current critic cache',**pair(old_meta,new_meta)),
        encoder=dict(rows=len(chosen),parameters=model.parameter_count,task_heads_bit_exact=True,
            forwards_before=2,forwards_after=1,**pair(old_encoder,new_encoder)),memory=memory('cpu'),
        scope='Same authored inputs and CPU settings within one process. Component microbenchmarks only; no game collection, optimizer updates, GPU or end-to-end throughput estimate.')
    write_json(out/'report.json',report)
    print({k:v['speedup'] for k,v in report.items() if isinstance(v,dict) and 'speedup' in v},flush=True)


if __name__=='__main__':main()
