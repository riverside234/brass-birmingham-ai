"""Isolated eight-iteration restart smoke and paired CPU preparation benchmarks."""
import argparse
import copy
import importlib.util
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
for path in ('src','tests/Recovery','tests/FeatureEngineering','tests/R2'):sys.path.insert(0,str(ROOT/path))


def benchmarks(output):
    import torch
    from brass_ai.recovery import calibration_cache,utility,updates,value_phase,ppg_config,ppg_replay
    from brass_ai.recovery.actor_value import TrainingPhases
    from brass_ai.step5.league import write_json
    from test_gae_value_phase import fixture
    from test_ppg_eight import setup,stage_only
    torch.set_num_threads(2)
    m,opt,c,(_,data)=fixture(preset='medium',games=16)
    c['update'].update(effective_batch_size=128,microbatch_size=64)
    ds=updates.Dataset(data,'cpu');weights=copy.deepcopy(m.state_dict());adam=copy.deepcopy(opt.state_dict())
    rng=torch.get_rng_state().clone();results={};build=calibration_cache.build
    for cached in (False,True):
        samples=[];calls=[]
        for repeat in range(4):
            m.load_state_dict(weights);m.utility_calibration=None;opt.load_state_dict(copy.deepcopy(adam));torch.set_rng_state(rng)
            with patch.object(calibration_cache,'build',build if cached else lambda *a:None),patch.object(m,'encode_state',wraps=m.encode_state) as enc:
                start=time.perf_counter();r=utility.calibrate(m,opt,ds,c['update']);elapsed=time.perf_counter()-start
            if repeat:samples.append(elapsed);calls.append(enc.call_count)
        results['calibration_cached' if cached else 'calibration_before']=dict(seconds=samples,median_seconds=statistics.median(samples),
            rows=len(data),steps=r['receipt']['steps'],encoder_calls=calls,cache=r.get('frozen_input_cache'))
    # Identical complete-legal distributions and batch sequence. Timed region
    # includes packing, reference probability transfer, forward and backward.
    ids=[i for i,d in enumerate(data) if len(d[3].choice_ids)>1]
    ref=value_phase.RawPolicyReference(m,ds,ids,c['update']);parts=list(ref.microbatches(ids[:128]))
    for cached in (False,True):
        samples=[]
        for repeat in range(7):
            m.zero_grad();start=time.perf_counter()
            for part in parts:
                if cached:
                    inputs=ref.inputs(part);loss=ref.loss(part,inputs,m)
                else:
                    inputs=ds.batch(part)[1];logits=m(*inputs)[0].float();old=torch.zeros_like(logits)
                    for j,i in enumerate(part):old[j,:len(ref.rows[i])]=ref.rows[i].to(logits.device)
                    logp=torch.log_softmax(logits,-1).masked_fill(~inputs[2],0.)
                    loss=(old*(old.clamp_min(1e-30).log()-logp)).sum(-1)
                loss.sum().backward()
            if repeat>1:samples.append(time.perf_counter()-start)
        results['cloning_cached' if cached else 'cloning_before']=dict(seconds=samples,median_seconds=statistics.median(samples),rows=sum(map(len,parts)),cache=ref.cache.report())
    m,_,c=setup();c=ppg_config.fixed_options(c);run=output/'read-benchmark';p=TrainingPhases(m,c,root=ROOT,run=run)
    for g in (1,2):
        stage_only(m,p,g)
        if g==1:write_json(run/'state.json',dict(iteration=g));p.buffer.committed(g)
    with p.buffer.datasets('cpu') as (critic,_,_):
        ids=[(i*29)%len(critic.data) for i in range(256)]
        for vectorized in (False,True):
            samples=[]
            for repeat in range(22):
                start=time.perf_counter()
                if vectorized:
                    critic.batch(ids)
                    for name in ('targets','utility','remaining','descriptors'):critic.data.column(ids,name)
                else:
                    updates.Dataset.batch(critic,ids,value_only=True)
                    for name in ('targets','utility','remaining','descriptors'):
                        [critic.data.locate(i)[0].array[critic.data.locate(i)[1]][name] for i in ids]
                if repeat>1:samples.append(time.perf_counter()-start)
            results['critic_batch_vectorized' if vectorized else 'critic_batch_before']=dict(seconds=samples,median_seconds=statistics.median(samples),rows=len(ids))
    return dict(device='cpu',threads=2,cuda_available=torch.cuda.is_available(),measurements=results,
        scope='Authored small fixtures, warmed paired CPU samples; calibration includes cache construction. Cloning cache construction separately reported. No production GPU speed prediction.')


def main():
    import torch
    from brass_ai.learning.contracts import require,project_path
    from brass_ai.step5.league import write_json,implementation_id
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    out=project_path(ROOT,args.output);require(out.is_relative_to(ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts directory')
    out.mkdir(parents=True);torch.set_num_threads(2);start=time.perf_counter()
    spec=importlib.util.spec_from_file_location('local_smoke',ROOT/'scripts/smoke-local-scoring.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    training=helper.training(out,throughput=True);benchmark=benchmarks(out)
    result=dict(passed=True,source=implementation_id(),seconds=time.perf_counter()-start,training=training,benchmark=benchmark)
    write_json(out/'report.json',result)
    print({k:round(v['median_seconds'],4) for k,v in benchmark['measurements'].items()},flush=True)
    print(str(out/'report.json'),flush=True)


if __name__=='__main__':main()
