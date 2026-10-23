"""Isolated CPU PPO/PPG cycle, exact resume, and paired cache-read benchmark."""
import argparse
import copy
from dataclasses import replace
import math
from pathlib import Path
import shutil
import statistics
import time

import numpy as np
import torch

from brass_ai.bank import sha
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery import ppg_packing as K,ppg_config,dcpg_config,ppg_replay
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.value_buffer import row_dtype
from brass_ai.step5.league import write_json,implementation_id
from test_ppg_sixteen import config
from test_ppg_eight import collect
from test_gae_value_phase import ppo,finish
from test_value_buffer import commit


def advance(model,opt,cfg,phase,run,g):
    behavior,data=collect(model,phase,g,branches=True)
    actor=ppo(model,opt,cfg,behavior,data,phase);before=copy.deepcopy(opt.state_dict())
    value=phase.fit(model);require(_same_payload(before,opt.state_dict()),'PPG changed PPO Adam')
    finish(phase,actor,g)
    checkpoint=run/f'generation-{g:04d}.pt'
    if g in (15,16):save_checkpoint(checkpoint,model,opt,step=g,training={KEY:phase.payload()})
    commit(run,phase,g)
    return checkpoint,dict(generation=g,ppo_steps=actor['optimizer_steps'],value=value)


def benchmark(out,phase):
    out.mkdir();receipt=phase.buffer.receipts[-1]
    rows=ppg_replay.Rows(phase.buffer.path(receipt['generation'],'critic.npy'),receipt)
    try:
        raw=np.empty(len(rows),dtype=row_dtype())
        for name in raw.dtype.names:raw[name]=rows.array.column(name,slice(None))
    finally:rows.close()
    # Repeat authored tensors to exercise page/gather overhead at a bounded size.
    # This is not a claim about compression ratios on production/native games.
    raw=np.tile(raw,64);np.save(out/'dense.npy',raw,allow_pickle=False)
    actions=raw['action'].copy();np.save(out/'actions.npy',actions,allow_pickle=False)
    dense=np.load(out/'dense.npy',mmap_mode='r');policy=np.load(out/'actions.npy',mmap_mode='r')
    began=time.perf_counter();layout=K.pack(lambda ext:out/ext,dense,policy,len(receipt['game_ids']))
    pack_seconds=time.perf_counter()-began
    packed=K.CriticArray(lambda ext:out/ext,dict(rows=len(dense),packing=layout))
    fields=('state','action','mapping','targets','utility','remaining','descriptors')
    rng=np.random.default_rng(614);batches=[rng.integers(len(dense),size=512) for _ in range(16)]
    try:
        for field in dense.dtype.names:
            require(packed.column(field,slice(None)).tobytes()==dense[field].tobytes(),'Packed tensor bits changed: '+field)
        def read_all(compact):
            for ids in batches:
                for name in fields:
                    a=packed.column(name,ids) if compact else dense[name][ids]
                    require(a.shape[0]==len(ids),'Benchmark gather lost rows')
        read_all(False);read_all(True)
        results={}
        for compact in (False,True):
            times=[]
            for _ in range(5):
                start=time.perf_counter();read_all(compact);times.append(time.perf_counter()-start)
            results['packed' if compact else 'dense']=dict(median_seconds=statistics.median(times),samples=times)
        return dict(rows=len(dense),batch_size=512,batches_per_trial=16,bit_exact=True,
            dense_bytes=dense.nbytes+policy.nbytes,packed_bytes=layout['packed_array_bytes'],
            packing_seconds=pack_seconds,reads=results,packing=layout,
            scope='Warm CPU mmap gathers, repeated authored full-width fixture tensors; same random row indices including duplicates. No production/GPU throughput or full-size memory claim.')
    finally:packed.close();dense._mmap.close();policy._mmap.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--insolvency-040',action='store_true');a=p.parse_args()
    out=project_path(SOURCE_ROOT,a.output);require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a fresh artifacts output')
    out.mkdir(parents=True);torch.set_num_threads(2);began=time.perf_counter();source=implementation_id()
    model,opt,cfg=config();cfg['value_phase']['batch_size']=128
    if a.insolvency_040:
        from brass_ai.recovery.config import insolvency_010_config,insolvency_020_config,insolvency_040_cap100_config,vp_cap_config
        from brass_ai.recovery.utility import reward_spec
        cfg=vp_cap_config(insolvency_040_cap100_config(insolvency_020_config(insolvency_010_config(cfg))))
        model.config=replace(model.config,utility=reward_spec(cfg))
    cfg['update'].update(effective_batch_size=512,critic_batch_size=512,microbatch_size=32)
    run=out/'continuous';phase=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run);reports=[]
    for g in range(1,16):
        checkpoint,r=advance(model,opt,cfg,phase,run,g);reports.append(r)
        require(not r['value']['scheduled'] and not r['value']['optimizer_steps'],'Early PPG phase')
    before_hash=sha(checkpoint);restart=out/'restart';shutil.copytree(run,restart)
    saved_rng=torch.get_rng_state().clone();other,raw=load_checkpoint(restart/checkpoint.name)
    torch.set_rng_state(saved_rng)
    _,r=advance(model,opt,cfg,phase,run,16);reports.append(r)
    expected=dict(weights=copy.deepcopy(model.state_dict()),ppo=copy.deepcopy(opt.state_dict()),
        value=copy.deepcopy(phase.optimizer.state_dict()),rng=torch.get_rng_state().clone(),
        schedule=copy.deepcopy(phase.schedule),beta=phase.beta,gae=copy.deepcopy(phase.state))
    other_opt=torch.optim.AdamW(other.parameters());resume_optimizer(raw,other_opt)
    resumed=TrainingPhases(other,cfg,raw=raw,generation=15,root=SOURCE_ROOT,run=restart)
    _,rr=advance(other,other_opt,cfg,resumed,restart,16)
    actual=dict(weights=other.state_dict(),ppo=other_opt.state_dict(),value=resumed.optimizer.state_dict(),
        rng=torch.get_rng_state(),schedule=resumed.schedule,beta=resumed.beta,gae=resumed.state)
    require(_same_payload(expected,actual),'Restart changed optimizer/model/RNG/evidence')
    require(sha(checkpoint)==before_hash,'Incoming checkpoint changed')
    value=r['value']
    require(value['scheduled'] and value['acceptance_generations']==list(range(16,0,-1))
        and value['acceptance_games']==96 and len(value['age_support'])==16,'Incomplete held-out window')
    require(all(math.isfinite(value[k]) and value[k]>=0 for k in ('probe_kl','age0_probe_kl')),'Invalid measured PPG KL')
    require(value['kl_cap'] is None and not value['policy_guard_enabled'] and not value['kl_guard_tripped']
        and value['beta_retries']==0 and phase.beta==cfg['value_phase']['beta_clone'],'Penalty-only PPG changed beta or applied a hard cap')
    selection=value['selection']
    require(selection['retained_loss']==selection['minimum_loss']<=selection['initial_loss']
        and selection['retained_budget_step']==selection['minimum_budget_step'],'PPG did not restore minimum held-out loss')
    bench=benchmark(out/'benchmark',phase)
    require(source==implementation_id(),'Source changed during smoke')
    report=dict(passed=True,seconds=time.perf_counter()-began,implementation=source,parameters=model.parameter_count,
        insolvency=cfg['insolvency'],
        exact_restart=True,incoming_checkpoint_unchanged=True,results=reports,benchmark=bench,
        scope='CPU authored fixtures, 18 games and reduced row budgets per iteration. Sixteen committed iterations, one real scheduled PPG phase, repeated after g15 restart. No production run or evaluation.')
    write_json(out/'report.json',report)
    print(dict(passed=True,seconds=report['seconds'],accepted_steps=value['optimizer_steps'],attempted_steps=value['attempted_steps'],
        holdout_games=value['acceptance_games'],overall_kl=value['probe_kl'],age0_kl=value['age0_probe_kl'],
        benchmark={k:v for k,v in bench.items() if k!='packing'}))


if __name__=='__main__':main()
