"""Isolated authored PPO/opening correction and an eight-age PPG phase with restart."""
import argparse
import copy
from pathlib import Path
import shutil
import time

import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.recovery import ppg_config as P,critic_data as C,gae,value_phase,snapshot_r2
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json,implementation_id
from test_ppg_eight import setup,collect
from test_gae_value_phase import ppo,finish
from test_opening_ppo import opening_rows,O
from test_value_buffer import source,commit


def advance(model,opt,cfg,phase,run,g):
    _,data=collect(model,phase,g,branches=True)
    behavior,data=opening_rows(model,data)
    phase.collection_source=dict(source(behavior),snapshot_rows=True)
    original=[copy.deepcopy(r['behavior']) for r,_,_,_ in data]
    actor=ppo(model,opt,cfg,behavior,data,phase);before=copy.deepcopy(opt.state_dict())
    require(original==[r['behavior'] for r,_,_,_ in data],'Behavior records were relabeled')
    value=phase.fit(model)
    require(_same_payload(before,opt.state_dict()),'PPG changed PPO Adam state')
    finish(phase,actor,g);path=run/f'generation-{g:04d}.pt'
    save_checkpoint(path,model,opt,step=g,training={KEY:phase.payload()});commit(run,phase,g)
    return path,dict(generation=g,actor=actor,value=value)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    out=project_path(SOURCE_ROOT,args.output)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts directory')
    out.mkdir(parents=True);torch.set_num_threads(2);began=time.perf_counter();identity=implementation_id()
    model,opt,cfg=setup(preset='medium');old=value_phase.disabled_options(cfg)
    old=C.fixed_config(old)
    initial=TrainingPhases(model,old)
    initial.state=gae.initial(old['advantage'],503);initial.critic_state=C.initial(old['critic_data'])
    initial.critic_state.update(generation=503)
    initial.critic_evidence['origin']=503;initial.beta=8.
    original=out/'original.pt';save_checkpoint(original,model,opt,step=503,training={KEY:initial.payload()})
    original_hash=sha(original)
    cfg=P.short_options(old);cfg['update']['opening_objective']=copy.deepcopy(O.DEFAULT);cfg['exploration']=0.
    # Fixture-sized tensors retain the requested eight-age schedule, all-eight
    # heldouts, acceptance=16, patience=1 and both exact .002 PPG safeguards.
    # A small row batch also crosses a complete acceptance interval.
    cfg['value_phase'].update(batch_size=32,clone_rows_per_age=48,clone_batch_size=24,
        probe_rows_per_age=12,current_probe_rows=48)
    model,raw=load_checkpoint(original);opt=torch.optim.AdamW(model.parameters());resume_optimizer(raw,opt)
    production=P.short_options(old);production['update']=copy.deepcopy(cfg['update'])
    run=out/'continuous';phase=TrainingPhases(model,production,raw=raw,generation=503,root=SOURCE_ROOT,run=run)
    phase.config=cfg;phase.options['value_phase']=copy.deepcopy(cfg['value_phase']);phase.buffer.config=cfg
    results=[]
    for g in range(504,511):
        checkpoint,r=advance(model,opt,cfg,phase,run,g);results.append(r)
        require(not r['value']['scheduled'],'PPG ran before eight fresh iterations')
    previous_hash=sha(checkpoint);copy_run=out/'restarted';shutil.copytree(run,copy_run)
    saved_rng=torch.get_rng_state().clone();other,other_raw=load_checkpoint(copy_run/checkpoint.name)
    torch.set_rng_state(saved_rng)
    _,r=advance(model,opt,cfg,phase,run,511);results.append(r)
    expected=dict(weights=copy.deepcopy(model.state_dict()),adam=copy.deepcopy(opt.state_dict()),
        value_adam=copy.deepcopy(phase.optimizer.state_dict()),rng=torch.get_rng_state().clone(),
        schedule=copy.deepcopy(phase.schedule),epochs=copy.deepcopy(phase.epoch_control))
    other_opt=torch.optim.AdamW(other.parameters());resume_optimizer(other_raw,other_opt)
    restarted=TrainingPhases(other,cfg,raw=other_raw,generation=510,root=SOURCE_ROOT,run=copy_run)
    _,rr=advance(other,other_opt,cfg,restarted,copy_run,511)
    actual=dict(weights=other.state_dict(),adam=other_opt.state_dict(),value_adam=restarted.optimizer.state_dict(),
        rng=torch.get_rng_state(),schedule=restarted.schedule,epochs=restarted.epoch_control)
    require(_same_payload(expected,actual),'Scheduled update differs after restart')
    value=r['value'];require(value['scheduled'] and value['acceptance_generations']==list(range(511,503,-1)),
        'PPG did not use all eight held-out iterations')
    require(value['acceptance_steps']==16 and value['attempted_steps']>=16,'PPG phase did not run a full acceptance interval')
    require(value['probe_kl']<=.002 and value['age0_probe_kl']<=.002,'Retained PPG KL exceeds cap')
    require(sha(original)==original_hash and sha(checkpoint)==previous_hash,'Incoming checkpoint changed')
    require(implementation_id()==identity,'Source changed during smoke')
    report=dict(passed=True,implementation=identity,seconds=time.perf_counter()-began,parameters=model.parameter_count,
        device='cpu',threads=2,exact_restart=True,original_checkpoint_unchanged=True,
        peak_cache_bytes=max(r['value']['buffer']['peak_disk_bytes'] for r in results),results=results,
        scope='Authored small mixed-table/branch fixtures, medium v11.2 model. Real PPO/PPG and disk checkpoints. No production weights, native games, full-size memory benchmark or strength evaluation.')
    write_json(out/'report.json',report)
    print(dict(passed=True,seconds=report['seconds'],parameters=model.parameter_count,exact_restart=True,
        ppg_attempted_steps=value['attempted_steps'],ppg_retained_steps=value['optimizer_steps'],
        kl=value['probe_kl'],age0_kl=value['age0_probe_kl'],peak_cache_bytes=report['peak_cache_bytes']))


if __name__=='__main__':main()
