"""Bounded authored PPO+scheduled PPG, eight-age fill and exact mid-cycle restart."""
import argparse
import copy
from pathlib import Path
import shutil
import time

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.recovery import ppg_config,value_phase,gae,critic_data,snapshot_r2
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.history import compact
from brass_ai.step5.league import write_json,implementation_id
from test_ppg_eight import setup,collect
from test_gae_value_phase import ppo,finish
from test_value_buffer import commit


def advance(model,opt,cfg,phase,run,g):
    behavior,data=collect(model,phase,g,branches=True)
    actor=ppo(model,opt,cfg,behavior,data,phase);before=copy.deepcopy(opt.state_dict())
    value=phase.fit(model)
    require(_same_payload(before,opt.state_dict()),'PPG changed PPO optimizer state')
    finish(phase,actor,g)
    output=run/f'generation-{g:04d}.pt'
    save_checkpoint(output,model,opt,step=g,training={KEY:phase.payload()})
    commit(run,phase,g)
    require(compact(dict(actor=actor,value_phase=value))['value_phase']==value,'Compact history dropped PPG evidence')
    return output,dict(generation=g,ppo_steps=actor['optimizer_steps'],value=value)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=project_path(SOURCE_ROOT,args.output)
    require(not out.exists(),'Use a new smoke output directory');out.mkdir(parents=True)
    torch.set_num_threads(2);began=time.perf_counter();source=implementation_id()
    model,opt,cfg=setup(preset='medium');old=copy.deepcopy(cfg)
    old['value_phase']=copy.deepcopy(value_phase.DEFAULT)
    # Reduce only authored fixture budgets, retaining eight-age replay/cadence/
    # validation, uniform weights, initial two epochs, patience two and both 0.002 guards.
    cfg=ppg_config.configure(old)
    cfg['value_phase'].update(batch_size=128,clone_rows_per_age=48,clone_batch_size=24,
        probe_rows_per_age=12,current_probe_rows=48,acceptance_steps=2)
    # Create an explicit small generation-418 state fixture, not production weights.
    initial=TrainingPhases(model,old)
    initial.state=gae.initial(old['advantage'],418)
    initial.critic_state=critic_data.initial(old['critic_data']);initial.critic_state['generation']=418
    initial.critic_state['weight_recovery']=snapshot_r2.initial(418,initial.critic_state['snapshot_weight'])
    initial.critic_evidence['origin']=418
    initial.beta=8.  # g418's saved cloning coefficient, preserved on adoption
    original=out/'original.pt';save_checkpoint(original,model,opt,step=418,training={KEY:initial.payload()})
    original_hash=sha(original)
    # The exact migration permits only the production recipe, so start that
    # schedule first, then use the explicitly authored small fixture options.
    model,raw=load_checkpoint(original);opt=torch.optim.AdamW(model.parameters());resume_optimizer(raw,opt)
    production=ppg_config.configure(old)
    run=out/'continuous';phase=TrainingPhases(model,production,raw=raw,generation=418,root=SOURCE_ROOT,run=run)
    phase.config=cfg;phase.options['value_phase']=copy.deepcopy(cfg['value_phase']);phase.buffer.config=cfg
    results=[]
    for g in range(419,426):
        checkpoint,r=advance(model,opt,cfg,phase,run,g);results.append(r)
        require(not r['value']['scheduled'] and r['value']['optimizer_steps']==0,'PPG ran without eight fresh iterations')
    checkpoint_hash=sha(checkpoint);copy_run=out/'restarted';shutil.copytree(run,copy_run)
    saved_rng=torch.get_rng_state().clone()
    restart_model,restart_raw=load_checkpoint(copy_run/checkpoint.name)
    # Loading constructs a model before applying saved weights. Preserve the
    # continuous branch's RNG; resume_optimizer restores the restarted branch.
    torch.set_rng_state(saved_rng)
    first,r=advance(model,opt,cfg,phase,run,426);results.append(r)
    expected=dict(weights=copy.deepcopy(model.state_dict()),ppo=copy.deepcopy(opt.state_dict()),
        value=copy.deepcopy(phase.optimizer.state_dict()),rng=torch.get_rng_state().clone(),schedule=copy.deepcopy(phase.schedule),
        epoch_control=copy.deepcopy(phase.epoch_control),beta=phase.beta)
    restart_opt=torch.optim.AdamW(restart_model.parameters());resume_optimizer(restart_raw,restart_opt)
    resumed=TrainingPhases(restart_model,cfg,raw=restart_raw,generation=425,root=SOURCE_ROOT,run=copy_run)
    other,rr=advance(restart_model,restart_opt,cfg,resumed,copy_run,426)
    actual=dict(weights=restart_model.state_dict(),ppo=restart_opt.state_dict(),value=resumed.optimizer.state_dict(),
        rng=torch.get_rng_state(),schedule=resumed.schedule,epoch_control=resumed.epoch_control,beta=resumed.beta)
    require(_same_payload(expected,actual),'Mid-cycle checkpoint restart changed the next scheduled update')
    require(r['value']['scheduled'] and r['value']['schedule']['origin']==418,'Wrong first scheduled phase')
    require(sha(checkpoint)==checkpoint_hash and sha(original)==original_hash,'Incoming checkpoint changed')
    require(r['value']['scheduled'] and len(r['value']['age_support'])==8,'Full eight-age phase did not run')
    require(r['value']['weighting']=='uniform' and r['value']['acceptance_games']==48
        and r['value']['acceptance_generations']==list(range(426,418,-1)),
        'PPG did not use uniform training weights and all eight ordinary holdout sets')
    require(r['value']['epoch_cap_used']==2 and r['value']['epoch_controller']['after']['phases']==1,
        'Adaptive epoch state was not committed with the first phase')
    require(implementation_id()==source,'Source changed during smoke')
    write_json(out/'report.json',dict(passed=True,seconds=time.perf_counter()-began,implementation=source,
        parameters=model.parameter_count,exact_restart=True,incoming_checkpoint_unchanged=True,
        peak_cache_bytes=max(r['value']['buffer']['peak_disk_bytes'] for r in results),
        results=results,restart_phase_memory=rr['value']['memory'],
        scope='CPU medium v11.2 authored 18-game fixtures, generations labeled 418-426 for transition testing; reduced row budgets. No production checkpoint, native full games, GPU throughput, full-size memory, evaluation or strength claim.'))
    print(dict(passed=True,seconds=time.perf_counter()-began,results=len(results)))


if __name__=='__main__':main()
