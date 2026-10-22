"""Small isolated g540 fresh-PPG reward migration and g548 exact-restart smoke."""
import argparse
import copy
from pathlib import Path
import shutil
import time

import torch

from brass_ai.bank import sha
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery import ppg_config as P,ppg_replay,ppg_epochs,critic_data,gae
from brass_ai.recovery.actor_value import TrainingPhases,KEY,ppo_evidence
from brass_ai.recovery.config import insolvency_010_config,insolvency_020_config
from brass_ai.recovery.opening_objective import DEFAULT as OPENING_OBJECTIVE
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.utility import migrate,reward_spec
from brass_ai.step5.league import write_json,implementation_id
from test_ppg_eight import setup,stage_only
from smoke_opening_ppo import advance


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    out=project_path(SOURCE_ROOT,args.output)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts directory')
    out.mkdir(parents=True);torch.set_num_threads(2);started=time.perf_counter();identity=implementation_id()
    model,opt,cfg=setup();old=critic_data.fixed_config(P.short_options(insolvency_010_config(cfg)))
    old['update']['opening_objective']=copy.deepcopy(OPENING_OBJECTIVE);old['exploration']=0.
    model._reviewed_cap_upgrade=True;migrate(model,opt,reward_spec(old))
    run=out/'continuous';previous=TrainingPhases(model,old,root=SOURCE_ROOT,run=run)
    previous.buffer=ppg_replay.Buffer(SOURCE_ROOT,run,old,generation=532,origin=532)
    for generation in range(533,541):
        stage_only(model,previous,generation)
        if generation<540:
            write_json(run/'state.json',dict(iteration=generation));previous.buffer.committed(generation)
    previous.state=gae.initial(old['advantage'],540);previous.critic_state['generation']=540
    previous.critic_evidence=ppo_evidence(540);previous.beta=8.
    clock=P.initial(532)
    for generation in range(533,541):clock=P.advance(clock,generation,P.due(clock))
    previous.pending_schedule=clock
    previous.pending_epoch_control=dict(ppg_epochs.initial(532,old['value_phase']),cap=1,
        phases=1,last_phase=540,window_origin=1)
    for optimizer in (opt,previous.optimizer):
        p=next(model.parameters());p.grad=torch.ones_like(p);optimizer.step();optimizer.zero_grad(set_to_none=True)
    original=out/'original-g540.pt';save_checkpoint(original,model,opt,step=540,training={KEY:previous.payload()})
    write_json(run/'state.json',dict(iteration=540,signature=dict(config=old)));previous.buffer.committed(540)
    original_hash=sha(original);cfg=P.coverage_options(insolvency_020_config(old))
    cfg['value_phase'].update(batch_size=32,clone_rows_per_age=48,clone_batch_size=24,
        probe_rows_per_age=12,current_probe_rows=48)
    model,raw=load_checkpoint(original);opt=torch.optim.AdamW(model.parameters());resume_optimizer(raw,opt)
    before=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
    model._reviewed_cap_upgrade=True;migrate(model,opt,reward_spec(cfg))
    # Resume validation binds production-size options; scale only the authored
    # smoke's tensor budgets after the reviewed actor/value migration.
    production=P.coverage_options(insolvency_020_config(old))
    phase=TrainingPhases(model,production,raw=raw,generation=540,root=SOURCE_ROOT,run=run)
    require(_same_payload(before,model.state_dict()) and _same_payload(adam,opt.state_dict()),'Resume reset weights/PPO Adam')
    require(torch.equal(rng,torch.get_rng_state()),'Resume changed RNG')
    phase.config=cfg;phase.options['value_phase']=copy.deepcopy(cfg['value_phase']);phase.buffer.config=cfg
    write_json(run/'state.json',dict(iteration=540,signature=dict(config=cfg)));cleanup=phase.startup(540)
    require(cleanup['deleted_bytes']>0 and phase.epoch_control['cap']==2,'Fresh reset failed')
    results=[]
    for generation in range(541,548):
        checkpoint,report=advance(model,opt,cfg,phase,run,generation);results.append(report)
        require(not report['value']['scheduled'],'PPG ran before eight fresh iterations')
    copied=out/'restarted';shutil.copytree(run,copied);saved_rng=torch.get_rng_state().clone()
    other,other_raw=load_checkpoint(copied/checkpoint.name);torch.set_rng_state(saved_rng)
    _,report=advance(model,opt,cfg,phase,run,548);results.append(report)
    expected=dict(weights=copy.deepcopy(model.state_dict()),adam=copy.deepcopy(opt.state_dict()),
        value_adam=copy.deepcopy(phase.optimizer.state_dict()),rng=torch.get_rng_state().clone(),
        schedule=copy.deepcopy(phase.schedule),epochs=copy.deepcopy(phase.epoch_control))
    other_opt=torch.optim.AdamW(other.parameters());resume_optimizer(other_raw,other_opt)
    restarted=TrainingPhases(other,cfg,raw=other_raw,generation=547,root=SOURCE_ROOT,run=copied)
    require(restarted.startup(547) is None,'Ordinary restart repeated fresh-cache reset')
    advance(other,other_opt,cfg,restarted,copied,548)
    actual=dict(weights=other.state_dict(),adam=other_opt.state_dict(),value_adam=restarted.optimizer.state_dict(),
        rng=torch.get_rng_state(),schedule=restarted.schedule,epochs=restarted.epoch_control)
    require(_same_payload(expected,actual),'PPG differs after restart')
    value=report['value'];require(value['acceptance_generations']==list(range(548,540,-1)),'Wrong validation ages')
    require(value['selection']['retained_vs_minimum']['reject'] is False,'Selected state outside one SE')
    require(value['probe_kl']<=.002 and value['age0_probe_kl']<=.002,'Retained KL guard violated')
    require(sha(original)==original_hash and implementation_id()==identity,'Incoming checkpoint/source changed')
    result=dict(passed=True,implementation=identity,seconds=time.perf_counter()-started,device='cpu',threads=2,
        parameters=model.parameter_count,exact_restart=True,original_checkpoint_unchanged=True,
        startup_cleanup=cleanup,results=results,
        scope='Authored small mixed-table/branch fixtures and real PPO/PPG/checkpoints. No production training/evaluation or strength claim.')
    write_json(out/'report.json',result)
    print(dict(passed=True,seconds=result['seconds'],exact_restart=True,ppg_steps=value['optimizer_steps'],
        attempted=value['attempted_steps'],selection=value['selection'],kl=value['probe_kl'],age0_kl=value['age0_probe_kl']))


if __name__=='__main__':main()
