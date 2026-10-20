"""Authored g185 checkpoint: reviewed source resume and two bounded CPU updates."""
import argparse
import copy
from pathlib import Path
import shutil
import time
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,digest,project_path,require
from brass_ai.recovery import gae,critic_data,il_kl_limit as limits,value_phase
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.g110_hotfix import verify_checkpoint
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.resume import load_resume,apply_optimizer_config
from brass_ai.step5.league import write_json,implementation_id
from test_critic_descriptors import mixed_fixture
from test_gae_value_phase import ppo,finish
from test_value_buffer import commit
from smoke_ppg_refinement import collect,payload
from test_auto_il_kl import measurement
from test_selfplay185_review import EXPORT


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--ppg-window',action='store_true',help='Exercise the signed PPG interval/GAE window continuation')
    args=parser.parse_args();out=project_path(SOURCE_ROOT,args.output)
    require(not out.exists(),'Use a new smoke directory');out.mkdir(parents=True)
    began=time.perf_counter();torch.set_num_threads(2);implementation=implementation_id()
    exported=read(EXPORT/'state.json');last=read(EXPORT/'iteration-0184/report.json')
    model,opt,cfg,_,_,_,_=mixed_fixture(preset='medium')
    cfg=value_phase.refined_options(cfg)
    cfg['il_kl_limit']=copy.deepcopy(exported['signature']['config']['il_kl_limit'])
    cfg['value_phase'].update(buffer_iterations=2,max_epochs=1,batch_size=128,clone_batch_size=16)
    cfg['update'].update(effective_batch_size=128,critic_batch_size=64,microbatch_size=16)
    oldsig=dict(exported['signature'],config=cfg)
    signature=dict(oldsig,implementation=implementation)
    run=out/'authored';phase=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
    phase.state=gae.initial(cfg['advantage'],generation=184)
    phase.critic_state=critic_data.initial(cfg['critic_data']);phase.critic_state['generation']=184
    phase.critic_state['weight_recovery']=critic_data.recovery_initial(184)
    phase.buffer.generation=184
    b,data=collect(model,phase,'authored-185-')
    update=ppo(model,opt,cfg,b,data,phase);phase.fit(model);finish(phase,update,185)
    old=run/'generation-0185.pt'
    identity=save_checkpoint(old,model,opt,step=185,training={
        KEY:phase.payload(), 'fresh_run':run.relative_to(SOURCE_ROOT).as_posix(),
        'signature':digest(oldsig),'il_kl_limit':copy.deepcopy(exported['il_kl_limit'])})
    commit(run,phase,185)
    entry=dict(path=old.relative_to(SOURCE_ROOT).as_posix(),sha256=sha(old),generation=185,model_id=identity)
    state=dict(exported,signature=oldsig,history=[entry],population=[entry],
               evaluation_anchor=entry,resume=[entry],source_transitions=[])
    if args.ppg_window:
        cfg=value_phase.interval_options(gae.health_options(cfg))
        signature=dict(oldsig,config=cfg,implementation=implementation)
    before=copy.deepcopy(state);results=[];finals=[]
    for branch in ('continuous','restarted'):
        dest=out/branch;shutil.copytree(run,dest)
        net,raw,changed=load_resume(SOURCE_ROOT,run,state,signature,reviewed=True)
        require(changed,'Missing reviewed source transition');verify_checkpoint(state,raw)
        adam=torch.optim.AdamW(net.parameters());resume_optimizer(raw,adam);apply_optimizer_config(adam,cfg)
        resumed=TrainingPhases(net,cfg,raw=raw,generation=185,root=SOURCE_ROOT,run=dest)
        require(_same_payload(adam.state_dict()['state'],raw['optimizer']['state']),'Adam moments changed')
        require(_same_payload(resumed.optimizer.state_dict(),raw['training'][KEY]['optimizer']),'Value optimizer changed')
        require(_same_payload(dict(net.state_dict()),raw['state_dict']),'Transition changed weights')
        require(torch.equal(torch.get_rng_state(),raw['rng']),'Transition changed RNG')
        if args.ppg_window:
            require(resumed.state['stage_evidence']==dict(contract='staged-gae-health-window/1.0.0',origin=185,entries=[]),
                    'Window migration backfilled evidence')
        limit=copy.deepcopy(raw['training']['il_kl_limit'])
        for generation in (186,187):
            b,data=collect(net,resumed,f'authored-{generation}-')
            update=ppo(net,adam,cfg,b,data,resumed);value=resumed.fit(net)
            # Stress the removed sales guard on the real g185 baseline, but
            # do not pretend this authored model supplies native-game health.
            current=dict(last['health_counts'],sells=0)
            limit,health=limits.advance(limit,measurement(.41,behavior=dict(ok=False,passing=dict(ok=False))),
                current,generation=generation,warmup_done=True)
            require(health['health']['ok'] and 'sells_per_seat' not in health['health']['checks']
                    and 'behavior' not in health['health'],'Retired IL action checks still block shared health')
            resumed.finish(generation=generation,update=update,anchor_gate=health['lr_gate'],rating=dict(ok=True))
            require(value['buffer']['previous_rows']>0 and value['buffer']['holdout_transition'] is None,
                    'Source-only health change discarded a compatible PPG cache')
            checkpoint=dest/f'generation-{generation:04d}.pt'
            save_checkpoint(checkpoint,net,adam,step=generation,training={KEY:resumed.payload(),'il_kl_limit':limit,
                'fresh_run':dest.relative_to(SOURCE_ROOT).as_posix(),'signature':digest(signature)})
            commit(dest,resumed,generation)
            results.append(dict(branch=branch,generation=generation,ppo_steps=update['optimizer_steps'],
                value_steps=value['optimizer_steps'],value_attempts=value['attempted_steps'],
                value_kl=value['probe_kl'],value_lr=value['learning_rate'],
                value_seconds=value['seconds'],accepted_intervals=value['accepted_intervals'],
                acceptance_steps=value['acceptance_steps'],stage_evidence=resumed.state.get('stage_evidence'),
                previous_rows=value['buffer']['previous_rows'],health=health['health'],
                limit=limit['limit'],group_limit=health['group_limit_after']))
            if branch=='restarted' and generation==186:
                net,raw=load_checkpoint(checkpoint);adam=torch.optim.AdamW(net.parameters())
                resume_optimizer(raw,adam);apply_optimizer_config(adam,cfg)
                resumed=TrainingPhases(net,cfg,raw=raw,generation=186,root=SOURCE_ROOT,run=dest)
                limit=copy.deepcopy(raw['training']['il_kl_limit'])
        finals.append(dict(payload(net,adam,resumed),limit=limit))
    require(_same_payload(*finals),'Checkpoint restart changed next update')
    require(state==before and sha(old)==entry['sha256'],'Original state/checkpoint changed')
    require(implementation_id()==implementation,'Source changed during smoke')
    write_json(out/'report.json',dict(passed=True,implementation=implementation,
        seconds=time.perf_counter()-began,parameters=model.parameter_count,ppg_window_upgrade=args.ppg_window,
        original_checkpoint_unchanged=True,exact_next_update_on_restart=True,results=results,
        scope='Authored v11.2 CPU weights/rows at g185, with real exported IL health baseline. Source admission and two resumed PPO/PPG updates per branch; not a live lab checkpoint, native game benchmark or strength evaluation.'))


if __name__=='__main__':main()
