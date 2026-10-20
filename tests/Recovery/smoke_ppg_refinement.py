"""Bounded authored g110->111->112 PPG continuation; no native games or ratings."""
import argparse
import copy
from dataclasses import replace
from pathlib import Path
import shutil
import time

import torch

from test_critic_descriptors import mixed_fixture,partition
from test_gae_value_phase import examples,ppo,finish
from test_value_buffer import source,commit
from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.recovery import gae,critic_data,critic_holdout,value_phase
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import insolvency_event_config
from brass_ai.recovery.utility import reward_spec,migrate
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json,implementation_id


def collect(model,phases,prefix):
    b,rows=examples(model,games=9,decisions=4,prefix=prefix)
    data,snap,_=partition(rows,prefix)
    phases.snapshot_data=snap;phases.collection_source=dict(source(b),snapshot_rows=True)
    return b,data


def payload(model,optimizer,phases):
    return dict(weights=copy.deepcopy(model.state_dict()),ppo=copy.deepcopy(optimizer.state_dict()),
        value=copy.deepcopy(phases.optimizer.state_dict()),gae=copy.deepcopy(phases.state),
        critic=copy.deepcopy(phases.critic_state),rng=torch.get_rng_state().clone())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=project_path(SOURCE_ROOT,args.output)
    require(not out.exists(),'Use a new smoke directory');out.mkdir(parents=True)
    torch.set_num_threads(2);began=time.perf_counter();source_id=implementation_id()
    model,opt,cfg,_,_,_,_=mixed_fixture(preset='medium')
    cfg['insolvency']=dict(event_cost=.005,max_cost=.075)
    cfg['search']['reward']['insolvency']=copy.deepcopy(cfg['insolvency'])
    cfg['value_phase'].update(buffer_iterations=2,max_epochs=1,batch_size=128,clone_batch_size=16)
    cfg['update'].update(effective_batch_size=128,critic_batch_size=64,microbatch_size=16)
    model.config=replace(model.config,utility=reward_spec(cfg));run=out/'old'
    phase=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
    # Authored progress fixture, not a claim to have trained 109 generations.
    phase.state=gae.initial(cfg['advantage'],generation=109)
    phase.critic_state=critic_data.initial(cfg['critic_data']);phase.critic_state['generation']=109
    phase.critic_state['weight_recovery']=critic_data.recovery_initial(109)
    phase.buffer.generation=109
    b,data=collect(model,phase,'old-');update=ppo(model,opt,cfg,b,data,phase)
    phase.fit(model);finish(phase,update,110)
    old=run/'generation-0110.pt';save_checkpoint(old,model,opt,step=110,training={KEY:phase.payload()})
    commit(run,phase,110);old_hash=sha(old)
    target=value_phase.refined_options(insolvency_event_config(cfg));results=[];finals=[]
    for branch in ('a','b'):
        dest=out/branch;shutil.copytree(run,dest)
        net,raw=load_checkpoint(old);adam=torch.optim.AdamW(net.parameters());resume_optimizer(raw,adam)
        before_weights=copy.deepcopy(net.state_dict());before_adam=copy.deepcopy(adam.state_dict());rng=torch.get_rng_state().clone()
        net._reviewed_cap_upgrade=True;migrate(net,adam,reward_spec(target))
        resumed=TrainingPhases(net,target,raw=raw,generation=110,root=SOURCE_ROOT,run=dest)
        old_value=copy.deepcopy(raw['training'][KEY]['optimizer']);old_value['param_groups'][0]['lr']=3e-6
        require(_same_payload(old_value,resumed.optimizer.state_dict()),'Value Adam moments changed')
        require(_same_payload(before_weights,net.state_dict()) and _same_payload(before_adam,adam.state_dict())
            and torch.equal(rng,torch.get_rng_state()),'Transition changed weights/PPO/RNG')
        for generation in (111,112):
            b,data=collect(net,resumed,f'new-{generation}-')
            update=ppo(net,adam,target,b,data,resumed);value=resumed.fit(net);finish(resumed,update,generation)
            if generation==111:
                require(value['buffer']['previous_rows']==0 and value['buffer']['holdout_transition'] is not None,
                    'Old holdout rows admitted')
            else:require(value['buffer']['previous_rows']>0 and value['buffer']['holdout_transition'] is None,
                    'Normal two-iteration reuse did not resume')
            require(value['learning_rate']==3e-6 and adam.param_groups[0]['lr']!=3e-6,'Value LR changed PPO plumbing')
            require(all(v['heldout_games']==2 for v in value['split']['by_table'].values()),'Missing stratified support')
            checkpoint=dest/f'generation-{generation:04d}.pt'
            save_checkpoint(checkpoint,net,adam,step=generation,training={KEY:resumed.payload()})
            commit(dest,resumed,generation)
            results.append(dict(branch=branch,generation=generation,ppo_steps=update['optimizer_steps'],
                value_steps=value['optimizer_steps'],value_attempts=value['attempted_steps'],
                value_lr=value['learning_rate'],value_stop=value['early_stop_reason'],probe_kl=value['probe_kl'],
                heldout_before=value['heldout_before']['loss'],heldout_after=value['heldout_after']['loss'],
                holdout=value['split']['by_table'],buffer=value['buffer']))
            if branch=='b' and generation==111:
                # Compare continued in-memory state against a real checkpoint
                # restore between the two new-policy iterations.
                net,raw=load_checkpoint(checkpoint);adam=torch.optim.AdamW(net.parameters());resume_optimizer(raw,adam)
                resumed=TrainingPhases(net,target,raw=raw,generation=111,root=SOURCE_ROOT,run=dest)
                require(resumed.refinement_migration is None,'PPG migration repeated after checkpoint')
        finals.append(payload(net,adam,resumed))
    require(_same_payload(*finals),'Restart changed the next update')
    require(sha(old)==old_hash and implementation_id()==source_id,'Old checkpoint/source changed')
    # Bounded metadata-only selection benchmark at actual collection sizes.
    selection_began=time.perf_counter();counts=[]
    for g in range(100):
        games={f'{g}-{kind}-{i}':kind for kind,n in zip(critic_holdout.TABLES,(160,144,16)) for i in range(n)}
        selection=critic_holdout.select(games,.05,critic_holdout.DEFAULT)
        counts.append({k:sum(games[x]==k for x in selection['heldout_games']) for k in critic_holdout.TABLES})
    require(all(c==dict(selfplay=8,historical=7,legacy=2) for c in counts),'Production-size holdout quota differs')
    write_json(out/'report.json',dict(passed=True,seconds=time.perf_counter()-began,implementation=source_id,
        parameters=model.parameter_count,exact_restart_next_update=True,original_checkpoint_unchanged=True,
        collection_selection_ms=(time.perf_counter()-selection_began)*1000/100,selection_counts=counts[0],results=results,
        scope='Authored v11.2 CPU checkpoint at g110; two resumed updates per branch, one branch restored at g111. No native games, full-lab throughput or playing-strength claim.'))


if __name__=='__main__':main()
