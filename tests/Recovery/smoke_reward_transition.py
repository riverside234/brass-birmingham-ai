"""Bounded old-reward -> new-reward PPO/PPG resume; authored data, no game campaign."""
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
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import insolvency_event_config
from brass_ai.recovery.utility import reward_spec,migrate
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json,implementation_id


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
    model.config=replace(model.config,utility=reward_spec(cfg))
    b,rows=examples(model,games=9,decisions=4,prefix='old-');data,snap,_=partition(rows,'old-')
    run=out/'before';phase=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
    phase.snapshot_data=snap;phase.collection_source=dict(source(b),snapshot_rows=True)
    update=ppo(model,opt,cfg,b,data,phase);phase.fit(model);finish(phase,update,1)
    checkpoint=run/'generation-1.pt';save_checkpoint(checkpoint,model,opt,step=1,training={KEY:phase.payload()})
    commit(run,phase,1);checkpoint_hash=sha(checkpoint)
    target=insolvency_event_config(cfg);results=[];last=None
    # Two independent resumes from exactly the same saved model/Adams/RNG.
    for name in ('continuation','retry'):
        dest=out/name;shutil.copytree(run,dest)
        net,raw=load_checkpoint(checkpoint);adam=torch.optim.AdamW(net.parameters());resume_optimizer(raw,adam)
        before_weights=copy.deepcopy(net.state_dict());before_adam=copy.deepcopy(adam.state_dict());rng=torch.get_rng_state().clone()
        net._reviewed_cap_upgrade=True;migration=migrate(net,adam,reward_spec(target))
        require(_same_payload(before_weights,net.state_dict()) and _same_payload(before_adam,adam.state_dict())
            and torch.equal(rng,torch.get_rng_state()),'Reward transition changed weights/Adam/RNG')
        resumed=TrainingPhases(net,target,raw=raw,generation=1,root=SOURCE_ROOT,run=dest)
        require(_same_payload(raw['training'][KEY]['optimizer'],resumed.optimizer.state_dict()),'Value Adam changed at resume')
        require(resumed.buffer.receipt is None and resumed.buffer.reward_transition is not None,'Old reward buffer was admitted')
        b,rows=examples(net,games=9,decisions=4,prefix='new-');data,snap,_=partition(rows,'new-')
        resumed.snapshot_data=snap;resumed.collection_source=dict(source(b),snapshot_rows=True)
        update=ppo(net,adam,target,b,data,resumed);value=resumed.fit(net);finish(resumed,update,2)
        commit(dest,resumed,2)
        payload=dict(weights=net.state_dict(),adam=adam.state_dict(),value_adam=resumed.optimizer.state_dict(),
            gae=resumed.state,critic=resumed.critic_state,rng=torch.get_rng_state())
        if last is not None:require(_same_payload(last,payload),'Independent reward continuation differs')
        last=copy.deepcopy(payload)
        require(not resumed.buffer.path(1,'npy').exists(),'Old reward cache survived new commit')
        results.append(dict(migration=migration,actor_steps=update['optimizer_steps'],
            value_steps=value['optimizer_steps'],buffer=value['buffer']))
    require(sha(checkpoint)==checkpoint_hash and implementation_id()==source_id,'Smoke changed checkpoint/source')
    write_json(out/'report.json',dict(passed=True,seconds=time.perf_counter()-began,parameters=model.parameter_count,
        implementation=source_id,exact_resumed_next_update=True,weights_both_optimizers_RNG_preserved_at_transition=True,
        original_checkpoint_unchanged=True,old_reward_rows_excluded=True,expired_cache_deleted=True,results=results,
        scope='Authored mixed-table v11.2 CPU rows; old-reward PPO/PPG checkpoint plus two identical new-reward continuations. No native games or strength claim.'))


if __name__=='__main__':main()
