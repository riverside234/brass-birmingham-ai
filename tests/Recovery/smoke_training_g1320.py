"""Small reward migration, fresh PPG cache and exact post-migration restart."""
import argparse
import copy
from pathlib import Path
import shutil
import time

import torch
from brass_ai.bank import sha
from brass_ai.learning.checkpoint import load_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require,digest
from brass_ai.recovery.actor_value import TrainingPhases
from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT as AUX,KEY as AUX_KEY
from brass_ai.recovery.ppo_preparation import PreparationPool
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.utility import reward_spec,migrate
from brass_ai.recovery.training_revision_g1320 import configure
from brass_ai.step5.league import write_json,implementation_id
from test_training_g1120_kl32 import smoke_config
from test_training_g1120 import collect_revised
import smoke_training_g850 as common

common.collect=collect_revised


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    out=project_path(SOURCE_ROOT,args.output)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use new isolated artifacts output')
    out.mkdir(parents=True);torch.set_num_threads(2);started=time.perf_counter();source=implementation_id()
    model,opt,old=smoke_config();old[AUX_KEY]=copy.deepcopy(AUX)
    old['update'].update(effective_batch_size=512,critic_batch_size=512,microbatch_size=32,preparation_workers=4)
    run=out/'continuous';p=TrainingPhases(model,old,root=SOURCE_ROOT,run=run);p.local_scoring=Companion(model,AUX)
    reports=[]
    with PreparationPool(4) as pool:
        for f in pool.ready:f.result()
        for g in (1,2):
            checkpoint,report=common.advance(model,opt,old,p,run,g,pool);reports.append(report)
        checksum=sha(checkpoint);saved_rng=torch.get_rng_state().clone()
        model,raw=load_checkpoint(checkpoint);opt=torch.optim.AdamW(model.parameters());resume_optimizer(raw,opt)
        torch.set_rng_state(saved_rng)
        new=configure(old);before=copy.deepcopy((model.state_dict(),opt.state_dict(),p.optimizer.state_dict(),torch.get_rng_state(),p.state,p.critic_state,p.beta,p.trips,p.local_scoring.payload(2)))
        model._reviewed_cap_upgrade=True;migration=migrate(model,opt,reward_spec(new))
        q=TrainingPhases(model,new,raw=raw,generation=2,root=SOURCE_ROOT,run=run)
        q.local_scoring=Companion(model,AUX,raw=raw,generation=2)
        require(_same_payload(before,(model.state_dict(),opt.state_dict(),q.optimizer.state_dict(),torch.get_rng_state(),q.state,q.critic_state,q.beta,q.trips,q.local_scoring.payload(2))),
            'Reward migration changed weights, Adam, RNG, beta or gate state')
        write_json(run/'state.json',dict(iteration=2,signature=dict(config=new)))
        cleanup=q.startup(2);require(cleanup['deleted_bytes']>0 and cleanup['next_phase']==3,'Old reward caches were not retired')
        require(sha(checkpoint)==checksum,'Incoming checkpoint was modified')
        checkpoint,report=common.advance(model,opt,new,q,run,3,pool);reports.append(report)
        require(report['value']['acceptance_generations']==[3],'First fresh phase reused previous reward targets')
        require(all(r['reward_id']==digest(reward_spec(new)) for r in q.buffer.receipts),'Stale reward cache')
        restart=out/'restart';shutil.copytree(run,restart)
        saved_rng=torch.get_rng_state().clone();other,new_raw=load_checkpoint(restart/checkpoint.name);torch.set_rng_state(saved_rng)
        _,report=common.advance(model,opt,new,q,run,4,pool);reports.append(report)
        expected=copy.deepcopy((model.state_dict(),opt.state_dict(),q.optimizer.state_dict(),torch.get_rng_state(),q.state,q.critic_state,q.beta,q.trips,q.local_scoring.payload(4)))
        other_opt=torch.optim.AdamW(other.parameters());resume_optimizer(new_raw,other_opt)
        r=TrainingPhases(other,new,raw=new_raw,generation=3,root=SOURCE_ROOT,run=restart)
        r.local_scoring=Companion(other,AUX,raw=new_raw,generation=3)
        require(r.startup(3) is None,'Reward refresh repeated after new checkpoint commit')
        common.advance(other,other_opt,new,r,restart,4,pool)
        actual=(other.state_dict(),other_opt.state_dict(),r.optimizer.state_dict(),torch.get_rng_state(),r.state,r.critic_state,r.beta,r.trips,r.local_scoring.payload(4))
        require(_same_payload(expected,actual),'Post-migration restart diverged')
        require(report['value']['acceptance_generations']==[3,4],'Two-age replay did not refill')
    require(source==implementation_id(),'Source changed during smoke')
    result=dict(passed=True,seconds=time.perf_counter()-started,source=source,exact_restart=True,
        device='cpu',cuda_available=torch.cuda.is_available(),reward_migration=migration,cache_cleanup=cleanup,iterations=reports,
        scope='Four small authored iterations: two old-reward, refresh, two new-reward; final iteration repeated after restart. No production checkpoint or native evaluation used.')
    write_json(out/'report.json',result)
    print(dict(passed=True,seconds=result['seconds'],steps=[x['value']['optimizer_steps'] for x in reports],
        acceptance_generations=[x['value']['acceptance_generations'] for x in reports],exact_restart=True),flush=True)


if __name__=='__main__':main()
