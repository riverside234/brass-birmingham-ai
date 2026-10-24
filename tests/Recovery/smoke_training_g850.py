"""Three bounded authored PPO/PPG iterations and exact restart after iteration 2."""
import argparse
import copy
from pathlib import Path
import shutil
import time

import torch
from brass_ai.bank import sha
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT as AUX,KEY as AUX_KEY
from brass_ai.recovery.ppo_preparation import PreparationPool
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery import updates
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import write_json,implementation_id
from test_training_g850 import smoke_config as config
from test_training_g850 import collect_revised as collect
from test_local_scoring import annotate
from test_gae_value_phase import finish
from test_value_buffer import commit


def advance(model,opt,c,p,run,g,pool):
    behavior,data=collect(model,p,g,branches=True);annotate(data);annotate(p.snapshot_data)
    reward=reward_spec(c)
    actor=updates.ppo_update(model,opt,[dict(split='train')],behavior,c['update'],learner_data=data,
        insolvency=reward['insolvency'],reward=reward,phases=p,preparation_pool=pool)
    before=copy.deepcopy(opt.state_dict());report=p.fit(model,preparation_pool=pool)
    require(_same_payload(before,opt.state_dict()),'PPG changed PPO Adam')
    require(report['learning_rate']==report['learning_rate_source']['ppo_lr']*.65,'PPG LR does not track actual PPO')
    require(report['scheduled'] and report['acceptance_generations']==list(range(max(1,g-1),g+1)),
        'Wrong PPG cadence/validation ages')
    require(len(p.buffer.pending)==min(2,g) and report['buffer']['packed_generations']==0,'Wrong replay window/format')
    require(report['probe_kl']<=.002+1e-7,'Retained phase exceeded hard KL cap')
    require(report['kl_check_steps']==16 and report['acceptance_steps']==16,'Wrong separated KL/loss cadence')
    require(report['acceptance_heldout_after']['loss']<=report['acceptance_heldout_before']['loss'],'Retained pooled critic loss increased')
    require(sum(x['rows'] for x in report['acceptance_support'])==report['acceptance_heldout_rows'],
        'Pooled holdout support differs')
    require(report['heldout_rows']==len(p.heldout_ids),'Gate evidence reused previous iteration')
    require(report['cpu_preparation']['workers']==4 and report['cpu_preparation']['batches']>0,'PPO preparation workers unused')
    finish(p,actor,g);checkpoint=run/f'generation-{g:04d}.pt'
    save_checkpoint(checkpoint,model,opt,step=g,training={KEY:p.payload(),AUX_KEY:p.local_scoring.payload(g)})
    commit(run,p,g);p.local_scoring.committed(g)
    return checkpoint,dict(generation=g,actor=actor,value=report)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    out=project_path(SOURCE_ROOT,args.output)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use new isolated artifacts output')
    out.mkdir(parents=True);torch.set_num_threads(2);started=time.perf_counter();source=implementation_id()
    model,opt,c=config();c[AUX_KEY]=copy.deepcopy(AUX)
    c['update'].update(effective_batch_size=512,critic_batch_size=512,microbatch_size=32,preparation_workers=4)
    run=out/'continuous';p=TrainingPhases(model,c,root=SOURCE_ROOT,run=run);p.local_scoring=Companion(model,AUX)
    reports=[]
    with PreparationPool(4) as pool:
        for f in pool.ready:f.result()
        for g in (1,2):
            checkpoint,report=advance(model,opt,c,p,run,g,pool);reports.append(report)
        checksum=sha(checkpoint);restart=out/'restart';shutil.copytree(run,restart)
        saved_rng=torch.get_rng_state().clone();other,raw=load_checkpoint(restart/checkpoint.name);torch.set_rng_state(saved_rng)
        _,report=advance(model,opt,c,p,run,3,pool);reports.append(report)
        expected=copy.deepcopy((model.state_dict(),opt.state_dict(),p.optimizer.state_dict(),torch.get_rng_state(),p.state,p.critic_state,p.beta,p.trips,p.local_scoring.payload(3)))
        other_opt=torch.optim.AdamW(other.parameters());resume_optimizer(raw,other_opt)
        q=TrainingPhases(other,c,raw=raw,generation=2,root=SOURCE_ROOT,run=restart)
        q.local_scoring=Companion(other,AUX,raw=raw,generation=2)
        _,resumed=advance(other,other_opt,c,q,restart,3,pool)
        actual=(other.state_dict(),other_opt.state_dict(),q.optimizer.state_dict(),torch.get_rng_state(),q.state,q.critic_state,q.beta,q.trips,q.local_scoring.payload(3))
        require(_same_payload(expected,actual),'Restart changed model, Adam, RNG, beta, gates or auxiliary state')
        require(sha(checkpoint)==checksum,'Incoming checkpoint changed')
        require([x['generation'] for x in p.buffer.receipts]==[2,3],'Expired replay was retained')
        require(not p.buffer.path(1,'critic.npy').exists(),'Expired cache not deleted after commit')
    require(source==implementation_id(),'Source changed during smoke')
    result=dict(passed=True,seconds=time.perf_counter()-started,source=source,exact_restart=True,
        device='cpu',cuda_available=torch.cuda.is_available(),parameters=model.parameter_count,iterations=reports,
        scope='Three authored small-fixture iterations, every-iteration auxiliary training, two dense ages, four preparation workers, ownership heads; third iteration repeated after restart. No production run changed.')
    write_json(out/'report.json',result)
    print(dict(passed=True,seconds=result['seconds'],steps=[r['value']['optimizer_steps'] for r in reports],
        beta=[r['value']['beta_clone_next'] for r in reports],exact_restart=True),flush=True)


if __name__=='__main__':main()
