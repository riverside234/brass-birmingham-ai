"""Small current-size fixed-snapshot PPO smoke with a serialized next-update restart."""
import argparse
import copy
import time

import torch

from brass_ai.bank import sha
from brass_ai.learning.checkpoint import load_checkpoint,resume_optimizer,save_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery import critic_data as C,snapshot_r2 as R,value_phase
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import insolvency_010_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.utility import migrate,reward_spec
from brass_ai.step5.league import write_json,implementation_id
from test_critic_descriptors import mixed_fixture,partition
from test_gae_value_phase import examples,ppo,finish
from test_ppg_window import upgraded


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    out=project_path(SOURCE_ROOT,args.output)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts directory')
    out.mkdir(parents=True);torch.set_num_threads(2);began=time.perf_counter()
    model,opt,cfg,behavior,data,snap,_=mixed_fixture(preset='medium')
    cfg=C.floor_config(value_phase.disabled_options(upgraded(cfg)))
    cfg['update'].update(effective_batch_size=128,critic_batch_size=64,microbatch_size=16)
    old=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=out)
    old.critic_state.update(snapshot_weight=0.,weight_recovery=R.initial(0,0.))
    old.snapshot_data=snap
    initial=ppo(model,opt,cfg,behavior,data,old);old.fit(model);finish(old,initial,1)
    path=out/'old-zero-weight.pt';save_checkpoint(path,model,opt,step=1,training={KEY:old.payload()})
    old_hash=sha(path);cfg=C.fixed_config(insolvency_010_config(cfg))
    model,raw=load_checkpoint(path);opt=torch.optim.AdamW(model.parameters());resume_optimizer(raw,opt)
    weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
    model._reviewed_cap_upgrade=True;migrate(model,opt,reward_spec(cfg))
    phase=TrainingPhases(model,cfg,raw=raw,generation=1,root=SOURCE_ROOT,run=out)
    require(_same_payload(weights,model.state_dict()) and _same_payload(adam,opt.state_dict()) and
        torch.equal(rng,torch.get_rng_state()),'Migration changed weights, PPO Adam or RNG')
    behavior,all_rows=examples(model,games=9,decisions=4,prefix='new-cost-')
    data,snap,_=partition(all_rows,'new-fixed-');phase.snapshot_data=snap
    first=ppo(model,opt,cfg,behavior,data,phase);value=phase.fit(model);finish(phase,first,2)
    require(first['critic_rows']['snapshot_distinct_used']>0 and first['critic_rows']['snapshot_weight']==1.,
        'Fixed weight did not train snapshot rows immediately')
    path2=out/'fixed-weight.pt';save_checkpoint(path2,model,opt,step=2,training={KEY:phase.payload()})
    frozen=sha(path2);restored,raw2=load_checkpoint(path2);other=torch.optim.AdamW(restored.parameters());resume_optimizer(raw2,other)
    resumed=TrainingPhases(restored,cfg,raw=raw2,generation=2,root=SOURCE_ROOT,run=out)
    behavior,all_rows=examples(model,games=9,decisions=4,prefix='next-cost-')
    data,snap,_=partition(all_rows,'next-fixed-');rng=torch.get_rng_state().clone();results=[]
    for net,optimizer,phases in ((model,opt,phase),(restored,other,resumed)):
        torch.set_rng_state(rng);phases.snapshot_data=snap
        update=ppo(net,optimizer,cfg,behavior,data,phases);phases.fit(net);finish(phases,update,3)
        results.append((update,torch.get_rng_state().clone()))
    require(_same_payload(model.state_dict(),restored.state_dict()) and _same_payload(opt.state_dict(),other.state_dict())
        and _same_payload(phase.payload(),resumed.payload()) and torch.equal(results[0][1],results[1][1]),
        'Serialized next update differs')
    require(sha(path)==old_hash and sha(path2)==frozen,'Original checkpoint changed')
    result=dict(passed=True,implementation=implementation_id(),parameters=model.parameter_count,device='cpu',threads=2,
        seconds=time.perf_counter()-began,learner_rows=len(data),snapshot_rows=len(snap),
        first_fixed_ppo=first,value_phase=value,resumed_next_update_bitwise_equal=True,checkpoint_bytes_unchanged=True,
        scope='Authored mixed-table tensors, medium v11.2 model, real PPO/Adam/checkpoint operations. No native games, GPU benchmark or strength evidence.')
    write_json(out/'report.json',result)
    print({k:v for k,v in result.items() if k not in ('first_fixed_ppo','value_phase')})


if __name__=='__main__':main()
