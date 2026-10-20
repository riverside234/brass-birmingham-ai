"""Isolated current-size PPO-only compute and exact next-update resume smoke."""
import argparse
import copy
from pathlib import Path
import time

import torch

from brass_ai.bank import sha
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,require,project_path
from brass_ai.recovery import value_phase
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json,implementation_id
from test_gae_value_phase import fixture,examples,ppo,finish
from test_ppg_window import upgraded


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    args=p.parse_args();out=project_path(SOURCE_ROOT,args.output)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts output')
    out.mkdir(parents=True);torch.set_num_threads(2);began=time.perf_counter()
    model,opt,cfg,(behavior,data)=fixture(preset='medium',games=8,decisions=4)
    cfg=value_phase.disabled_options(upgraded(cfg))
    cfg['update'].update(effective_batch_size=128,critic_batch_size=64,microbatch_size=16)
    phases=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=out)
    start=time.perf_counter();first=ppo(model,opt,cfg,behavior,data,phases);ppo_seconds=time.perf_counter()-start
    weights=copy.deepcopy(model.state_dict());adam=copy.deepcopy(opt.state_dict())
    value_adam=copy.deepcopy(phases.optimizer.state_dict());rng=torch.get_rng_state().clone()
    value=phases.fit(model)
    require(_same_payload(weights,model.state_dict()) and _same_payload(adam,opt.state_dict())
        and _same_payload(value_adam,phases.optimizer.state_dict()) and torch.equal(rng,torch.get_rng_state()),
        'Disabled PPG diagnostics mutated training state')
    finish(phases,first,1)
    path=out/'generation-0001.pt'
    save_checkpoint(path,model,opt,step=1,training={KEY:phases.payload()})
    write_json(out/'state.json',dict(iteration=1,signature=dict(config=cfg)))
    phases.committed(1)
    before=sha(path);restored,raw=load_checkpoint(path)
    other=torch.optim.AdamW(restored.parameters());resume_optimizer(raw,other)
    resumed=TrainingPhases(restored,cfg,raw=raw,generation=1,root=SOURCE_ROOT,run=out)
    behavior,data=examples(model,games=4,decisions=3,prefix='next-')
    rng=torch.get_rng_state().clone();results=[]
    for net,optimizer,phase in ((model,opt,phases),(restored,other,resumed)):
        torch.set_rng_state(rng);report=ppo(net,optimizer,cfg,behavior,data,phase)
        value_report=phase.fit(net);finish(phase,report,2)
        results.append((report,value_report,torch.get_rng_state().clone()))
    require(_same_payload(model.state_dict(),restored.state_dict())
        and _same_payload(opt.state_dict(),other.state_dict())
        and _same_payload(phases.payload(),resumed.payload())
        and torch.equal(results[0][2],results[1][2]),'Resumed next PPO update differs')
    require(sha(path)==before and not (out/'.ppg-value-cache').exists(),'Checkpoint or disabled cache changed')
    result=dict(passed=True,implementation=implementation_id(),model=model.config.to_dict(),
        parameters=model.parameter_count,device='cpu',threads=2,authored_games=8,rows=128,
        ppo_seconds=ppo_seconds,actor=first,value_phase=value,seconds=time.perf_counter()-began,
        ppg_steps=0,disabled_diagnostics_weights_adams_rng_unchanged=True,
        resumed_next_update_bitwise_equal=True,checkpoint_unchanged=True,no_ppg_disk_cache=True,
        scope='Authored Features-4 tensors, 1-4 legal candidates; current medium model, real PPO/critic diagnostics/checkpoint/resume. No native collection, live checkpoints or playing-strength evidence.')
    write_json(out/'report.json',result)
    print({k:result[k] for k in ('passed','parameters','ppo_seconds','seconds','ppg_steps','resumed_next_update_bitwise_equal')})


if __name__=='__main__':main()
