"""Isolated current-size PPG-disable/startup-cleanup and exact PPO restart smoke."""
import argparse
import copy
import time

import torch

from brass_ai.bank import sha
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery import value_phase
from brass_ai.recovery.config import ppo_microbatch_config
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json,implementation_id
from test_ppg_disabled import rolling_checkpoint,caches
from test_ppg_eight import collect
from test_gae_value_phase import ppo,finish


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=project_path(SOURCE_ROOT,args.output)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts directory')
    out.mkdir(parents=True);torch.set_num_threads(2);began=time.perf_counter();source=implementation_id()
    model,opt,old,raw=rolling_checkpoint('medium');cfg=ppo_microbatch_config(value_phase.disabled_options(old))
    original=out/'generation-0460.pt';save_checkpoint(original,model,opt,step=460,training=raw['training'])
    original_sha=sha(original);model,raw=load_checkpoint(original)
    opt=torch.optim.AdamW(model.parameters());resume_optimizer(raw,opt)
    bytes_expected=caches(out);write_json(out/'state.json',dict(iteration=460,signature=dict(config=cfg)))
    phase=TrainingPhases(model,cfg,raw=raw,generation=460,root=SOURCE_ROOT,run=out)
    cleanup=phase.startup(460)
    require(cleanup['deleted_bytes']==bytes_expected,'Startup did not retire the old auxiliary arrays')
    # A crash after source adoption/cleanup, before the next checkpoint, must
    # restore the same original PPG checkpoint without its deleted cache.
    again=TrainingPhases(model,cfg,raw=raw,generation=460,root=SOURCE_ROOT,run=out)
    require(_same_payload(phase.payload(),again.payload()) and not again.startup(460)['deleted_files'],
        'Pre-first-update restart differs after cache retirement')
    behavior,data=collect(model,phase,461,branches=True)
    started=time.perf_counter();actor=ppo(model,opt,cfg,behavior,data,phase);ppo_seconds=time.perf_counter()-started
    snapshot=(copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),
        copy.deepcopy(phase.optimizer.state_dict()),torch.get_rng_state().clone())
    value=phase.fit(model)
    require(_same_payload(snapshot,(model.state_dict(),opt.state_dict(),phase.optimizer.state_dict(),torch.get_rng_state())),
        'Disabled auxiliary diagnostics changed weights, Adam or RNG')
    finish(phase,actor,461)
    output=out/'generation-0461.pt';save_checkpoint(output,model,opt,step=461,training={KEY:phase.payload()})
    write_json(out/'state.json',dict(iteration=461,signature=dict(config=cfg)));phase.committed(461)
    output_sha=sha(output);restored,other_raw=load_checkpoint(output)
    other=torch.optim.AdamW(restored.parameters());resume_optimizer(other_raw,other)
    resumed=TrainingPhases(restored,cfg,raw=other_raw,generation=461,root=SOURCE_ROOT,run=out)
    resumed.startup(461)
    behavior,data=collect(model,phase,462,branches=True);resumed.snapshot_data=phase.snapshot_data
    rng=torch.get_rng_state().clone();results=[]
    for net,optimizer,phases in ((model,opt,phase),(restored,other,resumed)):
        torch.set_rng_state(rng);update=ppo(net,optimizer,cfg,behavior,data,phases)
        v=phases.fit(net);finish(phases,update,462)
        results.append(torch.get_rng_state().clone())
        require(v['optimizer_steps']==v['attempted_steps']==0,'Unexpected PPG work')
    require(_same_payload(model.state_dict(),restored.state_dict())
        and _same_payload(opt.state_dict(),other.state_dict())
        and _same_payload(phase.payload(),resumed.payload()) and torch.equal(*results),
        'Next PPO update differs after resume')
    require(sha(original)==original_sha and sha(output)==output_sha,'Incoming checkpoint changed')
    require(all([p.name for p in (out/name).iterdir()]==['owner.json']
        for name in ('.ppg-value-cache','.ppg-eight-cache')),'Auxiliary cache was recreated')
    require(implementation_id()==source,'Source changed during smoke')
    result=dict(passed=True,implementation=source,parameters=model.parameter_count,device='cpu',
        seconds=time.perf_counter()-began,ppo_seconds=ppo_seconds,ppo_steps=actor['optimizer_steps'],
        ppg_steps=0,cleanup=cleanup,exact_restart=True,incoming_checkpoint_unchanged=True,
        ppo_microbatch_size=cfg['update']['microbatch_size'],
        ppo_max_candidate_cells=cfg['update']['max_candidate_cells'],
        no_auxiliary_cache_recreated=True,value_phase=value,
        scope='Authored 18-game Features-4 fixture, current medium v11.2 model and real PPO; synthetic g460 state, no production weights or full-game evaluation.')
    write_json(out/'report.json',result)
    print({k:result[k] for k in ('passed','parameters','seconds','ppo_seconds','ppo_steps','ppg_steps','exact_restart')})


if __name__=='__main__':main()
