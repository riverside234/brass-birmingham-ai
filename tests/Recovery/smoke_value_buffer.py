"""Bounded medium-v11.1 two-iteration critic/cache/continuation smoke."""
import argparse
import copy
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import torch

from test_gae_value_phase import fixture,examples,ppo,finish
from test_value_buffer import source,commit
from brass_ai.bank import sha
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.value_buffer import row_dtype,ValueDataset
from brass_ai.step5.league import implementation_id,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    out=project_path(SOURCE_ROOT,a.output)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts output')
    out.mkdir(parents=True);torch.set_num_threads(2);began=time.perf_counter();implementation=implementation_id()
    model,opt,cfg,(behavior,data)=fixture(preset='medium',games=8,decisions=4)
    cfg['update'].update(effective_batch_size=128,critic_batch_size=64,microbatch_size=16)
    cfg['value_phase'].update(buffer_iterations=2,max_epochs=3,batch_size=128,clone_rows=128,clone_batch_size=32)
    run=out/'run';phases=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run);phases.collection_source=source(behavior)
    started=time.perf_counter();actor=ppo(model,opt,cfg,behavior,data,phases);seconds=time.perf_counter()-started
    state=copy.deepcopy(opt.state_dict());value=phases.fit(model);finish(phases,actor,1)
    require(_same_payload(state,opt.state_dict()),'Value phase modified PPO optimizer')
    path=run/'generation-0001.pt';save_checkpoint(path,model,opt,step=1,training={KEY:phases.payload()})
    checksum=sha(path);commit(run,phases,1);phases.release()
    first=dict(actor=actor,value_phase=value,ppo_seconds=seconds)
    other_run=out/'resumed';shutil.copytree(run,other_run)
    restored,raw=load_checkpoint(other_run/path.name);other=torch.optim.AdamW(restored.parameters());resume_optimizer(raw,other)
    resumed=TrainingPhases(restored,cfg,raw=raw,generation=1,root=SOURCE_ROOT,run=other_run)
    behavior,data=examples(model,games=8,decisions=4,prefix='second-');rng=torch.get_rng_state().clone();results=[]
    for net,adam,phase in ((model,opt,phases),(restored,other,resumed)):
        torch.set_rng_state(rng);phase.collection_source=source(behavior);started=time.perf_counter()
        r=ppo(net,adam,cfg,behavior,data,phase);seconds=time.perf_counter()-started
        state=copy.deepcopy(adam.state_dict());processed=[];real=ValueDataset.value_batches
        def batches(ds,parts,*args,**kwargs):
            for part,ix,inputs in real(ds,parts,*args,**kwargs):
                if torch.is_grad_enabled():processed.extend(i for i in part if i>=len(data))
                yield part,ix,inputs
        with patch.object(ValueDataset,'value_batches',batches):v=phase.fit(net)
        finish(phase,r,2)
        require(_same_payload(state,adam.state_dict()),'Value phase changed PPO optimizer')
        require(processed and v['previous_train_rows']==112,'Previous rows not processed by value backward pass')
        results.append(dict(actor=r,value_phase=v,ppo_seconds=seconds,rng=torch.get_rng_state().clone(),
                            previous_rows_processed_with_grad=len(set(processed)),previous_attempted_presentations=len(processed)))
    require(_same_payload(model.state_dict(),restored.state_dict()) and _same_payload(opt.state_dict(),other.state_dict())
        and _same_payload(phases.payload(),resumed.payload()) and torch.equal(results[0]['rng'],results[1]['rng']),
        'Resumed buffered update differs')
    for actual,phase,net,adam in ((run,phases,model,opt),(other_run,resumed,restored,other)):
        save_checkpoint(actual/'generation-0002.pt',net,adam,step=2,training={KEY:phase.payload()});commit(actual,phase,2)
        require(not phase.buffer.path(1,'npy').exists() and phase.buffer.path(2,'npy').exists(),'Cache rotation failed')
    require(sha(path)==checksum and implementation_id()==implementation,'Saved checkpoint or implementation changed')
    results[0].pop('rng');results[1].pop('rng')
    write_json(out/'report.json',dict(passed=True,implementation=implementation,parameters=model.parameter_count,
        model=model.config.to_dict(),device='cpu',threads=2,iterations=2,authored_games_per_iteration=8,
        rows_per_iteration=128,critic_row_bytes=row_dtype().itemsize,updates=[first,results[0]],
        resumed_next_update_bitwise_equal=True,ppo_optimizer_unchanged_by_value=True,
        expired_cache_deleted_after_commit=True,previous_checkpoint_unchanged=True,seconds=time.perf_counter()-began,
        scope='Authored Features-4 tensors with 1-4 legal actions, medium v11.1; disk-backed critic reuse and exact continuation. Not native gameplay, production throughput or playing-strength evidence.'))
    print('Buffered compute smoke:',out/'report.json',flush=True)


if __name__=='__main__':main()
