"""Isolated qualified-anchor hotfix, real 2,048-row cache and PPO/resume smoke."""
import argparse
import copy
import importlib.util
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest,project_path,require
from brass_ai.recovery import policy_anchor,il_kl_probe,exploration,updates,ratings
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.league import Behavior
from brass_ai.recovery.learning_rate import KEY as WARMUP
from brass_ai.recovery.lr_controller import KEY as LR
from brass_ai.recovery.resume import load_resume,apply_optimizer_config
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.telemetry import Timings
from brass_ai.step5.league import write_json,implementation_id
from test_auto_il_kl import ProbeTests,Rows
from test_gae_value_phase import fixture,examples,ppo
from test_policy_anchor import evaluated,install_fit


def behavior_rows(model,cfg,*,prefix):
    _,data=examples(model,games=8,decisions=4,prefix=prefix)
    behavior=Behavior(model_id(model),cfg['exploration'],dict(version=3))
    for row,_,_,_ in data:
        row['behavior'].update(policy_id=behavior.policy_id,exploration=.02,exploration_version=3,
            temperature=1.,component='policy',action_decision=5 if row['view']['context']['phase']=='action' else None)
    ds=updates.Dataset(data,'cpu');model.eval()
    with torch.no_grad():
        for part in ds.microbatches(range(len(data)),16,131072):
            ix,inputs,q=ds.batch(part,.02);p=updates.row_probabilities(ds,ix,model(*inputs)[0],q,behavior)
            require(torch.allclose(p.sum(-1),torch.ones(len(part))) and bool((p[~inputs[2]]==0).all()),'Invalid epsilon mixture')
            for j,i in enumerate(part):data[i][0]['behavior']['probability']=float(p[j,data[i][0]['chosen']])
    return behavior,data


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=project_path(SOURCE_ROOT,args.output)
    require(root.is_relative_to(SOURCE_ROOT/'artifacts') and not root.exists(),'Use a new isolated artifacts directory')
    root.mkdir(parents=True);run=root/'artifacts/step5/smoke';run.mkdir(parents=True)
    torch.set_num_threads(2);began=time.perf_counter()
    model,opt,_,_=fixture(preset='medium',games=8,decisions=4)
    il=copy.deepcopy(model)
    state,cfg,names,theta,draws=evaluated(root)
    cfg['threads']=2;cfg['update'].update(epochs=1,effective_batch_size=128,critic_batch_size=64,microbatch_size=16)
    validate_config(cfg)
    opt.param_groups[0][WARMUP]['completed_steps']=200
    opt.param_groups[0][LR]['multiplier']=1.5625
    behavior,data=behavior_rows(model,cfg,prefix='first-')
    first=ppo(model,opt,cfg,behavior,data)
    reference=copy.deepcopy(model)
    # Real cached tensors and raw original-IL probabilities. These authored
    # rows exercise the implementation, not a native validation population.
    base=ProbeTests().data();rows=Rows()
    for i in range(2048):
        row,targets,_,encoded=base[i%len(base)]
        rows.append((copy.deepcopy(row),targets,('authored-probe-'+str(i),i%4),encoded))
    source=root/'validation.jsonl.gz';source.write_bytes(b'authored validation loader fixture')
    manifest=root/'manifest.json';write_json(manifest,dict(entries=[dict(split='validation',path=source.name,sha256=sha(source))]))
    il_path=run/'il.pt';save_checkpoint(il_path,il,None,step=0)
    origin=dict(bc_run='artifacts/il',path=il_path.relative_to(root).as_posix(),manifest=manifest.name,
        manifest_sha256=sha(manifest),model_id=model_id(il),sha256=sha(il_path))
    with patch('brass_ai.recovery.demonstrations.load_dataset',return_value=(rows,{})):
        probe,probe_receipt=il_kl_probe.load(root,run,origin,il,cfg['il_kl_limit'],cfg['update'],1)
    state['il_kl_limit']['probe']=probe_receipt
    # Highest qualified checkpoint is g20, whereas the current learner is g30.
    theta[-1]=1.9;draws[:,-1]-=1.1
    signature=dict(config=cfg,implementation=implementation_id(),compatibility=compatibility(),imitation_source=origin)
    state.update(format='step5-recovery-league/2.2.0',signature=signature,
        next_seed=12345,context_counter=86400,population=[],
        exploration=exploration.initial(cfg,0),preservation=dict(gae=.95,snapshot_r2=.4,snapshot_weight=1.))
    for gen,net in ((0,il),(10,il),(20,reference)):
        name=ratings.IL if gen==0 else f'generation-{gen:06d}'
        path=run/f'frozen-{gen:06d}.pt';save_checkpoint(path,net,None,step=gen)
        state['ratings']['agents'][name]['checkpoint']=dict(ratings.receipt(root,path),generation=gen,model_id=model_id(net))
    state['evaluation_anchor']=copy.deepcopy(state['ratings']['agents'][ratings.IL]['checkpoint'])
    current=run/'generation-0030.pt'
    save_checkpoint(current,model,opt,step=30,training=dict(fresh_run=run.relative_to(root).as_posix(),
        signature=digest(signature),il_kl_limit=state['il_kl_limit'],preservation=state['preservation']))
    cp=dict(ratings.receipt(root,current),generation=30,model_id=model_id(model))
    state['history']=[cp];state['ratings']['agents'][names[-1]]['checkpoint']=cp
    install_fit(root,state,names,theta,draws)
    write_json(run/'state.json',state)
    before=copy.deepcopy(state);checkpoint_hash=sha(current);weights=copy.deepcopy(model.state_dict())
    adam=copy.deepcopy(opt.state_dict());rng=torch.get_rng_state().clone()
    spec=importlib.util.spec_from_file_location('anchor_hotfix',SOURCE_ROOT/'scripts/joint7-qualified-anchor.py')
    hotfix=importlib.util.module_from_spec(spec);spec.loader.exec_module(hotfix)
    # Only completed-IL lineage discovery is substituted for this authored run.
    # The actual checkpoint loader, probe loader, hotfix and cache all execute.
    with patch('brass_ai.recovery.demonstrations.completed_imitation',return_value=(il,origin)):
        result=hotfix.run(root,run.relative_to(root).as_posix(),'configs/step5/smoke.json',apply=True,expected_generation=30)
        payload=(run/'state.json').read_bytes()
        again=hotfix.run(root,run.relative_to(root).as_posix(),'configs/step5/smoke.json',apply=True,expected_generation=30)
    state=read(run/'state.json')
    require(payload==(run/'state.json').read_bytes() and again['transition']['decision']=='hold','Hotfix was not idempotent')
    require(result['active_reference']['checkpoint']['generation']==20,'Wrong qualified reference')
    require(sha(current)==checkpoint_hash and _same_payload(weights,model.state_dict()) and
        _same_payload(adam,opt.state_dict()) and torch.equal(rng,torch.get_rng_state()),'Hotfix mutated learner/Adam/RNG')
    for key in before:
        if key not in ('ratings','il_kl_limit'):require(state[key]==before[key],'Hotfix changed '+key)
    for key in ('baseline','baseline_rows','rolling','probe','last_generation'):
        require(state['il_kl_limit'][key]==before['il_kl_limit'][key],'Rebased independent health')
    restored,raw,_=load_resume(root,run,state,signature,reviewed=True)
    policy_anchor.verify_checkpoint(state,raw)
    other=torch.optim.AdamW(restored.parameters());resume_optimizer(raw,other);apply_optimizer_config(other,cfg)
    require(other.param_groups[0][LR]['multiplier']==1.5625 and
        other.param_groups[0][WARMUP]==opt.param_groups[0][WARMUP],'LR state reset')
    active=state['policy_anchor']['active'];probabilities=policy_anchor.values(root,active,probe)
    with patch.object(policy_anchor,'load_checkpoint',side_effect=AssertionError('cache was rebuilt')):
        require(policy_anchor.cache(root,run,active,probe,cfg['update'],'cpu')==active['cache'],'Cache not reused')
    behavior,data=behavior_rows(model,cfg,prefix='resumed-');ds=updates.Dataset(data,'cpu')
    mismatch=Behavior(model_id(model),.03,dict(version=3))
    try:updates.policy_measure(model,ds,range(len(data)),mismatch,cfg['update'],Timings(),verify=True)
    except ValueError:pass
    else:raise AssertionError('Admitted incompatible epsilon behavior')
    rng=torch.get_rng_state().clone();reports=[]
    for net,optimizer in ((model,opt),(restored,other)):
        torch.set_rng_state(rng);reports.append(ppo(net,optimizer,cfg,behavior,data))
    require(_same_payload(model.state_dict(),restored.state_dict()) and _same_payload(opt.state_dict(),other.state_dict()),
        'Next resumed PPO update differs')
    dual=probe.measure(model,cfg['update'],reference=probabilities)
    output=dict(passed=True,implementation=implementation_id(),parameters=model.parameter_count,threads=2,
        seconds=time.perf_counter()-began,epsilon=.02,authored_games_per_update=8,rows_per_update=128,
        probe_positions=2048,group_positions={k:v['positions'] for k,v in dual['groups'].items()},
        first_update=first,next_update=reports[0],anchor_transition=result['transition'],final_probe=dual,
        idempotent_hotfix=True,exact_next_ppo_resume=True,checkpoint_unchanged=True,cache_reused=True,
        preserved_health_warmup_lr_rng=True,incompatible_behavior_rejected=True,
        scope='Current medium utility model; authored Features-4 tensors and synthetic paired rating evidence. No native games, live checkpoints or playing-strength claims.')
    write_json(root/'report.json',output)
    print({k:output[k] for k in ('passed','parameters','seconds','probe_positions','exact_next_ppo_resume')})


if __name__=='__main__':main()
