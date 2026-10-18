"""Isolated medium-v11.1 compute smoke, optional two-game trainer/resume smoke."""
import argparse
import copy
from pathlib import Path
import time

import torch

from test_gae_value_phase import fixture,examples,ppo,finish
from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import latest_model_config,validate_config
from brass_ai.recovery.demonstrations import completed_imitation
from brass_ai.recovery.league import train
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import implementation_id,write_json


def compute(out):
    began=time.perf_counter();torch.set_num_threads(2)
    model,opt,cfg,(behavior,data)=fixture(preset='medium',games=8,decisions=4)
    cfg['update'].update(effective_batch_size=128,critic_batch_size=64,microbatch_size=16)
    cfg['value_phase'].update(max_epochs=3,batch_size=128,clone_rows=128,clone_batch_size=32)
    phases=TrainingPhases(model,cfg);start=time.perf_counter()
    first=ppo(model,opt,cfg,behavior,data,phases);ppo_seconds=time.perf_counter()-start
    ppo_state=copy.deepcopy(opt.state_dict());value=phases.fit(model);finish(phases,first,1)
    require(_same_payload(ppo_state,opt.state_dict()),'Value phase modified PPO optimizer/controller/warmup')
    path=out/'generation-0001.pt';save_checkpoint(path,model,opt,step=1,training={KEY:phases.payload()})
    before=sha(path);restored,raw=load_checkpoint(path);other=torch.optim.AdamW(restored.parameters());resume_optimizer(raw,other)
    resumed=TrainingPhases(restored,cfg,raw=raw,generation=1)
    behavior,data=examples(model,games=4,decisions=3);rng=torch.get_rng_state().clone();results=[]
    for net,adam,phase in ((model,opt,phases),(restored,other,resumed)):
        torch.set_rng_state(rng);r=ppo(net,adam,cfg,behavior,data,phase);v=phase.fit(net);finish(phase,r,2)
        results.append((r,v,torch.get_rng_state().clone()))
    require(_same_payload(model.state_dict(),restored.state_dict()) and _same_payload(opt.state_dict(),other.state_dict())
        and _same_payload(phases.payload(),resumed.payload()) and torch.equal(results[0][2],results[1][2]),
        'Next resumed PPO/value update differs')
    require(sha(path)==before,'Smoke changed its saved checkpoint')
    report=dict(passed=True,implementation=implementation_id(),model=model.config.to_dict(),parameters=model.parameter_count,
        device='cpu',threads=2,authored_games=8,rows=128,ppo_seconds=ppo_seconds,value_phase=value,
        actor=first,resumed_next_update_bitwise_equal=True,ppo_optimizer_unchanged_by_value=True,
        checkpoint_bytes=path.stat().st_size,seconds=time.perf_counter()-began,
        scope='Authored Features-4 tensors, 1-4 legal candidates, medium v11.1, CPU; full PPO/value/checkpoint computation only. Not native-game or production-throughput/strength evidence.')
    write_json(out/'report.json',report);print('Compute smoke:',out/'report.json',flush=True)


def native(out,imitation,donor):
    _,origin=completed_imitation(SOURCE_ROOT,imitation)
    cfg=latest_model_config(read(SOURCE_ROOT/'configs/step5/joint7-latest.json'),origin)
    cfg.update(games_per_iteration=1,threads=2,audit_workers=1,profile_steps=0,seed=18570000,
        rehearsal_manifest=origin['manifest'])
    cfg['execution'].update(processes=1,hosts_per_process=2,heavy_processes=1,worker_threads=1,
        batch_size=64,max_candidate_cells=131072,shared_slot_bytes=2097152)
    cfg['update'].update(effective_batch_size=512,critic_batch_size=64,microbatch_size=16,max_candidate_cells=8192,epochs=1)
    cfg['value_phase'].update(max_epochs=1,batch_size=128,clone_rows=64,clone_batch_size=16)
    cfg['rehearsal_sampling'].update(rows_per_iteration=64,shard_draws=2,max_cache_bytes=64*1024**2)
    cfg['retention']['enabled']=False
    # Ratings stay enabled for the gate contract; none are due before iteration
    # ten. Never call extra evaluation just to advance this diagnostic gate.
    validate_config(cfg);source=implementation_id()
    donor=project_path(SOURCE_ROOT,donor);state=read(donor/'state.json')
    probe=read(donor/'il-kl/manifest.json')
    protected=[donor/'state.json',donor/'il-kl/manifest.json',
        project_path(SOURCE_ROOT,probe['cache']['path']),
        project_path(SOURCE_ROOT,state['history'][-1]['path']),project_path(SOURCE_ROOT,origin['path']),
        project_path(SOURCE_ROOT,origin['manifest'])]
    hashes={str(p):sha(p) for p in protected};write_json(out/'config.json',cfg);began=time.perf_counter()
    train(SOURCE_ROOT,out/'run',cfg,iterations=1,device='cpu',imitation_run=imitation,fresh=True,il_kl_probe_from_run=donor)
    first=read(out/'run/state.json')['history'][-1]
    # Exercise recovery of this smoke's disposable cache, not any donor data.
    cache=(out/'run/.ppg-value-cache/generation-000001.npy').resolve()
    require(cache.is_relative_to((out/'run').resolve()) and cache.is_file(),'Missing owned PPG smoke cache')
    first_cache_sha=sha(cache);cache.unlink()
    train(SOURCE_ROOT,out/'run',cfg,iterations=1,device='cpu',imitation_run=imitation,expected_generation=1)
    state=read(out/'run/state.json');reports=[read(out/'run'/f'iteration-{i:04d}'/'report.json') for i in range(2)]
    require(state['iteration']==2 and state['games_done']==2,'Native smoke progress differs')
    require(not state['ratings']['slots'] and not state['evaluations'],'Unexpected evaluation in bounded smoke')
    require(sha(project_path(SOURCE_ROOT,first['path']))==first['sha256'],'Previous checkpoint changed')
    require(hashes=={p:sha(Path(p)) for p in hashes},'Donor or completed IL changed')
    require(source==implementation_id(),'Implementation changed during smoke')
    _,raw=load_checkpoint(project_path(SOURCE_ROOT,state['history'][-1]['path']))
    require(raw['training'][KEY]['gae']['generation']==2,'GAE state not checkpointed')
    buffer=reports[1]['value_phase']['buffer']
    require(buffer['rebuilt_from_replay'] and buffer['previous_rows']==reports[0]['value_phase']['current_train_rows'],
            'Previous cache was not rebuilt and admitted exactly')
    require(not cache.exists() and (cache.parent/'generation-000002.npy').is_file(),
            'Expired cache survived commit or next iteration cache was deleted')
    write_json(out/'report.json',dict(passed=True,implementation=source,seconds=time.perf_counter()-began,
        iterations=2,games=2,source_files_unchanged=True,previous_checkpoint_unchanged=True,
        no_evaluation_games=True,updates=[dict(actor=r['actor'],value_phase=r['value_phase'],timings=r['timings']) for r in reports],
        deleted_cache_rebuilt_exactly=True,rebuilt_cache_sha256=first_cache_sha,expired_cache_deleted_after_commit=True,
        scope='Two one-game CPU iterations through the real trainer, fresh diagnostic IL then same-run resume. 5% hash holdout may be empty; value training then skips explicitly. No production checkpoints or strength campaign.'))
    print('Native smoke:',out/'report.json',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--native',action='store_true');p.add_argument('--imitation-run');p.add_argument('--probe-from-run')
    a=p.parse_args();out=project_path(SOURCE_ROOT,a.output)
    torch.set_num_threads(2)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts output')
    out.mkdir(parents=True)
    if a.native:
        require(a.imitation_run and a.probe_from_run,'Native smoke needs completed diagnostic IL and probe donor')
        native(out,a.imitation_run,a.probe_from_run)
    else:compute(out)


if __name__=='__main__':main()
