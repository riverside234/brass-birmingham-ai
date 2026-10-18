"""Bounded v11.2 mixed-row compute smoke or two native fresh/resume iterations."""
import argparse
import copy
from pathlib import Path
import time
import torch

from test_critic_descriptors import mixed_fixture,partition
from test_gae_value_phase import examples,ppo,finish
from test_value_buffer import source,commit
from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint,resume_optimizer
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.config import latest_model_config,validate_config
from brass_ai.recovery.demonstrations import completed_imitation
from brass_ai.recovery.retention import _same_payload
from brass_ai.step5.league import write_json,implementation_id


def compute(out):
    began=time.perf_counter();implementation=implementation_id()
    model,opt,cfg,b,data,snap,excluded=mixed_fixture(preset='medium')
    cfg['value_phase'].update(buffer_iterations=2,max_epochs=2,batch_size=128,clone_batch_size=16)
    cfg['update'].update(effective_batch_size=128,critic_batch_size=64,microbatch_size=16)
    run=out/'run';phase=TrainingPhases(model,cfg,root=SOURCE_ROOT,run=run)
    phase.snapshot_data=snap;phase.collection_source=dict(source(b),snapshot_rows=True)
    t=time.perf_counter();r=ppo(model,opt,cfg,b,data,phase);seconds=time.perf_counter()-t
    ppo_adam=copy.deepcopy(opt.state_dict());v=phase.fit(model);finish(phase,r,1)
    require(_same_payload(ppo_adam,opt.state_dict()),'Value phase changed PPO Adam')
    cp=run/'generation-1.pt';save_checkpoint(cp,model,opt,step=1,training={KEY:phase.payload()});commit(run,phase,1)
    first=dict(actor=r,value_phase=v,ppo_seconds=seconds);checkpoint_sha=sha(cp)
    import shutil
    restored_run=out/'resumed';shutil.copytree(run,restored_run)
    restored,raw=load_checkpoint(cp);adam=torch.optim.AdamW(restored.parameters());resume_optimizer(raw,adam)
    resumed=TrainingPhases(restored,cfg,raw=raw,generation=1,root=SOURCE_ROOT,run=restored_run)
    b,rows=examples(model,games=9,decisions=4,prefix='second-');data,snap,excluded=partition(rows,'second-')
    rng=torch.get_rng_state().clone();results=[]
    for net,optimizer,p in ((model,opt,phase),(restored,adam,resumed)):
        torch.set_rng_state(rng);p.snapshot_data=snap;p.collection_source=dict(source(b),snapshot_rows=True)
        t=time.perf_counter();r=ppo(net,optimizer,cfg,b,data,p);elapsed=time.perf_counter()-t
        v=p.fit(net);finish(p,r,2);results.append(dict(actor=r,value_phase=v,ppo_seconds=elapsed))
        commit(p.buffer.run,p,2)
    require(_same_payload(model.state_dict(),restored.state_dict()) and _same_payload(opt.state_dict(),adam.state_dict())
        and _same_payload(phase.optimizer.state_dict(),resumed.optimizer.state_dict()) and phase.state==resumed.state
        and phase.critic_state==resumed.critic_state,'Mixed-row continuation differs')
    require(sha(cp)==checkpoint_sha and not phase.buffer.path(1,'npy').exists(),'Checkpoint/cache lifecycle failed')
    require(implementation==implementation_id(),'Source changed during smoke')
    write_json(out/'report.json',dict(passed=True,implementation=implementation,parameters=model.parameter_count,
        rows_per_collection=dict(learner=len(data),snapshot=len(snap),excluded=len(excluded)),iterations=[first,results[0]],
        exact_resumed_next_update=True,expired_cache_deleted=True,checkpoint_unchanged=True,
        seconds=time.perf_counter()-began,
        scope='Authored Features-4 mixed tables, medium v11.2 CPU/two threads. Snapshot mode forced on to exercise rows; production starts with auto gate. No strength evidence.'))


def native(out,imitation,workers=2):
    from brass_ai.recovery.league import train
    _,origin=completed_imitation(SOURCE_ROOT,imitation)
    cfg=latest_model_config(read(SOURCE_ROOT/'configs/step5/joint7-latest.json'),origin)
    cfg.update(games_per_iteration=1,threads=2,audit_workers=workers,profile_steps=0,seed=18570000,rehearsal_manifest=origin['manifest'])
    cfg['execution'].update(processes=1,hosts_per_process=2,heavy_processes=1,worker_threads=1,
        batch_size=64,max_candidate_cells=131072,shared_slot_bytes=2097152)
    cfg['update'].update(effective_batch_size=512,critic_batch_size=64,microbatch_size=16,max_candidate_cells=8192,epochs=1)
    cfg['value_phase'].update(max_epochs=1,batch_size=128,clone_rows=64,clone_batch_size=16)
    cfg['rehearsal_sampling'].update(rows_per_iteration=64,shard_draws=2,max_cache_bytes=64*1024**2)
    cfg['retention']['enabled']=False
    validate_config(cfg);implementation=implementation_id();write_json(out/'config.json',cfg)
    protected={str(project_path(SOURCE_ROOT,origin[k])):sha(project_path(SOURCE_ROOT,origin[k])) for k in ('path','manifest')}
    began=time.perf_counter();run=out/'run'
    train(SOURCE_ROOT,run,cfg,iterations=1,device='cpu',imitation_run=imitation,fresh=True)
    first=read(run/'state.json')['history'][-1];cache=run/'.ppg-value-cache/generation-000001.npy'
    require(cache.resolve().is_relative_to(out.resolve()),'Unsafe diagnostic cache path');cache.unlink()
    train(SOURCE_ROOT,run,cfg,iterations=1,device='cpu',imitation_run=imitation,expected_generation=1)
    state=read(run/'state.json');reports=[read(run/f'iteration-{i:04d}/report.json') for i in range(2)]
    require(state['iteration']==2 and state['games_done']==2 and not state['ratings']['slots'],'Unexpected native smoke progress/evaluation')
    require(sha(project_path(SOURCE_ROOT,first['path']))==first['sha256'],'Native smoke changed committed checkpoint')
    require(protected=={p:sha(Path(p)) for p in protected},'Completed IL or dataset changed')
    require(reports[1]['value_phase']['buffer']['rebuilt_from_replay'] and not cache.exists(),'Cache rebuild/cleanup failed')
    for report in reports:
        sampling=report['actor']['rehearsal_sampling']
        require(sampling['workers_requested']==workers and sampling['reused_reader_pool']==(workers>1)
                and sampling['cache_deleted'],'Rehearsal reader pool/cleanup integration differs')
        timings=report['timings']
        require(all(timings[k]>=0 for k in ('rehearsal_prepare_seconds','ppo_seconds','value_phase_seconds',
                'search_and_distill_seconds','post_ppo_diagnostics_seconds','final_diagnostics_seconds')),
                'Missing/invalid separate iteration timings')
        if not report['value_phase']['optimizer_steps']:
            final=report['actor']['final_categorical_probe'];initial=report['actor']['categorical_probe']
            require(final['reused_post_ppo'] and all(final[k]==v for k,v in initial.items() if k!='scope'),
                    'Unchanged post-PPO metrics were not reused exactly')
    require(implementation==implementation_id(),'Source changed during native smoke')
    write_json(out/'report.json',dict(passed=True,implementation=implementation,seconds=time.perf_counter()-began,
        iterations=2,games=2,source_files_unchanged=True,cache_rebuilt_and_expired_deleted=True,
        updates=[dict(actor=r['actor'],value_phase=r['value_phase'],fingerprints=r['seat_fingerprints'],composition=r['composition'],timings=r['timings']) for r in reports],
        scope='Real trainer and native games, fresh upgraded diagnostic IL then resume. One-game hash holdouts can be empty; gates must remain closed without evidence. No evaluation campaign.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--native',action='store_true');p.add_argument('--imitation-run');p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();out=project_path(SOURCE_ROOT,a.output);torch.set_num_threads(2)
    require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use an unused isolated artifacts directory')
    out.mkdir(parents=True)
    if a.native:native(out,a.imitation_run,a.workers)
    else:compute(out)
