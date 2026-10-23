"""Bounded dense-eight PPO/PPG restart smoke and paired input benchmarks."""
import argparse
import copy
from dataclasses import replace
from pathlib import Path
import shutil
import statistics
import time

import numpy as np
import torch
from brass_ai.bank import sha
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint,resume_optimizer
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery import ppg_packing as K,ppg_replay,updates
from brass_ai.recovery.actor_value import TrainingPhases,KEY
from brass_ai.recovery.local_scoring_auxiliary import Companion,DEFAULT as AUX,KEY as AUX_KEY
from brass_ai.recovery.ppo_preparation import PreparationPool,Preparation
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.utility import reward_spec
from brass_ai.step5.league import write_json,implementation_id
from test_ppg_dense_eight import config
from test_ppg_eight import collect
from test_gae_value_phase import finish
from test_value_buffer import commit
from test_local_scoring import annotate


def advance(model,opt,cfg,phase,run,g,pool):
    behavior,data=collect(model,phase,g,branches=True);annotate(data);annotate(phase.snapshot_data)
    reward=reward_spec(cfg)
    actor=updates.ppo_update(model,opt,[dict(split='train')],behavior,cfg['update'],learner_data=data,
        insolvency=reward['insolvency'],reward=reward,phases=phase,preparation_pool=pool)
    before=copy.deepcopy(opt.state_dict());value=phase.fit(model,preparation_pool=pool)
    require(_same_payload(before,opt.state_dict()),'PPG changed PPO Adam')
    finish(phase,actor,g);checkpoint=run/f'generation-{g:04d}.pt'
    if g in (7,8):save_checkpoint(checkpoint,model,opt,step=g,training={KEY:phase.payload(),AUX_KEY:phase.local_scoring.payload(g)})
    commit(run,phase,g);phase.local_scoring.committed(g)
    return checkpoint,dict(generation=g,ppo_steps=actor['optimizer_steps'],value=value)


def benchmark(out,phase,model,pool):
    out.mkdir();receipt=copy.deepcopy(phase.buffer.receipts[-1]);dense=out/'dense';packed=out/'packed'
    dense.mkdir();packed.mkdir()
    original=np.load(phase.buffer.path(receipt['generation'],'critic.npy'),mmap_mode='r')
    try:array=np.tile(original,32)
    finally:original._mmap.close()
    np.save(dense/'critic.npy',array,allow_pickle=False);receipt['rows']=len(array)
    for target in (dense,packed):shutil.copy2(phase.buffer.path(receipt['generation'],'local-scoring.json'),target/'local-scoring.json')
    layout=K.pack(lambda ext:packed/ext,array,array['action'].copy(),len(receipt['game_ids']))
    packed_receipt=dict(receipt,contract=K.CONTRACT,packing=layout)
    rows=[ppg_replay.Rows(dense/'critic.npy',receipt),ppg_replay.Rows(packed/'critic.npy',packed_receipt)]
    datasets=[ppg_replay.CriticDataset([r],'cpu') for r in rows]
    rng=np.random.default_rng(650);parts=[rng.integers(len(array),size=512).tolist() for _ in range(32)]
    # Identical request order, duplicates, every FP32 input/target/descriptor.
    expected=[]
    for part in parts:
        ds=datasets[1]
        expected.append((ds.batch(part)[:2],tuple(t[part] for t in ds.targets),ds.utility[part],ds.remaining[part],ds.descriptors[part]))
    results={}
    try:
        for name,ds,parallel in [('packed_sync',datasets[1],False),('dense_sync',datasets[0],False),('dense_4_workers',datasets[0],True)]:
            with Preparation(pool,directory=out) as prep:
                durations=[]
                for repeat in range(4):
                    start=time.perf_counter()
                    for j,(part,ix,inputs) in enumerate(ds.value_batches(parts,model,prep if parallel else None)):
                        got=((ix,inputs),tuple(t[ix] for t in ds.targets),ds.utility[ix],ds.remaining[ix],ds.descriptors[ix])
                        if repeat==0:require(_same_payload(expected[j],got),'Prepared benchmark tensors differ')
                    if repeat:durations.append(time.perf_counter()-start)
                results[name]=dict(seconds=durations,median_seconds=statistics.median(durations),
                    preparation=prep.report() if parallel else None)
        # Ownership metadata and label preparation: old row reconstruction vs
        # batched small columns, same labels and masks, no model/GPU work.
        ds=datasets[0];companion=phase.local_scoring;ids=parts[0]
        fast=ds.scoring_metadata
        def slow(chosen):
            from brass_ai.recovery.local_scoring_targets import map_for
            values=[]
            for i in chosen:
                row=ds.rows[i];t=map_for(row);m=ds.encoded[i].stable_to_relative;context=row['view']['context']
                values.append((t,(row['_local_scoring']['envelope']['sha256'],t['era'],int(row['seat']),tuple(m)),
                    (ds.data[i][2][0],context['era'],int(context['revision']),int(row['seat'])),m))
            return values
        for name,fn in [('ownership_row_metadata',slow),('ownership_batch_metadata',fast)]:
            times=[]
            for repeat in range(7):
                start=time.perf_counter();meta=fn(ids)
                for kind,count in [('slots',49),('links',69)]:companion.target_arrays(meta,kind,count)
                if repeat:times.append(time.perf_counter()-start)
            results[name]=dict(seconds=times,median_seconds=statistics.median(times))
        return dict(rows=len(array),batch_size=512,batches=32,measurements=results,
            dense_critic_bytes=(dense/'critic.npy').stat().st_size,packed_array_bytes=layout['packed_array_bytes'],
            bit_exact=True,scope='Warm CPU input preparation, same authored tensors and row indices. Includes IPC/transfer for process arm; workers warmed before timing. Does not measure production GPU throughput or whole-phase speed.')
    finally:
        for r in rows:r.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);args=p.parse_args()
    out=project_path(SOURCE_ROOT,args.output);require(out.is_relative_to(SOURCE_ROOT/'artifacts') and not out.exists(),'Use a new isolated artifacts output')
    out.mkdir(parents=True);torch.set_num_threads(2);start=time.perf_counter();source=implementation_id()
    m,opt,c=config();c['value_phase']['batch_size']=128
    from brass_ai.recovery.config import insolvency_010_config,insolvency_020_config,insolvency_040_cap100_config,vp_cap_config
    c=vp_cap_config(insolvency_040_cap100_config(insolvency_020_config(insolvency_010_config(c))))
    m.config=replace(m.config,utility=reward_spec(c));c[AUX_KEY]=copy.deepcopy(AUX)
    c['update'].update(effective_batch_size=512,critic_batch_size=512,microbatch_size=32,preparation_workers=4)
    run=out/'continuous';phase=TrainingPhases(m,c,root=SOURCE_ROOT,run=run);phase.local_scoring=Companion(m,c[AUX_KEY]);reports=[]
    with PreparationPool(4) as pool:
        for f in pool.ready:f.result()
        for g in range(1,8):
            checkpoint,r=advance(m,opt,c,phase,run,g,pool);reports.append(r)
            require(not r['value']['scheduled'],'PPG ran before eight fresh iterations')
        checksum=sha(checkpoint);restart=out/'restart';shutil.copytree(run,restart)
        saved_rng=torch.get_rng_state().clone();other,raw=load_checkpoint(restart/checkpoint.name);torch.set_rng_state(saved_rng)
        _,r=advance(m,opt,c,phase,run,8,pool);reports.append(r)
        expected=copy.deepcopy((m.state_dict(),opt.state_dict(),phase.optimizer.state_dict(),torch.get_rng_state(),
            phase.schedule,phase.state,phase.local_scoring.payload(8)))
        other_opt=torch.optim.AdamW(other.parameters());resume_optimizer(raw,other_opt)
        resumed=TrainingPhases(other,c,raw=raw,generation=7,root=SOURCE_ROOT,run=restart)
        resumed.local_scoring=Companion(other,c[AUX_KEY],raw=raw,generation=7)
        advance(other,other_opt,c,resumed,restart,8,pool)
        actual=(other.state_dict(),other_opt.state_dict(),resumed.optimizer.state_dict(),torch.get_rng_state(),
            resumed.schedule,resumed.state,resumed.local_scoring.payload(8))
        require(_same_payload(expected,actual),'Restart changed model, Adam, RNG, companion or evidence')
        require(sha(checkpoint)==checksum,'Incoming checkpoint changed')
        v=r['value'];require(v['acceptance_generations']==list(range(8,0,-1)) and v['acceptance_games']==48,'Incomplete eight-age holdout')
        require(v['kl_cap'] is None and phase.beta==1. and v['beta_retries']==0,'Penalty-only/fixed beta changed')
        require(v['cpu_preparation']['workers']==4 and v['cpu_preparation']['input_bytes']==0,'Dense worker path was not used')
        bench=benchmark(out/'benchmark',phase,m,pool)
    require(source==implementation_id(),'Source changed during smoke')
    result=dict(passed=True,source=source,seconds=time.perf_counter()-start,exact_restart=True,incoming_checkpoint_unchanged=True,
        parameters=m.parameter_count,device='cpu',cuda_available=torch.cuda.is_available(),iterations=reports,benchmark=bench,
        scope='Eight authored small-fixture iterations, one real PPG phase repeated after g7 restart; no production checkpoints or long evaluation touched.')
    write_json(out/'report.json',result)
    print(dict(passed=True,seconds=result['seconds'],steps=v['optimizer_steps'],attempts=v['attempted_steps'],
        loss=v['selection'],benchmark={k:round(x['median_seconds'],5) for k,x in bench['measurements'].items()}),flush=True)


if __name__=='__main__':main()
