"""Single FC2 lifecycle smoke: tiny BC, one ordinary game/PPO step, leaf/export.

Uses only a new scratch directory and the current target. No timed search,
evaluation games, historical-model loading, or additional PPO iterations.
"""
import argparse
import copy
import multiprocessing as mp
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,digest,read_json,require
from brass_ai.learning.checkpoint import load_checkpoint,resume_optimizer
from brass_ai.learning.batching import model_id
from brass_ai.recovery.demonstrations import train_bc,completed_imitation,load_dataset
from brass_ai.recovery.config import latest_model_config
from brass_ai.recovery.league import train
from brass_ai.recovery.resume import load_resume
from brass_ai.recovery.retention import _same_payload
from brass_ai.recovery.utility import reward_spec,calibration_ready
from brass_ai.step5.league import implementation_id,write_json


def leaf_worker(channels,identity,row,reward,queue):
    """The teacher's remote evaluator boundary, with no tree or search clock."""
    from brass_ai.recovery.inference import RemoteEvaluator
    from brass_ai.learning.batching import Request
    endpoint=channels.acquire()
    try:
        request=Request('compact-leaf',identity,row['view'],row['choices'],time.monotonic()+30,
            remaining=(reward['insolvency']['max_cost'],)*4)
        evaluator=RemoteEvaluator(endpoint,identity)
        result=evaluator.predict([request])[0]
        require(result['status']=='ok' and result['model_id']==identity,'Teacher leaf model/status differs')
        require(result['reward_id']==digest(reward),'Teacher leaf utility contract differs')
        require(len(result['probabilities'])==len(row['choices']) and np.isfinite(result['utility']).all(),
                'Teacher leaf shape/utility differs')
        queue.put(dict(passed=True,response=result,telemetry=evaluator.telemetry))
    except Exception as error:queue.put(dict(passed=False,error=repr(error)))
    finally:endpoint.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,help='Tiny audited heuristic source: one train and one validation game')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--resume-debug',action='store_true',help='Reuse this incomplete current-source smoke after a harness fix; never collect another game')
    args=parser.parse_args();out=args.output.resolve()
    require(not out.exists() or args.resume_debug and not(out/'report.json').exists(),
            'Use a new isolated smoke directory, or explicitly resume an incomplete debug fixture')
    source=read_json(args.manifest)
    require(len(source['entries'])==2 and {e['split'] for e in source['entries']}=={'train','validation'},
            'Smoke requires exactly one complete heuristic game per train/validation partition')
    out.mkdir(parents=True,exist_ok=args.resume_debug);began=time.monotonic();source_id=implementation_id();torch.set_num_threads(2)
    clock=out/'clock.json'
    if not clock.exists():write_json(clock,dict(started_unix=time.time()))
    from brass_ai.feature_conversion import convert
    if args.resume_debug and (out/'data/manifest.json').exists():
        conversion=read_json(out/'data/conversion.json')
        require(conversion['compatibility']==compatibility() and conversion['source_manifest_sha256']==sha(args.manifest),
                'Debug resume data/source differs')
    else:conversion=convert(SOURCE_ROOT,args.manifest,out/'data',workers=1)
    manifest=(out/'data/manifest.json').relative_to(SOURCE_ROOT).as_posix()
    cfg=read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json');before=copy.deepcopy(cfg)
    update=copy.deepcopy(cfg['update'])
    update.update(effective_batch_size=512,microbatch_size=16,max_candidate_cells=8192,
                  critic_batch_size=64,kl_probe_rows=32,epochs=1)
    selected={}
    def tiny_loader(*a,**kw):
        data,meta=load_dataset(*a,**kw);rows=data[:8]
        selected[a[2]]=dict(available=len(data),used=len(rows))
        return rows,meta
    il=out/'imitation'
    # This selection is a test fixture only. The source still passes full native
    # audit; production dataset loading and completion/handoff guards are intact.
    if args.resume_debug and (il/'report.json').exists():
        bc=read_json(il/'report.json')
        require(bc['signature']['update']==update and bc['epochs']==1,'Debug IL fixture differs')
        for partition in ('train','validation'):
            selected[partition]=dict(used=bc['history'][0][partition]['decisions'],
                available='one complete audited source game; see conversion manifest')
    else:
        with patch('brass_ai.recovery.demonstrations.load_dataset',side_effect=tiny_loader):
            bc=train_bc(SOURCE_ROOT,il,manifest,None,update,epochs=1,min_epochs=1,patience=1,workers=1)
    original,origin=completed_imitation(SOURCE_ROOT,il)
    require(bc['history'][0]['train']['optimizer_steps']==1,'Expected one tiny IL update')
    cfg=latest_model_config(cfg,origin)
    cfg['rehearsal_manifest']=origin['manifest']
    write_json(out/'resolved-production-config.json',cfg)
    require(all(cfg[k]==before[k] for k in before.keys()-{'imitation_checkpoint','rehearsal_manifest'}),
            'Config generation changed an unrelated training setting')
    cfg.update(games_per_iteration=1,threads=2,audit_workers=1,rehearsal_weight=0.,update=update)
    cfg['execution'].update(processes=1,hosts_per_process=1,heavy_processes=1)
    cfg['evaluation_schedule']['enabled']=False;cfg['retention']['enabled']=False
    write_json(out/'smoke-config.json',cfg);run=out/'league'
    if args.resume_debug and (run/'state.json').exists():
        state=read_json(run/'state.json')
        require(state['iteration']==state['games_done']==1 and state['signature']['implementation']==source_id
                and state['signature']['config']==cfg,'Debug league state differs')
        report=read_json(run/'iteration-0000/report.json')
    else:
        with patch('brass_ai.recovery.ordinary.training_table',return_value=('selfplay',True)):
            result=train(SOURCE_ROOT,run,cfg,iterations=1,imitation_run=il,fresh=True)
        report=result['summaries'][0];state=read_json(run/'state.json')
    initial,initial_raw=load_checkpoint(run/'initial.pt')
    require(_same_payload(initial.state_dict(),original.state_dict()),'Fresh IL handoff changed weights')
    require(initial_raw['optimizer']['state']=={},'PPO inherited the IL optimizer')
    require(state['iteration']==state['games_done']==1 and report['actor']['optimizer_steps']==1,
            'Smoke exceeded one ordinary game/PPO update')
    require(report['sampling']['learner_trajectories']==4,'Smoke did not collect all four current-model seats')
    old_files={p:sha(p) for p in run.rglob('*.pt')};old_state=copy.deepcopy(state)
    model,raw,changed=load_resume(SOURCE_ROOT,run,state,state['signature'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
    resume_optimizer(raw,optimizer)
    require(not changed and _same_payload(raw['optimizer'],optimizer.state_dict()) and
            torch.equal(raw['rng'],torch.get_rng_state()),'Matching Adam/RNG resume differs')
    require(state==old_state and read_json(run/'state.json')==old_state and all(sha(p)==s for p,s in old_files.items()),
            'Resume modified progress, population or checkpoint lineage')
    require(calibration_ready(model),'Do not bypass teacher utility calibration')
    from brass_ai.recovery.replay import read_shards,load_cache
    games=read_shards(sorted(run.glob('iteration-*/*.jsonl.gz')),partition='train')
    require(len(games)==1,'Smoke collected extra games')
    row=games[0]['records'][0]
    # A foreign attestation is rejected without trusting an old tensor sidecar.
    require(load_cache(next(run.glob('iteration-*/*.jsonl.gz')),games,dict(validator='0'*64)) is None,
            'Incompatible encoded cache was reused')
    from brass_ai.recovery.inference import InferenceService
    process_context=mp.get_context('spawn');queue=process_context.Queue()
    with InferenceService(cfg['execution'],'cpu') as service:
        identity=service.register(model);channels=service.channels(1)
        worker=process_context.Process(target=leaf_worker,args=(channels,identity,row,reward_spec(cfg),queue))
        worker.start()
        try:leaf=queue.get(timeout=45)
        finally:
            worker.join(timeout=10)
            if worker.is_alive():worker.terminate();worker.join(timeout=5)
        require(worker.exitcode==0 and leaf['passed'],'Teacher worker failed: '+str(leaf))
        leaf['service']=service.report()
    queue.close();queue.join_thread()
    # Export the affected production actor path, with two small dynamic shapes.
    from brass_ai.personality.model import import_parent
    from brass_ai.personality.checkpoint import save,load
    from brass_ai.personality.laptop.export import export,samples,Graph,compare,Named
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(661);actor,import_receipt=import_parent(model)
    require(_same_payload(actor.core.state_dict(),model.state_dict()),'Personality handoff changed parent')
    require(not(out/'actor.pt').exists(),'Inspect a previous actor/export failure before continuing')
    save(out/'actor.pt',actor,training=dict(diagnostic=True,untrained_goals=True))
    actor,_=load(out/'actor.pt')
    batches=samples(model.config.to_dict(),True)[:2]
    with patch('brass_ai.personality.laptop.export.samples',return_value=batches):
        portable=export(actor,out/'actor.onnx')
    graph=Graph(out/'actor.onnx');named=Named(actor.eval())
    for batch in batches:
        with torch.no_grad():expected=named(*(torch.from_numpy(batch[k]) for k in graph.inputs))
        compare(dict(zip(graph.outputs,[x.numpy() for x in expected])),graph.run(batch),batch['legal_mask'])
    require(source_id==implementation_id(),'Source changed during smoke')
    require(read_json(SOURCE_ROOT/'configs/step5/joint7-latest.json')==before,'Production recipe changed')
    write_json(out/'report.json',dict(format='feature-compaction-smoke/1.0.0',passed=True,implementation=source_id,
        compatibility=compatibility(),elapsed_seconds=time.time()-read_json(clock)['started_unix'],
        final_invocation_seconds=time.monotonic()-began,resumed_debug=args.resume_debug,conversion=conversion,
        IL=dict(epochs=1,optimizer_steps=1,selection=selected,origin=origin,parameters=original.parameter_count),
        PPO=dict(iterations=1,games=1,optimizer_steps=report['actor']['optimizer_steps'],
            initial_weights_exact=True,initial_optimizer_empty=True,sampling=report['sampling'],update=report['actor']),
        resume=dict(weights_exact=model_id(model)==raw['model_id'],adam_states=len(optimizer.state),rng_exact=True,
            committed_iteration=state['iteration'],completed_games=state['games_done'],
            population_generations=[p['generation'] for p in state['population']],state_unchanged=True,checkpoint_bytes_unchanged=True),
        teacher_leaf=leaf,personality_import=import_receipt,export=portable,
        fixture_only=dict(IL_first_rows_per_partition=8,forced_latest_selfplay=True,update=update,
            execution=cfg['execution'],rehearsal_disabled=True,scheduled_evaluation_disabled=True,retention_disabled=True),
        unexercised=['long IL/PPO training','historical/legacy ordinary tables','timed teacher search/collection/distillation',
            'scheduled evaluation','GPU/OpenVINO/Unity deployment','strength evaluation','performance profiling'],
        scope='Current-target CPU debug fixture; one collected game, one PPO optimizer step, no strength or speedup claim.'))
    print('Compaction smoke passed:',out/'report.json',flush=True)


if __name__=='__main__':main()
