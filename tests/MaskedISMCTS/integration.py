"""Small real-engine exit check, deliberately not a playing-strength tournament."""
import copy
import json
from pathlib import Path
import random
import time

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.checkpoint import load_checkpoint,save_checkpoint
from brass_ai.learning.contracts import require
from brass_ai.learning.export import export_onnx
from brass_ai.learning.replay import validate_teacher
from brass_ai.recovery.league import train
from brass_ai.recovery.replay import read_shards
from brass_ai.recovery.retention import plan
from brass_ai.recovery.updates import distill_update
from brass_ai.recovery.evaluation import evaluate
from brass_ai.search.common import SearchConfig
from brass_ai.search.ismcts import MaskedSearchPolicy,information_key
from brass_ai.step5.evaluation import evaluation_config
from brass_ai.step5.league import implementation_id,write_json


def run(root):
    root=Path(root);out=root/'artifacts/masked-exit'/('integration-'+str(time.time_ns()))
    cfg=read(root/'configs/step5/recovery-smoke.json')
    cfg.update(model_size='medium',games_per_iteration=1,threads=2)
    cfg['execution'].update(processes=1,heavy_processes=1,search_workers_per_turn=1)
    archived_demos=root/'artifacts/recovery/exit-8c3613bff0/demonstrations.json'
    if archived_demos.exists():
        cfg['imitation'].update(manifest=archived_demos.relative_to(root).as_posix(),epochs=3,min_epochs=3,patience=1)
        cfg['rehearsal_manifest']=cfg['imitation']['manifest']
    run_path=(out/'fresh-run').relative_to(root).as_posix()
    # The first two scheduled tables are pure self-play; they exercise completed
    # four-seat terminal targets, new-run population construction and exact resume.
    first=train(root,run_path,cfg,iterations=1)
    second=train(root,run_path,cfg,iterations=1)
    require(second['completed_iteration']==2,'Same-run resume did not advance')
    state=read(root/run_path/'state.json')
    require(state['origin']['initialization']=='random' and state['origin']['parameters']==1840109,'Fresh residual initialization')
    try:train(root,(out/'forbidden').relative_to(root),cfg,initial=first['checkpoint']['path'])
    except ValueError as e:require('forbids' in str(e),'Unexpected checkpoint-rejection failure')
    else:raise AssertionError('Previous-checkpoint import accepted')
    games=read_shards(list((root/run_path/'iteration-0001').glob('*.jsonl.gz')),partition='train')
    require(len(games)==1 and len(games[0]['result']['seats'])==4,'Four seats must finish')
    model,raw=load_checkpoint(root/second['checkpoint']['path'])
    # Work-bound three trajectories allow the CPU check to finish on this laptop;
    # the v3 lab configuration remains exactly one second and is not calibrated.
    row=games[0]['records'][0]
    search=MaskedSearchPolicy(model,root,config=SearchConfig(seconds=20,simulations=3,pending_paths=3,batch_size=3,turn_interval=1))
    try:
        a=search.game.from_observation(row['view'],901)
        b=search.simulation_games[0].from_observation(row['view'],902)
        require(information_key(a.views[row['seat']],a.legal,a.recall[row['seat']])==information_key(b.views[row['seat']],b.legal,b.recall[row['seat']]),'Hidden-world root information differs')
        require(all(v['seat']==s and 'own_hand' in v and 'checkpoint' not in v for s,v in enumerate(a.views)),'Opponent permitted view contract')
        tree={}
        selected,_,teacher=search.decide(row['view'],row['choices'],random.Random(1701),time.monotonic()+20,tree=tree)
        write_json(out/'search-diagnostic.json',dict(statistics=search.last_statistics,teacher=teacher))
        require(teacher is not None,'No meaningful completed search teacher')
        validate_teacher(teacher,len(row['choices']))
        stats=search.last_statistics
        require(stats['completed']==3 and stats['opponent_nodes']==0,'Completed masked trajectories')
        require(stats['batching']['max_batch']>=2,'Single CPU worker never batched paths')
        require(stats['batching']['elapsed_seconds']>=stats['batching']['inference_seconds'],'Search timing excludes part of the completed work')
        require(all(e.pending==0 for node in tree.values() for e in node.edges.values()),'Pending reservations remain')
        # Distillation is a separate loss, with external actions rather than fake
        # NN behavior probabilities. Full terminal targets remain four-seat.
        teacher_game=copy.deepcopy(dict(games[0]));teacher_game['records']=copy.deepcopy(games[0]['records'])
        for r in teacher_game['records']:r['teacher']=None
        target=next(r for r in teacher_game['records'] if r['step']==row['step'])
        target.update(teacher=teacher,chosen=selected)
        target['behavior']=dict(kind='external',policy_id=search.policy_id,probability=1.,temperature=1.)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
        distilled=distill_update(model,optimizer,[teacher_game],cfg['update'],steps=1)
        require(distilled['updated'] and distilled['optimizer_steps']==1,'Search loss failed')
        search.cancel();before=stats['completed']
        search.decide(row['view'],row['choices'],random.Random(1702),time.monotonic()+1,tree=tree)
        require(search.last_statistics['completed']==before,'Cancellation manufactured evidence')
        encoded=search.base.evaluator.encoder
        batches=[[encoded.encode(row['view'],row['choices'])],
                 [encoded.encode(r['view'],r['choices']) for r in games[0]['records'][:2]]]
        exported=export_onnx(model,out/'fresh-residual.onnx',batches)
    finally:search.close()
    evalcfg=evaluation_config(cfg,setups=1,lineups=['recent-neural'],include_legacy=False,panel='development')
    report=evaluate(root,(out/'evaluation').relative_to(root),second['checkpoint']['path'],[first['checkpoint']['path']],evalcfg)
    require(report['failed_games']==0,'Seat-rotated development evaluation failed')
    cleanup=plan(root,root/run_path,cfg['retention'])
    # Applying retention must retain current resume and population files.
    from brass_ai.recovery.retention import prune
    prune(root,root/run_path,cfg['retention'],apply=True)
    require((root/second['checkpoint']['path']).exists() and all((root/e['path']).exists() for e in state['population']),'Cleanup removed active weights')
    result=dict(passed=True,scope='compact functional exit, not strength qualification',implementation=implementation_id(),
        parameters=model.parameter_count,fresh_run=run_path,resumed_iteration=2,training_games=2,evaluation_games=report['games'],imitation=state['imitation'],
        completed_search=stats['completed'],teacher_alternatives=sum(v>0 for v in teacher['visits']),batching=stats['batching'],
        distilled=distilled['optimizer_steps'],onnx=exported,cleanup_preserved=True)
    write_json(out/'report.json',result)
    return result
