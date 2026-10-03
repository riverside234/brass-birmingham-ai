"""Small real four-player checks, not a playing-strength tournament."""
from pathlib import Path
import uuid
import torch

from brass_ai.bank import read,sha
from brass_ai.step5.league import write_json,implementation_id
from brass_ai.recovery.demonstrations import generate,audit
from brass_ai.recovery.league import train
from brass_ai.recovery.evaluation import evaluate


def run(root,device='cpu',mono=None):
    root=Path(root).resolve();torch.set_num_threads(2)
    directory=root/'artifacts/recovery'/('exit-'+uuid.uuid4().hex[:10])
    directory.mkdir(parents=True)
    original_pool_sha=sha(root/'training/opponents/pool.json')
    config=read(root/'configs/step5/recovery-smoke.json')
    config['model_size']='medium'
    config['execution'].update(processes=2,hosts_per_process=1,heavy_processes=1)
    config['games_per_iteration']=2
    generated=[]
    for partition in ('train','validation'):
        output=directory/('demos-'+partition)
        generate(root,output,1,config['execution'],partition=partition,mono=mono,budget=config['legacy_budget'])
        generated+=list(output.glob('*.jsonl.gz'))
    manifest=directory/'demonstrations.json'
    evidence=audit(root,generated,manifest,mono=mono,workers=2,budget=config['legacy_budget'])
    update=dict(config['update'],effective_batch_size=64,microbatch_size=8,max_candidate_cells=8192)
    config['update']=update
    config['imitation'].update(manifest=manifest.relative_to(root).as_posix(),epochs=1,min_epochs=1,patience=1)
    config['rehearsal_manifest']=config['imitation']['manifest']
    config['evaluation_schedule'].update(enabled=True,monitor_every_games=4,monitor_setups=1)
    first=train(root,directory/'league',config,iterations=1,device=device,mono=mono)
    second=train(root,directory/'league',config,iterations=1,device=device,mono=mono)
    state=read(directory/'league/state.json')
    assert state['iteration']==2 and state['games_done']==4
    assert len(state['evaluations'])==1 and state['evaluations'][0]['games']==4
    assert first['checkpoint']['model_id']!=second['checkpoint']['model_id']
    eval_config=read(root/'configs/step5/development-evaluation.json')
    eval_config.update(setups=1,include_legacy=False,lineups=['recent-neural'],threads=2,execution=config['execution'],legacy_budget=config['legacy_budget'])
    ex=config['execution']
    eval_config.update(workers=ex['processes']*ex['hosts_per_process'],legacy_workers=ex['heavy_processes'],
        search_workers=ex['heavy_processes'],batch_size=ex['batch_size'],max_candidate_cells=ex['max_candidate_cells'])
    result=evaluate(root,directory/'evaluation',second['checkpoint']['path'],[state['evaluation_anchor']['path']],eval_config,device=device,mono=mono)
    assert not result['promotion_eligible'] and result['failed_games']==0
    assert sha(root/'training/opponents/pool.json')==original_pool_sha
    record=dict(format='recovery-exit/1.0.0',implementation=implementation_id(),device=device,
        demonstration_games=len(evidence['entries']),bc_epochs=len(state['imitation']['history']),
        model_parameters=state['origin']['parameters'],teacher_workers=config['legacy_budget']['search_workers'],
        teacher_seconds=config['legacy_budget']['seconds'],
        training_games=state['games_done'],resumed_iteration=state['iteration'],
        optimizer_steps=[r['actor']['optimizer_steps'] for r in first['summaries']+second['summaries']],
        evaluation_games=sum(r['games'] for r in result['results'].values()),central_inference=result['inference'],
        source_bank_unchanged=True,promotion=False,directory=str(directory))
    write_json(directory/'exit.json',record)
    return record
