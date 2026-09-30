"""Frozen seat-rotated evaluation on the same persistent actor/inference pipeline."""
from pathlib import Path
import time

import torch

from brass_ai.bank import read,sha,admission
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import digest,project_path,require
from brass_ai.learning.replay import read_shards
from brass_ai.step5.evaluation import (evaluation_config,evaluation_jobs,check_evaluation_batch,
    outcome,summarize_outcomes)
from brass_ai.step5.league import seeds_for,write_json,implementation_id,pottery_metrics
from .diagnostics import economy
from .execution import ActorPool,neural_spec
from .inference import InferenceService


def evaluate(root,output,candidate_path,references,config,*,setups=None,seed=None,
             partition=None,include_legacy=None,device='cpu',mono=None,search=None,focus=None):
    root=Path(root);directory=project_path(root,output)
    config=evaluation_config(config,setups=setups,seed=seed,partition=partition,
        include_legacy=include_legacy,use_search=search,focus=focus)
    torch.set_num_threads(config['threads'])
    require(references,'Frozen references required')
    seeds,_=seeds_for(config['seed'],config['setups'],config['partition'])
    entries=[];modes={}
    for path in [candidate_path]+list(references):
        with torch.random.fork_rng(devices=[]): model,raw=load_checkpoint(project_path(root,path),device='cpu')
        entries.append(dict(path=str(path),sha256=sha(project_path(root,path)),model_id=raw['model_id']))
        modes[raw['model_id']]=raw['training'].get('mode')
    require(entries[0]['model_id'] not in [e['model_id'] for e in entries[1:]],'Candidate cannot be its own reference')
    pool=read(root/'training/opponents/pool.json')
    profiles=sorted(pool['training']);held=sorted(pool['evaluation'])
    lineups=['recent-neural']
    if len({e['model_id'] for e in entries[1:]})>1: lineups.append('mixed-neural')
    if config['include_legacy']:
        lineups.append('mixed-v3')
        if config['panel']=='full': lineups.append('heldout-pottery-v3')
    spec=dict(format='step5-evaluation/2.0.0',purpose='evaluation',entries=entries,
        config=config,seeds=seeds,lineups=lineups,implementation=implementation_id(),legacy_pool=pool)
    if (directory/'spec.json').exists(): require(read(directory/'spec.json')==spec,'Evaluation specification changed')
    else:
        require(not directory.exists() or not any(directory.iterdir()),'Use a fresh evaluation directory')
        write_json(directory/'spec.json',spec)
    results={};execution={};evidence={};began=time.monotonic()
    with InferenceService(config['execution'],device) as service:
        for e in entries: service.register_checkpoint(project_path(root,e['path']))
        with ActorPool(root,config['host'],service,config['execution'],mono=mono) as ordinary,ActorPool(root,config['host'],service,config['execution'],mono=mono,heavy=True) as heavy:
            for name in lineups:
                jobs=[];lineup_began=time.monotonic()
                for i,job in enumerate(evaluation_jobs(seeds)):
                    if config['use_search']:
                        candidate=dict(kind='s2',model_id=entries[0]['model_id'],focus=config['focus'],search=config['search'])
                    else: candidate=neural_spec(entries[0]['model_id'],0,config['focus'])
                    if name.endswith('neural'):
                        refs=entries[1:2] if name=='recent-neural' else entries[1:]
                        opponents=[neural_spec(refs[s%len(refs)]['model_id'],0,'neutral') for s in range(3)]
                    else:
                        chosen=held if name=='heldout-pottery-v3' else profiles
                        require(chosen,'Missing evaluation bank')
                        purpose='evaluation' if name=='heldout-pottery-v3' else 'training'
                        opponents=[dict(kind='legacy',policy_id=chosen[(i//4*3+s)%len(chosen)],purpose=purpose) for s in range(3)]
                        for p in opponents: admission(root,p['policy_id'],purpose)
                    provenance=dict(producer='recovery-evaluation/2.0.0',purpose='evaluation',kind=name,
                        seed=job['seed'],sampling_seed=job['sampling_seed'],rotation=job['seat'],spec=digest(spec))
                    jobs.append(dict(index=i,seed=job['seed'],rotation=job['seat'],sampling_seed=job['sampling_seed'],
                        policies=[candidate]+opponents,path=str(directory/f'{name}-{i:06d}.jsonl.gz'),provenance=provenance))
                actor=heavy if config['use_search'] or name.endswith('v3') else ordinary
                records=actor.collect(jobs)
                games=read_shards([r['path'] for r in records])
                candidate_id=games[0]['opponents'][jobs[0]['rotation']]
                check_evaluation_batch(games,evaluation_jobs(seeds),candidate_id,config['partition'])
                by_group={digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=s)):s for s in seeds}
                rows=sorted((outcome(g,candidate_id,by_group[g['setup_group']]) for g in games),key=lambda r:(r['seed'],r['seat']))
                results[name]=dict(**summarize_outcomes(rows),outcomes=rows,
                    diversity=pottery_metrics(games,{candidate_id}),economy=economy(games,{candidate_id}))
                execution[name]=dict(mode='spawned-private-hosts-central-inference',workers=actor.width*actor.lanes,
                    games=len(games),seconds=time.monotonic()-lineup_began,cached_games=sum(r['cached'] for r in records))
                evidence.update({Path(r['path']).name:r['sha256'] for r in records})
        telemetry=service.report()
    trained={e['model_id'] for e in entries[1:] if modes[e['model_id']] in ('step5-population-ppo','step5-reptile')}
    neural=[r for k,r in results.items() if k.endswith('neural')]
    eligible=(config['panel']=='full' and config['partition']=='validation' and config['setups']>=30 and len(trained)>=2
        and all(r['win_share_setup_bootstrap_95'][0]>.25 for r in neural)
        and not any(r['diversity']['pottery_warning'] for r in results.values()))
    report=dict(format='step5-evaluation-report/1.0.0',spec=spec,results=results,execution=execution,
        promotion_eligible=eligible,failed_games=0,candidate_sha256=entries[0]['sha256'],raw_record_sha256=evidence,
        inference=telemetry,elapsed_seconds=time.monotonic()-began,
        limitation='Development panels cannot promote. BC references do not satisfy trained-population gates. Cached replay time is not live throughput.')
    write_json(directory/'report.json',report);return report
