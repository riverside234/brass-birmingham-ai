"""Frozen seat-rotated evaluation on the same persistent actor/inference pipeline."""
from pathlib import Path
import time

import torch

from brass_ai.bank import read,sha,admission
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import digest,project_path,require
from .replay import read_shards
from brass_ai.step5.evaluation import (evaluation_config,evaluation_jobs,check_evaluation_batch,
    outcome,summarize_outcomes)
from brass_ai.step5.league import seeds_for,write_json,implementation_id,pottery_metrics
from .diagnostics import economy
from .execution import ActorPool,neural_spec
from .inference import InferenceService


def evaluate(root,output,candidate_path,references,config,*,setups=None,seed=None,
             partition=None,include_legacy=None,device='cpu',mono=None,search=None,focus=None,lineups=None):
    root=Path(root);directory=project_path(root,output)
    if lineups is not None: include_legacy=any(n.endswith('v3') for n in lineups)
    config=evaluation_config(config,setups=setups,seed=seed,partition=partition,
        include_legacy=include_legacy,use_search=search,focus=focus,lineups=lineups)
    if lineups is not None:
        config.update(lineups=lineups,include_legacy=any(n.endswith('v3') for n in lineups))
        from brass_ai.step5.evaluation import validate_evaluation_config
        validate_evaluation_config(config)
    torch.set_num_threads(config['threads'])
    require(references,'Frozen references required')
    seeds,_=seeds_for(config['seed'],config['setups'],config['partition'])
    entries=[];modes={}
    for path in [candidate_path]+list(references):
        with torch.random.fork_rng(devices=[]): model,raw=load_checkpoint(project_path(root,path),device='cpu')
        entries.append(dict(path=str(path),sha256=sha(project_path(root,path)),model_id=raw['model_id']))
        modes[raw['model_id']]=raw['training'].get('mode')
        require(model.config.architecture=='brass-four-seat-mlp-resnet/1.0.0' and raw['training'].get('fresh_run'),'Evaluation accepts only checkpoints from a new fresh residual run')
        if len(entries)==1 and config['use_search'] and config['search'].get('reward') is not None:
            from .utility import calibration_ready
            require(model.config.utility==config['search']['reward'] and calibration_ready(model),
                    'Combined search evaluation requires a calibrated matching-reward checkpoint')
        if entries[:-1]:require(raw['training']['fresh_run']==fresh_run,'Evaluation cannot import another run population')
        fresh_run=raw['training']['fresh_run']
    require(entries[0]['model_id'] not in [e['model_id'] for e in entries[1:]],'Candidate cannot be its own reference')
    pool=read(root/'training/opponents/pool.json')
    profiles=sorted(pool['training']);held=sorted(pool['evaluation'])
    lineups=config.get('lineups',['recent-neural']).copy()
    if 'lineups' not in config and len({e['model_id'] for e in entries[1:]})>1: lineups.append('mixed-neural')
    if 'lineups' not in config and config['include_legacy']:
        lineups.append('mixed-v3')
        if config['panel']=='full': lineups.append('heldout-pottery-v3')
    require('mixed-neural' not in lineups or len({e['model_id'] for e in entries[1:]})>=2,
        'mixed-neural requires two distinct frozen references')
    from .legacy_search import legacy_spec
    budget=config.get('legacy_budget')
    teacher_specs={p:legacy_spec(root,p,budget) for p in profiles} if 'mixed-v3' in lineups else {}
    held_specs={p:legacy_spec(root,p,budget,'evaluation') for p in held} if 'heldout-pottery-v3' in lineups else {}
    spec=dict(format='step5-evaluation/2.0.0',purpose='evaluation',entries=entries,
        config=config,seeds=seeds,lineups=lineups,implementation=implementation_id(),legacy_pool=pool,teachers=teacher_specs,held=held_specs)
    if (directory/'report.json').exists() and read(directory/'report.json').get('raw_records_pruned'):
        report=read(directory/'report.json');require(report['spec']==spec,'Evaluation specification changed')
        return dict(report,cached_summary=True,promotion_eligible=False)
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
                        candidate=dict(kind='masked_ismcts',model_id=entries[0]['model_id'],focus=config['focus'],search=config['search'])
                    else: candidate=neural_spec(entries[0]['model_id'],0,config['focus'])
                    if name.endswith('neural'):
                        refs=entries[1:2] if name=='recent-neural' else entries[1:]
                        opponents=[neural_spec(refs[s%len(refs)]['model_id'],0,'neutral') for s in range(3)]
                    else:
                        chosen=held if name=='heldout-pottery-v3' else profiles
                        require(chosen,'Missing evaluation bank')
                        purpose='evaluation' if name=='heldout-pottery-v3' else 'training'
                        specs=held_specs if name=='heldout-pottery-v3' else teacher_specs
                        opponents=[specs[chosen[(i//4*3+s)%len(chosen)]] for s in range(3)]
                    provenance=dict(producer='recovery-evaluation/2.0.0',purpose='evaluation',kind=name,
                        seed=job['seed'],sampling_seed=job['sampling_seed'],rotation=job['seat'],spec=digest(spec))
                    jobs.append(dict(index=i,seed=job['seed'],rotation=job['seat'],sampling_seed=job['sampling_seed'],
                        policies=[candidate]+opponents,path=str(directory/f'{name}-{i:06d}.jsonl.gz'),provenance=provenance))
                actor=heavy if config['use_search'] or name.endswith('v3') else ordinary
                records=actor.collect(jobs)
                games=read_shards([r['path'] for r in records],receipts={r['path']:r.get('features') for r in records})
                # Shards/worker completions need not have scheduling order.
                # Attribute the candidate through its job/game receipt rather
                # than interpreting the first replay as the first rotation.
                scheduled={j['index']:j for j in jobs}
                require(len(records)==len(jobs) and {r['job'] for r in records}==set(scheduled),
                        'Missing, duplicate or foreign evaluation receipts')
                rotations={r['game_id']:scheduled[r['job']]['rotation'] for r in records}
                require(len(rotations)==len(records) and {g['game_id'] for g in games}==set(rotations),
                        'Missing or duplicate evaluation replay identities')
                candidate_ids={g['opponents'][rotations[g['game_id']]] for g in games}
                require(len(candidate_ids)==1,'Evaluation candidate identity changed between rotations')
                candidate_id=candidate_ids.pop()
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
    eligible=(config['panel']=='full' and set(lineups)=={'recent-neural','mixed-neural','mixed-v3','heldout-pottery-v3'} and
        config['partition']=='validation' and config['setups']>=30 and len(trained)>=2
        and all(r['win_share_setup_bootstrap_95'][0]>.25 for r in neural)
        and not any(r['diversity']['pottery_warning'] for r in results.values()))
    report=dict(format='step5-evaluation-report/1.0.0',spec=spec,results=results,execution=execution,
        promotion_eligible=eligible,failed_games=0,candidate_sha256=entries[0]['sha256'],raw_record_sha256=evidence,
        inference=telemetry,elapsed_seconds=time.monotonic()-began,
        games=sum(r['games'] for r in results.values()),setups_per_lineup=config['setups'],lineup_count=len(lineups),
        limitation='Development panels cannot promote. BC references do not satisfy trained-population gates. Cached replay time is not live throughput.')
    write_json(directory/'report.json',report)
    if config.get('retention',{}).get('enabled'):
        from .retention import prune
        report['retention']=prune(root,directory,config['retention'],apply=True)
    return report
