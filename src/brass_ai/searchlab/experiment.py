"""Paired equal-cap experiments, detached from all active Step 5 training runs."""
from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import random
import statistics
import time
import uuid

import torch

from brass_ai.host import RulesHost
from brass_ai.learning.checkpoint import load_checkpoint, save_checkpoint
from brass_ai.learning.contracts import digest, project_path, read_json, require
from brass_ai.learning.replay import write_shard, read_shards, split_for
from brass_ai.qualification import grouped_interval
from brass_ai.step5.league import seeds_for, write_json, pottery_metrics
from brass_ai.step5.policy import NeuralPolicy
from brass_ai.step5.search import SearchPolicy, SearchConfig, turn_key
from brass_ai.step5.updates import distill_update
from .runtime import require_gate, implementation, sha
from .s3 import S3Policy, TreeConfig


def validate_config(config):
    require(config['format']=='searchlab-experiment/1.0.0' and config['players']==4,'Four-player experiment required')
    require(config['seconds']>0 and 0<config['validation_reserve_fraction']<.5,'Invalid whole-turn budget')
    require(type(config['threads']) is int and config['threads']>0 and config['promotion_min_setups']>=30,
            'Invalid threads or weakened promotion threshold')
    TreeConfig(simulations=config['simulations'],future_turns=config['future_turns']).validate()
    SearchConfig(proposals=config['proposals'],finalists=config['finalists'],pilot_samples=config['pilot_samples'],
        confirmation_samples=config['confirmation_samples'],seconds=config['seconds']).validate()
    return config


def make_policy(kind,model,root,device,config,opponents,mono=None):
    seconds=config['seconds']*(1-config['validation_reserve_fraction'])
    common=SearchConfig(seconds=seconds,proposals=config['proposals'],finalists=config['finalists'],
        pilot_samples=config['pilot_samples'],confirmation_samples=config['confirmation_samples'],turn_interval=1)
    if kind=='s2':return SearchPolicy(model,root/'artifacts/r1-host/guarded',device,config=common,mono=mono,focus='neutral',opponents=opponents)
    if kind=='s3':return S3Policy(model,root/'artifacts/r1-host/guarded',device,config=common,mono=mono,
        focus='neutral',opponents=opponents,tree=TreeConfig(simulations=config['simulations'],future_turns=config['future_turns']))
    require(kind=='r4','Unknown variant')
    from .ismcts import OpenSpielPolicy,SpielSearchConfig
    return OpenSpielPolicy(model,root,device,config=SpielSearchConfig(seconds=seconds,
        simulations=config['simulations'],future_turns=config['future_turns'],proposals=config['proposals']),
        mono=mono,focus='neutral',opponents=opponents)


def match(root,seed,table,seat,cap,*,mono=None):
    """Measure view/choices, policy, inference and live validation per root turn."""
    group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed))
    game=dict(game_id=uuid.uuid4().hex,setup_group=group,players=4,split=split_for(group),
              opponents=[p.policy_id for p in table],result=None,records=[])
    rngs=[random.Random(digest(dict(seed=seed,rotation=seat,seat=s))) for s in range(4)]
    with RulesHost(root/'artifacts/r1-host/guarded',mono=mono) as host:
        context=host.reset(seed)
        for step in range(2000):
            if context['terminal']:break
            began=time.monotonic();actor=context['actor'];policy=table[actor]
            view,choices=host.observe(actor),host.choices()
            chosen,probability,teacher=policy.select_batch([(view,choices,rngs[actor])])[0]
            require(type(chosen) is int and 0<=chosen<len(choices),'Illegal policy choice')
            context=host.apply(context,choices[chosen]);elapsed=time.monotonic()-began
            style,intent=policy.context_features(view,rngs[actor])
            game['records'].append(dict(step=step,seat=actor,view=view,choices=choices,chosen=chosen,
                teacher=teacher,style=style,intent=intent,seconds=elapsed,
                behavior=dict(kind=policy.kind,policy_id=policy.policy_id,probability=probability,temperature=1.)))
        require(context['terminal'],'Incomplete game; no result or promotion allowed')
        game['result']=host.call('result')
    return game,result_from_game(game,seed,seat,cap)


def result_from_game(game,seed,seat,cap):
    turns=defaultdict(float)
    for row in game['records']:
        if row['seat']==seat:turns[turn_key(row['view']['context'])]+=row['seconds']
    result=game['result']['seats'][seat]
    timings=list(turns.values())
    return dict(seed=seed,seat=seat,win=result['utility'],rank=result['rank'],vp=result['vp'],
        turn_seconds=timings,whole_turn_cap=cap,over_budget=sum(t>cap for t in timings),
        max_turn_seconds=max(timings),pottery=pottery_metrics([game],{game['opponents'][seat]}))


def paired_summary(rows,variants,setups,partition,minimum=30):
    require('s2' in variants and len(set(variants))==len(variants),'Missing or duplicate comparison control')
    by_key={(r['variant'],r['seed'],r['seat']):r for r in rows}
    require(len(by_key)==len(rows),'Duplicate comparison result')
    seeds=sorted({r['seed'] for r in rows})
    require(len(seeds)==setups and all((v,s,p) in by_key for v in variants for s in seeds for p in range(4)),
            'Incomplete paired rotations')
    summaries={}
    for variant in variants:
        selected=[r for r in rows if r['variant']==variant]
        deltas=[dict(seed=r['seed'],delta=r['win']-by_key['s2',r['seed'],r['seat']]['win']) for r in selected]
        interval=grouped_interval(deltas,'delta')
        baseline=[r for r in rows if r['variant']=='s2']
        opportunities=sum(r['pottery']['legal_build_opportunities']['Pottery'] for r in selected)
        builds=sum(r['pottery']['builds']['Pottery'] for r in selected)
        pottery_warning=opportunities>=30 and builds==0
        eligible=(variant!='s2' and partition=='validation' and setups>=minimum and interval is not None
            and interval[0]>0 and all(r['over_budget']==0 for r in selected+baseline)
            and not pottery_warning)
        summaries[variant]=dict(games=len(selected),win_share=statistics.mean(r['win'] for r in selected),
            mean_rank=statistics.mean(r['rank'] for r in selected),mean_vp=statistics.mean(r['vp'] for r in selected),
            paired_win_delta=statistics.mean(r['delta'] for r in deltas),paired_setup_bootstrap_95=interval,
            over_budget_turns=sum(r['over_budget'] for r in selected),
            max_turn_seconds=max(r['max_turn_seconds'] for r in selected),pottery_builds=builds,
            pottery_opportunities=opportunities,pottery_warning=pottery_warning,eligible=eligible)
    return summaries


def compare(root,checkpoint,references,output,config,*,setups=1,seed=3910000,partition='validation',
            variants=('s2','r4','s3'),device='cpu',mono=None):
    root=Path(root).resolve();config=validate_config(config);gate=require_gate(root,device,mono);directory=project_path(root,output)
    require('s2' in variants and len(set(variants))==len(variants) and set(variants)<= {'s2','r4','s3'},'Include the matched S2 control')
    require(partition in ('validation','test') and setups>0 and references,'Held-out setups and frozen opponents required')
    torch.set_num_threads(config['threads'])
    model,raw=load_checkpoint(project_path(root,checkpoint),device=device)
    loaded=[load_checkpoint(project_path(root,p),device=device) for p in references]
    seeds,_=seeds_for(seed,setups,partition)
    spec=dict(format='searchlab-paired/1.0.0',implementation=implementation(root),correctness=gate['identity'],
        checkpoint=dict(path=checkpoint,sha256=sha(project_path(root,checkpoint)),model_id=raw['model_id']),
        references=[dict(path=p,sha256=sha(project_path(root,p))) for p in references],
        config=config,device=device,variants=list(variants),seeds=seeds,partition=partition,rotations=4)
    spec_path=directory/'spec.json'
    if spec_path.exists():require(read_json(spec_path)==spec,'Comparison specification changed; use a new directory')
    else:
        require(not directory.exists() or not any(directory.iterdir()),'Use a new experiment directory, separate from main training')
        write_json(spec_path,spec)
    rows=[];evidence={}
    jobs=[(variant,s,p) for s in seeds for p in range(4) for variant in variants]
    random.Random(71823).shuffle(jobs)  # Avoid always timing one variant first.
    for variant,s,p in jobs:
        name=f'{variant}-{s}-seat-{p}';record_path=directory/(name+'.json');shard=directory/(name+'.jsonl.gz')
        if record_path.exists():
            record=read_json(record_path)
            require(record['spec']==digest(spec) and record['replay_sha256']==sha(shard),'Cached evidence changed')
        elif shard.exists():
            # Recover the crash window between atomic replay publication and
            # metadata publication. Recompute scores/timing from validated data.
            import gzip
            with gzip.open(shard,'rt',encoding='utf-8') as stream:header=json.loads(next(stream))
            require(header['provenance']==dict(producer='searchlab/1.0.0',purpose='evaluation',spec=digest(spec)),
                    'Orphan replay belongs to another experiment')
            episodes=read_shards([shard]);require(len(episodes)==1,'Expected one complete match')
            require(episodes[0]['setup_group']==digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=s)),
                    'Cached replay has wrong setup')
            record=dict(spec=digest(spec),result=dict(variant=variant,**result_from_game(episodes[0],s,p,config['seconds'])),
                replay_sha256=sha(shard),telemetry=dict(recovered_complete_replay=True))
            write_json(record_path,record)
        else:
            with ExitStack() as stack:
                rollouts=[NeuralPolicy(m,device,exploration=.05,focus='neutral') for m,_ in loaded]
                candidate=make_policy(variant,model,root,device,config,rollouts,mono)
                stack.callback(candidate.close)
                opponents=[NeuralPolicy(loaded[i%len(loaded)][0],device,exploration=0,focus='neutral') for i in range(3)]
                table=opponents.copy();table.insert(p,candidate)
                try:
                    game,result=match(root,s,table,p,config['seconds'],mono=mono)
                    write_shard(shard,[game],dict(producer='searchlab/1.0.0',purpose='evaluation',spec=digest(spec)))
                    record=dict(spec=digest(spec),result=dict(variant=variant,**result),
                        replay_sha256=sha(shard),telemetry=candidate.telemetry)
                    write_json(record_path,record)
                except Exception as error:
                    write_json(directory/(name+'.failure.json'),dict(spec=digest(spec),error=str(error)))
                    raise
        rows.append(record['result']);evidence[record_path.name]=sha(record_path)
        print(json.dumps(dict(completed=len(rows),total=len(jobs),variant=variant,rank=rows[-1]['rank'])),flush=True)
    failures=len(list(directory.glob('*.failure.json')))
    report=dict(format='searchlab-paired-report/1.0.0',spec=spec,failed_games=failures,results=rows,
        summary=paired_summary(rows,variants,setups,partition,config['promotion_min_setups']),evidence=evidence)
    if failures:
        for result in report['summary'].values():result['eligible']=False
    write_json(directory/'report.json',report)
    return report


def promote(root,report_path,variant,output,*,device='cpu',mono=None):
    report_path=project_path(root,report_path);report=read_json(report_path)
    gate=require_gate(root,device,mono)
    require(report['spec']['implementation']==implementation(root) and report['spec']['correctness']==gate['identity'],'Stale correctness gate')
    require(variant in ('r4','s3') and report['failed_games']==0,'Variant/failure gate rejected')
    spec=report['spec'];minimum=read_json(Path(root)/'configs/searchlab/experiment.json')['promotion_min_setups']
    recalculated=paired_summary(report['results'],spec['variants'],len(spec['seeds']),spec['partition'],minimum)
    require(recalculated[variant]['eligible'],'No held-out equal-budget advantage; not promoted')
    require(sha(project_path(root,spec['checkpoint']['path']))==spec['checkpoint']['sha256'],'Evaluated checkpoint changed')
    for entry in spec['references']:
        require(sha(project_path(root,entry['path']))==entry['sha256'],'Evaluated opponent changed')
    expected={f"{r['variant']}-{r['seed']}-seat-{r['seat']}.json":r for r in report['results']}
    require(set(expected)==set(report['evidence']),'Missing match evidence')
    for name,fingerprint in report['evidence'].items():
        record_path=report_path.parent/name;record=read_json(record_path)
        require(sha(record_path)==fingerprint and record['spec']==digest(spec) and record['result']==expected[name],
                'Match evidence changed or differs from summary')
        require(sha(record_path.with_suffix('.jsonl.gz'))==record['replay_sha256'],'Match replay changed')
    record=dict(format='searchlab-promotion/1.0.0',variant=variant,report_sha256=sha(report_path),
        checkpoint=spec['checkpoint'],config=spec['config'],scope='This frozen experiment only; main training and Unity are unchanged')
    target=project_path(root,output)
    if target.exists():require(read_json(target)==record,'Refusing to overwrite a different selection or training file')
    else:write_json(target,record)
    return record


def distill_s3(root,checkpoint,output,config,*,games=2,seed=4920000,device='cpu',mono=None):
    """An additional, isolated S3 teacher update; never mutate the input run."""
    from brass_ai.learning.runner import collect_games
    root=Path(root).resolve();config=validate_config(config);require_gate(root,device,mono);directory=project_path(root,output)
    require(games>0,'Positive game count required');torch.set_num_threads(config['threads'])
    require(not directory.exists() or not any(directory.iterdir()),'Use a new S3 experiment output')
    model,raw=load_checkpoint(project_path(root,checkpoint),device=device)
    initial_sha=sha(project_path(root,checkpoint));seeds,_=seeds_for(seed,games,'train')
    rollouts=[NeuralPolicy(model,device,exploration=.05,focus='neutral')]
    policy=make_policy('s3',model,root,device,config,rollouts,mono)
    try:
        episodes=collect_games(root/'artifacts/r1-host/guarded',seeds,[policy]*4,workers=1,sampling_seed=seed,mono=mono)
        write_shard(directory/'teacher.jsonl.gz',episodes,dict(producer='searchlab/1.0.0',purpose='training',variant='s3'))
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        result=distill_update(model,optimizer,episodes,steps=4,weight=.25)
        save_checkpoint(directory/'candidate.pt',model,optimizer,training=dict(mode='searchlab-s3-distill',
            parent=raw['model_id'],implementation=implementation(root),evidence=result,promoted=False))
        require(sha(project_path(root,checkpoint))==initial_sha,'Source training checkpoint changed')
        report=dict(updated=result,source_unchanged=True,promoted=False,telemetry=policy.telemetry,
            initial_sha256=initial_sha,config=config,device=device,implementation=implementation(root))
        write_json(directory/'report.json',report);return report
    finally:policy.close()
