"""Paired equal-cap experiments, detached from all active Step 5 training runs."""
from collections import defaultdict
from contextlib import ExitStack
import gzip
import json
import os
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
from .config import validate_config
from .execution import run_jobs


def neural_policy(model, device, config, *, exploration):
    return NeuralPolicy(model, device, exploration=exploration, focus='neutral',
        batch_size=config['batch_size'], max_candidate_cells=config['max_candidate_cells'],
        evaluator=remote_evaluator(model,config))


def make_policy(kind,model,root,device,config,opponents,mono=None):
    seconds=config['seconds']*(1-config['validation_reserve_fraction'])
    common=SearchConfig(seconds=seconds,proposals=config['proposals'],finalists=config['finalists'],
        pilot_samples=config['pilot_samples'],confirmation_samples=config['confirmation_samples'],turn_interval=1)
    evaluator=remote_evaluator(model,config)
    if kind=='s2':policy=SearchPolicy(model,root/'artifacts/r1-host/guarded',device,config=common,mono=mono,focus='neutral',opponents=opponents,evaluator=evaluator)
    elif kind=='s3':policy=S3Policy(model,root/'artifacts/r1-host/guarded',device,config=common,mono=mono,
        focus='neutral',opponents=opponents,tree=TreeConfig(simulations=config['simulations'],future_turns=config['future_turns']),evaluator=evaluator)
    else:
        require(kind=='r4','Unknown variant')
        from .ismcts import OpenSpielPolicy,SpielSearchConfig
        policy=OpenSpielPolicy(model,root,device,config=SpielSearchConfig(seconds=seconds,
            simulations=config['simulations'],future_turns=config['future_turns'],proposals=config['proposals']),
            mono=mono,focus='neutral',opponents=opponents,evaluator=evaluator)
    # The evaluator is private to this policy; the optional service groups
    # requests by frozen model identity across independent search workers.
    policy.base.evaluator.batch_size=config.get('batch_size',32)
    policy.base.evaluator.max_candidate_cells=config.get('max_candidate_cells',8192)
    if 'execution' in config:
        from brass_ai.recovery.telemetry import instrument_host
        host=policy.game.host if kind=='r4' else policy.host
        instrument_host(host)
        _WORKER.setdefault('host_metrics',[]).append(host.recovery_metrics)
    return policy


def match(root,seed,table,seat,cap,*,mono=None):
    """Measure view/choices, policy, inference and live validation per root turn."""
    group=digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=seed))
    game=dict(game_id=uuid.uuid4().hex,setup_group=group,players=4,split=split_for(group),
              opponents=[p.policy_id for p in table],result=None,records=[])
    rngs=[random.Random(digest(dict(seed=seed,rotation=seat,seat=s))) for s in range(4)]
    with RulesHost(root/'artifacts/r1-host/guarded',mono=mono) as host:
        if _WORKER.get('endpoint'):
            from brass_ai.recovery.telemetry import instrument_host
            instrument_host(host);_WORKER.setdefault('host_metrics',[]).append(host.recovery_metrics)
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


def paired_summary(rows,variants,setups,partition,minimum=30,*,workers=1):
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
        eligible=(workers==1 and variant!='s2' and partition=='validation' and setups>=minimum and interval is not None
            and interval[0]>0 and all(r['over_budget']==0 for r in selected+baseline)
            and not pottery_warning)
        summaries[variant]=dict(games=len(selected),win_share=statistics.mean(r['win'] for r in selected),
            mean_rank=statistics.mean(r['rank'] for r in selected),mean_vp=statistics.mean(r['vp'] for r in selected),
            paired_win_delta=statistics.mean(r['delta'] for r in deltas),paired_setup_bootstrap_95=interval,
            over_budget_turns=sum(r['over_budget'] for r in selected),
            max_turn_seconds=max(r['max_turn_seconds'] for r in selected),pottery_builds=builds,
            pottery_opportunities=opportunities,pottery_warning=pottery_warning,eligible=eligible,
            timing_qualified=workers==1)
    return summaries


_WORKER = {}
_LAST_INFERENCE = {}


def remote_evaluator(model,config):
    if not _WORKER.get('endpoint'): return None
    from brass_ai.recovery.inference import RemoteEvaluator
    from brass_ai.learning.batching import model_id
    identity=_WORKER['model_ids'].get(id(model)) or model_id(model)
    return RemoteEvaluator(_WORKER['endpoint'],identity,config['batch_size'],config['max_candidate_cells'])


def experiment_jobs(function,jobs,workers,root,checkpoint,references,config,device,mono):
    global _LAST_INFERENCE
    args=(str(root),checkpoint,references,config,device,mono)
    _LAST_INFERENCE={}
    if 'execution' not in config:
        yield from run_jobs(function,jobs,workers,init_worker,args)
        return
    from brass_ai.recovery.inference import InferenceService
    with InferenceService(config['execution'],device) as service:
        for entry in [checkpoint]+list(references):
            path=project_path(root,entry['path'])
            require(sha(path)==entry['sha256'],'Frozen experiment source changed')
            require(service.register_checkpoint(path)==entry['model_id'],'Experiment model differs')
        lease=service.channels(workers)
        try:
            yield from run_jobs(function,jobs,workers,init_worker,args+(lease,))
        finally:
            _LAST_INFERENCE=service.report()


def checkpoint_entry(root, path):
    filename=project_path(root,path)
    _,raw=load_checkpoint(filename,device='cpu')
    return dict(path=str(Path(path).as_posix()),sha256=sha(filename),model_id=raw['model_id'])


def init_worker(root,checkpoint,references,config,device,mono,lease=None):
    """Process-local frozen weights; hosts, trees and contexts are never shared."""
    root=Path(root);torch.set_num_threads(config['threads'])
    endpoint=lease.acquire() if lease else None
    if endpoint: device='cpu'
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    def load(entry):
        path=project_path(root,entry['path'])
        require(sha(path)==entry['sha256'],'Frozen experiment checkpoint changed')
        model,raw=load_checkpoint(path,device=device)
        require(raw['model_id']==entry['model_id'],'Frozen model identity changed')
        return model.eval().requires_grad_(False)
    _WORKER.clear()
    _WORKER.update(root=root,model=load(checkpoint),references=[load(p) for p in references],
                   config=config,device=device,mono=mono,endpoint=endpoint)
    _WORKER['model_ids']={id(_WORKER['model']):checkpoint['model_id']}
    _WORKER['model_ids'].update({id(m):e['model_id'] for m,e in zip(_WORKER['references'],references)})


def worker_metrics():
    return dict(pid=os.getpid(),cuda_peak_allocated_bytes=
        torch.cuda.max_memory_allocated() if _WORKER['device']=='cuda' else None,
        hosts=[m.report() for m in _WORKER.get('host_metrics',[])])


def comparison_jobs(seeds,variants):
    jobs=[dict(variant=v,seed=s,seat=p) for s in seeds for p in range(4) for v in variants]
    random.Random(71823).shuffle(jobs)
    return jobs


def comparison_job(job):
    _WORKER['host_metrics']=[]
    root,model,loaded,config,device,mono=(_WORKER[k] for k in ('root','model','references','config','device','mono'))
    variant,s,p=job['variant'],job['seed'],job['seat']
    directory=Path(job['directory']);name=f'{variant}-{s}-seat-{p}'
    record_path=directory/(name+'.json');shard=directory/(name+'.jsonl.gz')
    provenance=dict(producer='searchlab/1.1.0',purpose='evaluation',spec=job['spec'],
                    job=dict(variant=variant,seed=s,seat=p))
    if record_path.exists():
        record=read_json(record_path)
        require(record['spec']==job['spec'] and record['replay_sha256']==sha(shard),'Cached evidence changed')
    elif shard.exists():
        with gzip.open(shard,'rt',encoding='utf-8') as stream:header=json.loads(next(stream))
        require(header['provenance']==provenance,'Orphan replay belongs to another experiment/job')
        episodes=read_shards([shard]);require(len(episodes)==1,'Expected one complete match')
        require(episodes[0]['setup_group']==digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=s)),
                'Cached replay has wrong setup')
        record=dict(spec=job['spec'],result=dict(variant=variant,**result_from_game(episodes[0],s,p,config['seconds'])),
            replay_sha256=sha(shard),telemetry=dict(recovered_complete_replay=True),worker=worker_metrics())
        write_json(record_path,record)
    else:
        try:
            with ExitStack() as stack:
                rollouts=[neural_policy(m,device,config,exploration=.05) for m in loaded]
                candidate=make_policy(variant,model,root,device,config,rollouts,mono)
                stack.callback(candidate.close)
                opponents=[neural_policy(loaded[i%len(loaded)],device,config,exploration=0) for i in range(3)]
                table=opponents.copy();table.insert(p,candidate)
                game,result=match(root,s,table,p,config['seconds'],mono=mono)
                write_shard(shard,[game],provenance)
                record=dict(spec=job['spec'],result=dict(variant=variant,**result),
                    replay_sha256=sha(shard),telemetry=candidate.telemetry,worker=worker_metrics())
                write_json(record_path,record)
        except Exception as error:
            failure=directory/(name+'.failure.json')
            if not failure.exists():write_json(failure,dict(spec=job['spec'],error=str(error)))
            raise
    require(all(record['result'][k]==job[k] for k in ('variant','seed','seat')),'Cached match assignment changed')
    return dict(result=record['result'],name=record_path.name,sha256=sha(record_path),worker=record.get('worker'))


def compare(root,checkpoint,references,output,config,*,setups=None,seed=3910000,partition=None,
            variants=None,device='cpu',mono=None):
    root=Path(root).resolve();config=validate_config(config);gate=require_gate(root,device,mono);directory=project_path(root,output)
    setups=config['setups'] if setups is None else setups
    partition=partition or config['default_partition'];variants=variants or config['default_variants']
    require('s2' in variants and len(set(variants))==len(variants) and set(variants)<= {'s2','r4','s3'},'Include the matched S2 control')
    require(partition in ('validation','test') and type(setups) is int and setups>0 and references,'Held-out setups and frozen opponents required')
    torch.set_num_threads(config['threads'])
    seeds,_=seeds_for(seed,setups,partition)
    spec=dict(format='searchlab-paired/1.1.0',implementation=implementation(root),correctness=gate['identity'],
        checkpoint=checkpoint_entry(root,checkpoint),references=[checkpoint_entry(root,p) for p in references],
        config=config,device=device,variants=list(variants),seeds=seeds,partition=partition,rotations=4)
    spec_path=directory/'spec.json'
    if spec_path.exists():require(read_json(spec_path)==spec,'Comparison specification changed; use a new directory')
    else:
        require(not directory.exists() or not any(directory.iterdir()),'Use a new experiment directory, separate from main training')
        write_json(spec_path,spec)
    rows=[];evidence={};worker_evidence={};began=time.monotonic()
    jobs=[dict(j,directory=str(directory),spec=digest(spec)) for j in comparison_jobs(seeds,variants)]
    workers=min(config['workers'],len(jobs))
    print(f'Comparing {len(jobs)} games with {workers} worker processes; whole-turn cap {config["seconds"]}s',flush=True)
    try:
        for completed in experiment_jobs(comparison_job,jobs,workers,root,spec['checkpoint'],spec['references'],config,device,mono):
            rows.append(completed['result']);evidence[completed['name']]=completed['sha256']
            worker_evidence[completed['name']]=completed['worker']
            print(json.dumps(dict(completed=len(rows),total=len(jobs),variant=rows[-1]['variant'],rank=rows[-1]['rank'])),flush=True)
    finally:_WORKER.clear()
    rows.sort(key=lambda r:(r['seed'],r['seat'],r['variant']))
    failures=len(list(directory.glob('*.failure.json')))
    report=dict(format='searchlab-paired-report/1.1.0',spec=spec,failed_games=failures,results=rows,
        summary=paired_summary(rows,variants,setups,partition,config['promotion_min_setups'],workers=config['workers']),
        evidence=evidence,inference=_LAST_INFERENCE,
        execution=dict(workers=workers,start_method='spawn' if workers>1 else 'in-process',
            seconds=time.monotonic()-began,worker_evidence=worker_evidence,
            mode='parallel-screening' if config['workers']>1 else 'serial-timing',
            note='Cached jobs inflate resume throughput; parallel timing cannot qualify promotion.'))
    if failures:
        for result in report['summary'].values():result['eligible']=False
    write_json(directory/'report.json',report)
    return report


def promote(root,report_path,variant,output,*,device='cpu',mono=None):
    report_path=project_path(root,report_path);report=read_json(report_path)
    gate=require_gate(root,device,mono)
    require(report['spec']['implementation']==implementation(root) and report['spec']['correctness']==gate['identity'],'Stale correctness gate')
    require(variant in ('r4','s3') and report['failed_games']==0,'Variant/failure gate rejected')
    spec=report['spec'];config=validate_config(spec['config'])
    require(config['workers']==1,'Parallel screening requires a separate --workers 1 equal-budget confirmation')
    minimum=max(30,config['promotion_min_setups'])
    recalculated=paired_summary(report['results'],spec['variants'],len(spec['seeds']),spec['partition'],minimum,workers=config['workers'])
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


def distill_job(job):
    _WORKER['host_metrics']=[]
    from brass_ai.learning.runner import collect_games
    root,model,config,device,mono=(_WORKER[k] for k in ('root','model','config','device','mono'))
    directory=Path(job['directory']);path=directory/f'teacher-{job["seed"]}.jsonl.gz'
    record_path=path.with_suffix('').with_suffix('.json')
    provenance=dict(producer='searchlab/1.1.0',purpose='training',variant='s3',spec=job['spec'],
                    seed=job['seed'],sampling_seed=job['sampling_seed'])
    if record_path.exists():
        record=read_json(record_path)
        require(record['provenance']==provenance and record['replay_sha256']==sha(path),'Teacher cache changed')
        return record
    if path.exists():
        with gzip.open(path,'rt',encoding='utf-8') as stream:header=json.loads(next(stream))
        require(header['provenance']==provenance,'Orphan teacher replay differs')
        episodes=read_shards([path],partition='train')
        require(len(episodes)==1 and episodes[0]['setup_group']==digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed'])),
                'Cached teacher setup differs')
        telemetry=dict(recovered_complete_replay=True)
    else:
        rollouts=[neural_policy(model,device,config,exploration=.05)]
        with ExitStack() as stack:
            policy=make_policy('s3',model,root,device,config,rollouts,mono);stack.callback(policy.close)
            episodes=collect_games(root/'artifacts/r1-host/guarded',[job['seed']],[policy]*4,
                workers=1,sampling_seed=job['sampling_seed'],mono=mono)
            write_shard(path,episodes,provenance);telemetry=policy.telemetry
    record=dict(provenance=provenance,replay=path.name,replay_sha256=sha(path),
                telemetry=telemetry,worker=worker_metrics())
    write_json(record_path,record)
    return record


def distill_s3(root,checkpoint,output,config,*,games=None,seed=4920000,device='cpu',mono=None):
    """Parallel frozen S3 teachers, then one coordinator updates a separate copy."""
    root=Path(root).resolve();config=validate_config(config);gate=require_gate(root,device,mono);directory=project_path(root,output)
    games=config['distill_games'] if games is None else games
    require(type(games) is int and games>0,'Positive game count required');torch.set_num_threads(config['threads'])
    source=checkpoint_entry(root,checkpoint);seeds,_=seeds_for(seed,games,'train')
    spec=dict(format='searchlab-distill/1.1.0',purpose='training',variant='s3',checkpoint=source,
        config=config,seeds=seeds,sampling_seed=seed,device=device,implementation=implementation(root),correctness=gate['identity'])
    if (directory/'spec.json').exists():
        require(read_json(directory/'spec.json')==spec,'S3 distillation spec changed; use a new directory')
    else:
        require(not directory.exists() or not any(directory.iterdir()),'Use a new S3 experiment output')
        write_json(directory/'spec.json',spec)
    if (directory/'report.json').exists():
        report=read_json(directory/'report.json')
        require(report['spec']==digest(spec) and report['candidate_sha256']==sha(directory/'candidate.pt'),'S3 result changed')
        for record in report['collection']:
            require(sha(directory/record['replay'])==record['replay_sha256'],'S3 teacher replay changed')
        return report
    jobs=[dict(seed=s,sampling_seed=seed*7+i,directory=str(directory),spec=digest(spec)) for i,s in enumerate(seeds)]
    workers=min(games,config['distill_workers']);began=time.monotonic();records=[]
    print(f'Collecting {games} S3 teacher games with {workers} worker processes',flush=True)
    try:
        for record in experiment_jobs(distill_job,jobs,workers,root,source,[],config,device,mono):
            records.append(record);print(f'S3 teacher games: {len(records)}/{games}',flush=True)
    finally:_WORKER.clear()
    collection_seconds=time.monotonic()-began
    records.sort(key=lambda r:r['provenance']['seed'])
    episodes=read_shards([directory/r['replay'] for r in records],partition='train')
    require(len(episodes)==games and len({g['setup_group'] for g in episodes})==games,'Incomplete or duplicate teacher collection')
    require(sha(project_path(root,checkpoint))==source['sha256'],'Source training checkpoint changed')
    model,raw=load_checkpoint(project_path(root,checkpoint),device=device)
    torch.manual_seed(seed)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'])
    began=time.monotonic()
    if 'update' in config:
        from brass_ai.recovery.updates import distill_update as recovery_distill
        result=recovery_distill(model,optimizer,episodes,config['update'],steps=config['distill_steps'],weight=config['distill_weight'])
    else:
        result=distill_update(model,optimizer,episodes,steps=config['distill_steps'],weight=config['distill_weight'],
                             batch_size=config['minibatch_size'],max_candidate_cells=config['max_candidate_cells'])
    update_seconds=time.monotonic()-began
    save_checkpoint(directory/'candidate.pt',model,optimizer,training=dict(mode='searchlab-s3-distill',
        parent=raw['model_id'],implementation=spec['implementation'],evidence=result,promoted=False))
    require(sha(project_path(root,checkpoint))==source['sha256'],'Source training checkpoint changed')
    report=dict(format='searchlab-distill-report/1.1.0',spec=digest(spec),updated=result['updated'],update=result,
        source_unchanged=True,promoted=False,collection=records,initial_sha256=source['sha256'],
        candidate_sha256=sha(directory/'candidate.pt'),config=config,device=device,implementation=spec['implementation'],
        execution=dict(workers=workers,start_method='spawn' if workers>1 else 'in-process',games=games,
            collection_seconds=collection_seconds,update_seconds=update_seconds,
            update_cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
            note='Cached games are not fresh throughput. Central inference is enabled only by execution schema 1.2.0.'))
    report['inference']=_LAST_INFERENCE
    write_json(directory/'report.json',report);return report


def smoke_job(variant):
    """One accepted live decision, using the same process initializer as games."""
    root,model,config,device,mono=(_WORKER[k] for k in ('root','model','config','device','mono'))
    _WORKER['host_metrics']=[]
    with RulesHost(root/'artifacts/r1-host/guarded',mono=mono) as live, ExitStack() as stack:
        context=live.reset(61302)
        rollouts=[neural_policy(model,device,config,exploration=.05)]
        policy=make_policy(variant,model,root,device,config,rollouts,mono);stack.callback(policy.close)
        began=time.monotonic();view=live.observe(context['actor']);choices=live.choices()
        index,_,teacher=policy.select_batch([(view,choices,random.Random(72050))])[0]
        require(type(index) is int and 0<=index<len(choices),'Illegal search output')
        live.apply(context,choices[index])
        return dict(variant=variant,legal=True,seconds=time.monotonic()-began,
                    teacher=teacher is not None,telemetry=policy.telemetry,worker=worker_metrics())


def quick_smoke(root,checkpoint,config,*,workers=1,device='cpu',mono=None):
    root=Path(root).resolve();config=validate_config(config)
    require(type(workers) is int and workers>0,'Positive smoke worker count required')
    source=checkpoint_entry(root,checkpoint)
    try:
        rows=list(experiment_jobs(smoke_job,['s2','r4','s3'],min(workers,3),root,source,[],config,device,mono))
    finally:_WORKER.clear()
    require(sha(project_path(root,checkpoint))==source['sha256'],'Smoke changed the source checkpoint')
    return dict(format='searchlab-quick-smoke/1.1.0',passed=True,model_id=source['model_id'],
        decisions=sorted(rows,key=lambda r:r['variant']),workers=min(workers,3),inference=_LAST_INFERENCE,promoted=False,
        note='One decision per variant; no complete-game or strength claim.')
