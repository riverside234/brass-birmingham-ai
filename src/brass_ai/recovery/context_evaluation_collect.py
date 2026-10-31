"""Validated replay -> candidate-only compact evidence; no training tensors."""
import copy
import gzip
import json
import time
from collections import Counter
from pathlib import Path
from brass_ai.bank import sha
from brass_ai.learning.contracts import digest, project_path, require
from brass_ai.learning.replay import terminal_targets
from brass_ai.learning.reward import terminal_components, terminal_utility
from brass_ai.step5.policy import context_for
from . import domino_features as features
from .context_evaluation_config import CONTRACT
from .domino_config import NAMES
from .ratings import write_once, receipt


def summary(game,job,spec,context_id,parent_index,replay):
    require(not any(k in game for k in ('branch','branches','branch_statistics')) and
        not job.get('branching') and 'diversity_assignment' not in job['provenance'],
        'Training branching/random contexts are forbidden in context evaluation')
    parent=spec['parent'];line=parent['lineups'][parent_index];seat=line['seat']
    require(line['kind']=='rating' and line['agents'].count(parent['candidate'])==1 and
        job['rotation']==seat and job['seed']==line['seed'] and
        job['sampling_seed']==line['seed']*7+seat and job.get('rng_index',0)==0,
        'Context evaluation RNG/rotation differs from neutral')
    require(game['setup_group']==digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed'])),
            'Context setup differs')
    assigned=[job['policies'][(s-seat)%4] for s in range(4)]
    require(game['opponents']==[p['policy_id'] for p in assigned], 'Context opponents differ')
    from .execution import neural_spec
    expected=[]
    for k,a in enumerate(line['agents']):
        ref=parent['agents'][a]
        p=(neural_spec(ref['checkpoint']['model_id'],0.,NAMES[context_id] if k==0 else 'neutral',temperature=1.)
            if ref['kind']=='model' else copy.deepcopy(ref['policy']))
        if ref['kind']=='v4' and p.get('search_runtime'):p['frozen_rating']=True
        expected.append(p)
    require(job['policies']==expected, 'Context policy/checkpoint/search settings differ')
    events=[];actions=Counter();industry={'build':Counter(),'develop':Counter()};trace=[]
    for row in game['records']:
        p=assigned[row['seat']]
        if p['kind']=='neural':
            style,intent=context_for(p['focus']);b=row['behavior']
            require(row['style']==style and row['intent']==intent and b['kind']=='model' and
                b['policy_id']==p['policy_id'] and b['temperature']==1. and
                b.get('exploration',0.)==0. and row['teacher'] is None and
                0.<b['probability']<=1., 'Replay context/behavior does not match panel job')
        if row['seat']!=seat:continue
        choice=row['choices'][row['chosen']];kind=choice['kind']
        # Development can remove a second tile in a continuation. Count both
        # industry selections, while action-family frequency stays main-only.
        if kind in industry:
            require(choice.get('industry') in features.INDUSTRIES,'Missing build/develop industry')
            industry[kind][choice['industry']]+=1
        event=features.event(row)
        if event is None:continue
        events.append(event);actions[kind]+=1
        trace.append(dict(step=row['step'],phase=features.PHASES[event['phase']],choice=choice,
            cash=row['view']['players'][seat]['cash'],income=row['view']['players'][seat]['income']))
    require(events,'Candidate has no main actions')
    psi,counts=features.occupancy(events);targets=terminal_targets(game['result'])
    from .insolvency import returns
    _,costs=returns(game['records'],spec['reward']['insolvency']);cost=costs[seat]
    utility=terminal_utility(game['result'],spec['reward'])[seat]-cost['cost']
    # No intrinsic reward, and the capped cost is subtracted exactly once.
    return dict(contract=CONTRACT,generation=spec['generation'],panel_digest=digest(spec),
        parent_digest=spec['parent_digest'],parent_index=parent_index,context_id=context_id,context=NAMES[context_id],
        game_id=game['game_id'],setup_id=game['setup_group'],setup_seed=job['seed'],seat=seat,
        sampling_seed=job['sampling_seed'],rng_index=job.get('rng_index',0),
        opponents=game['opponents'],policies=copy.deepcopy(job['policies']),
        candidate=copy.deepcopy(parent['agents'][parent['candidate']]['checkpoint']),
        runtime=spec['runtime'],implementation=spec['implementation'],replay=replay,
        win_share=targets['win_share'][seat],vp=targets['vp'][seat],placement=targets['placement'][seat],
        outrank_share=terminal_components(game['result'])['placement'][seat],task_return=float(utility),
        insolvency_events=cost['events'],insolvent=int(cost['events']>0),insolvency_cost=cost['cost'],
        zero_vp=int(targets['vp'][seat]==0),passes=actions['pass'],main_actions=sum(actions.values()),
        actions=dict(actions),industry={k:dict(v) for k,v in industry.items()},
        descriptor=psi.tolist(),phase_counts=counts,missing_phases=[features.PHASES[i] for i,n in enumerate(counts) if not n],
        trace=trace)


def verify_summary(row,job,spec,z,index):
    require(row['panel_digest']==digest(spec) and row['parent_digest']==spec['parent_digest'] and
        row['parent_index']==index and row['context_id']==z and row['context']==NAMES[z] and
        row['runtime']==spec['runtime'] and row['implementation']==spec['implementation'] and
        row['candidate']==spec['parent']['agents'][spec['parent']['candidate']]['checkpoint'] and
        row['policies']==job['policies'] and row['opponents']==[job['policies'][(s-job['rotation'])%4]['policy_id'] for s in range(4)] and
        row['sampling_seed']==job['sampling_seed'] and row['rng_index']==job.get('rng_index',0) and
        row['setup_seed']==job['seed'] and row['seat']==job['rotation'], 'Incompatible context summary receipt')


def extract_records(root,records,jobs,spec,directory,*,neutral=False,on_progress=None):
    from .replay import read_shards
    by_index={j['index']:j for j in jobs};groups={}
    for record in records:groups.setdefault(record['path'],[]).append(record)
    result=[];began=time.perf_counter()
    for path,items in groups.items():
        checksum=sha(path);require(all(checksum==r['sha256'] for r in items),'Context replay changed')
        with gzip.open(path,'rt',encoding='utf-8') as stream:header=json.loads(next(stream))
        proof=header['provenance'];recorded={r['index']:r for r in proof['jobs']}
        games={g['game_id']:g for g in read_shards([path],partition='validation',receipts={path:items[0].get('features')})}
        for r in items:
            job=by_index[r['job']]
            require(recorded[job['index']]['provenance']==job['provenance'],'Replay job provenance differs')
            z=0 if neutral else job['provenance']['context_id']
            index=job['index'] if neutral else job['provenance']['parent_index']
            value=summary(games[r['game_id']],job,spec,z,index,dict(path=Path(path).relative_to(root).as_posix(),sha256=checksum))
            save_summary(directory,value);result.append(value)
        if on_progress:on_progress(len(result))
    return result,time.perf_counter()-began


def save_summary(directory,value):
    path=directory/'evidence'/f'context-{value["context_id"]}-parent-{value["parent_index"]:06d}.json'
    write_once(path,value)
    write_once(path.with_suffix('.receipt.json'),dict(summary=receipt(directory,path)))


def load_summaries(directory,spec,*,neutral_only=False):
    from brass_ai.bank import read
    from .context_evaluation_spec import jobs_for
    from .ratings import jobs_for as base_jobs
    jobs={(0,i):j for i,j in enumerate(base_jobs(spec['parent'],directory))
        if i in {i for s in spec['selection']['setups'] for i in s['indices']}}
    if not neutral_only:
        jobs.update({(j['provenance']['context_id'],j['provenance']['parent_index']):j for j in jobs_for(spec,directory/'raw')})
    values=[]
    for (z,i),job in sorted(jobs.items()):
        path=directory/'evidence'/f'context-{z}-parent-{i:06d}.json'
        if not path.exists():continue
        require(path.with_suffix('.receipt.json').is_file() and read(path.with_suffix('.receipt.json'))==
            dict(summary=receipt(directory,path)), 'Missing/changed context summary receipt')
        value=read(path);verify_summary(value,job,spec,z,i);values.append(value)
    return values


def collect(root,directory,spec,*,device,mono=None,inference_devices=None,on_records=None):
    from .inference import InferenceService
    from .execution import ActorPool
    from .context_evaluation_spec import jobs_for,validate_runtime
    validate_runtime(root,spec);jobs=jobs_for(spec,directory/'raw');began=time.perf_counter();extraction=0.;counts=Counter()
    sampled_rss=[];disk_samples=[]
    # The same complete ordered job recipe on every retry lets ActorPool verify
    # completed shards and run only missing jobs, even after Ctrl-C.
    def ingest(records):
        nonlocal extraction
        _,elapsed=extract_records(root,records,jobs,spec,directory)
        extraction+=elapsed
        counts['unique']+=len(records);counts['cached']+=sum(bool(r['cached']) for r in records)
        from .telemetry import memory
        rss=memory('cpu').get('host_resident_bytes')
        if rss is not None:sampled_rss.append(rss)
        disk_samples.append(sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()))
        if on_records:on_records(records)
    with InferenceService(spec['execution'],device,inference_devices=inference_devices) as service:
        for ref in spec['parent']['agents'].values():
            if ref['kind']=='model':
                e=ref['checkpoint']
                if e['model_id'] not in service.models:service.register_checkpoint(project_path(root,e['path']))
        with ActorPool(root,spec['parent']['host'],service,spec['execution'],mono=mono) as actors:
            records=actors.collect(jobs,on_records=ingest)
        inference=service.report()
    require(len(records)==len(jobs),'Incomplete context collection')
    elapsed=time.perf_counter()-began;fresh=counts['unique']-counts['cached']
    return dict(collection_seconds=elapsed-extraction,extraction_seconds=extraction,collection_total_seconds=elapsed,
        new_games_per_second=fresh/elapsed if elapsed>0 else None,
        sampled_peak_host_rss_bytes=max(sampled_rss) if sampled_rss else None,
        sampled_peak_disk_bytes=max(disk_samples) if disk_samples else None,
        memory_scope='Main-process RSS sampled after completed shard extraction; GPU allocator peaks in inference.memory '
            'are process-lifetime counters, not isolated panel peaks. No profiler counters are reset.',
        reused_completed_jobs=counts['cached'],new_completed_jobs=counts['unique']-counts['cached'],
        unique_games=counts['unique'],inference=inference,
        actor_worker_seconds=sum(r.get('worker_seconds',0) for r in records),
        actor_python_cpu_seconds=sum(r.get('actor_python_cpu_seconds') or 0 for r in records),
        scope='Additional six contexts only; CPU time excludes native host processes')
