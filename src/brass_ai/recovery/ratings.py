"""Immutable held-out rating slots; never used by PPO or history sampling."""
import copy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import digest,project_path,require
from brass_ai.learning.replay import terminal_targets
from brass_ai.step5.league import seeds_for,write_json,implementation_id
from .plackett_luce import IL,V4,estimate,check_dependencies
from .quality_config import validate_rating

HISTORY_ITERATIONS=100
ARCHIVE_FORMAT='plackett-luce-history/1.0.0'


def archived_history(root,state):
    reference=state.get('archive')
    if reference is None:return dict(format=ARCHIVE_FORMAT,slots=[],games=[])
    path=project_path(root,reference['path'])
    require(sha(path)==reference['sha256'],'Archived rating evidence changed')
    data=read(path)
    require(set(data)=={'format','slots','games'} and data['format']==ARCHIVE_FORMAT and
            isinstance(data['slots'],list) and isinstance(data['games'],list),'Invalid rating history archive')
    generations=[s['generation'] for s in data['slots']]
    require(generations==sorted(set(generations)) and all(type(g) is int and g>0 for g in generations),
            'Invalid archived rating generations')
    require({g['slot'] for g in data['games']}==set(generations),'Archived rating games/slots differ')
    return data


def panel_history(root,state):
    return [s['panels'] for s in archived_history(root,state)['slots']]+[
        read(project_path(root,s['panels']['path'])) for s in state['slots']]


def compact_history(root,run,state):
    """Publish compact cumulative evidence before retiring full slot folders."""
    rating=state['ratings'];cutoff=state['iteration']-HISTORY_ITERATIONS
    # Keep the latest fit available even for a custom cadence over 100 rounds.
    expired=[s for s in rating['slots'] if s['generation']<=cutoff and s['report']!=rating['latest_report']]
    if not expired:return
    from .retention import owned,safe,active
    root,run=owned(root,run)
    require(not active(run,allow_owner=True),'Cannot compact ratings in another active run')
    require(read(run/'state.json')==state,'Run advanced during rating compaction')
    archive=copy.deepcopy(archived_history(root,rating))
    for slot in expired:
        # Exact receipts remain mandatory before any source evidence is retired.
        for key in ('records','report','spec','panels','bootstrap'):
            path=safe(run,project_path(root,slot[key]['path']))
            require(sha(path)==slot[key]['sha256'],'Committed rating evidence changed')
        rows=[json.loads(line) for line in project_path(root,slot['records']['path']).read_text(encoding='utf-8').splitlines()]
        require(rows and all(r['slot']==slot['generation'] for r in rows),'Retired rating games/slot differ')
        archive['games'].extend(rows)
        archive['slots'].append(dict(generation=slot['generation'],
            panels=read(project_path(root,slot['panels']['path'])),
            original_sha256={k:slot[k]['sha256'] for k in ('records','report','spec','panels','bootstrap')}))
    require(len({g['game_id'] for g in archive['games']})==len(archive['games']),'Rating archive repeats a game')
    # Immutable content-addressed publication keeps the old state usable if the
    # process stops before the atomic state replacement. Cleanup runs afterward.
    path=safe(run,run/'ratings'/('history-'+digest(archive)+'.json'))
    write_once(path,archive)
    expired_generations={s['generation'] for s in expired}
    updated=copy.deepcopy(rating)
    updated.update(archive=receipt(root,path),slots=[s for s in rating['slots'] if s['generation'] not in expired_generations])
    committed=dict(state,ratings=updated)
    write_json(run/'state.json',committed)
    state['ratings']=updated


def initialize(root,run,state,config):
    c=validate_rating(config['rating'])
    if c['enabled']:
        check_dependencies()
        try:import matplotlib
        except ImportError as error:raise ValueError('Strength plots need matplotlib; install requirements-step5.txt') from error
    from .legacy_search import legacy_spec
    # Bank admission "evaluation" means the reserved Pottery opponent, not a
    # held-out game. Balanced belongs to the training bank, as in existing
    # mixed-v4 panels; these jobs still carry evaluation provenance/splits.
    v4=legacy_spec(root,c['v4_profile'],config['legacy_budget'],purpose='training',search_free=True,
                   work_budget=config['legacy_opponent_work_budget'])
    from .quality_pool import reference
    anchor=next(m for m in state['population'] if m['generation']==0)
    seeds,_=seeds_for(c['seed'],c['setups'],'validation')
    return dict(contract=c['contract'],config=copy.deepcopy(c),seeds=seeds,slots=[],
        agents={IL:dict(kind='model',checkpoint=reference(anchor)),V4:dict(kind='v4',policy=v4)},
        latest_report=None,scope='Independent held-out evidence; never a sampling score')


def validate_state(root,state,config):
    require(state['contract']==config['contract'] and state['config']==config,'Rating config differs')
    seeds,_=seeds_for(config['seed'],config['setups'],'validation')
    require(state['seeds']==seeds and IL in state['agents'] and V4 in state['agents'],'Rating reference/setup identity differs')
    archived=archived_history(root,state)['slots']
    generations=[s['generation'] for s in archived+state['slots']]
    require(generations==sorted(set(generations)) and all(g%config['every_iterations']==0 for g in generations),
            'Invalid committed rating slots')
    for slot in state['slots']:
        for key in ('records','report','spec','panels','bootstrap'):
            receipt=slot[key];require(sha(project_path(root,receipt['path']))==receipt['sha256'],'Committed rating evidence changed')
    return state


def write_once(path,data):
    """Slot files are immutable; retries must reproduce their exact contents."""
    raw=(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if path.exists():require(path.read_bytes()==raw,'Immutable rating file changed: '+str(path));return
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(raw);temp.replace(path)


def receipt(root,path):return dict(path=path.relative_to(root).as_posix(),sha256=sha(path))


def append_games(path,rows):
    require(len({r['game_id'] for r in rows})==len(rows),'Duplicate rating game id')
    raw=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in sorted(rows,key=lambda r:r['game_id'])).encode()
    if path.exists():require(path.read_bytes()==raw,'Append-only rating slot differs');return
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.jsonl.tmp');temp.write_bytes(raw);temp.replace(path)


def read_games(root,state):
    result=list(archived_history(root,state)['games'])
    for slot in state['slots']:
        item=slot['records'];path=project_path(root,item['path'])
        require(sha(path)==item['sha256'],'Append-only rating games changed')
        result.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
    require(len({r['game_id'] for r in result})==len(result),'Rating store repeats a game')
    return result


def slot_spec(root,run,rating,entry,config):
    generation=entry['generation'];agent=f'generation-{generation:06d}';c=config['rating']
    agents=copy.deepcopy(rating['agents']);agents[agent]=dict(kind='model',checkpoint=copy.deepcopy(entry))
    previous=read(project_path(root,rating['latest_report']['path'])) if rating['latest_report'] else None
    scores=previous['ratings'] if previous else {}
    checkpoints=[a for a in agents if a not in (IL,V4,agent)]
    last=max(checkpoints,key=lambda a:agents[a]['checkpoint']['generation']) if checkpoints else None
    estimate_theta=scores[last]['theta'] if last else 0.
    nearest=sorted(checkpoints,key=lambda a:(abs(scores[a]['theta']-estimate_theta),a))[:c['nearest_opponents']]
    # Startup has no rated checkpoints. Explicit reference fallback keeps the
    # comparison graph connected and never loads an unqualified external model.
    available=nearest or [IL,V4];lineups=[]
    for i,seed in enumerate(rating['seeds']):
        rng=random.Random(digest(dict(seed=c['seed'],setup=seed,slot=generation)))
        opponents=rng.sample(available,2) if len(available)>=2 else available*2
        base=[agent,IL if i%2==0 else V4]+opponents
        for seat in range(4):lineups.append(dict(seed=seed,seat=seat,kind='rating',agents=base))
    for seed in rating['seeds'][:c['panel_setups']]:
        for seat in range(4):
            lineups.append(dict(seed=seed,seat=seat,kind='v4-panel',agents=[agent,V4,V4,V4]))
            lineups.append(dict(seed=seed,seat=seat,kind='il-panel',agents=[agent,agent,IL,IL]))
    used={a for row in lineups for a in row['agents']}
    return dict(format='plackett-luce-slot/1.0.0',generation=generation,candidate=agent,
        agents={a:agents[a] for a in sorted(used)},lineups=lineups,config=copy.deepcopy(c),
        estimated_theta=estimate_theta,estimate_scope='Latest rated checkpoint, or IL zero before the first slot',
        nearest=nearest,startup_reference_fallback=not nearest,implementation=implementation_id(),
        execution=copy.deepcopy(config['execution']),host=config['host'],partition='validation')


def jobs_for(spec,directory):
    from .execution import neural_spec
    jobs=[]
    for i,row in enumerate(spec['lineups']):
        policies=[]
        for agent in row['agents']:
            ref=spec['agents'][agent]
            policies.append(neural_spec(ref['checkpoint']['model_id'],0.,'neutral') if ref['kind']=='model' else ref['policy'])
        provenance=dict(producer='plackett-luce-rating/1.0.0',purpose='evaluation',kind=row['kind'],
            seed=row['seed'],sampling_seed=row['seed']*7+row['seat'],rotation=row['seat'],spec=digest(spec))
        jobs.append(dict(index=i,seed=row['seed'],rotation=row['seat'],sampling_seed=provenance['sampling_seed'],
            policies=policies,path=str(directory/f'game-{i:06d}.jsonl.gz'),provenance=provenance))
    return jobs


def game_record(game,row,spec):
    targets=terminal_targets(game['result']);rng=random.Random(digest(dict(
        contract=spec['config']['contract'],seed=spec['config']['seed'],slot=spec['generation'],
        setup=game['setup_group'],seat=row['seat'],kind=row['kind'])))
    order=[]
    for rank in sorted(set(targets['placement'])):
        tied=[s for s in range(4) if targets['placement'][s]==rank];rng.shuffle(tied);order.extend(tied)
    agents=[row['agents'][(s-row['seat'])%4] for s in range(4)]
    return dict(game_id=game['game_id'],setup_id=game['setup_group'],setup_seed=row['seed'],slot=spec['generation'],
        candidate_seat=row['seat'],kind=row['kind'],agents=agents,order=order,
        official_placement=targets['placement'],vp=targets['vp'],win_share=targets['win_share'],spec_digest=digest(spec))


def collect_slot(root,directory,spec,config,device,mono,inference_devices):
    from .inference import InferenceService
    from .execution import ActorPool
    from .replay import read_shards
    jobs=jobs_for(spec,directory);rows=[]
    with InferenceService(config['execution'],device,inference_devices=inference_devices) as service:
        for ref in spec['agents'].values():
            if ref['kind']=='model':
                e=ref['checkpoint'];path=project_path(root,e['path'])
                require(sha(path)==e['sha256'],'Rated checkpoint changed');service.register_checkpoint(path)
        with ActorPool(root,config['host'],service,config['execution'],mono=mono) as actors:
            records=actors.collect(jobs)
        require(len(records)==len(jobs) and {r['job'] for r in records}==set(range(len(jobs))),
                'Incomplete rating slot')
        loaded={}
        for record in records:
            job=jobs[record['job']]
            require(sha(record['path'])==record['sha256'],'Rating replay changed')
            if record['path'] not in loaded:
                games=read_shards([record['path']],partition='validation',receipts={record['path']:record.get('features')})
                loaded[record['path']]={g['game_id']:g for g in games}
            game=loaded[record['path']][record['game_id']]
            expected=[job['policies'][(s-job['rotation'])%4]['policy_id'] for s in range(4)]
            require(game['opponents']==expected and game['setup_group']==digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed'])),
                    'Rating game seat/setup identity differs')
            rows.append(game_record(game,spec['lineups'][record['job']],spec))
    return rows


def panel_report(rows,candidate,health,seed):
    from brass_ai.qualification import grouped_interval
    result={}
    for name in ('v4-panel','il-panel'):
        games=[r for r in rows if r['kind']==name];samples=[]
        for game in games:
            seats=[s for s,a in enumerate(game['agents']) if a==candidate]
            samples.append(dict(seed=game['setup_seed'],win=sum(game['win_share'][s] for s in seats),
                                vp=sum(game['vp'][s] for s in seats)/len(seats)))
        result[name]=dict(games=len(games),setups=len({r['setup_id'] for r in games}),
            learner_win_share=sum(r['win'] for r in samples)/len(samples),
            win_share_setup_bootstrap_95=grouped_interval(samples,'win',seed=seed),
            vp_per_learner_seat=sum(r['vp'] for r in samples)/len(samples),qualified=None)
    return dict(candidate=candidate,panels=result,all_learner_health=copy.deepcopy(health),
                qualified=None,reason='Deployment requires human review; no invented panel pass thresholds')


def alerts(report,state,panels,previous_panels):
    checkpoints=[a for a in report['ratings'] if a not in (IL,V4)]
    latest=max(checkpoints,key=lambda a:state['agents'][a]['checkpoint']['generation'])
    best=max(checkpoints,key=lambda a:report['ratings'][a]['points'])
    current=report['ratings'][latest];best_rating=report['ratings'][best]['points']
    series=[p['panels']['v4-panel']['learner_win_share'] for p in previous_panels+[panels]]
    slipping=len(series)>=3 and series[-3]>series[-2]>series[-1]
    return dict(regression=bool(current['points_ci95'][1]<best_rating-50),candidate=latest,best=best,
        regression_threshold_points=50,v4_watch=slipping,v4_recent_win_shares=series[-3:],
        v4_watch_action='Review three-slot decline; training mixture stays 50/45/5',
        highest_lower_bound=max(checkpoints,key=lambda a:report['ratings'][a]['points_ci95'][0]),
        deployment_choice=None,deployment_status='Pending separate panel review; no automatic publication')


def plot_report(path,report,agents):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    points=sorted((entry['checkpoint']['generation'],name) for name,entry in agents.items() if entry['kind']=='model' and name!=IL)
    fig,ax=plt.subplots(figsize=(8,4.5))
    if points:
        x=np.asarray([g for g,_ in points]);y=np.asarray([report['ratings'][a]['points'] for _,a in points])
        ci=np.asarray([report['ratings'][a]['points_ci95'] for _,a in points])
        # Percentile intervals can exclude the MAP estimate; draw endpoints
        # directly rather than forcing nonnegative error-bar magnitudes.
        ax.vlines(x,ci[:,0],ci[:,1],alpha=.6,color='#176b9b');ax.plot(x,y,'o-',color='#176b9b',label='Checkpoint (95% setup bootstrap)')
    ax.axhline(0,color='gray',linestyle=':',label='IL = 0')
    v4=report['ratings'][V4];ax.axhline(v4['points'],color='#b65f28',linestyle='--',label='v4, no search')
    ax.axhspan(*v4['points_ci95'],alpha=.10,color='#b65f28')
    ax.set(xlabel='Committed training generation',ylabel='PL points (173.7 × theta)',title='Held-out strength estimates')
    ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(path);plt.close(fig)


def schedule(root,run,state,config,*,device='cpu',mono=None,inference_devices=None):
    rating=validate_state(root,state['ratings'],config['rating']);c=config['rating']
    if not c['enabled']:return None
    completed={s['generation'] for s in archived_history(root,rating)['slots']+rating['slots']}
    available={e['generation']:e for e in state['history']}
    last_report=None
    for generation in range(c['every_iterations'],state['iteration']+1,c['every_iterations']):
        if generation in completed:continue
        require(generation in available,'Missing checkpoint for uncommitted rating slot '+str(generation))
        began=time.perf_counter();entry=available[generation];directory=run/'ratings'/'slots'/f'generation-{generation:06d}'
        from .retention import slim_checkpoint
        retained=state['population']+[a['checkpoint'] for a in rating['agents'].values() if a['kind']=='model']
        previous=next((m for m in retained if m['model_id']==entry['model_id'] and m.get('weights_only')),None)
        entry=slim_checkpoint(root,run,entry,previous)
        spec=slot_spec(root,run,rating,entry,config);write_once(directory/'spec.json',spec)
        records_path=directory/'games.jsonl'
        manifest_path=directory/'games-manifest.json'
        if records_path.exists() and manifest_path.exists():
            manifest=read(manifest_path)
            require(manifest==dict(spec_digest=digest(spec),records=receipt(root,records_path)),
                    'Uncommitted rating game receipt differs')
            rows=[json.loads(line) for line in records_path.read_text(encoding='utf-8').splitlines()]
            expected={(r['kind'],r['seed'],r['seat']) for r in spec['lineups']}
            require(len(rows)==len(expected) and {(r['kind'],r['setup_seed'],r['candidate_seat']) for r in rows}==expected
                    and all(r['slot']==generation and r['spec_digest']==digest(spec) for r in rows),'Incomplete cached rating games')
        else:
            rows=collect_slot(root,directory,spec,config,device,mono,inference_devices);append_games(records_path,rows)
            write_once(manifest_path,dict(spec_digest=digest(spec),records=receipt(root,records_path)))
        working=copy.deepcopy(rating);working['agents'][spec['candidate']]=spec['agents'][spec['candidate']]
        # Only the 128 main games enter PL; diagnostic panels stay separate.
        all_rows=read_games(root,rating)+rows
        fitted,samples=estimate([r for r in all_rows if r['kind']=='rating'],working['agents'],c)
        recent=state.get('recent_training',[])
        health=next((r['il_kl_limit']['health'] for r in reversed(recent) if r.get('il_kl_limit') and r['iteration']+1==generation),None)
        panels=panel_report(rows,spec['candidate'],health,c['seed']);write_once(directory/'panels.json',panels)
        previous_panels=panel_history(root,rating)
        fitted.update(generation=generation,candidate=spec['candidate'],alerts=alerts(fitted,working,panels,previous_panels),
            protocol=dict(main_games=c['setups']*4,panel_games=c['panel_setups']*8,search=False,exploration=0.,focus='neutral'),
            spec=receipt(root,directory/'spec.json'),records=receipt(root,records_path),panels=receipt(root,directory/'panels.json'))
        # Deterministic report bytes make an interrupted commit idempotent.
        write_once(directory/'report.json',fitted)
        samples_path=directory/'bootstrap.npz'
        if samples_path.exists():
            with np.load(samples_path,allow_pickle=False) as old:
                require(np.array_equal(old['theta'],samples) and old['agents'].tolist()==list(fitted['ratings']),
                        'Uncommitted rating bootstrap differs')
        else:
            temporary=samples_path.with_suffix('.tmp')
            with temporary.open('wb') as handle:np.savez(handle,theta=samples,agents=np.asarray(list(fitted['ratings'])))
            temporary.replace(samples_path)
        plot_report(directory/'progress.svg',fitted,working['agents'])
        slot=dict(generation=generation,records=receipt(root,records_path),report=receipt(root,directory/'report.json'),
                  spec=receipt(root,directory/'spec.json'),panels=receipt(root,directory/'panels.json'),
                  bootstrap=receipt(root,samples_path))
        working['slots'].append(slot);working['latest_report']=slot['report'];rating=working;state['ratings']=rating
        write_json(run/'state.json',state)
        print(f'Rating generation {generation}: {c["setups"]*4} main + {c["panel_setups"]*8} panel games; '
              f'{fitted["ratings"][spec["candidate"]]["points"]:.1f} PL points; '
              f'{c["bootstrap"]} bootstrap refits; {time.perf_counter()-began:.1f}s',flush=True)
        last_report=fitted
    if config.get('retention',{}).get('enabled',True):compact_history(root,run,state)
    return last_report
