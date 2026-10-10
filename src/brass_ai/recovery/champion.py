"""Frozen incumbent selection, independent of the continuing on-policy learner.

The gate uses two seats per model in ordinary four-player games, never teams.
All six seat pairs share each held-out setup; uncertainty is setup-clustered.
"""
from itertools import combinations
from pathlib import Path
import shutil
import statistics
import time

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import digest,project_path,require
from brass_ai.learning.replay import terminal_targets
from brass_ai.qualification import grouped_interval
from brass_ai.step5.league import seeds_for,write_json,implementation_id
from .execution import neural_spec
from .replay import read_shards


CONTRACT='four-player-champion/1.0.0'


def enabled(config):return config.get('champion',{}).get('enabled',False)


def publish_best(root,run,state):
    """Repairable convenience copy; state points at immutable protected weights."""
    entry=state['champion'];source=project_path(root,entry['path'])
    require(source.is_relative_to(run) and sha(source)==entry['sha256'],'Champion checkpoint changed')
    target=run/'best.pt'
    if not target.exists() or sha(target)!=entry['sha256']:
        temporary=run/'best.pt.tmp';shutil.copyfile(source,temporary);temporary.replace(target)
    write_json(run/'best.json',dict(format=CONTRACT,checkpoint=entry,
        selected_games=state['champion_gate']['accepted_games'],gate=state['champion_gate'].get('accepted_gate'),
        scope='Best accepted policy in this run; not a global-strength or game-release qualification'))


def initialize(root,run,state):
    if not enabled(state['signature']['config']):return
    if 'champion' not in state:
        from .retention import slim_checkpoint
        state['champion']=slim_checkpoint(root,run,state['evaluation_anchor'])
        interval=state['signature']['config']['champion']['every_games']
        state['champion_gate']=dict(next_games=(state['games_done']//interval+1)*interval,
            accepted_games=0,accepted_gate=None,recent=[])
        write_json(run/'state.json',state)
    publish_best(root,run,state)


def selfplay_policies(learner,incumbent):
    # PPO uses only the learner seat. Incumbent actions are opponent data, not
    # mislabeled on-policy samples from the current optimizer's parameters.
    return [learner]+[neural_spec(incumbent['model_id'],.05,'neutral')]*3


def jobs_for(seeds,candidate,incumbent,directory,spec_id):
    a=neural_spec(candidate['model_id'],0,'neutral');b=neural_spec(incumbent['model_id'],0,'neutral')
    jobs=[]
    for seed in seeds:
        for pair in combinations(range(4),2):
            index=len(jobs);sampling_seed=seed*7
            provenance=dict(purpose='evaluation',kind='champion-gate',seed=seed,
                sampling_seed=sampling_seed,spec=spec_id,candidate_seats=list(pair))
            jobs.append(dict(index=index,seed=seed,rotation=0,sampling_seed=sampling_seed,
                policies=[a if s in pair else b for s in range(4)],
                path=str(directory/f'gate-{index:06d}.jsonl.gz'),provenance=provenance))
    return jobs


def gate_rows(games,records,jobs):
    schedule={j['index']:j for j in jobs}
    require(len(records)==len(jobs) and {r['job'] for r in records}==set(schedule),'Incomplete champion receipts')
    by_id={r['game_id']:schedule[r['job']] for r in records}
    require(len(by_id)==len(records) and len(games)==len(jobs) and
            {g['game_id'] for g in games}==set(by_id),'Missing/duplicate champion games')
    rows=[]
    for game in games:
        job=by_id[game['game_id']]
        require(game['split']=='validation' and game['setup_group']==digest(dict(rules='steam-brass-patch-1.1.0',setup_seed=job['seed']))
                and game['opponents']==[p['policy_id'] for p in job['policies']], 'Champion lineup/setup changed')
        targets=terminal_targets(game['result']);pair=job['provenance']['candidate_seats']
        other=[s for s in range(4) if s not in pair]
        row=dict(seed=job['seed'],candidate_seats=pair,win=sum(targets['win_share'][s] for s in pair))
        for key,field in [('vp','vp'),('rank','placement')]:
            row['candidate_'+key]=statistics.mean(targets[field][s] for s in pair)
            row['incumbent_'+key]=statistics.mean(targets[field][s] for s in other)
        rows.append(row)
    return sorted(rows,key=lambda r:(r['seed'],r['candidate_seats']))


def decision(rows,config):
    groups={r['seed'] for r in rows}
    require(len(groups)==config['setups'] and len(rows)==6*len(groups) and
            all(sorted(tuple(r['candidate_seats']) for r in rows if r['seed']==seed)==list(combinations(range(4),2))
                for seed in groups),'Champion gate needs all six seat pairs per setup')
    share=statistics.mean(r['win'] for r in rows);interval=grouped_interval(rows,'win',samples=4000)
    accepted=share>=config['min_win_share'] and interval is not None and interval[0]>.5
    return dict(accepted=accepted,games=len(rows),setups=len(groups),candidate_win_share=share,
        setup_bootstrap_95=interval,threshold=config['min_win_share'],
        **{k:statistics.mean(r[k] for r in rows) for k in ('candidate_vp','incumbent_vp','candidate_rank','incumbent_rank')},
        scope='Two independent seats per model, neutral policy play, no search; repeated training-selection screen, not formal promotion')


def schedule(root,run,state,service,actor):
    config=state['signature']['config']
    if not enabled(config):return None
    gate=state['champion_gate'];cfg=config['champion'];done=state['games_done']
    if done<gate['next_games']:return None
    candidate=state['history'][-1];incumbent=state['champion'];threshold=gate['next_games']
    if candidate['model_id']==incumbent['model_id']:
        gate['next_games']=(done//cfg['every_games']+1)*cfg['every_games']
        write_json(run/'state.json',state)
        return dict(accepted=False,reason='Candidate weights equal incumbent',games=0)
    for entry in (candidate,incumbent):
        path=project_path(root,entry['path'])
        require(path.is_relative_to(run) and sha(path)==entry['sha256'],'Gate checkpoint changed')
        if entry['model_id'] not in service.models:service.register_checkpoint(path)
    # Advancing seeds makes repeat attempts independent; never replay a favorable
    # batch until it passes. At most one gate per committed iteration.
    seeds,_=seeds_for(cfg['seed']+threshold, cfg['setups'],'validation')
    directory=run/'evaluations'/f'champion-{threshold:08d}'
    spec=dict(format=CONTRACT,candidate=candidate,incumbent=incumbent,seeds=seeds,
              config=cfg,implementation=implementation_id(),execution=config['execution'])
    if (directory/'spec.json').exists():require(read(directory/'spec.json')==spec,'Champion gate specification changed')
    else:write_json(directory/'spec.json',spec)
    if (directory/'report.json').exists():
        report=read(directory/'report.json')
        require(report['spec_id']==digest(spec),'Champion report specification changed')
        result=decision(report['outcomes'],cfg)
        require(result==report['result'],'Champion result changed')
    else:
        print(f'Champion gate: generation {candidate["generation"]} vs {incumbent["generation"]}, {6*len(seeds)} policy games',flush=True)
        jobs=jobs_for(seeds,candidate,incumbent,directory,digest(spec));began=time.monotonic()
        records=actor.collect(jobs)
        games=read_shards([r['path'] for r in records],partition='validation',receipts={r['path']:r.get('features') for r in records})
        rows=gate_rows(games,records,jobs);result=decision(rows,cfg)
        report=dict(format=CONTRACT,spec_id=digest(spec),result=result,outcomes=rows,games_done=done,
            failed_games=0,seconds=time.monotonic()-began)
        write_json(directory/'report.json',report)
    receipt=dict(games_done=done,candidate_generation=candidate['generation'],incumbent_generation=incumbent['generation'],
                 **result,report=(directory/'report.json').relative_to(root).as_posix(),report_sha256=sha(directory/'report.json'))
    if result['accepted']:
        from .retention import slim_checkpoint
        state['champion']=slim_checkpoint(root,run,candidate)
        gate.update(accepted_games=done,accepted_gate=receipt)
    gate['next_games']=(done//cfg['every_games']+1)*cfg['every_games']
    gate['recent']=(gate['recent']+[receipt])[-16:]
    write_json(run/'state.json',state)
    publish_best(root,run,state)
    print(f'Champion gate: {"accepted" if result["accepted"] else "retained incumbent"}; candidate win share {result["candidate_win_share"]:.3f}',flush=True)
    return receipt
