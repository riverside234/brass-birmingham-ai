"""Seat-rotated, setup-clustered neural/legacy comparisons kept out of training."""
from __future__ import annotations

from contextlib import ExitStack
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import statistics
from threading import Event
import time

import torch

from brass_ai.bank import read, sha, require
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import SOURCE_ROOT, digest, project_path
from brass_ai.learning.legacy_policy import LegacyPolicy
from brass_ai.learning.replay import read_shards, write_shard, terminal_targets
from brass_ai.qualification import grouped_interval
from .collector import collect_games
from .league import implementation_id, pottery_metrics, seeds_for, write_json, validate_config
from .policy import NeuralPolicy, INDUSTRIES
from brass_ai.search.common import SearchConfig


def outcome(game, candidate_id, seed):
    require(game['opponents'].count(candidate_id) == 1, 'Evaluation requires one candidate seat')
    seat = game['opponents'].index(candidate_id)
    targets = terminal_targets(game['result'])
    rivals = [v for s, v in enumerate(targets['vp']) if s != seat]
    return dict(seed=seed, seat=seat, win=targets['win_share'][seat],
                rank=targets['placement'][seat], vp=targets['vp'][seat],
                margin=targets['vp'][seat] - statistics.mean(rivals),
                margin_to_best=targets['vp'][seat] - max(rivals))


def summarize_outcomes(rows):
    require(rows, 'No complete evaluation games')
    return dict(games=len(rows), setups=len({r['seed'] for r in rows}),
                win_share=statistics.mean(r['win'] for r in rows),
                win_share_setup_bootstrap_95=grouped_interval(rows, 'win'),
                mean_rank=statistics.mean(r['rank'] for r in rows),
                mean_vp=statistics.mean(r['vp'] for r in rows),
                mean_margin=statistics.mean(r['margin'] for r in rows),
                mean_margin_to_best=statistics.mean(r['margin_to_best'] for r in rows),
                seat_counts=dict(Counter(str(r['seat']) for r in rows)))


def validate_evaluation_config(config):
    expected = {'format', 'players', 'host', 'setups', 'seed', 'partition', 'workers', 'batch_size',
                'max_candidate_cells', 'threads', 'parallel_hosts', 'legacy_workers', 'search_workers',
                'include_legacy', 'use_search', 'focus', 'search'}
    from brass_ai.recovery.config import nn_work_budget_config,legacy_opponent_work_budget_config,time_budget
    for key,validator in (('nn_work_budget',nn_work_budget_config),
                          ('legacy_opponent_work_budget',legacy_opponent_work_budget_config)):
        if key in config:
            expected.add(key);validator(config[key])
    if config.get('format') in ('step5-evaluation-config/1.1.0','step5-evaluation-config/1.3.0'):
        from brass_ai.recovery.config import execution_config
        expected |= {'execution','panel'}
        execution_config(config['execution'])
        require(config['panel'] in ('development','full'),'Invalid evaluation panel')
        ex=config['execution']
        require(config['workers']==ex['processes']*ex['hosts_per_process'] and
            config['legacy_workers']==config['search_workers']==ex['heavy_processes'] and
            config['batch_size']==ex['batch_size'] and config['max_candidate_cells']==ex['max_candidate_cells'],
            'Evaluation aliases differ from central execution config')
        require(config['parallel_hosts'],'Schema 1.1.0 uses private spawned parallel hosts')
    if config.get('format')=='step5-evaluation-config/1.3.0':
        from brass_ai.recovery.config import RETENTION,bounded_fields,legacy_budget_config
        expected|={'lineups','retention','legacy_budget'}
        require(type(config['lineups']) is list and config['lineups'] and len(set(config['lineups']))==len(config['lineups']) and
            set(config['lineups'])<={'recent-neural','mixed-neural','mixed-v3','heldout-pottery-v3'},'Invalid explicit lineups')
        require(config['include_legacy']==any(n.endswith('v3') for n in config['lineups']),'Legacy flag differs from selected lineups')
        require('heldout-pottery-v3' not in config['lineups'] or config['panel']=='full','Holdout belongs to formal panel')
        bounded_fields(config['retention'],RETENTION,'retention');legacy_budget_config(config['legacy_budget'])
        available=config['execution']['cpu_budget']-config['execution']['cpu_reserve']
        require(config['legacy_budget']['global_workers']==available and
            max(config['execution']['search_workers_per_turn'],config['legacy_budget']['search_workers'])<=available,
            'Evaluation search group exceeds queued CPU admission allowance')
    require(set(config) == expected, 'Unknown or missing evaluation config field')
    require(config['format'] in ('step5-evaluation-config/1.3.0',) and config['players'] == 4,
            'Evaluation requires the four-player config')
    for key in ('setups', 'workers', 'batch_size', 'max_candidate_cells', 'threads', 'legacy_workers', 'search_workers'):
        require(type(config[key]) is int and config[key] > 0, 'Invalid evaluation ' + key)
    for key in ('parallel_hosts', 'include_legacy', 'use_search'):
        require(type(config[key]) is bool, 'Invalid evaluation ' + key)
    require(type(config['seed']) is int and 0 <= config['seed'] < 2**31, 'Invalid evaluation seed')
    require(config['partition'] in ('validation', 'test'), 'Evaluation must use held-out setups')
    require(config['focus'] in ('neutral',) + INDUSTRIES, 'Evaluation needs a fixed neural focus')
    SearchConfig(**config['search']).validate()
    if 'nn_work_budget' in config and not time_budget(config['nn_work_budget']):
        count=config['nn_work_budget']['simulations_per_turn']
        workers=config['execution']['search_workers_per_turn']
        require(workers*config['search'].get('minimum_plan_visits',2)<=count<=workers*config['search']['simulations'],
                'Evaluation NN work budget cannot support workers or exceeds simulation cap')
    if config.get('legacy_opponent_work_budget') is not None and not time_budget(config['legacy_opponent_work_budget']):
        count=config['legacy_opponent_work_budget']['simulations_per_turn']
        workers=config['legacy_budget']['search_workers']
        require(count%workers==0 and count//workers>=4,'Evaluation legacy work needs matched four-attempt blocks')
    return config


def evaluation_config(config, **overrides):
    if config.get('format','').startswith('step5-evaluation-config/'):
        result = dict(config)
    elif config.get('format') in ('step5-training/2.2.0',):
        result = read(SOURCE_ROOT / 'configs/step5/development-evaluation.json')
        result.update(host=config['host'],threads=config['threads'],search=config['search'],
            execution=config['execution'])
        ex=config['execution']
        result.update(workers=ex['processes']*ex['hosts_per_process'],legacy_workers=ex['heavy_processes'],
            search_workers=ex['heavy_processes'],batch_size=ex['batch_size'],max_candidate_cells=ex['max_candidate_cells'])
        if config['format']=='step5-training/2.2.0': result.update(retention=config['retention'],legacy_budget=config['legacy_budget'])
        elif result.get('format')=='step5-evaluation-config/1.3.0':
            result['legacy_budget']=dict(result['legacy_budget'],role='reference')
    else:raise ValueError('Historical search/evaluation recipes are retired; choose a fresh 2.2/1.3 config')
    from brass_ai.recovery.config import NN_TIME_BUDGET,LEGACY_TIME_BUDGET,nn_work_budget_config,legacy_opponent_work_budget_config
    # Active evaluations adopt the same versioned clocks. Archived specifications
    # remain distinct and cannot be silently reused in an existing output folder.
    if config.get('nn_work_budget') is not None:nn_work_budget_config(config['nn_work_budget'])
    legacy_opponent_work_budget_config(config.get('legacy_opponent_work_budget'))
    result['nn_work_budget']=dict(NN_TIME_BUDGET)
    result['legacy_opponent_work_budget']=dict(LEGACY_TIME_BUDGET)
    result['search']=dict(result['search'],seconds=NN_TIME_BUDGET['seconds_per_turn'])
    result.update({key: value for key, value in overrides.items() if value is not None})
    if 'lineups' in result and overrides.get('lineups') is None and overrides.get('include_legacy') is not None:
        result['lineups']=[n for n in result['lineups'] if not n.endswith('v3')]
        if result['include_legacy']:
            result['lineups'].append('mixed-v3')
            if result['panel']=='full': result['lineups'].append('heldout-pottery-v3')
    return validate_evaluation_config(result)


def evaluation_jobs(seeds):
    return [dict(seed=seed, seat=seat, sampling_seed=seed * 7 + seat) for seed in seeds for seat in range(4)]


def check_evaluation_batch(games, jobs, candidate_id, partition):
    expected = {(digest(dict(rules='steam-brass-patch-1.1.0', setup_seed=j['seed'])), j['seat']) for j in jobs}
    actual = []
    require(len(expected) == len(jobs), 'Duplicate evaluation job')
    for game in games:
        require(game['split'] == partition, 'Evaluation setup crossed partition')
        require(game['opponents'].count(candidate_id) == 1, 'Evaluation requires one candidate seat')
        actual.append((game['setup_group'], game['opponents'].index(candidate_id)))
    require(len(actual) == len(jobs) and set(actual) == expected, 'Missing, duplicate or misrotated evaluation games')


def evaluate(root,output,candidate_path,references,config,**kwargs):
    from brass_ai.recovery.evaluation import evaluate as run
    return run(root,output,candidate_path,references,config,**kwargs)


def promote(root, checkpoint, report_path, output):
    root = Path(root).resolve(); report = read(project_path(root, report_path))
    require(report['format'] == 'step5-evaluation-report/1.0.0' and report['promotion_eligible'],
            'Candidate has not passed neural-population promotion gates')
    require(sha(project_path(root, checkpoint)) == report['candidate_sha256'], 'Evaluation was for other weights')
    record = dict(format='step5-promoted/1.0.0', checkpoint=checkpoint,
                  checkpoint_sha256=report['candidate_sha256'], report=report_path,
                  report_sha256=sha(project_path(root, report_path)),
                  scope='Neural population validation only; laptop release and human playtests remain separate')
    write_json(project_path(root, output), record)
    return record
