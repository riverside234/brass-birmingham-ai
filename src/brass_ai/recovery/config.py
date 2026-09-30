"""Strict recovery schemas; optimizer batch and memory microbatch are distinct."""
import math
from brass_ai.learning.contracts import require
from brass_ai.step5.search import SearchConfig


EXECUTION = dict(processes=8, hosts_per_process=4, heavy_processes=8,
    batch_size=64, max_candidate_cells=131072, flush_ms=2.0, worker_threads=1,
    max_decisions=2000, max_models=16)
UPDATE = dict(effective_batch_size=2048, microbatch_size=512, max_candidate_cells=131072,
    epochs=2, clip=.2, target_kl=.01, entropy_weight=.005, value_weight=.5,
    grad_clip=5.0, normalize_advantages=True, precision='fp32', bucket=True)
CURRICULUM = [
    dict(until=2000, selfplay=40, historical=20, legacy=40),
    dict(until=5000, selfplay=55, historical=25, legacy=20),
    dict(until=10000, selfplay=65, historical=25, legacy=10),
    dict(until=None, selfplay=70, historical=25, legacy=5)]

EXECUTION_V21 = dict(EXECUTION, processes=32, hosts_per_process=1, flush_ms=1.0,
    shard_games=8, shard_bytes=268435456, feature_cache=True, policy_cache=32,search_workers_per_turn=4)
UPDATE_V21 = dict(UPDATE, actor_nonforced=True, critic_batch_size=2048, kl_probe_rows=256)
RETENTION = dict(enabled=True, resume_checkpoints=2, replay_rounds=4, quarantine_hours=24,
    evaluation_samples=4, profile_traces=2)
POPULATION = dict(recent=8, anchors=4)
EVALUATION_SCHEDULE = dict(enabled=True, monitor_every_games=1600, broader_every_games=6400,
    monitor_setups=2, broader_setups=2, seed=910000)
LEGACY_BUDGET = dict(role='lab', seconds=1.0, search_workers=4, global_workers=32,
    work_multiplier=3,calibration='artifacts/search-budgets/lab-v3-calibration.json')


def positive(value, name, integer=False):
    require(type(value) is int if integer else type(value) in (int, float), 'Invalid '+name)
    require(math.isfinite(value) and value > 0, 'Invalid '+name)


def bounded_fields(config, default, name):
    require(type(config) is dict and set(config)==set(default),'Invalid '+name+' fields')
    for key,value in config.items():
        expected=default[key]
        if type(expected) is bool: require(type(value) is bool,'Invalid '+name+'.'+key)
        elif type(expected) in (int,float): positive(value,name+'.'+key,integer=type(expected) is int)
        else: require(type(value) is str and value,'Invalid '+name+'.'+key)
    return config


def legacy_budget_config(config):
    bounded_fields(config,LEGACY_BUDGET,'legacy_budget')
    require(config['role'] in ('lab','laptop','reference'),'Unknown budget role')
    return config


def execution_config(config):
    require(set(config) in (set(EXECUTION),set(EXECUTION_V21)), 'Unknown/missing execution field')
    for key in set(config)-{'feature_cache'}:
        positive(config[key], key, integer=key != 'flush_ms')
    if 'feature_cache' in config: require(type(config['feature_cache']) is bool,'Invalid feature cache flag')
    require(config['flush_ms'] <= 50, 'Excessive inference queue delay')
    return config


def update_config(config):
    require(set(config) in (set(UPDATE),set(UPDATE_V21)), 'Unknown/missing update field')
    if 'actor_nonforced' in config:
        require(type(config['actor_nonforced']) is bool,'Invalid actor sampling flag')
        for key in ('critic_batch_size','kl_probe_rows'): positive(config[key],key,integer=True)
    for key in ('effective_batch_size', 'microbatch_size', 'max_candidate_cells', 'epochs'):
        positive(config[key], key, integer=True)
    for key in ('clip', 'target_kl', 'grad_clip'):
        positive(config[key], key)
    for key in ('entropy_weight', 'value_weight'):
        require(type(config[key]) in (int,float) and math.isfinite(config[key]) and config[key] >= 0, 'Invalid '+key)
    require(0 < config['clip'] < 1 and config['target_kl'] <= .1, 'Invalid PPO guard')
    require(type(config['normalize_advantages']) is bool and type(config['bucket']) is bool, 'Invalid update boolean')
    require(config['precision'] in ('fp32','bf16'), 'Unsupported precision')
    return config


def curriculum_config(stages):
    require(isinstance(stages,list) and stages, 'Missing curriculum')
    previous=0
    for i, stage in enumerate(stages):
        require(set(stage)=={'until','selfplay','historical','legacy'}, 'Invalid curriculum fields')
        require(all(type(stage[k]) is int and 0 <= stage[k] <= 100 for k in ('selfplay','historical','legacy')), 'Invalid table quota')
        require(sum(stage[k] for k in ('selfplay','historical','legacy'))==100, 'Quotas must sum to 100')
        require(all(stage[k]%5==0 for k in ('selfplay','historical','legacy')), 'Twenty-table quotas must use multiples of five')
        if i == len(stages)-1:
            require(stage['until'] is None and stage['legacy']==5, 'Mature curriculum must end at five percent v3 tables')
        else:
            require(type(stage['until']) is int and stage['until'] > previous, 'Unordered curriculum')
            previous=stage['until']
    return stages


def validate_config(config):
    fields={'format','players','model_size','host','games_per_iteration','threads','learning_rate',
        'weight_decay','exploration','seed','execution','update','curriculum','search',
        'search_ready_games','search_every_games','search_games','distill_steps','distill_weight',
        'rehearsal_manifest','rehearsal_weight','rehearsal_until','profile_steps'}
    version=config.get('format')
    if version=='step5-training/2.1.0':
        fields|={'retention','population','evaluation_schedule','legacy_budget','audit_workers',
                 'overlap_pools','search_wall_fraction'}
        for key,default in [('retention',RETENTION),('population',POPULATION),('evaluation_schedule',EVALUATION_SCHEDULE),('legacy_budget',LEGACY_BUDGET)]:
            bounded_fields(config[key],default,key)
        legacy_budget_config(config['legacy_budget'])
        require(config['legacy_budget']['search_workers']*config['execution']['heavy_processes']<=config['legacy_budget']['global_workers'],
                'Heavy search workers exceed global CPU allowance')
        require(config['execution']['search_workers_per_turn']*config['execution']['heavy_processes']<=config['legacy_budget']['global_workers'],
                'Neural search workers exceed global CPU allowance')
        positive(config['audit_workers'],'audit_workers',integer=True)
        require(type(config['overlap_pools']) is bool and 0<config['search_wall_fraction']<=.25,'Invalid overlap/search fraction')
        require(set(config['update'])==set(UPDATE_V21) and set(config['execution'])==set(EXECUTION_V21),'2.1 requires revised update/execution')
    require(set(config)==fields and version in ('step5-training/2.0.0','step5-training/2.1.0') and config['players']==4,
        'Recovery requires a strict four-player 2.0/2.1 config')
    require(config['model_size'] in ('small','medium','large'), 'Unknown model size')
    for key in ('games_per_iteration','threads','search_games','distill_steps','rehearsal_until'):
        positive(config[key],key,integer=True)
    for key in ('search_ready_games','search_every_games','seed','profile_steps'):
        require(type(config[key]) is int and 0 <= config[key] < 2**31, 'Invalid '+key)
    positive(config['learning_rate'],'learning_rate')
    for key in ('weight_decay','exploration','rehearsal_weight'):
        require(type(config[key]) in (int,float) and math.isfinite(config[key]) and 0 <= config[key] <= .5,'Invalid '+key)
    require(0 < config['distill_weight'] <= 1,'Invalid distillation weight')
    require(config['rehearsal_manifest'] is None or isinstance(config['rehearsal_manifest'],str),'Invalid rehearsal manifest')
    execution_config(config['execution']); update_config(config['update'])
    curriculum_config(config['curriculum']); SearchConfig(**config['search']).validate()
    return config


def table_kind(index, stages):
    stage = next(s for s in stages if s['until'] is None or index < s['until'])
    # Twenty evenly spaced quota points, interleaved across tables.
    point = (index * 35) % 100
    if point < stage['selfplay']: return 'selfplay'
    if point < stage['selfplay']+stage['historical']: return 'historical'
    return 'legacy'
