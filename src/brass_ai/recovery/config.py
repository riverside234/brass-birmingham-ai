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


def positive(value, name, integer=False):
    require(type(value) is int if integer else type(value) in (int, float), 'Invalid '+name)
    require(math.isfinite(value) and value > 0, 'Invalid '+name)


def execution_config(config):
    require(set(config) == set(EXECUTION), 'Unknown/missing execution field')
    for key in EXECUTION:
        positive(config[key], key, integer=key != 'flush_ms')
    require(config['flush_ms'] <= 50, 'Excessive inference queue delay')
    return config


def update_config(config):
    require(set(config) == set(UPDATE), 'Unknown/missing update field')
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
    require(set(config)==fields and config['format']=='step5-training/2.0.0' and config['players']==4,
        'Recovery requires the strict four-player 2.0.0 config')
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
