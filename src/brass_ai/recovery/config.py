"""Strict recovery schemas; optimizer batch and memory microbatch are distinct."""
import math
import copy
from brass_ai.learning.contracts import require
from brass_ai.search.common import SearchConfig

# Training-only defaults; evaluation/laptop SearchConfig defaults stay intact.
SEARCH_EVERY_GAMES=160
SEARCH_COLLECTIONS_PER_ITERATION=2
# None disables only cumulative admission; each reserved turn still has a clock.
SEARCH_WALL_FRACTION=None
SEARCH_SECONDS=12.0
SEARCH_WORKERS=4
# Completed evidence in the saved 12-second lab searches suggests these
# provisional work caps (rounded phase upper quartiles, at least four paths per
# worker). These are attempts across the WHOLE forest/turn, not per worker.
TEACHER_WORK_BUDGET = dict(contract='teacher-turn-work/1.1.0',mode='simulations',watchdog_seconds=120.0,
    simulations_per_phase=dict(canal_early=32,canal_late=16,rail_early=16,rail_late=16))
LEGACY_OPPONENT_WORK_BUDGET = dict(contract='legacy-opponent-turn-work/1.1.0',
    mode='simulations',simulations_per_turn=16,watchdog_seconds=120.0)
NN_WORK_BUDGET = dict(contract='nn-turn-work/1.0.0',mode='simulations',
    simulations_per_turn=16,watchdog_seconds=120.0)
# Keep the work contracts above readable for archived data and explicit small
# comparisons. These versioned whole-turn clocks are the active Step 5 defaults.
TEACHER_TIME_BUDGET_V1 = dict(contract='teacher-turn-time/1.0.0',mode='time',seconds_per_turn=12.0)
TEACHER_PHASE_TIME_BUDGET_V1 = dict(contract='teacher-turn-time/1.1.0',mode='time',
    seconds_per_phase=dict(canal_early=12.0,canal_late=12.0,rail_early=12.0,rail_late=24.0))
TEACHER_PHASE_TIME_BUDGET_V2 = dict(contract='teacher-turn-time/1.1.0',mode='time',
    seconds_per_phase=dict(canal_early=12.0,canal_late=12.0,rail_early=16.0,rail_late=24.0))
TEACHER_TIME_BUDGET = dict(contract='teacher-turn-time/1.1.0',mode='time',
    seconds_per_phase=dict(canal_early=12.0,canal_late=16.0,rail_early=20.0,rail_late=20.0))
LEGACY_TIME_BUDGET = dict(contract='legacy-opponent-turn-time/1.0.0',mode='time',seconds_per_turn=1.0)
NN_TIME_BUDGET = dict(contract='nn-turn-time/1.0.0',mode='time',seconds_per_turn=12.0)
ARCHIVED_TIME_CONFIG_MIGRATION = 'step5-whole-turn-time/1.0.0'
ARCHIVED_PHASE_TIME_CONFIG_MIGRATION = 'step5-phase-teacher-time/1.0.0'
ARCHIVED_EARLY_RAIL_TIME_CONFIG_MIGRATION = 'step5-phase-teacher-time/1.1.0'
TIME_CONFIG_MIGRATION = 'step5-phase-teacher-time/1.2.0'


def time_budget(value):
    return isinstance(value,dict) and value.get('mode')=='time'


def _time_fields(value,default):
    require(value==default and set(value)==set(default) and
            type(value['seconds_per_turn']) in (int,float),'Invalid whole-turn time contract')
    return value
PPO_LEARNING_RATE=1e-5
# A full 80-turn collection supplies 8 batches of 16 using all 80 positions,
# then 48 second presentations. Missing evidence reduces or skips updates;
# a surviving decision is never repeated to fill an otherwise empty batch.
DISTILLATION=dict(min_samples=16,min_games=2,min_samples_per_phase=2,batch_size=16,max_reuse=2)


def teacher_search_config(options):
    """Training defaults without changing standalone/evaluation search clocks."""
    return SearchConfig(**dict({'seconds':SEARCH_SECONDS,'workers':SEARCH_WORKERS,
                               'pending_paths':2,'batch_size':2},**options)).validate()


def teacher_work_budget_config(value):
    """Validate current and archived receipts without relabelling old evidence."""
    if value is None:return None
    if time_budget(value):
        if value.get('contract')==TEACHER_TIME_BUDGET_V1['contract']:
            return _time_fields(value,TEACHER_TIME_BUDGET_V1)
        require(set(value)==set(TEACHER_TIME_BUDGET) and value.get('contract')==TEACHER_TIME_BUDGET['contract'],
                'Invalid phase-specific teacher time contract')
        phases=value['seconds_per_phase']
        require(type(phases) is dict and set(phases)==set(TEACHER_TIME_BUDGET['seconds_per_phase']),
                'Missing teacher time phase')
        require(all(type(n) in (int,float) and math.isfinite(n) and 0<n<=120 for n in phases.values()),
                'Teacher phase seconds must be finite and at most 120')
        return value
    _work_fields(value,TEACHER_WORK_BUDGET,'teacher-turn-work/1.0.0')
    phases=value['simulations_per_phase']
    require(type(phases) is dict and set(phases)==set(TEACHER_WORK_BUDGET['simulations_per_phase']),
            'Missing teacher work phase')
    require(all(type(n) is int and 1<=n<=100000 for n in phases.values()),
            'Invalid teacher phase simulation budget')
    return value


def teacher_time_seconds(value,phase=None):
    """Resolve one shared turn clock; no phase means the maximum failure bound."""
    teacher_work_budget_config(value)
    require(time_budget(value),'Expected teacher time budget')
    if 'seconds_per_turn' in value:return value['seconds_per_turn']
    phases=value['seconds_per_phase']
    return max(phases.values()) if phase is None else phases[phase]


def search_wall_fraction(value):
    require(value is None or (type(value) in (int,float) and math.isfinite(value) and 0<value<=.90),
            'Invalid search fraction: use null/None or a number in (0, .90]')
    return value


def legacy_opponent_work_budget_config(value):
    """Lab-only searched opponents, demonstrations and frozen evaluations."""
    if value is None:return None
    if time_budget(value):return _time_fields(value,LEGACY_TIME_BUDGET)
    _work_fields(value,LEGACY_OPPONENT_WORK_BUDGET,'legacy-opponent-turn-work/1.0.0')
    require(type(value['simulations_per_turn']) is int and 4<=value['simulations_per_turn']<=100000,
            'Invalid legacy opponent simulation budget')
    return value


def _work_fields(value,default,archived=None):
    require(type(value) is dict,'Invalid work budget')
    old=value.get('contract')==archived and archived is not None
    require(set(value)==set(default)-({'watchdog_seconds'} if old else set()),'Invalid work budget fields')
    require(value['contract'] in ((default['contract'],archived) if archived is not None else (default['contract'],)) and value['mode']=='simulations',
            'Unknown work budget contract/mode')
    if not old:
        watchdog=value['watchdog_seconds']
        require(type(watchdog) in (int,float) and math.isfinite(watchdog) and 30<=watchdog<=600,
                'Work watchdog must be finite and between 30 and 600 seconds')


def nn_work_budget_config(value):
    if time_budget(value):return _time_fields(value,NN_TIME_BUDGET)
    _work_fields(value,NN_WORK_BUDGET)
    require(type(value['simulations_per_turn']) is int and 1<=value['simulations_per_turn']<=100000,
            'Invalid NN turn simulation budget')
    return value


def current_work_budget(value,default):
    """Explicitly upgrade active configuration, never stored traces/teacher labels."""
    result=copy.deepcopy(default if value is None else value)
    if 'watchdog_seconds' in default and default['contract'].endswith('/1.1.0') and result.get('contract')==default['contract'].replace('/1.1.0','/1.0.0'):
        result.update(contract=default['contract'],watchdog_seconds=default['watchdog_seconds'])
    return result


def time_budget_config(config):
    """Explicit active-recipe upgrade; same-run signature changes require review."""
    result=archived_early_rail_time_budget_config(config)
    # Upgrade only the exact preceding default, preserving custom phase clocks.
    if result['teacher_work_budget']==TEACHER_PHASE_TIME_BUDGET_V2:
        result['teacher_work_budget']=copy.deepcopy(TEACHER_TIME_BUDGET)
    return result


def archived_early_rail_time_budget_config(config):
    """Keep the preceding 12/12/16/24 migration's receipts reproducible."""
    result=archived_phase_time_budget_config(config)
    if result['teacher_work_budget']==TEACHER_PHASE_TIME_BUDGET_V1:
        result['teacher_work_budget']=copy.deepcopy(TEACHER_PHASE_TIME_BUDGET_V2)
    return result


def archived_phase_time_budget_config(config):
    """Keep the first phase migration's 12/12/12/24 receipts reproducible."""
    result=copy.deepcopy(config)
    work=teacher_work_budget_config(result.get('teacher_work_budget'))
    archived=not work or work.get('contract')!=TEACHER_PHASE_TIME_BUDGET_V1['contract']
    for key,default,validator in (('teacher_work_budget',TEACHER_PHASE_TIME_BUDGET_V1,teacher_work_budget_config),
                        ('legacy_opponent_work_budget',LEGACY_TIME_BUDGET,legacy_opponent_work_budget_config)):
        validator(result.get(key))
        if key!='teacher_work_budget' or archived:result[key]=copy.deepcopy(default)
    result['search']['seconds']=SEARCH_SECONDS
    fraction=search_wall_fraction(result.get('search_wall_fraction',SEARCH_WALL_FRACTION))
    result['search_wall_fraction']=None if archived else fraction
    if archived and result.get('search_games')==4:result['search_games']=5
    return result


def archived_time_budget_config(config):
    """Exact old migration, solely for validating historical source receipts."""
    result=copy.deepcopy(config)
    for key,default,validator in (('teacher_work_budget',TEACHER_TIME_BUDGET_V1,teacher_work_budget_config),
                        ('legacy_opponent_work_budget',LEGACY_TIME_BUDGET,legacy_opponent_work_budget_config)):
        validator(result.get(key));result[key]=copy.deepcopy(default)
    result['search']['seconds']=SEARCH_SECONDS
    fraction=search_wall_fraction(result.get('search_wall_fraction',.50))
    require(fraction is not None,'Archived migration requires a numeric search fraction')
    result['search_wall_fraction']=min(fraction,.50)
    return result


# Import compatibility for preflight scripts; no silent fixed-work continuation.
fixed_work_config=time_budget_config


EXECUTION = dict(processes=8, hosts_per_process=4, heavy_processes=8,
    batch_size=64, max_candidate_cells=131072, flush_ms=2.0, worker_threads=1,
    max_decisions=2000, max_models=72)
UPDATE = dict(effective_batch_size=2048, microbatch_size=512, max_candidate_cells=131072,
    epochs=2, clip=.2, target_kl=.01, entropy_weight=.005, value_weight=.5,
    grad_clip=5.0, normalize_advantages=True, precision='fp32', bucket=True)
CURRICULUM = [
    dict(until=2000, selfplay=40, historical=20, legacy=40),
    dict(until=5000, selfplay=55, historical=25, legacy=20),
    dict(until=10000, selfplay=65, historical=25, legacy=10),
    dict(until=None, selfplay=70, historical=25, legacy=5)]

EXECUTION_V21 = dict(EXECUTION, processes=24, hosts_per_process=2, flush_ms=1.0,
    shard_games=8, shard_bytes=268435456, feature_cache=True, policy_cache=32,search_workers_per_turn=4,
    cpu_budget=64,cpu_reserve=4)
UPDATE_V21 = dict(UPDATE, actor_nonforced=True, critic_batch_size=2048, kl_probe_rows=2048)
RETENTION = dict(enabled=True, resume_checkpoints=2, replay_rounds=4, quarantine_hours=24,
    evaluation_samples=4, profile_traces=2, history_reports=128)
CHAMPION_LEGACY = dict(enabled=True, every_games=6400, setups=8, min_win_share=.55, seed=1910000)
CHAMPION_PROTOCOL = 'two-stage-champion/1.0.0'
CHAMPION = dict(CHAMPION_LEGACY, protocol=CHAMPION_PROTOCOL, confirmation_setups=64,
    screen_min_win_share=.56, min_ci_lower=.49)
POPULATION = dict(recent=16, anchors=48)
EVALUATION_SCHEDULE = dict(enabled=True, monitor_every_games=6400, broader_every_games=25600,
    monitor_setups=1, broader_setups=1, seed=910000)
LEGACY_BUDGET = dict(role='lab', seconds=1.0, search_workers=4, global_workers=60, work_multiplier=1)
LEGACY_OPPONENTS = dict(search_free_until_games=10000)
IMITATION = dict(manifest=None,epochs=10,patience=2,min_epochs=3,learning_rate=1e-4)
INSOLVENCY = dict(event_cost=.005,max_cost=.075)
CHAMPION_OPPONENTS = dict(contract='staged-champion-opponents/1.0.0',stage_iterations=5,max_champion_opponents=2)
CHAMPION_MIX_CONFIG_MIGRATION = 'step5-staged-champion-opponents/1.0.0'
LEGACY_BOOST = dict(stages=[dict(games=20000,selfplay=15,historical=25,legacy=60),
                          dict(games=10000,selfplay=45,historical=25,legacy=30)])


def champion_mix_config(config):
    """Reviewed upgrade of the preceding default; keep custom reward recipes."""
    result=copy.deepcopy(config)
    if result.get('champion',{}).get('enabled'):
        result.setdefault('champion_opponents',copy.deepcopy(CHAMPION_OPPONENTS))
    if result.get('insolvency')==dict(event_cost=.005,max_cost=.025):
        old=result.get('search',{}).get('reward')
        if old is not None:
            require(old['insolvency']==result['insolvency'],'Search/learner insolvency differs')
            old['insolvency']=copy.deepcopy(INSOLVENCY)
        result['insolvency']=copy.deepcopy(INSOLVENCY)
    return result


def positive(value, name, integer=False):
    require(type(value) is int if integer else type(value) in (int, float), 'Invalid '+name)
    require(math.isfinite(value) and value > 0, 'Invalid '+name)


def bounded_fields(config, default, name):
    allowed=[set(default)]
    if name=='retention':allowed.append(set(default)-{'history_reports'})
    require(type(config) is dict and set(config) in allowed,'Invalid '+name+' fields')
    for key,value in config.items():
        expected=default[key]
        if type(expected) is bool: require(type(value) is bool,'Invalid '+name+'.'+key)
        elif name=='retention' and key=='quarantine_hours':
            require(type(value) is int and value>=0,'Invalid retention.quarantine_hours')
        elif type(expected) in (int,float): positive(value,name+'.'+key,integer=type(expected) is int)
        else: require(type(value) is str and value,'Invalid '+name+'.'+key)
    return config


def legacy_budget_config(config):
    bounded_fields(config,LEGACY_BUDGET,'legacy_budget')
    require(config['role'] in ('lab','laptop','reference'),'Unknown budget role')
    require(config['seconds']==1.0,'Training/evaluation v3 allowance must be exactly one second')
    require(config['search_workers']<=4 and config['work_multiplier']<=16,'Invalid legacy CPU allowance')
    return config


def champion_config(config):
    """Keep archived single-stage evidence readable; new recipes opt in explicitly."""
    staged='protocol' in config
    bounded_fields(config,CHAMPION if staged else CHAMPION_LEGACY,'champion')
    require(config['setups']>=8 and .5<config['min_win_share']<1,
            'Champion gate requires at least eight independent setups and a winning majority')
    if staged:
        require(config['protocol']==CHAMPION_PROTOCOL,'Unknown champion protocol')
        require(config['setups']==8 and config['confirmation_setups']==64,
                'Two-stage champion gate requires 8 screening and 64 new confirmation setups')
        require(.5<config['screen_min_win_share']<1 and 0<config['min_ci_lower']<1,
                'Invalid champion screening/confidence thresholds')
    return config


def execution_config(config):
    optional={'shared_slot_bytes','inference_routing','replica_backlog'}
    require(set(config)-optional in (set(EXECUTION),set(EXECUTION_V21)), 'Unknown/missing execution field')
    for key in set(config)-{'feature_cache','shared_slot_bytes','inference_routing'}:
        positive(config[key], key, integer=key != 'flush_ms')
    if 'shared_slot_bytes' in config:
        require(type(config['shared_slot_bytes']) is int and (config['shared_slot_bytes']==0 or 65536<=config['shared_slot_bytes']<=16777216),'Invalid shared slot capacity')
    if 'inference_routing' in config:require(config['inference_routing'] in ('demand','round_robin'),'Invalid inference routing')
    if 'feature_cache' in config: require(type(config['feature_cache']) is bool,'Invalid feature cache flag')
    require(config['flush_ms'] <= 50, 'Excessive inference queue delay')
    if 'cpu_budget' in config:
        require(config['cpu_reserve']<config['cpu_budget'],'CPU reserve exceeds allocation')
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
    if 'legacy_opponent_work_budget' in config:
        fields.add('legacy_opponent_work_budget')
        legacy_work=legacy_opponent_work_budget_config(config['legacy_opponent_work_budget'])
        if legacy_work is not None and not time_budget(legacy_work):
            workers=config['legacy_budget']['search_workers']
            count=legacy_work['simulations_per_turn']
            require(type(workers) is int and 1<=workers<=4,'Invalid legacy work worker count')
            require(count%workers==0 and count//workers>=4,
                    'Legacy work budget must retain a four-attempt paired block per worker')
    if 'teacher_work_budget' in config:
        fields.add('teacher_work_budget')
        work=teacher_work_budget_config(config['teacher_work_budget'])
        if work is not None and not time_budget(work):
            search=teacher_search_config(config['search'])
            workers=config['execution'].get('search_workers_per_turn',search.workers)
            require(all(workers*search.minimum_plan_visits<=n<=workers*search.simulations
                        for n in work['simulations_per_phase'].values()),
                    'Teacher work budget cannot support workers or exceeds simulation cap')
    if 'champion' in config:
        fields.add('champion')
        champion_config(config['champion'])
    if 'champion_opponents' in config:
        fields.add('champion_opponents')
        mix=config['champion_opponents']
        require(set(mix)==set(CHAMPION_OPPONENTS) and mix['contract']==CHAMPION_OPPONENTS['contract'],
                'Invalid champion opponent mixture')
        require(type(mix['stage_iterations']) is int and mix['stage_iterations']>0 and
                type(mix['max_champion_opponents']) is int and mix['max_champion_opponents'] in (1,2),
                'Champion rollout requires positive stages and a one/two opponent limit')
        require(config.get('champion',{}).get('enabled'),'Champion mixture needs champion selection')
    if 'reward' in config:
        fields.add('reward')
        from .utility import reward_spec
        spec=reward_spec(config)
        require(config['search'].get('reward')==spec and config['search'].get('contract')=='masked-ismcts/1.2.0',
                'PPO/search reward contracts must match')
    if 'insolvency' in config:
        fields.add('insolvency')
        bounded_fields(config['insolvency'],INSOLVENCY,'insolvency')
        require(config['insolvency']['event_cost']<=config['insolvency']['max_cost']<=.1,
                'Insolvency costs must stay small relative to winning')
    if 'legacy_boost' in config:
        fields.add('legacy_boost')
        boost=config['legacy_boost']
        require(type(boost) is dict and set(boost)=={'stages'} and type(boost['stages']) is list and
                1<=len(boost['stages'])<=2,'Invalid searched-v4 stages')
        for stage in boost['stages']:
            require(type(stage) is dict and set(stage)=={'games','selfplay','historical','legacy'},
                    'Invalid searched-v4 stage fields')
            positive(stage['games'],'legacy_boost.games',integer=True)
            require(stage['games']%20==0 and
                    all(type(stage[k]) is int and 0<=stage[k]<=100 and stage[k]%5==0
                        for k in ('selfplay','historical','legacy')) and
                    sum(stage[k] for k in ('selfplay','historical','legacy'))==100,
                    'Invalid bounded searched-v4 quota window')
        require(sum(s['games'] for s in boost['stages'])<=30000,'Excessive searched-v4 recovery duration')
    if 'search_collections_per_iteration' in config:
        fields.add('search_collections_per_iteration')
        require(type(config['search_collections_per_iteration']) is int and
                1<=config['search_collections_per_iteration']<=4,'Invalid teacher collection limit')
    if 'distillation' in config:
        fields.add('distillation')
        bounded_fields(config['distillation'],DISTILLATION,'distillation')
    if version=='step5-training/2.2.0':
        fields|={'retention','population','evaluation_schedule','legacy_budget','audit_workers',
                 'overlap_pools','search_wall_fraction','initialization','architecture','parameter_limit','imitation','legacy_opponents'}
        require(config['initialization']=='random' and config['architecture']=='brass-four-seat-mlp-resnet/1.0.0'
                and config['parameter_limit']==2000000,'Fresh residual-model contract required')
        require(type(config['imitation']) is dict and set(config['imitation'])==set(IMITATION),'Invalid imitation fields')
        imitation=config['imitation']
        require(imitation['manifest'] is None or type(imitation['manifest']) is str,'Invalid imitation manifest')
        for key in ('epochs','patience','min_epochs'):positive(imitation[key],key,integer=True)
        positive(imitation['learning_rate'],'imitation.learning_rate')
        require(imitation['epochs']>=imitation['min_epochs'],'Imitation minimum exceeds epochs')
        for key,default in [('retention',RETENTION),('population',POPULATION),('evaluation_schedule',EVALUATION_SCHEDULE),('legacy_budget',LEGACY_BUDGET)]:
            bounded_fields(config[key],default,key)
        legacy_budget_config(config['legacy_budget'])
        bounded_fields(config['legacy_opponents'],LEGACY_OPPONENTS,'legacy_opponents')
        available=config['execution']['cpu_budget']-config['execution']['cpu_reserve']
        require(config['legacy_budget']['global_workers']==available and
                max(config['legacy_budget']['search_workers'],config['execution']['search_workers_per_turn'])<=available,
                'Search group exceeds queued CPU admission allowance')
        positive(config['audit_workers'],'audit_workers',integer=True)
        # Historical configs remain readable; active entry points explicitly
        # migrate the cumulative cap before signing/resuming the run.
        require(type(config['overlap_pools']) is bool,'Invalid overlap flag')
        search_wall_fraction(config['search_wall_fraction'])
        require(config['execution']['max_models']>=sum(config['population'].values())+3,
                'Model cache must cover frozen population plus learner/teacher versions')
        require(set(config['update'])==set(UPDATE_V21) and
                set(config['execution'])-{'shared_slot_bytes','inference_routing','replica_backlog'}==set(EXECUTION_V21),
                '2.1 requires revised update/execution')
    require(set(config)==fields and version=='step5-training/2.2.0' and config['players']==4,
        'Use the fresh four-player Masked ISMCTS 2.2 recipe; historical runs cannot migrate in place')
    require(config['model_size'] in ('small','medium'), 'Residual model size must fit the 2M cap')
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


def table_rotation(index):
    """Rotate each 20-table quota position through all four physical seats.

    index % 4 aliases the quota period: at five percent legacy tables the
    learner otherwise always occupies the same seat. This stays resumable from
    the absolute game counter and changes no quota or setup-seed allocation.
    """
    return (index + index // 20) % 4
