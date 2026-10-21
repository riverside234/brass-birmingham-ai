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
TEACHER_SEARCH_WORKERS=8
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
TEACHER_PHASE_TIME_BUDGET_V3 = dict(contract='teacher-turn-time/1.1.0',mode='time',
    seconds_per_phase=dict(canal_early=12.0,canal_late=16.0,rail_early=20.0,rail_late=20.0))
TEACHER_PHASE_TIME_BUDGET_V4 = dict(contract='teacher-turn-time/1.1.0',mode='time',
    seconds_per_phase=dict(canal_early=9.0,canal_late=9.0,rail_early=12.0,rail_late=16.0))
TEACHER_PHASE_TIME_BUDGET_V5 = dict(contract='teacher-turn-time/1.1.0',mode='time',
    seconds_per_phase=dict(canal_early=9.0,canal_late=12.0,rail_early=12.0,rail_late=12.0))
TEACHER_TIME_BUDGET = copy.deepcopy(TEACHER_PHASE_TIME_BUDGET_V5)
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
# Enabled in fresh self-play recipes; no additional games or optimizer steps.
# Diagnostics are independently opt-in
# until their full-iteration overhead has been measured on representative runs.
TEACHER_VALUE=dict(contract='teacher-outcome-value/1.0.0',enabled=True,loss_weight=.1,
    teacher_fraction=.25,samples_per_game=24,max_samples=128,max_reuse=2,
    diagnostics=False,probe_positions=128)


def teacher_value_config(value=None):
    # Fresh recipes explicitly write TEACHER_VALUE. Omission belongs to older
    # signed configurations and must retain their policy-only teaching behavior.
    value=copy.deepcopy(dict(TEACHER_VALUE,enabled=False) if value is None else value)
    require(type(value) is dict and set(value)==set(TEACHER_VALUE) and
            value['contract']==TEACHER_VALUE['contract'],'Invalid teacher auxiliary value contract')
    require(all(type(value[k]) is bool for k in ('enabled','diagnostics')),'Invalid teacher value switches')
    for key,limit in (('samples_per_game',128),('max_samples',128),('max_reuse',2),('probe_positions',128)):
        require(type(value[key]) is int and 1<=value[key]<=limit,'Invalid bounded teacher value '+key)
    for key in ('loss_weight','teacher_fraction'):
        require(type(value[key]) in (int,float) and math.isfinite(value[key]) and 0<value[key]<1,
                'Invalid teacher value '+key)
    return value


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
    if latest_model_recipe(config):
        # This recipe already pins its clocks. Do not replay an archived clock
        # migration merely because its explicit values match an older default.
        teacher_work_budget_config(config['teacher_work_budget'])
        return copy.deepcopy(config)
    result=archived_early_rail_time_budget_config(config)
    # Upgrade only the exact preceding default, preserving custom phase clocks.
    if result['teacher_work_budget']==TEACHER_PHASE_TIME_BUDGET_V2:
        result['teacher_work_budget']=copy.deepcopy(TEACHER_PHASE_TIME_BUDGET_V3)
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
    grad_clip=1.0, normalize_advantages=True, precision='fp32', bucket=True)
ARCHIVED_CURRICULUM = [
    dict(until=20000, selfplay=15, historical=25, legacy=60),
    dict(until=100000, selfplay=45, historical=25, legacy=30),
    dict(until=None, selfplay=60, historical=25, legacy=15)]
PREVIOUS_FIXED_CURRICULUM = [dict(until=None,selfplay=60,historical=30,legacy=10)]
CURRICULUM = [dict(until=None,selfplay=50,historical=45,legacy=5)]
ARCHIVED_LATEST_MODEL_RECIPE='latest-model-selfplay/1.0.0'
PREVIOUS_FIXED_RECIPE='latest-model-selfplay/2.0.0'
LATEST_MODEL_RECIPE='latest-model-selfplay/3.0.0'
ORDINARY_LEARNER_EXPLORATION=.03

EXECUTION_V21 = dict(EXECUTION, processes=24, hosts_per_process=2, flush_ms=1.0,
    shard_games=8, shard_bytes=268435456, feature_cache=True, policy_cache=32,search_workers_per_turn=4,
    cpu_budget=64,cpu_reserve=4)
UPDATE_V21 = dict(UPDATE, actor_nonforced=True, critic_batch_size=2048, kl_probe_rows=2048)
RETENTION = dict(enabled=True, resume_checkpoints=2, replay_rounds=4, quarantine_hours=24,
    evaluation_samples=4, profile_traces=2, history_reports=128)
CHAMPION_LEGACY = dict(enabled=True, every_games=6400, setups=8, min_win_share=.55, seed=1910000)
CHAMPION_PROTOCOL = 'two-stage-champion/1.0.0'
CHAMPION = dict(CHAMPION_LEGACY, enabled=False, protocol=CHAMPION_PROTOCOL, confirmation_setups=64,
    screen_min_win_share=.56, min_ci_lower=.49)
POPULATION = dict(recent=2, anchors=22)
EVALUATION_SCHEDULE = dict(enabled=False, monitor_every_games=6400, broader_every_games=25600,
    monitor_setups=1, broader_setups=1, seed=910000)
LEGACY_BUDGET = dict(role='lab', seconds=1.0, search_workers=4, global_workers=60, work_multiplier=1)
LEGACY_OPPONENTS = dict(search_free_until_games=10000)
IMITATION = dict(manifest=None,epochs=10,patience=2,min_epochs=3,learning_rate=6e-5)
INSOLVENCY = dict(event_cost=.0075,max_cost=.075)
INSOLVENCY_EVENT_CONFIG_MIGRATION = 'step5-insolvency-event-cost/1.0.0'


def insolvency_event_config(config):
    """Explicit fresh/config-helper upgrade; never silently edit a resumed run."""
    result=copy.deepcopy(config)
    if result.get('insolvency')==dict(event_cost=.005,max_cost=.075):
        search=result.get('search',{}).get('reward')
        if search is not None:
            require(search['insolvency']==result['insolvency'],'Search/learner insolvency differs')
            search['insolvency']=copy.deepcopy(INSOLVENCY)
        result['insolvency']=copy.deepcopy(INSOLVENCY)
    return result
CHAMPION_OPPONENTS = dict(contract='staged-champion-opponents/1.0.0',stage_iterations=5,max_champion_opponents=2)
CHAMPION_MIX_CONFIG_MIGRATION = 'step5-staged-champion-opponents/1.0.0'
LEGACY_BOOST = dict(stages=[dict(games=20000,selfplay=15,historical=25,legacy=60),
                          dict(games=80000,selfplay=45,historical=25,legacy=30)])


def latest_model_recipe(config):
    return config.get('selfplay_recipe') in (ARCHIVED_LATEST_MODEL_RECIPE,PREVIOUS_FIXED_RECIPE,LATEST_MODEL_RECIPE)


def fixed_selfplay_recipe(config):
    return config.get('selfplay_recipe') in (PREVIOUS_FIXED_RECIPE,LATEST_MODEL_RECIPE)


def quality_selfplay_recipe(config):
    return config.get('selfplay_recipe')==LATEST_MODEL_RECIPE


def learner_seats_per_table(config,kind):
    return 4 if kind=='selfplay' else 2 if fixed_selfplay_recipe(config) else 1


def v4_search_enabled(config):
    """Missing means the historical setting; fresh recipes explicitly disable it."""
    value=config.get('v4_search_enabled',True)
    require(type(value) is bool,'Invalid v4_search_enabled flag')
    return value


def latest_model_config(config,imitation_checkpoint):
    """Generate a fresh IL recipe from current settings, never a run snapshot."""
    result=copy.deepcopy(config)
    from brass_ai.learning.model import ARCHITECTURE,PARAMETER_LIMIT
    from .learning_rate import DEFAULT_WARMUP
    from .lr_controller import AUTO_CONTROLLER,AUTO_CONTRACT
    from .il_kl_limit import DEFAULT as AUTO_IL_KL,current_config as current_il_kl_config
    from .rehearsal import DEFAULT_FLOOR
    from .rehearsal_window import DEFAULT as REHEARSAL_SAMPLING
    from .gae import DEFAULT as ADVANTAGE
    from .ppg_config import DEFAULT as VALUE_PHASE
    from .critic_data import DEFAULT as CRITIC_DATA
    from brass_ai.learning.seat_descriptors import DEFAULT as SEAT_DESCRIPTORS
    from .quality_config import TABLE_PLAN,OPPONENT_POOL,RATING,sampling_config
    # The width-512 learning-rate adjustment was introduced with v11. Moving
    # the local addition in v11.1 must not apply that adjustment a second time.
    if result.get('architecture') not in (ARCHITECTURE,'brass-four-seat-mlp-resnet/11.0.0','brass-four-seat-mlp-resnet/11.1.0'):
        result['learning_rate']*=.6
        result['imitation']['learning_rate']*=.6
        result['update']['grad_clip']=min(result['update']['grad_clip'],1.)
    result.setdefault('learning_rate_warmup',copy.deepcopy(DEFAULT_WARMUP))
    result.update(architecture=ARCHITECTURE,parameter_limit=PARAMETER_LIMIT)
    result.update(selfplay_recipe=LATEST_MODEL_RECIPE,curriculum=copy.deepcopy(CURRICULUM),
        champion=dict(result.get('champion',CHAMPION),enabled=False),
        evaluation_schedule=dict(result.get('evaluation_schedule',EVALUATION_SCHEDULE),enabled=False),
        teacher_work_budget=copy.deepcopy(TEACHER_TIME_BUDGET),search_wall_fraction=None,
        imitation_checkpoint=copy.deepcopy(imitation_checkpoint),exploration=ORDINARY_LEARNER_EXPLORATION,
        teacher_enabled=False,v4_search_enabled=False,
        population=copy.deepcopy(POPULATION),legacy_opponents=dict(search_free_until_games=0))
    # This helper explicitly creates a new recipe. Continuation admits only
    # exact source/config reviews and their separately versioned migrations.
    result['ppo_lr_controller']=dict(AUTO_CONTROLLER,**result.get('ppo_lr_controller',{}))
    result['ppo_lr_controller']['contract']=AUTO_CONTRACT
    if result['ppo_lr_controller']['contract']==AUTO_CONTRACT:
        result.setdefault('il_kl_limit',copy.deepcopy(AUTO_IL_KL))
        result['il_kl_limit']=current_il_kl_config(result['il_kl_limit'])
    result.setdefault('rehearsal_floor',copy.deepcopy(DEFAULT_FLOOR))
    result.setdefault('rehearsal_sampling',copy.deepcopy(REHEARSAL_SAMPLING))
    result['advantage']=copy.deepcopy(ADVANTAGE)
    result['critic_data']=copy.deepcopy(CRITIC_DATA)
    result['seat_descriptors']=copy.deepcopy(SEAT_DESCRIPTORS)
    result.setdefault('value_phase',dict(copy.deepcopy(VALUE_PHASE),learning_rate=.5*result['learning_rate']))
    result.setdefault('table_plan',copy.deepcopy(TABLE_PLAN))
    result.setdefault('opponent_pool',copy.deepcopy(OPPONENT_POOL))
    result.setdefault('rating',copy.deepcopy(RATING))
    result.pop('legacy_boost',None)
    result.setdefault('teacher_value',copy.deepcopy(TEACHER_VALUE))
    result.pop('champion_opponents',None)
    result['execution']['teacher_search_workers_per_turn']=TEACHER_SEARCH_WORKERS
    from .value_phase import refined_options,interval_options
    from .exploration import configure
    from .v4_share import configure as configure_v4_share
    from .policy_anchor import configure as configure_anchor
    from .ppg_config import configure as configure_ppg
    return validate_config(configure_ppg(configure_anchor(configure_v4_share(configure(interval_options(refined_options(insolvency_event_config(champion_mix_config(sampling_config(result))))))))))


def champion_mix_config(config):
    """Reviewed upgrade of the preceding default; keep custom reward recipes."""
    result=copy.deepcopy(config)
    if not latest_model_recipe(result) and result.get('champion',{}).get('enabled'):
        result.setdefault('champion_opponents',copy.deepcopy(CHAMPION_OPPONENTS))
    if result.get('insolvency')==dict(event_cost=.005,max_cost=.025):
        # Preserve the exact historical cap-only migration. A separate,
        # explicitly reviewed transition changes the per-event coefficient.
        upgraded=dict(event_cost=.005,max_cost=.075)
        old=result.get('search',{}).get('reward')
        if old is not None:
            require(old['insolvency']==result['insolvency'],'Search/learner insolvency differs')
            old['insolvency']=copy.deepcopy(upgraded)
        result['insolvency']=copy.deepcopy(upgraded)
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
        elif name=='legacy_opponents' and key=='search_free_until_games':
            require(type(value) is int and value>=0,'Invalid legacy_opponents.search_free_until_games')
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


def search_workers_per_turn(config,*,teacher=False):
    ordinary=config.get('search_workers_per_turn',1)
    return config.get('teacher_search_workers_per_turn',ordinary) if teacher else ordinary


def execution_config(config):
    optional={'shared_slot_bytes','inference_routing','replica_backlog','teacher_search_workers_per_turn'}
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
    if 'teacher_search_workers_per_turn' in config:
        require(config['teacher_search_workers_per_turn']<=16,'Teacher root forest supports at most 16 workers')
        if 'cpu_budget' in config:
            require(config['teacher_search_workers_per_turn']<=config['cpu_budget']-config['cpu_reserve'],
                    'Teacher search group exceeds queued CPU admission allowance')
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
            require(stage['until'] is None, 'Mature curriculum must have no end')
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
    from .exploration import validate as validate_exploration,BLOCKS
    validate_exploration(config)
    fields.update(k for k in BLOCKS if k in config)
    require(('critic_data' in config)==('seat_descriptors' in config),'Critic/descriptor blocks must be supplied together')
    if 'critic_data' in config:
        from .critic_data import validate_config as validate_critic
        from brass_ai.learning.seat_descriptors import validate_config as validate_descriptors
        from brass_ai.learning.model import ARCHITECTURE
        fields.update(('critic_data','seat_descriptors'))
        validate_critic(config['critic_data']);validate_descriptors(config['seat_descriptors'])
        require(config.get('architecture')==ARCHITECTURE and quality_selfplay_recipe(config)
            and 'value_phase' in config and 'il_kl_limit' in config,
            'Opponent critic rows require fresh v11.2 quality-pool training and fixed IL probe')
        require(config['value_phase']['holdout_fraction']==config['critic_data']['holdout_fraction']
            and config.get('advantage',{}).get('table_gate') is not None,'Critic holdout/GAE table contract differs')
    if 'advantage' in config:
        from .gae import validate as validate_advantage
        fields.add('advantage');validate_advantage(config['advantage'],combined='reward' in config)
        require(config['update'].get('actor_nonforced'),'GAE requires the nonforced actor recipe')
    if 'value_phase' in config:
        from .value_phase import validate as validate_value_phase
        fields.add('value_phase');value=validate_value_phase(config['value_phase'])
        require('advantage' in config and 'reward' in config and config['update'].get('actor_nonforced'),
                'Value phase requires GAE plumbing and combined-return critic')
        from .value_phase import stratified
        if stratified(value):
            require(value['learning_rate']<=config['learning_rate'],'Value phase LR must not exceed the base PPO learning rate')
        else:require(value['learning_rate']==config['learning_rate'],'Value phase must use the base PPO learning rate')
        if value['enabled'] and value['buffer_iterations']==2:
            require(type(config.get('retention',{}).get('replay_rounds')) is int and
                    config['retention']['replay_rounds']>=2,'Two-iteration PPG requires at least 2 replay rounds')
    require(('advantage' in config)==('value_phase' in config),'GAE/value configuration blocks must be supplied together')
    if config.get('advantage',{}).get('gate',{}).get('mode')=='auto':
        require('il_kl_limit' in config and config.get('rating',{}).get('enabled'),
                'Automatic GAE requires critic holdout diagnostics, calibrated health and scheduled ratings')
    if 'v4_search_enabled' in config:
        fields.add('v4_search_enabled');v4_search_enabled(config)
    if 'learning_rate_warmup' in config:
        from .learning_rate import validate_warmup
        fields.add('learning_rate_warmup');validate_warmup(config['learning_rate_warmup'])
    if 'ppo_lr_controller' in config:
        from .lr_controller import validate_config as validate_controller
        fields.add('ppo_lr_controller');validate_controller(config['ppo_lr_controller'],config['update'])
        require('learning_rate_warmup' in config,'PPO LR controller requires persisted warmup state')
    if 'rehearsal_floor' in config:
        from .rehearsal import validate as validate_floor
        fields.add('rehearsal_floor');validate_floor(config['rehearsal_floor'])
    if 'rehearsal_sampling' in config:
        from .rehearsal_window import validate as validate_sampling
        fields.add('rehearsal_sampling');validate_sampling(config['rehearsal_sampling'])
    from .lr_controller import AUTO_CONTRACT
    auto=config.get('ppo_lr_controller',{}).get('contract')==AUTO_CONTRACT
    require(auto==('il_kl_limit' in config),'Calibrated IL-KL requires LR controller 1.2 and its limit block together')
    if auto:
        from .il_kl_limit import validate as validate_limit
        fields.add('il_kl_limit');limit=validate_limit(config['il_kl_limit']);controller=config['ppo_lr_controller']
        require(controller['anchor_kl_budget']==limit['initial_limit'] and
                controller['anchor_group_kl_budget']==limit['group_factor']*limit['initial_limit'] and
                controller['anchor_group_min_positions']==limit['group_min_positions'],
                'Initial overall/group IL-KL settings must match the calibrated limits')
        require(fixed_selfplay_recipe(config) and config.get('teacher_enabled') is False,
                'Auto IL-KL is for fresh latest-model self-play with teaching disabled')
    quality=quality_selfplay_recipe(config)
    if 'policy_anchor' in config or 'rating_coverage' in config:
        from .policy_anchor import validate as validate_anchor
        fields|={'policy_anchor','rating_coverage'}
        validate_anchor(config)
    if 'panel_gates' in config:
        from .panel_gates import validate as validate_panels
        from .v4_share import CURRENT_CONTRACT
        fields.add('panel_gates');validate_panels(config['panel_gates'])
        require(quality and config.get('rating',{}).get('enabled') is True and
            config.get('opponent_pool',{}).get('contract')=='pfsp-pool/1.0.0' and
            config.get('v4_share_gate',{}).get('contract')==CURRENT_CONTRACT and
            config.get('entropy_control',{}).get('deployment_temperature_check',{}).get('enabled') is False,
            'Panel retirement requires PFSP, the 90% v4 share gate and disabled temperature comparisons')
    if 'v4_share_gate' in config:
        from .v4_share import validate as validate_share
        fields.add('v4_share_gate');validate_share(config['v4_share_gate'])
        require(quality and config.get('rating',{}).get('enabled') and config.get('v4_search_enabled') is False,
            'V4 share gate requires fixed 50/45/5 and scheduled search-free v4 ratings')
    require(quality==all(k in config for k in ('table_plan','opponent_pool','rating')) and
            (quality or not any(k in config for k in ('table_plan','opponent_pool','rating'))),
            'Table plan, quality pool and ratings require the fresh 3.0 recipe together')
    if quality:
        from .quality_config import TABLE_PLAN,validate_pool,validate_rating,maximum_members
        fields|={'table_plan','opponent_pool','rating'}
        require(config['table_plan']==TABLE_PLAN,'Quality recipe requires base 50/45/5 and six seat pairs')
        validate_pool(config['opponent_pool']);validate_rating(config['rating'])
        require(auto and sum(config['population'].values())==maximum_members(config['opponent_pool']),
                'Quality pool requires calibrated health and matching population capacity')
    if 'teacher_enabled' in config:
        fields.add('teacher_enabled');require(type(config['teacher_enabled']) is bool,'Invalid teacher enabled flag')
    if 'teacher_value' in config:
        fields.add('teacher_value')
        auxiliary=teacher_value_config(config['teacher_value'])
        require(not auxiliary['enabled'] or ('reward' in config and config['update'].get('actor_nonforced')),
                'Teacher value requires the current combined-return critic')
    if 'selfplay_recipe' in config:
        fields|={'selfplay_recipe','imitation_checkpoint'}
        require(latest_model_recipe(config),'Unknown self-play recipe')
        require(config.get('champion',{}).get('enabled') is False and 'champion_opponents' not in config,
                'Latest-model self-play forbids champion gates and mixing')
        if fixed_selfplay_recipe(config):
            expected_curriculum=CURRICULUM if quality else PREVIOUS_FIXED_CURRICULUM
            require(config['curriculum']==expected_curriculum and 'legacy_boost' not in config and
                    config['legacy_opponents']==dict(search_free_until_games=0),
                    'Fixed self-play curriculum differs from its versioned recipe')
        else:
            require(config['curriculum']==ARCHIVED_CURRICULUM and config.get('legacy_boost')==LEGACY_BOOST,
                    'Archived latest-model curriculum and v4 window must agree at 20000/100000 games')
        require(config.get('search_wall_fraction') is None,'Latest-model recipe disables the cumulative teacher wall limit')
        origin=config.get('imitation_checkpoint')
        require(type(origin) is dict and set(origin)=={'bc_run','path','sha256','model_id','manifest','manifest_sha256','bc_signature'},
                'Pin the completed imitation checkpoint and lineage')
        require(all(type(v) is str and v for v in origin.values()),'Invalid imitation checkpoint identity')
        require(all(len(origin[k])==64 and all(c in '0123456789abcdef' for c in origin[k])
                    for k in ('sha256','model_id','manifest_sha256','bc_signature')),'Invalid imitation checkpoint hashes')
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
            workers=search_workers_per_turn(config['execution'],teacher=True)
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
        require(sum(s['games'] for s in boost['stages'])<=100000,'Excessive searched-v4 recovery duration')
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
        from brass_ai.learning.model import ARCHITECTURE,PARAMETER_LIMIT
        require(config['initialization']=='random' and (config['architecture'],config['parameter_limit']) in
                ((ARCHITECTURE,PARAMETER_LIMIT),('brass-four-seat-mlp-resnet/5.0.0',3000000),
                 ('brass-four-seat-mlp-resnet/4.0.0',3000000),
                 ('brass-four-seat-mlp-resnet/3.0.0',3000000),
                 ('brass-four-seat-mlp-resnet/2.0.0',3000000),
                 ('brass-four-seat-mlp-resnet/1.0.0',2000000)),
                'Fresh versioned residual-model contract required')
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
                set(config['execution'])-{'shared_slot_bytes','inference_routing','replica_backlog','teacher_search_workers_per_turn'}==set(EXECUTION_V21),
                '2.1 requires revised update/execution')
    require(set(config)==fields and version=='step5-training/2.2.0' and config['players']==4,
        'Use the fresh four-player Masked ISMCTS 2.2 recipe; historical runs cannot migrate in place')
    require(config['model_size'] in ('small','medium'), 'Unknown residual model preset')
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
