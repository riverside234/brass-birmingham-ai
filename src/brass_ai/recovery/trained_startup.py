"""Explicit startup permissions for a NEW run seeded by trained g1290 weights.

Manual panel retirement is scheduling permission, never statistical evidence.
Ownership/VP heads start fresh, separately from the trained main model.
"""
import copy
from brass_ai.learning.contracts import digest,require

KEY='trained_startup'
CONTRACT='trained-selfplay-startup/1.0.0'
MANUAL='manual-training-qualification/1.0.0'
REASON='Explicit user-authorized g1290 restart; no qualification games were run'
DEFAULT=dict(contract=CONTRACT,source_generation=1290,branching='immediate',
    panels='manual-retire-il-v4',competitive_pl='manual-enable',
    main_gae_lambda=.98,ppo_warmup_steps=100,authority='user',reason=REASON)


def enabled(config):return config.get(KEY)==DEFAULT


def configure(config):
    from . import exploration,gae
    result=copy.deepcopy(config);result[KEY]=copy.deepcopy(DEFAULT)
    result['rehearsal_weight']=0.
    result['exploration_rollout']=copy.deepcopy(exploration.TRAINED_ROLLOUT)
    result['advantage'].update(contract=gae.TRAINED_CONTRACT,**{'lambda':.98})
    result['learning_rate_warmup']['selfplay_steps']=100
    validate(result)
    return result


def validate(config):
    from . import neutral_training,policy_observations,exploration,gae,v4_share,panel_gates,rating_coverage
    require(enabled(config),'Invalid trained startup settings')
    require(neutral_training.enabled(config) and policy_observations.enabled(config),
        'Trained startup requires neutral contexts/fingerprints and AnchorKL off')
    revised=config['advantage']['contract']==gae.DIRECT_095_CONTRACT
    fork=config['branching'] in (exploration.FORK_T15_BRANCH,dict(exploration.FORK_T15_BRANCH,rate_per_decision=0.))
    from . import entropy_schedule as anneal
    from .ppg_g210 import BETA1_CONTRACT,QUARTER_CONTRACT,EARLY_CONTRACT,TWO_ITER_CONTRACT,EVERY_ITER_CONTRACT,TEACHER_FLOOR_CONTRACT,KL_STOP_CONTRACT
    quarter=config['value_phase']['contract'] in (QUARTER_CONTRACT,EARLY_CONTRACT,TWO_ITER_CONTRACT,EVERY_ITER_CONTRACT,TEACHER_FLOOR_CONTRACT,KL_STOP_CONTRACT)
    scheduled=anneal.KEY in config['update']
    if scheduled:anneal.validate(config)  # Runtime restore also verifies the applied-step receipt.
    if quarter:require(fork and scheduled and config['update'][anneal.KEY]['contract'] in (anneal.EXTENDED_CONTRACT,anneal.AUTO_CONTRACT),
        'Quarter-beta trained recipe requires the extended insolvency schedule')
    require(config['exploration_rollout']==exploration.TRAINED_ROLLOUT
        and config['branching']==(config['branching'] if fork else exploration.T2_ONLY_BRANCH if revised else exploration.NEUTRAL_BRANCH) and config['opening']==exploration.THREE_OPENING
        and config['rehearsal_weight']==0. and config['diversity']['mode']=='off'
        and config['context_evaluation']['mode']=='off' and config['exploration']==(0. if quarter else .02 if fork else .05)
        and config['update']['fixed_entropy']==(exploration.FIXED_ENTROPY_0025 if fork and not scheduled else exploration.FIXED_ENTROPY_005)
        and (scheduled or config['insolvency']==dict(event_cost=.02,max_cost=.075)), 'Trained rollback recipe differs')
    require(config['advantage']['contract'] in (gae.TRAINED_CONTRACT,gae.FLOOR_CONTRACT,gae.DIRECT_095_CONTRACT) and config['advantage']['lambda']==(.95 if revised else .98)
        and config['advantage']['gate']==gae.GATE and config['advantage']['table_gate']==gae.TABLE_GATE,
        'Trained startup must preserve automatic GAE health/support gates')
    if revised:
        require(config['value_phase']['contract'] in (BETA1_CONTRACT,QUARTER_CONTRACT,EARLY_CONTRACT,TWO_ITER_CONTRACT,EVERY_ITER_CONTRACT,TEACHER_FLOOR_CONTRACT,KL_STOP_CONTRACT),
            'Revised trained recipe requires a reviewed loss16/KL32 PPG contract')
    require(config['learning_rate_warmup']['selfplay_steps']==100,'Trained startup needs 100 optimizer-step PPO warmup')
    require(config['panel_gates']==panel_gates.DEFAULT and config['v4_share_gate']==v4_share.CURRENT_DEFAULT
        and rating_coverage.previous(config)['rating_coverage']==(rating_coverage.WITH_SEED if fork else rating_coverage.DEFAULT)
        and config['rating']['enabled']
        and config['games_per_iteration']==320,'Manual startup requires the existing 320-game/PL protocols')
    require(('v4_history' in config)==fork and (not fork or revised),'Incomplete g570 trained recipe')


def authorization(state,config):
    validate(config)
    origin=state['origin']
    require(origin['seed_generation']==1290 and origin['initialization']=='fresh-selfplay-weights',
        'Startup authorization requires the bound g1290 seed')
    return dict(contract=CONTRACT,authority='user',reason=REASON,generation=0,
        source_generation=1290,source_model_id=origin['parent_model_id'],
        manifest_sha256=config['fresh_selfplay_seed']['sha256'],config_sha256=digest(config[KEY]))


def validate_authorization(value):
    require(type(value) is dict and set(value)=={'contract','authority','reason','generation','source_generation',
        'source_model_id','manifest_sha256','config_sha256'} and value['contract']==CONTRACT
        and value['authority']=='user' and value['reason']==REASON and type(value['generation']) is int and value['generation']==0
        and value['source_generation']==1290 and value['config_sha256']==digest(DEFAULT),
        'Invalid manual startup authorization')
    from .fresh_selfplay import hex64
    require(all(hex64(value[k]) for k in ('source_model_id','manifest_sha256','config_sha256')),
        'Invalid startup identity hash')
    return value


def manual(value):return isinstance(value,dict) and value.get('contract')==MANUAL


def trigger(name,reference,auth):
    return dict(contract=MANUAL,kind='manual_override',generation=0,panel=name,reference=reference,
        authorization=copy.deepcopy(validate_authorization(auth)),measured=False)


def validate_trigger(value):
    from .fresh_selfplay import hex64
    require(type(value) is dict and set(value)=={'contract','kind','generation','panel','reference','authorization','measured'}
        and manual(value) and value['kind']=='manual_override' and type(value['generation']) is int and value['generation']==0
        and value['panel'] in ('il-panel','v4-panel') and hex64(value['reference'])
        and value['measured'] is False,'Invalid manual panel override; measurements are forbidden')
    validate_authorization(value['authorization'])
    return value


def initialize_gates(state,config):
    from . import panel_gates,v4_share,rating_coverage
    auth=authorization(state,config);rating=state['ratings']
    require(state['iteration']==state['games_done']==0 and not rating['slots'],
        'Manual startup cannot rewrite an existing run')
    rating['startup_override']=copy.deepcopy(auth)
    rating['panel_gates']=panel_gates.initial(0)
    rating['panel_gates']['retired']={n:trigger(n,panel_gates.reference(rating,n),auth) for n in panel_gates.NAMES}
    rating['coverage']=dict(contract=rating_coverage.CONTRACT,activated=dict(generation=0,manual_override=copy.deepcopy(auth)))
    state['v4_share_gate']=dict(v4_share.initial(config,0,0),activated=dict(generation=0,games=0),manual_override=copy.deepcopy(auth))
    # This is the first planned distribution, not a receipt of played games.
    # Bind the IL latch immediately so saved next_planning matches collection.
    from . import quality_pool,pfsp_pool
    pool=state['quality_pool']
    pool['planning']=quality_pool.distribution(pool,il_gate=rating['panel_gates']['retired']['il-panel'])
    pfsp_pool.seal(pool)
    state['startup_initialization']=dict(authorization=auth,branching_from_collection=1,main_gae_lambda=.98,
        warmup_unit='optimizer_steps',warmup_steps=100,
        measured_qualification=False,health_evidence_inherited=False)
    panel_gates.validate_state(rating['panel_gates'],0)
    v4_share.validate_state(state['v4_share_gate'],config,0,0)


def verify_gates(state,config):
    from . import panel_gates
    auth=authorization(state,config)
    require(state['ratings'].get('startup_override')==auth
        and state.get('startup_initialization',{}).get('authorization')==auth
        and state.get('v4_share_gate',{}).get('manual_override')==auth
        and state['ratings'].get('coverage',{}).get('activated')==dict(generation=0,manual_override=auth),
        'Startup authorization differs from signed seed')
    require(state['ratings'].get('panel_gates',{}).get('retired')=={
        n:trigger(n,panel_gates.reference(state['ratings'],n),auth) for n in panel_gates.NAMES},
        'Manual startup panel retirement changed')
    return auth


def verify_manual(rating,name,value):
    from .panel_gates import reference
    validate_trigger(value)
    require(value['panel']==name and value['reference']==reference(rating,name)
        and value['authorization']==rating.get('startup_override'),'Manual override reference/authorization differs')

