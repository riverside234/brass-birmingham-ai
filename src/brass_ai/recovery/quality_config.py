"""Versioned table, opponent-pool and held-out rating contracts."""
import copy
import math
from brass_ai.learning.contracts import require

TABLE_PLAN=dict(quotas=dict(selfplay=50,historical=45,legacy=5),
                learner_seats=dict(selfplay=4,historical=2,legacy=2),pair_rotation='six-pairs')
LEGACY_POOL=dict(contract='quality-score-pool/1.0.0',max_members=24,pinned=['generation-0'],
    pinned_seat_share=.2,eta=.01,uniform_floor=.2,admit_every_iterations=5,
    admit_health_iterations=5,admit_min_kl=.02,evict_min_age_iterations=20,protect_newest=2)
OPPONENT_POOL=dict(contract='pfsp-pool/1.0.0',pinned=['generation-0'],
    pinned_seat_share=dict(rule='reciprocal',offset=1,max=None),
    max_scored_members=[dict(until_games=100000,max=16),dict(until_games=None,max=23)],
    outrank_rate=dict(half_life_iterations=5,prior_rate=.5,prior_comparisons=10),
    weighting=dict(function='hard',power=2),uniform_floor=.2,
    admit_every_iterations=5,admit_health_iterations=5,admit_min_kl=.02,
    evict=dict(min_age_iterations=20,protect_newest=2,min_decayed_comparisons=15,redundancy_band=.05),
    alarm=dict(rate_below=.4,min_decayed_comparisons=20,iterations=3))
CONFIG_MIGRATION='pfsp-and-rehearsal/1.0.0'
RATING=dict(contract='plackett-luce-rating/1.0.0',enabled=True,every_iterations=10,setups=32,
    seat_rotations=4,references=['generation-0','v4'],nearest_opponents=8,
    theta_prior_sd=2.,seat_prior_sd=.5,bootstrap=200,elo_scale=173.7,seed=910517,
    panel_setups=8,v4_profile='bank-3dc65253a610191d66a5/balanced-v3-masked-economy')


def pfsp(c):return c.get('contract')==OPPONENT_POOL['contract']


def maximum_members(c):
    return 1+max(row['max'] for row in c['max_scored_members']) if pfsp(c) else c['max_members']


def sampling_config(config):
    """Explicit fresh generation/reviewed upgrade; never rewrite a saved run."""
    result=copy.deepcopy(config)
    if result.get('opponent_pool')==LEGACY_POOL:result['opponent_pool']=copy.deepcopy(OPPONENT_POOL)
    from .rehearsal import DEFAULT_FLOOR
    if result.get('rehearsal_floor')==dict(DEFAULT_FLOOR,interval_iterations=30):
        result['rehearsal_floor']=copy.deepcopy(DEFAULT_FLOOR)
    return result


def validate_pool(c):
    require(type(c) is dict,'Invalid opponent-pool config')
    if pfsp(c):return validate_pfsp(c)
    require(set(c)==set(LEGACY_POOL) and c['contract']==LEGACY_POOL['contract'],
            'Invalid quality-score pool contract/fields')
    require(c['pinned']==['generation-0'],'Quality pool must pin generation zero')
    for key in ('max_members','admit_every_iterations','admit_health_iterations','evict_min_age_iterations','protect_newest'):
        require(type(c[key]) is int and c[key]>0,'Invalid quality-pool '+key)
    require(3<=c['max_members']<=24 and c['protect_newest']<c['max_members'],'Quality pool capacity exceeds 24')
    for key in ('pinned_seat_share','eta','uniform_floor','admit_min_kl'):
        require(type(c[key]) in (int,float) and math.isfinite(c[key]) and c[key]>0,'Invalid quality-pool '+key)
    require(c['pinned_seat_share']<1 and c['uniform_floor']<=1,'Invalid quality-pool probability')
    return c


def validate_pfsp(c):
    require(set(c)==set(OPPONENT_POOL) and c['pinned']==['generation-0'],'Invalid PFSP contract/fields/pinned anchor')
    def fields(name):
        require(type(c[name]) is dict and set(c[name])==set(OPPONENT_POOL[name]),'Invalid PFSP '+name+' fields')
        return c[name]
    def positive(value,name):
        require(type(value) in (int,float) and math.isfinite(value) and value>0,'Invalid PFSP '+name)
    share=fields('pinned_seat_share')
    require(share['rule']=='reciprocal' and type(share['offset']) is int and share['offset']==1,'Invalid PFSP IL share rule')
    if share['max'] is not None:
        positive(share['max'],'IL share cap');require(share['max']<=1,'Invalid PFSP IL share cap')
    caps=c['max_scored_members']
    require(type(caps) is list and len(caps)==2 and all(type(r) is dict and set(r)=={'until_games','max'} for r in caps),
            'Invalid PFSP capacity schedule')
    require(type(caps[0]['until_games']) is int and caps[0]['until_games']==100000 and caps[1]['until_games'] is None
        and all(type(r['max']) is int and 2<=r['max']<=23 for r in caps) and caps[0]['max']<=caps[1]['max'],
        'Invalid PFSP capacity bounds')
    rate=fields('outrank_rate')
    for key,value in rate.items():positive(value,key)
    require(rate['prior_rate']<1,'Invalid PFSP prior rate')
    w=fields('weighting');positive(w['power'],'weighting power')
    require(w['function'] in ('hard','variance'),'PFSP weighting must be hard or variance (x*(1-x))')
    require(w['function']!='variance' or w['power']==1,'Variance weighting is exactly x*(1-x); use power=1')
    for key in ('uniform_floor','admit_min_kl'):positive(c[key],key)
    require(c['uniform_floor']<=1,'Invalid PFSP uniform floor')
    for key in ('admit_every_iterations','admit_health_iterations'):
        require(type(c[key]) is int and c[key]>0,'Invalid PFSP '+key)
    evict=fields('evict');alarm=fields('alarm')
    for key in ('min_age_iterations','protect_newest'):
        require(type(evict[key]) is int and evict[key]>0,'Invalid PFSP eviction '+key)
    require(evict['protect_newest']<=caps[0]['max'],'PFSP protection exceeds capacity')
    for key in ('min_decayed_comparisons','redundancy_band'):positive(evict[key],key)
    require(evict['redundancy_band']<=1,'Invalid PFSP redundancy band')
    for key in ('rate_below','min_decayed_comparisons'):positive(alarm[key],key)
    require(alarm['rate_below']<1 and type(alarm['iterations']) is int and alarm['iterations']>0,'Invalid PFSP alarm')
    return c


def validate_rating(c):
    require(type(c) is dict and set(c)==set(RATING) and c['contract']==RATING['contract'],
            'Invalid Plackett-Luce rating contract/fields')
    require(type(c['enabled']) is bool and c['references']==['generation-0','v4'] and c['seat_rotations']==4,
            'Rating requires fixed IL/v4 references and four seat rotations')
    for key in ('every_iterations','setups','nearest_opponents','bootstrap','panel_setups'):
        require(type(c[key]) is int and c[key]>0,'Invalid rating '+key)
    require(c['setups']%2==0 and c['panel_setups']<=c['setups'],'Rating needs balanced reference setups')
    require(type(c['seed']) is int and 0<=c['seed']<2**31,'Invalid rating seed')
    for key in ('theta_prior_sd','seat_prior_sd','elo_scale'):
        require(type(c[key]) in (int,float) and math.isfinite(c[key]) and c[key]>0,'Invalid rating '+key)
    require(type(c['v4_profile']) is str and '/' in c['v4_profile'],'Pin the rating v4 profile')
    return c
