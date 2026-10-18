"""Versioned fresh-run table, quality-pool and held-out rating contracts."""
import math
from brass_ai.learning.contracts import require

TABLE_PLAN=dict(quotas=dict(selfplay=50,historical=45,legacy=5),
                learner_seats=dict(selfplay=4,historical=2,legacy=2),pair_rotation='six-pairs')
OPPONENT_POOL=dict(contract='quality-score-pool/1.0.0',max_members=24,pinned=['generation-0'],
    pinned_seat_share=.2,eta=.01,uniform_floor=.2,admit_every_iterations=5,
    admit_health_iterations=5,admit_min_kl=.02,evict_min_age_iterations=20,protect_newest=2)
RATING=dict(contract='plackett-luce-rating/1.0.0',enabled=True,every_iterations=10,setups=32,
    seat_rotations=4,references=['generation-0','v4'],nearest_opponents=8,
    theta_prior_sd=2.,seat_prior_sd=.5,bootstrap=200,elo_scale=173.7,seed=910517,
    panel_setups=8,v4_profile='bank-3dc65253a610191d66a5/balanced-v3-masked-economy')


def validate_pool(c):
    require(type(c) is dict and set(c)==set(OPPONENT_POOL) and c['contract']==OPPONENT_POOL['contract'],
            'Invalid quality-score pool contract/fields')
    require(c['pinned']==['generation-0'],'Quality pool must pin generation zero')
    for key in ('max_members','admit_every_iterations','admit_health_iterations','evict_min_age_iterations','protect_newest'):
        require(type(c[key]) is int and c[key]>0,'Invalid quality-pool '+key)
    require(3<=c['max_members']<=24 and c['protect_newest']<c['max_members'],'Quality pool capacity exceeds 24')
    for key in ('pinned_seat_share','eta','uniform_floor','admit_min_kl'):
        require(type(c[key]) in (int,float) and math.isfinite(c[key]) and c[key]>0,'Invalid quality-pool '+key)
    require(c['pinned_seat_share']<1 and c['uniform_floor']<=1,'Invalid quality-pool probability')
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
