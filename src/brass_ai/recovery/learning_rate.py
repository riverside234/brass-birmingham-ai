"""Linear warmup of shared-network updates, persisted inside Adam groups.

Head-only utility calibration keeps its existing temporary rate and does not
advance this counter. There is no post-warmup decay or restart on resume.
"""
import copy
import math
from brass_ai.learning.contracts import require

DEFAULT_WARMUP=dict(contract='linear-update-warmup/1.0.0',imitation_steps=500,selfplay_steps=100,start_factor=.1)
KEY='brass_learning_rate_warmup'


def validate_warmup(config):
    require(type(config) is dict and set(config)==set(DEFAULT_WARMUP) and
            config['contract']==DEFAULT_WARMUP['contract'],'Invalid learning-rate warmup contract')
    require(all(type(config[k]) is int and 0<=config[k]<=100000 for k in ('imitation_steps','selfplay_steps')),
            'Invalid warmup step count')
    require(type(config['start_factor']) in (int,float) and math.isfinite(config['start_factor']) and
            0<config['start_factor']<=1,'Invalid warmup start factor')
    return config


def rate(state,step):
    if state['steps']==0:return state['base_lr']
    progress=min(max(step-1,0)/max(1,state['steps']-1),1.) if state['steps']>1 else 1.
    return state['base_lr']*(state['start_factor']+(1-state['start_factor'])*progress)


def configure(optimizer,base_lr,steps,start_factor,kind):
    validate_warmup(dict(DEFAULT_WARMUP,imitation_steps=steps,selfplay_steps=steps,start_factor=start_factor))
    require(type(base_lr) in (int,float) and math.isfinite(base_lr) and base_lr>0,'Invalid warmup base rate')
    require(kind in ('imitation','selfplay') and len(optimizer.param_groups)==1,'Invalid warmup optimizer role')
    group=optimizer.param_groups[0]
    expected=dict(contract=DEFAULT_WARMUP['contract'],kind=kind,base_lr=base_lr,steps=steps,start_factor=start_factor)
    previous=group.get(KEY)
    if previous is None:
        require(not optimizer.state,'Missing warmup state in a trained optimizer; do not restart its schedule')
        previous=dict(expected,completed_steps=0);group[KEY]=previous
    require(set(previous)==set(expected)|{'completed_steps'} and all(previous[k]==v for k,v in expected.items()) and
            type(previous['completed_steps']) is int and previous['completed_steps']>=0,'Warmup checkpoint/config differs')
    group['lr']=rate(previous,max(1,previous['completed_steps']))
    return report(optimizer)


def step(optimizer):
    for group in optimizer.param_groups:
        if KEY in group:group['lr']=rate(group[KEY],group[KEY]['completed_steps']+1)
    optimizer.step()
    for group in optimizer.param_groups:
        if KEY in group:group[KEY]['completed_steps']+=1


def report(optimizer):
    if optimizer is None:return None
    values=[dict(copy.deepcopy(g[KEY]),effective_lr=g['lr'],next_lr=rate(g[KEY],g[KEY]['completed_steps']+1))
            for g in optimizer.param_groups if KEY in g]
    return values[0] if len(values)==1 else None
