"""Committed, rollback-aware auxiliary fitting evidence; independent of GAE/PPO."""
import copy
from brass_ai.learning.contracts import require

CONTRACT='local-scoring-accepted-warmup/1.0.0'


def initial():
    return dict(contract=CONTRACT,generation=0,qualifying_iterations=0,fitting_steps=0,
        released_at=None,encoder_stage=0)


def validate(state,config,generation):
    require(type(state) is dict and set(state)==set(initial()) and state['contract']==CONTRACT,
        'Missing/invalid accepted auxiliary warmup state')
    require(all(type(state[k]) is int and state[k]>=0 for k in
        ('generation','qualifying_iterations','fitting_steps','encoder_stage'))
        and state['generation']==generation and state['qualifying_iterations']<=generation
        and state['fitting_steps']>=state['qualifying_iterations'], 'Invalid auxiliary fitting counters')
    q=state['qualifying_iterations'];released=state['released_at']
    ready=q>=config['head_only_iterations'] and state['fitting_steps']>=config['minimum_fitting_steps']
    if released is None:
        require(not ready and state['encoder_stage']==0,'Missing auxiliary warmup release')
    else:
        require(ready and type(released) is int and config['head_only_iterations']<=released<=q
            and state['encoder_stage']==min(config['ramp_iterations'],q-released+1),
            'Auxiliary ramp differs from committed fitting evidence')
    return state


def advance(state,config,generation,steps,games):
    validate(state,config,generation-1)
    require(type(steps) is int and type(games) is int and steps>=0 and games>=0
        and bool(steps)==bool(games),'Invalid retained auxiliary fitting work')
    result=copy.deepcopy(state);result['generation']=generation
    if steps:
        result['qualifying_iterations']+=1;result['fitting_steps']+=steps
        if result['released_at'] is None:
            if (result['qualifying_iterations']>=config['head_only_iterations']
                and result['fitting_steps']>=config['minimum_fitting_steps']):
                result['released_at']=result['qualifying_iterations'];result['encoder_stage']=1
        else:result['encoder_stage']=min(config['ramp_iterations'],result['encoder_stage']+1)
    return validate(result,config,generation)


def scale(state,config):return state['encoder_stage']/config['ramp_iterations']


def report(state,config):
    multiplier=scale(state,config)
    return dict(state,stage='head_only' if multiplier==0 else 'full' if multiplier==1 else 'ramp',
        encoder_gradient_multiplier=multiplier,minimum_fitting_iterations=config['head_only_iterations'],
        minimum_fitting_steps=config['minimum_fitting_steps'],ramp_iterations=config['ramp_iterations'])
