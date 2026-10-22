"""Explicit eight-iteration auxiliary recipe; old signed recipes stay valid."""
import copy
import math

from brass_ai.learning.contracts import require
from .critic_holdout import DEFAULT as HOLDOUT

CONTRACT='ppg-value-phase/5.0.0'
MIGRATION='ppg-eight-iteration-replay/1.0.0'
DEFAULT=dict(contract=CONTRACT,enabled=False,buffer_iterations=8,max_epochs=2,batch_size=4096,
    clone_rows_per_age=2048,probe_rows_per_age=512,current_probe_rows=2048,clone_batch_size=512,
    beta_clone=1.,beta_clone_max=32.,kl_cap=.002,learning_rate=3e-6,holdout_fraction=.05,
    holdout=copy.deepcopy(HOLDOUT),acceptance_steps=32,weighting='uniform',phase_interval=8,
    validation_iterations=8,patience=2,minimum_stratum_rows=1,
    epoch_controller=dict(contract='ppg-epoch-cap/1.0.0',minimum=1,maximum=6,
        window_phases=4,qualifying_phases=3,late_fraction=.75),
    rate_schedule=dict(contract='ppg-phase-rate/1.0.0',reset_adam=True,
        warmup_start=.1,decay='linear',end_factor=.3))


def active(config):return config.get('contract')==CONTRACT


def configure(options):
    """Explicit opt-in; fresh self-play defaults leave the auxiliary phase off."""
    result=copy.deepcopy(options)
    old=result['value_phase']
    if active(old):
        validate(old);old['enabled']=True
        return result
    from .value_phase import validate as validate_old
    validate_old(old)
    require(old.get('holdout')==HOLDOUT,'Eight-iteration PPG requires the existing 8/8/2 holdout')
    result['value_phase']=dict(copy.deepcopy(DEFAULT),enabled=True,learning_rate=old['learning_rate'],
        beta_clone=old['beta_clone'])
    return result


def validate(c):
    require(type(c) is dict and set(c)==set(DEFAULT) and c['contract']==CONTRACT,
        'Invalid eight-iteration PPG fields/contract')
    require(type(c['enabled']) is bool,'Invalid PPG enabled flag')
    for k in ('buffer_iterations','max_epochs','batch_size','clone_rows_per_age','probe_rows_per_age',
              'current_probe_rows','clone_batch_size','acceptance_steps','phase_interval',
              'validation_iterations','patience','minimum_stratum_rows'):
        require(type(c[k]) is int and c[k]>0,'Invalid PPG integer: '+k)
    for k in ('beta_clone','beta_clone_max','kl_cap','learning_rate','holdout_fraction'):
        require(type(c[k]) in (int,float) and math.isfinite(c[k]) and c[k]>0,'Invalid PPG number: '+k)
    require(c['buffer_iterations']==8 and c['max_epochs']==2 and c['phase_interval']==8
        and c['validation_iterations']==8 and c['patience']==2 and c['weighting']=='uniform'
        and c['minimum_stratum_rows']==1 and c['kl_cap']==.002 and c['holdout']==HOLDOUT
        and c['holdout_fraction']==.05 and 1<=c['acceptance_steps']<=32
        and c['batch_size']<=4096 and c['clone_batch_size']<=c['clone_rows_per_age']<=2048
        and c['probe_rows_per_age']<=512 and c['probe_rows_per_age']<=c['current_probe_rows']<=2048
        and 1<=c['beta_clone']<=c['beta_clone_max']==32.,'Invalid eight-iteration PPG bounds')
    gate=c['epoch_controller']
    require(type(gate) is dict and gate==DEFAULT['epoch_controller']
        and all(type(gate[k]) is int for k in ('minimum','maximum','window_phases','qualifying_phases'))
        and type(gate['late_fraction']) in (int,float),
        'Invalid PPG epoch controller')
    rate=c['rate_schedule']
    require(type(rate) is dict and rate==DEFAULT['rate_schedule'] and rate['reset_adam'] is True
        and all(type(rate[k]) in (int,float) for k in ('warmup_start','end_factor')),
        'Invalid PPG phase learning-rate schedule')
    return c


def phase_rate(config,step,planned_steps):
    """One-based planned step; retries replay the same rate without extending it."""
    require(type(step) is int and type(planned_steps) is int and 1<=step<=planned_steps,
        'Invalid PPG phase-rate progress')
    rate=config['rate_schedule'];warmup=min(config['acceptance_steps'],planned_steps)
    if step<=warmup:
        factor=rate['warmup_start']+(1-rate['warmup_start'])*(step-1)/max(1,warmup-1)
    else:factor=1-(1-rate['end_factor'])*(step-warmup)/(planned_steps-warmup)
    return config['learning_rate']*factor


def weight(age,config):
    require(type(age) is int and 0<=age<config['buffer_iterations'],'Invalid PPG replay age')
    return 1.


def initial(generation):
    return dict(contract=MIGRATION,origin=generation,generation=generation,last_phase=generation,phases=0)


def schedule(state,generation):
    require(type(state) is dict and set(state)==set(initial(0)) and state['contract']==MIGRATION
        and all(type(state[k]) is int for k in ('origin','generation','last_phase','phases'))
        and 0<=state['origin']<=state['last_phase']<=generation==state['generation']
        and state['last_phase']==state['origin']+DEFAULT['phase_interval']*state['phases']
        and generation-state['last_phase']<DEFAULT['phase_interval'],'Invalid PPG committed schedule')
    return state


def due(state):
    schedule(state,state['generation'])
    return state['generation']+1-state['last_phase']==DEFAULT['phase_interval']


def advance(state,generation,ran):
    require(generation==state['generation']+1 and ran==due(state),'PPG schedule skipped/repeated a phase')
    result=dict(state,generation=generation)
    if ran:result.update(last_phase=generation,phases=state['phases']+1)
    return schedule(result,generation)
