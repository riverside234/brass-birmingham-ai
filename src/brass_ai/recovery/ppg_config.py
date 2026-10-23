"""Explicit eight-iteration auxiliary recipe; old signed recipes stay valid."""
import copy
import math

from brass_ai.learning.contracts import require
from .critic_holdout import DEFAULT as HOLDOUT
from . import dcpg_config

CONTRACT='ppg-value-phase/5.0.0'
SHORT_CONTRACT='ppg-value-phase/6.0.0'
SHORT_MIGRATION='ppg-sixteen-step-acceptance/1.0.0'
COVERAGE_CONTRACT='ppg-value-phase/7.0.0'
COVERAGE_MIGRATION='ppg-one-se-coverage/1.0.0'
FIXED_CONTRACT='ppg-value-phase/8.0.0'
FIXED_MIGRATION='ppg-fixed-five-32-step-guards/1.0.0'
LINKED_CONTRACT='ppg-value-phase/9.0.0'
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


def active(config):return config.get('contract') in (CONTRACT,SHORT_CONTRACT,COVERAGE_CONTRACT,FIXED_CONTRACT,LINKED_CONTRACT,dcpg_config.CONTRACT)


def coverage(config):return config.get('contract') in (COVERAGE_CONTRACT,FIXED_CONTRACT,LINKED_CONTRACT,dcpg_config.CONTRACT)


def fixed(config):return config.get('contract') in (FIXED_CONTRACT,LINKED_CONTRACT,dcpg_config.CONTRACT)


def linked(config):return config.get('contract') in (LINKED_CONTRACT,dcpg_config.CONTRACT)


def fixed_options(options):
    """Reviewed scheduling-only change; existing replay/targets stay compatible."""
    if linked(options['value_phase']):
        validate(options['value_phase']);return copy.deepcopy(options)
    result=coverage_options(options)
    from .ppg_selection import RELAXED_DEFAULT
    result['value_phase'].update(contract=FIXED_CONTRACT,max_epochs=5,epoch_controller=None,
        acceptance_steps=32,kl_check_steps=32,patience=1,selection=copy.deepcopy(RELAXED_DEFAULT))
    validate(result['value_phase']);return result


def coverage_options(options):
    from .ppg_selection import DEFAULT as SELECTION
    result=copy.deepcopy(options)
    if coverage(result['value_phase']):
        validate(result['value_phase']);return result
    result=short_options(result)
    result['value_phase'].update(contract=COVERAGE_CONTRACT,selection=copy.deepcopy(SELECTION),
        epoch_controller=dict(copy.deepcopy(DEFAULT['epoch_controller']),contract='ppg-epoch-cap/2.0.0',minimum=2))
    validate(result['value_phase']);return result


def short_options(options):
    """Explicit enable/upgrade; preserve the archived 32-step migration exactly."""
    result=configure(options)
    if coverage(result['value_phase']):return result
    result['value_phase'].update(contract=SHORT_CONTRACT,enabled=True,acceptance_steps=16,patience=1)
    validate(result['value_phase'])
    return result


def configure(options):
    """Archived explicit opt-in; preserve its signed 32-step/patience-2 recipe."""
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
    require(type(c) is dict and set(c)==set(DEFAULT)|({'selection'} if coverage(c) else set())|({'kl_check_steps'} if fixed(c) else set())|({'learning_rate_source'} if linked(c) else set())|({'learning_rate_factor'} if dcpg_config.active(c) else set()) and active(c),
        'Invalid eight-iteration PPG fields/contract')
    require(type(c['enabled']) is bool,'Invalid PPG enabled flag')
    for k in ('buffer_iterations','max_epochs','batch_size','clone_rows_per_age','probe_rows_per_age',
              'current_probe_rows','clone_batch_size','acceptance_steps','phase_interval',
              'validation_iterations','patience','minimum_stratum_rows'):
        require(type(c[k]) is int and c[k]>0,'Invalid PPG integer: '+k)
    for k in ('beta_clone','beta_clone_max','kl_cap','holdout_fraction')+(() if linked(c) else ('learning_rate',)):
        require(type(c[k]) in (int,float) and math.isfinite(c[k]) and c[k]>0,'Invalid PPG number: '+k)
    short=c['contract']!=CONTRACT
    delayed=dcpg_config.active(c)
    if delayed:
        require(type(c['learning_rate_factor']) in (int,float) and c['learning_rate_factor']==.35,
            'Delayed critic requires 0.35 times actual PPO LR')
    if linked(c):
        require(c['learning_rate'] is None and c['learning_rate_source']=='ppo-last-applied',
            'Linked PPG requires the actual PPO peak, not a configured fixed LR')
    if coverage(c):
        from .ppg_selection import DEFAULT as OLD_SELECTION,RELAXED_DEFAULT
        SELECTION=RELAXED_DEFAULT if fixed(c) else OLD_SELECTION
        require(c['selection']==SELECTION and type(c['selection']['standard_errors']) in (int,float),
            'Invalid one-SE PPG selection contract')
    require(c['buffer_iterations']==8 and c['max_epochs']==(5 if fixed(c) else 2) and c['phase_interval']==(4 if linked(c) and not delayed else 8)
        and c['validation_iterations']==8 and c['patience']==(1 if short else 2) and c['weighting']=='uniform'
        and c['minimum_stratum_rows']==1 and c['kl_cap']==.002 and c['holdout']==HOLDOUT
        and c['holdout_fraction']==.05 and 1<=c['acceptance_steps']<=(16 if short and not fixed(c) else 32)
        and c['batch_size']<=4096 and c['clone_batch_size']<=c['clone_rows_per_age']<=2048
        and c['probe_rows_per_age']<=512 and c['probe_rows_per_age']<=c['current_probe_rows']<=2048
        and 1<=c['beta_clone']<=c['beta_clone_max']==32.,'Invalid eight-iteration PPG bounds')
    gate=c['epoch_controller']
    if fixed(c):
        require(gate is None and type(c['kl_check_steps']) is int and c['kl_check_steps']==32
            and c['acceptance_steps']==(16 if delayed else 32),'Invalid fixed PPG loss/KL intervals')
    else:
        expected_gate=dict(DEFAULT['epoch_controller'],contract='ppg-epoch-cap/2.0.0',minimum=2) if coverage(c) else DEFAULT['epoch_controller']
        require(type(gate) is dict and gate==expected_gate
            and all(type(gate[k]) is int for k in ('minimum','maximum','window_phases','qualifying_phases'))
            and type(gate['late_fraction']) in (int,float),
            'Invalid PPG epoch controller')
    rate=c['rate_schedule']
    require(type(rate) is dict and rate==(dcpg_config.SCHEDULE if delayed else DEFAULT['rate_schedule']) and rate['reset_adam'] is True
        and all(type(rate[k]) in (int,float) for k in ('warmup_start','end_factor')),
        'Invalid PPG phase learning-rate schedule')
    return c


def phase_rate(config,step,planned_steps,*,peak=None):
    """One-based planned step; retries replay the same rate without extending it."""
    require(type(step) is int and type(planned_steps) is int and 1<=step<=planned_steps,
        'Invalid PPG phase-rate progress')
    if dcpg_config.active(config):
        require(type(peak) in (int,float) and math.isfinite(peak) and peak>0,'Missing scaled actual PPO rate')
        return peak
    rate=config['rate_schedule'];warmup=min(config['acceptance_steps'],planned_steps)
    if step<=warmup:
        factor=rate['warmup_start']+(1-rate['warmup_start'])*(step-1)/max(1,warmup-1)
    else:factor=1-(1-rate['end_factor'])*(step-warmup)/(planned_steps-warmup)
    if linked(config):
        require(type(peak) in (int,float) and math.isfinite(peak) and peak>0,'Missing actual PPO peak rate')
    else:peak=config['learning_rate']
    return peak*factor


def weight(age,config):
    require(type(age) is int and 0<=age<config['buffer_iterations'],'Invalid PPG replay age')
    return 1.


def initial(generation,config=None):
    if dcpg_config.active(config or {}):return dcpg_config.initial(generation)
    if linked(config or {}):
        from .ppg_phasic import initial as linked_initial
        return linked_initial(generation)
    return dict(contract=MIGRATION,origin=generation,generation=generation,last_phase=generation,phases=0)


def schedule(state,generation):
    from . import ppg_phasic
    require(type(state) is dict,'Invalid PPG schedule')
    if state.get('contract')==dcpg_config.CLOCK:return dcpg_config.schedule(state,generation)
    if state.get('contract')==ppg_phasic.CLOCK:return ppg_phasic.schedule(state,generation)
    require(type(state) is dict and set(state)==set(initial(0)) and state['contract']==MIGRATION
        and all(type(state[k]) is int for k in ('origin','generation','last_phase','phases'))
        and 0<=state['origin']<=state['last_phase']<=generation==state['generation']
        and state['last_phase']==state['origin']+DEFAULT['phase_interval']*state['phases']
        and generation-state['last_phase']<DEFAULT['phase_interval'],'Invalid PPG committed schedule')
    return state


def next_phase(state):
    from . import ppg_phasic
    if state['contract']==dcpg_config.CLOCK:return dcpg_config.next_phase(state)
    if state['contract']==ppg_phasic.CLOCK:return ppg_phasic.next_phase(state)
    return state['last_phase']+DEFAULT['phase_interval']


def due(state):
    schedule(state,state['generation'])
    return state['generation']+1==next_phase(state)


def advance(state,generation,ran):
    require(generation==state['generation']+1 and ran==due(state),'PPG schedule skipped/repeated a phase')
    result=dict(state,generation=generation)
    if ran:result.update(last_phase=generation,phases=state['phases']+1)
    return schedule(result,generation)
