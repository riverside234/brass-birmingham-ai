"""Explicit delayed-critic objective and eight-iteration auxiliary transition."""
import copy
import math

from brass_ai.learning.contracts import require

CONTRACT='ppg-value-phase/10.0.0'
MIGRATION='dcpg-preservation-eight-iteration/1.0.0'
CLOCK='ppg-committed-cadence/3.0.0'
RATE='ppg-ppo-constant-rate/2.0.0'
GRADIENT=dict(contract='ppo-critic-gradient/2.0.0',mode='preserve-outputs',
    scope='outcome-utility-and-local-scoring-heads',beta=1.,
    reference='post-calibration-pre-advantages-eval',descriptors='true-no-dropout',
    losses='existing-head-forms-and-scales',fresh_targets='value-phase-only')
SCHEDULE=dict(contract='ppg-phase-rate/2.0.0',reset_adam=True,
    warmup_start=1.,decay='none',end_factor=1.)


def active(config):return config.get('contract')==CONTRACT


def enabled(update):return update.get('value_gradient',{}).get('mode')=='preserve-outputs'


def options(config):
    from . import ppg_config,ppg_phasic
    if active(config['value_phase']):
        ppg_config.validate(config['value_phase']);return copy.deepcopy(config)
    result=ppg_phasic.options(config)
    result['value_phase'].update(contract=CONTRACT,phase_interval=8,acceptance_steps=16,
        learning_rate_factor=.35,rate_schedule=copy.deepcopy(SCHEDULE))
    ppg_config.validate(result['value_phase']);return result


def configure(config):
    result=options(config);result['update']['value_gradient']=copy.deepcopy(GRADIENT)
    return result


def initial(generation):
    return dict(contract=CLOCK,origin=generation,generation=generation,last_phase=generation,phases=0,
        cadence=dict(origin=generation,previous_phases=0,previous_last_phase=generation,first_phase=generation+8))


def migrate_clock(previous,generation):
    from . import ppg_config
    ppg_config.schedule(previous,generation)
    if previous['contract']==CLOCK:return copy.deepcopy(previous)
    result=dict(copy.deepcopy(previous),contract=CLOCK,cadence=dict(origin=generation,
        previous_phases=previous['phases'],previous_last_phase=previous['last_phase'],
        first_phase=max(generation+1,previous['last_phase']+8,previous['origin']+8)))
    return schedule(result,generation)


def next_phase(state):
    return state['cadence']['first_phase']+8*(state['phases']-state['cadence']['previous_phases'])


def schedule(state,generation):
    require(type(state) is dict and set(state)==set(initial(0)) and state['contract']==CLOCK,
        'Invalid delayed-critic schedule fields')
    c=state['cadence']
    require(type(c) is dict and set(c)==set(initial(0)['cadence'])
        and all(type(v) is int for v in c.values())
        and all(type(state[k]) is int for k in ('origin','generation','last_phase','phases')),
        'Invalid delayed-critic schedule counters')
    done=state['phases']-c['previous_phases']
    require(0<=state['origin']<=c['previous_last_phase']<=c['origin']<=generation==state['generation']
        and 0<=c['previous_phases']<=state['phases']
        and c['first_phase']==max(c['origin']+1,c['previous_last_phase']+8,state['origin']+8)
        and state['last_phase']==(c['first_phase']+8*(done-1) if done else c['previous_last_phase'])
        and state['last_phase']<=generation<next_phase(state),'Invalid delayed-critic committed schedule')
    return state


def rate_receipt(ppo_lr,generation,steps):
    result=dict(contract=RATE,peak_lr=ppo_lr*.35,ppo_lr=ppo_lr,factor=.35,
        generation=generation,source='ppo-last-applied',ppo_steps=steps)
    return validate_rate(result,generation)


def validate_rate(r,generation):
    require(type(r) is dict and set(r)=={'contract','peak_lr','ppo_lr','factor','generation','source','ppo_steps'}
        and r['contract']==RATE and r['source']=='ppo-last-applied'
        and type(r['generation']) is int and 0<=r['generation']<=generation
        and type(r['ppo_steps']) is int and r['ppo_steps']>0
        and all(type(r[k]) in (int,float) and math.isfinite(r[k]) and r[k]>0 for k in ('peak_lr','ppo_lr','factor'))
        and r['factor']==.35 and r['peak_lr']==r['ppo_lr']*.35,'Invalid constant PPG/PPO LR receipt')
    return r
