"""Reviewed four-iteration PPG cadence, PPO-linked peak, and critic stop-gradient."""
import copy
import math
from dataclasses import fields,replace

from brass_ai.learning.contracts import require

MIGRATION='ppg-four-iteration-ppo-peak-detach/1.0.0'
CLOCK='ppg-committed-cadence/2.0.0'
RATE_KEY='brass_ppg_peak_rate'
RATE_CONTRACT='ppg-ppo-peak/1.0.0'
GRADIENT=dict(contract='ppo-critic-gradient/1.0.0',mode='detach-encoder',
    scope='outcome-utility-and-local-scoring-heads')


def options(config):
    from . import ppg_config as p
    result=p.fixed_options(config)
    result['value_phase'].update(contract=p.LINKED_CONTRACT,phase_interval=4,
        learning_rate=None,learning_rate_source='ppo-last-applied')
    p.validate(result['value_phase']);return result


def configure(config):
    result=options(config)
    result['update']['value_gradient']=copy.deepcopy(GRADIENT)
    return result


def validate_gradient(config):
    from .dcpg_config import GRADIENT as PRESERVATION
    require(type(config) is dict and config in (GRADIENT,PRESERVATION)
        and ('beta' not in config or type(config['beta']) in (int,float)),'Invalid PPO critic gradient contract')
    return config


def detached(config):
    if 'value_gradient' not in config:return False
    validate_gradient(config['value_gradient']);return config['value_gradient']['mode']=='detach-encoder'


def stop_value_gradient(encoded):
    # Detach the representation, not the predictions or head parameters. The
    # policy forward owns its original tensors and keeps its normal gradients.
    return replace(encoded,**{f.name:getattr(encoded,f.name).detach() for f in fields(encoded)})


def initial(generation=0):
    return dict(contract=CLOCK,origin=generation,generation=generation,last_phase=generation,phases=0,
        cadence=dict(origin=generation,previous_phases=0,previous_last_phase=generation,first_phase=generation+8))


def migrate_clock(previous,generation):
    from . import ppg_config as p
    p.schedule(previous,generation)
    if previous['contract']==CLOCK:return copy.deepcopy(previous)
    # Do not fabricate a phase at the already committed adoption generation.
    # Keep the previous due date if it is sooner; require eight cached ages for
    # the first phase of a fresh/short run, then add four fresh ages each time.
    first=max(previous['origin']+8,min(generation+4,p.next_phase(previous)))
    result=dict(copy.deepcopy(previous),contract=CLOCK,cadence=dict(origin=generation,
        previous_phases=previous['phases'],previous_last_phase=previous['last_phase'],first_phase=first))
    return schedule(result,generation)


def next_phase(state):
    c=state['cadence'];done=state['phases']-c['previous_phases']
    return c['first_phase']+4*done


def schedule(state,generation):
    expected=set(initial())
    require(type(state) is dict and set(state)==expected and state['contract']==CLOCK,
        'Invalid linked PPG schedule fields')
    c=state['cadence']
    require(type(c) is dict and set(c)==set(initial()['cadence'])
        and all(type(c[k]) is int for k in c)
        and all(type(state[k]) is int for k in ('origin','generation','last_phase','phases')),
        'Invalid linked PPG schedule counters')
    done=state['phases']-c['previous_phases']
    require(0<=state['origin']<=c['previous_last_phase']<=c['origin']<=generation==state['generation']
        and 0<=c['previous_phases']<=state['phases']
        and max(c['origin']+1,state['origin']+8)<=c['first_phase']<=max(c['origin']+4,state['origin']+8)
        and state['last_phase']==(c['first_phase']+4*(done-1) if done else c['previous_last_phase'])
        and state['last_phase']<=generation<next_phase(state),'Invalid linked PPG committed schedule')
    return state


def rate_receipt(peak,generation,source,steps=0):
    result=dict(contract=RATE_CONTRACT,peak_lr=peak,generation=generation,source=source,ppo_steps=steps)
    validate_rate(result,generation);return result


def validate_rate(receipt,generation):
    from . import dcpg_config
    if type(receipt) is dict and receipt.get('contract')==dcpg_config.RATE:
        return dcpg_config.validate_rate(receipt,generation)
    require(type(receipt) is dict and set(receipt)=={'contract','peak_lr','generation','source','ppo_steps'}
        and receipt['contract']==RATE_CONTRACT and type(receipt['peak_lr']) in (int,float)
        and math.isfinite(receipt['peak_lr']) and receipt['peak_lr']>0
        and type(receipt['generation']) is int and 0<=receipt['generation']<=generation
        and type(receipt['ppo_steps']) is int and receipt['ppo_steps']>=0
        and receipt['source'] in ('legacy-value-rate','initial-ppo-base','ppo-last-applied')
        and (receipt['source']!='ppo-last-applied' or receipt['ppo_steps']>0),
        'Invalid persisted PPG peak-rate receipt')
    return receipt
