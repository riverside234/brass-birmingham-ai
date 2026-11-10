"""A 4.0e-6 floor on the PPO learning rate, adopted once at a committed boundary from g1307.

Only the controller's lower bound moves: its minimum multiplier goes from .5 (a
3.0e-6 floor on the 6.0e-6 base rate) to 4.0e-6 / base. The base rate, the
persisted multiplier, its windows and cooldown are kept, so the next PPO step
uses exactly the rate it would have used before.
"""
import copy

from brass_ai.learning.contracts import require

CONTRACT='training-revision-g1307/1.0.0'
FIRST_GENERATION=1307
LR_FLOOR=4e-6
PREVIOUS_MIN_MULTIPLIER=.5


def due(state):
    return state['iteration']>=FIRST_GENERATION


def floor_multiplier(config):
    return LR_FLOOR/config['learning_rate']


def applied(config):
    controller=config.get('ppo_lr_controller')
    return controller is not None and controller['min_multiplier']==floor_multiplier(config)


def configure(config):
    """Raise only the floor; afterwards the committed recipe is returned unchanged."""
    if applied(config):return copy.deepcopy(config)
    from .lr_controller import ECONOMIC_CONTRACT,validate_config
    result=copy.deepcopy(config)
    controller=validate_config(result['ppo_lr_controller'],result['update'])
    require(controller['contract']==ECONOMIC_CONTRACT and controller['min_multiplier']==PREVIOUS_MIN_MULTIPLIER
        and controller['mode']=='apply' and floor_multiplier(result)<=1,
        'The 4.0e-6 PPO LR floor replaces the reviewed .5 multiplier floor of the applied controller')
    controller['min_multiplier']=floor_multiplier(result)
    validate_config(controller,result['update'])
    return result


def previous(config):
    """The recipe before this revision, used only when comparing reviewed transitions."""
    if not applied(config):return config
    result=copy.deepcopy(config)
    result['ppo_lr_controller']['min_multiplier']=PREVIOUS_MIN_MULTIPLIER
    return result


def compatible(before,after):
    """Adoption is one-way."""
    return not applied(before) or applied(after)


def summary(config):
    return dict(contract=CONTRACT,ppo_lr_floor=LR_FLOOR,min_multiplier=config['ppo_lr_controller']['min_multiplier'],
        base_learning_rate=config['learning_rate'],previous_floor=PREVIOUS_MIN_MULTIPLIER*config['learning_rate'])
