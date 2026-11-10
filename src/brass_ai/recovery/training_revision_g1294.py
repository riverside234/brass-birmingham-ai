"""One reviewed recipe revision adopted once, at a committed boundary after g1294.

PPO entropy rises from .001 to .005 over the next 100 iterations; a hit on the PPG
KL cap stops that phase and keeps the update; the search head trains at most 64
steps per iteration. Weights, Adam, RNG, replay and the anneal history are kept.
"""
import copy

from brass_ai.learning.contracts import require

CONTRACT='training-revision-g1294/1.0.0'
# Older boundaries (restores, historical fixtures) keep their recipe.
FIRST_GENERATION=1294


def due(state):
    return state['iteration']>=FIRST_GENERATION


def applied(config):
    from .ppg_g210 import keeps_update_at_kl_cap
    return keeps_update_at_kl_cap(config.get('value_phase',{}))


def raise_block(config):
    from . import entropy_schedule as S
    return config['update'][S.KEY].get(S.RAISE)


def configure(config,state):
    """Add the revision once; afterwards the committed recipe is returned unchanged."""
    from . import entropy_schedule as S,ppg_g210 as P
    from .search_value_config import max_steps_options
    if applied(config):return copy.deepcopy(config)
    require(due(state),'The g1294 training revision starts at g1294 or later')
    schedule=S.restore_run(config,state)
    require(schedule['config']['mode']=='on' and schedule['progress']==schedule['config']['ramp_iterations'] and not S.waiting(schedule),
        'The training revision needs the completed entropy anneal and reward steps')
    result=max_steps_options(P.kl_stop_options(S.raise_config(config,state['iteration'])))
    S.validate(result)
    return result


def previous(config):
    """The recipe before this revision, used only when comparing reviewed transitions."""
    if not applied(config):return config
    from . import entropy_schedule as S,ppg_g210 as P
    from .search_value_config import MAX_STEPS,defaults
    options=config['teacher_phase']['search_value']
    require(raise_block(config) is not None and options['steps_per_iteration']==MAX_STEPS,'Incomplete g1294 training revision')
    result=S.without_raise(config)
    result['value_phase'].update(contract=P.TEACHER_FLOOR_CONTRACT)
    original=defaults()
    result['teacher_phase']['search_value'].update(steps_per_iteration=original['steps_per_iteration'])
    result['teacher_phase']['search_value']['trunk']['intervals']=original['trunk']['intervals']
    return result


def compatible(before,after):
    """Adoption is one-way, and the raise cannot be re-timed once it is committed."""
    return not applied(before) or (applied(after) and raise_block(before)==raise_block(after))


def summary(config):
    from . import entropy_schedule as S
    block=raise_block(config)
    return dict(contract=CONTRACT,entropy=dict(start_generation=block['start_generation'],ramp_iterations=block['ramp_iterations'],
        start=S.RAISE_FROM,final=S.RAISE_TARGET),ppg_kl_cap='stop-and-keep',search_head_max_steps=config['teacher_phase']['search_value']['steps_per_iteration'])
