"""Resume the seated g1190 restart with new opening sampling and a .5 PPG floor."""
from itertools import takewhile

from brass_ai.learning.contracts import require
from . import adaptive_teacher_beta, exploration, opening_objective, ordinary_cotton, training_revision_g1491 as teacher
from .search_value_schedule import dedicated_holdout


def applied(config):
    revision = config.get('teacher_phase', {}).get(teacher.KEY, {})
    return revision.get('exploration', {}).get('game_opening') == teacher.GAME_OPENING


def previous_opening(config):
    """Project the ordinary opening back to the reviewed teacher-on base."""
    if (applied(config) and config.get('opening') == exploration.THREE_T2_OPENING
            and config['update'].get('opening_objective') == opening_objective.T2_THREE):
        return dict(config, opening=exploration.THREE_OPENING,
                    update=dict(config['update'], opening_objective=opening_objective.FORK_T15))
    return config


def teacher_weight(config, state, weight):
    """Request one measurement at adoption, including recovery before the next checkpoint."""
    if not applied(config) or not weight:
        return weight
    generation = state['iteration']
    request = weight.get(adaptive_teacher_beta.REMEASUREMENT)
    if request and request['requested_generation'] >= generation:
        return weight
    if applied(state['signature']['config']):
        transitions = takewhile(lambda t: t['iteration'] == generation,
                                reversed(state.get('source_transitions', ())))
        if all(applied(t['from_signature']['config']) for t in transitions):
            return weight
    return dict(weight, **{adaptive_teacher_beta.REMEASUREMENT:
                           dict(requested_generation=generation, previous_beta=weight['beta'])})


def configure(config):
    phase = config['teacher_phase']
    search = phase['search_value']
    value = config['value_phase']
    adopted = applied(config)
    cotton = (ordinary_cotton.LIMITED if teacher.cotton_only(config) else
              dict(ordinary_cotton.SETTINGS, probability=.10 if adopted else .15))
    require(teacher.seated_cotton(config) and phase['mode'] == 'on' and dedicated_holdout(search)
            and value['max_epochs'] == 1 and config['branching']['rate_per_decision'] == 0,
            'Opening sampling revision requires the seated teacher-on recipe and one-epoch PPG')
    require(config['opening'] == (exploration.THREE_T2_OPENING if adopted else exploration.THREE_OPENING)
            and config['update']['opening_objective'] == (opening_objective.T2_THREE if adopted else opening_objective.FORK_T15)
            and config['table_plan'].get(ordinary_cotton.KEY) == cotton
            and value['beta_clone_min'] == search['schedule']['ppg_beta_floor_after_stage_b'] == (.5 if adopted else .25),
            'Opening sampling revision needs the complete preceding or revised recipe')
    if adopted:
        return config
    marker = phase[teacher.KEY]
    teacher_opening = dict(marker['exploration'], game_opening=dict(teacher.GAME_OPENING))
    schedule = dict(search['schedule'], ppg_beta_floor_after_stage_b=.5)
    return ordinary_cotton.configure(dict(config, opening=dict(exploration.THREE_T2_OPENING),
        update=dict(config['update'], opening_objective=dict(opening_objective.T2_THREE)),
        value_phase=dict(value, beta_clone_min=.5),
        teacher_phase=dict(phase, search_value=dict(search, schedule=schedule),
                           **{teacher.KEY: dict(marker, exploration=teacher_opening)})), probability=.10)
