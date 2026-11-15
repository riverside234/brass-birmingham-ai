"""g2600 recipe: T1.3 openings, long C_scale .20 everywhere, PPG floor .25 and one epoch.

Adopted once at a committed boundary by ``teacher_search_retune.prepare(g2600=True)``,
which retires the teacher label and search-value buffers, re-measures teacher beta and
rewrites the committed checkpoint. Applied on top of the g2400 cotton-network recipe:

* PPO opening: temperature 1.3 for each neural seat's first three actions with the
  reviewed FORK_T15 objective (the pre-g2090 opening; ordinary branching stays off).
* Teacher tables: the same three opening actions at temperature 1.3 (visit counts
  raised to 1/1.3) for the single cotton seat, in training and held-out tables.
* Teacher search: long C_scale .20 on standard and cotton tables; short stays .70, K=8.
* PPG: beta floor .25 in every search-head stage; ordinary PPG at most one epoch.
"""
from brass_ai.learning.contracts import require
from . import exploration, opening_objective, training_revision_g1491 as teacher, training_revision_g2400 as network

GAME_OPENING = teacher.T13_GAME_OPENING
SEARCH = teacher.G2600_SEARCH
BETA_FLOOR = .25
MAX_EPOCHS = 1


def applied(config):
    revision = config.get('teacher_phase', {}).get(teacher.KEY, {})
    return revision.get('exploration', {}).get('game_opening') == GAME_OPENING


def configure(config):
    if applied(config):
        return config
    from brass_ai.rust_backend.lab_teacher_activation import is_on
    phase = config['teacher_phase']
    search = phase['search_value']
    value = config['value_phase']
    marker = phase[teacher.KEY]
    require(network.applied(config) and is_on(config) and teacher.count_search(config) == teacher.UPDATED_SEARCH
            and marker['exploration'].get('game_opening') == teacher.GAME_OPENING
            and config['opening'] == exploration.THREE_T2_OPENING
            and config['update']['opening_objective'] == opening_objective.T2_THREE
            and value['max_epochs'] == 2 and value['beta_clone_min'] == .5
            and search['schedule']['ppg_beta_floor_after_stage_b'] == .5 and search['max_epochs'] == 1,
            'The g2600 recipe requires the committed g2400 cotton-network recipe')
    schedule = dict(search['schedule'], ppg_beta_floor_after_stage_b=BETA_FLOOR)
    return dict(config, opening=dict(exploration.THREE_OPENING),
                update=dict(config['update'], opening_objective=dict(opening_objective.FORK_T15)),
                value_phase=dict(value, max_epochs=MAX_EPOCHS, beta_clone_min=BETA_FLOOR),
                teacher_phase=dict(phase, search_value=dict(search, schedule=schedule),
                                   **{teacher.KEY: teacher.g2600_marker(marker)}))


def summary(config):
    profile = teacher.count_search(config)
    return dict(ppo_opening=dict(temperature=config['opening']['temperature'],
                                 actions_per_seat=config['opening']['action_decisions_per_seat']),
                teacher_opening=dict(GAME_OPENING), long_c_scale=profile['c_scale'], short_c_scale=profile['fast_c_scale'],
                ppg_beta_floor=config['value_phase']['beta_clone_min'],
                ppg_beta_floor_after_stage_b=config['teacher_phase']['search_value']['schedule']['ppg_beta_floor_after_stage_b'],
                ordinary_ppg_max_epochs=config['value_phase']['max_epochs'])
