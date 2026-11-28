"""g3658: PPG beta floor .125 (was .25), and the PPG beta set to .125 once at adoption.

Adopted once at a committed boundary with --ppg-beta-floor 0.125 (continue-rust-g1491.py), then kept.
The recipe moves both floors that hold the ordinary value phase's cloning weight:

* value_phase.beta_clone_min .25 -> .125 (ppg-value-phase/26.0.0, the KL cap stops and keeps);
* the search-head schedule's Stage C floor, ppg_beta_floor_after_stage_b .25 -> .125
  (search_value_schedule.beta_floor; it otherwise kept beta at .25 in Stage C).

The first resume under the new floor sets the restored beta to .125 (ppg_g210.EIGHTH_FLOOR_MIGRATION),
keeping the trip history, both Adams, RNG, replay and cadence. Afterwards beta moves as before: a clean
phase below half the KL cap halves it down to .125, KL hits raise it. Collection, labels, search and
the teacher job identities do not change.
"""
import copy

from brass_ai.learning.contracts import require

FLOOR = .125
BEFORE = .25
STAGE_FLOOR = 'ppg_beta_floor_after_stage_b'


def _schedule(config):
    return config['teacher_phase']['search_value']['schedule']


def applied(config):
    """True on the lineage that adopted the .125 floor, including after g6670 raised it back to .25 (its recipe
    projection restores .125, so the transitions and the recipe stages read the same history)."""
    from .ppg_g210 import KL_STOP_CONTRACT
    from .training_revision_g6670 import raises_ppg_floor
    value = (config or {}).get('value_phase') or {}
    return value.get('contract') == KL_STOP_CONTRACT and (value.get('beta_clone_min') == FLOOR or raises_ppg_floor(config))


def configure(config):
    """The recipe with both floors at .125; a recipe already on it is returned unchanged."""
    from .ppg_g210 import KL_STOP_CONTRACT
    from .search_value_schedule import dedicated_holdout
    search = config['teacher_phase']['search_value']
    if applied(config):
        require(_schedule(config).get(STAGE_FLOOR) == FLOOR, 'The .125 PPG floor needs the search-head Stage C floor .125')
        return config
    value = config['value_phase']
    require(value['contract'] == KL_STOP_CONTRACT and value['beta_clone_min'] == BEFORE
            and dedicated_holdout(search) and _schedule(config).get(STAGE_FLOOR) == BEFORE,
            'The .125 PPG beta floor follows the committed .25 floor (KL-cap stop PPG, refreshed search-head schedule)')
    result = copy.deepcopy(config)
    result['value_phase']['beta_clone_min'] = FLOOR
    result['teacher_phase']['search_value']['schedule'][STAGE_FLOOR] = FLOOR
    return result


def previous(config):
    """The recipe with the .25 floors, used only when comparing reviewed transitions. With g6670 the value-phase
    floor is .25 already, and both sides of a comparison get the same .25 Stage C floor."""
    if not applied(config): return config
    result = copy.deepcopy(config)
    result['value_phase']['beta_clone_min'] = BEFORE
    schedule = result.get('teacher_phase', {}).get('search_value', {}).get('schedule')
    if schedule is not None and schedule.get(STAGE_FLOOR) == FLOOR: schedule[STAGE_FLOOR] = BEFORE
    return result


def summary(config):
    return dict(ppg_beta_floor=config['value_phase']['beta_clone_min'],
                stage_c_floor=_schedule(config).get(STAGE_FLOOR),
                adoption='beta set to .125 once at the first resume; trips, Adams, RNG and replay kept')
