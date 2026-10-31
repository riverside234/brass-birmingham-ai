"""Versioned policy-only teacher phase. Historical Masked recipes stay valid."""
import copy
import math

from brass_ai.learning.contracts import digest, require
from brass_ai.rust_backend.lab_contract import ALGORITHM, SESSION, SOURCE, ARCHIVE, admit_search_request

CONTRACT = 'teacher-phase/1.0.0'
LABEL_SOURCE = 'search-v1-improved-policy/1.0.0'
SCHEDULE = 'four-phase-teacher-turns/2.0.0'
MIGRATION = 'teacher-phase-adoption/1.0.0'
PROFILE = dict(contract='brass-lab-search-v1/1.0.0', method='search_v1', algorithm=ALGORITHM,
               session_contract=SESSION, v205_enabled=False, native_source_sha256=SOURCE,
               source_archive_sha256=ARCHIVE)


def defaults(reward, *, diversity=False):
    result = dict(contract=CONTRACT, mode='off', tables=dict(train=28, held_out=4), schedule=SCHEDULE,
        unsearched=dict(temperature=1., epsilon=0., opening_temperature=None, branching=False),
        search=dict(method='search_v1', algorithm=ALGORITHM, session=SESSION, v205_enabled=False,
            allowance_seconds=4., sessions=1, threads_per_session=1, pending_paths=1, k=16,
            c1=50, c2=.1, gumbel_noise=True, prior_temperature=1., watchdog_seconds=30.,
            reward=copy.deepcopy(reward)),
        labels=dict(source=LABEL_SOURCE, decisions='action-stage-both-actions', evidence='completed-trajectories'),
        buffer=dict(iterations=4, age_weights=[4, 3, 2, 1], max_staleness_ratio=1.),
        term=dict(rows_per_step=64, share_target=.25, shadow_iterations=2, beta_bounds=[.01, 10.]))
    if diversity:
        from . import teacher_diversity as D
        result.update(contract=D.CONTRACT, diversity=D.options())
        result['labels']['source'] = D.LABEL
    return result


def number(value, low, high, name):
    require(type(value) in (int, float) and math.isfinite(value) and low < value <= high,
            'Invalid teacher phase ' + name)


def validate(config):
    from .utility import reward_spec
    from .config import teacher_value_config
    from . import teacher_diversity as D
    c = config['teacher_phase']; expected = defaults(reward_spec(config), diversity=D.enabled(config))
    require(isinstance(expected['search']['reward'], dict), 'Teacher phase requires the combined PPO utility reward')
    require(type(c) is dict and set(c) == set(expected) and c['contract'] == expected['contract'],
            'Invalid teacher phase contract/fields')
    require(config.get('teacher_enabled') is False, 'Teacher phase excludes legacy teacher_enabled')
    require(c['mode'] in ('off', 'shadow', 'on'), 'Invalid teacher phase mode')
    for key in ('tables', 'unsearched', 'labels', 'search'):
        # Canonical JSON distinguishes booleans from numeric settings.
        require(digest(c[key]) == digest(expected[key]), 'Unreviewed teacher phase ' + key)
    require(c['schedule'] == SCHEDULE, 'Invalid teacher phase schedule')
    if D.enabled(config):
        require(digest(c['diversity']) == digest(D.options()) and config.get('diversity',{}).get('mode') == 'apply',
                'Unreviewed teacher diversity objective or missing DOMiNO apply')
        require(not teacher_value_config(config.get('teacher_value'))['enabled'], 'Diversity teacher is policy-only')
    # The dispatch API forbids override keys (even v205_enabled=False). Verify
    # the full phase above, then pass its method/settings without that marker.
    admit_search_request(PROFILE, {k: v for k, v in c['search'].items() if k != 'v205_enabled'})
    require(config['update'].get('actor_nonforced') is True, 'Teacher phase requires guarded nonforced PPO')
    require(c['mode'] == 'off' or not teacher_value_config(config.get('teacher_value'))['enabled'],
            'Teacher phase is policy-only; disable teacher_value')
    b, t = c['buffer'], c['term']
    require(type(b) is dict and set(b) == set(expected['buffer']) and type(b['iterations']) is int
            and b['iterations'] == 4, 'Invalid teacher buffer window')
    weights = b['age_weights']
    require(type(weights) is list and len(weights) == b['iterations'], 'Invalid teacher age weights')
    for w in weights: number(w, 0, 1e6, 'age weight')
    require(weights == sorted(weights, reverse=True), 'Teacher age weights must be non-increasing')
    number(b['max_staleness_ratio'], 0, 100, 'staleness ratio')
    require(type(t) is dict and set(t) == set(expected['term']), 'Invalid teacher term fields')
    require(type(t['rows_per_step']) is int and 1 <= t['rows_per_step'] <= config['update']['effective_batch_size'],
            'Teacher rows exceed effective_batch_size')
    number(t['share_target'], 0, 1, 'share target')
    require(type(t['shadow_iterations']) is int and t['shadow_iterations'] == 2
            and digest(t['beta_bounds']) == digest([.01, 10.]), 'Invalid teacher calibration settings')
    return c


def adopt(config):
    from .config import teacher_value_config
    from .utility import reward_spec
    result = copy.deepcopy(config)
    require(result.get('teacher_enabled') is False, 'Disable legacy teachers before adoption')
    result.setdefault('teacher_phase', defaults(reward_spec(result)))
    result['teacher_value'] = dict(teacher_value_config(result.get('teacher_value')), enabled=False)
    validate(result)
    return result


def toggle_compatible(before, after):
    """Exactly mode/share changes; never adoption, source or reward changes."""
    if 'teacher_phase' not in before or 'teacher_phase' not in after: return False
    try:
        validate(before); validate(after)
    except (ValueError, KeyError, TypeError): return False
    expected = copy.deepcopy(before)
    expected['teacher_phase']['mode'] = after['teacher_phase']['mode']
    expected['teacher_phase']['term']['share_target'] = after['teacher_phase']['term']['share_target']
    return expected == after


def mode_only_signature(before, after):
    return ({k: v for k, v in before.items() if k != 'config'} ==
            {k: v for k, v in after.items() if k != 'config'} and
            toggle_compatible(before['config'], after['config']))


def buffer_identity(config, *, regime=None):
    """Learning-rate, exploration, entropy, mode and PPG changes retain labels."""
    from brass_ai.learning.contracts import compatibility
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    from .domino_state import regime as evidence_regime
    c = validate(config)
    return digest(dict(learning=compatibility(), architecture=config['architecture'],
        model_size=config['model_size'], backend=PROFILE, native_teacher=source_fingerprint(),
        search=c['search'], labels=c['labels'], schedule=c['schedule'],
        **(dict(diversity=c['diversity'], descriptor=config['diversity']['descriptor_manifest_sha256'],
                regime=regime or evidence_regime(config))
           if 'diversity' in c else {})))
