"""g2880: search-value returns use lambda .95 (was .8 since the 32-wide head refresh).

Each training fit recomputes every buffered game's lambda-return from its stored
per-turn utility, root bootstrap and final utility, so the existing buffer is
retargeted exactly; nothing retires. Dedicated held-out games score the realized
final utility-to-go, which does not depend on lambda. The head, stage, blend,
trunk scale, optimizer and teacher labels are kept. Only .8 -> .95 is allowed.
"""
from brass_ai.learning.contracts import require
from . import training_revision_g2654 as g2654

LAMBDA = .95
PREVIOUS = .8


def _target(config):
    return config.get('teacher_phase', {}).get('search_value', {}).get('target', {})


def applied(config):
    return _target(config).get('lambda') == LAMBDA


def configure(config):
    if applied(config):
        return config
    from .search_value_schedule import dedicated_holdout
    phase = config['teacher_phase']
    options = phase['search_value']
    require(g2654.applied(config) and dedicated_holdout(options) and options['target']['lambda'] == PREVIOUS,
            'Search-value lambda .95 requires the g2654 recipe with the dedicated-holdout head at lambda .8')
    target = dict(options['target'], **{'lambda': LAMBDA})
    return dict(config, teacher_phase=dict(phase, search_value=dict(options, target=target)))


def summary(config):
    return dict(search_value_lambda=_target(config).get('lambda'))
