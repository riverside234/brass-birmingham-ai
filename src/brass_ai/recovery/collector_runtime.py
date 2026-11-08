"""Process-local collector options, separate from frozen learning identities.

Training installs the explicit execution settings (or their historical defaults)
before scheduling work. Spawned workers receive a detached copy; their threads
share it. Environment switches remain available to standalone diagnostic tools.
"""
from contextlib import contextmanager
import os

from brass_ai.learning.contracts import require

DEFAULTS = dict(collector_validation='production_fast', replay_bindings=True)
MODES = ('production_fast', 'strict', 'profile')
_current = None


def validate(execution):
    if 'collector_validation' in execution:
        require(type(execution['collector_validation']) is str and execution['collector_validation'] in MODES,
                'Invalid collector_validation')
    if 'replay_bindings' in execution:
        require(type(execution['replay_bindings']) is bool, 'Invalid replay_bindings')


def from_config(config):
    execution = config.get('execution', {})
    validate(execution)
    return {k: execution.get(k, v) for k, v in DEFAULTS.items()}


def current():
    if _current is not None:
        return dict(_current)
    flag = os.environ.get('BRASS_REPLAY_BINDINGS', '1')
    require(flag in ('0', '1'), 'Invalid BRASS_REPLAY_BINDINGS')
    result = dict(collector_validation=os.environ.get('BRASS_COLLECTOR_VALIDATION', 'production_fast'),
                  replay_bindings=flag == '1')
    validate(result)
    return result


def install(settings):
    global _current
    require(type(settings) is dict and set(settings) == set(DEFAULTS), 'Invalid collector runtime settings')
    validate(settings)
    _current = dict(settings)


def option(name):
    # This accessor is used per row/array. Snapshot copies belong at worker
    # launch boundaries, rather than in the replay capability hot path.
    return (_current if _current is not None else current())[name]


@contextmanager
def scope(settings):
    global _current
    previous = _current
    install(settings)
    try:
        yield
    finally:
        _current = previous


def recipe(config):
    """Remove only operational fields; pending jobs retain their exact recipe."""
    execution = config.get('execution', {})
    validate(execution)
    if not set(DEFAULTS).intersection(execution):
        return config
    return dict(config, execution={k: v for k, v in execution.items() if k not in DEFAULTS})


def configured(config, settings=None):
    """Persist explicit defaults, preserving other execution and recipe fields."""
    settings = from_config(config) if settings is None else settings
    require(type(settings) is dict and set(settings) == set(DEFAULTS), 'Invalid collector runtime settings')
    validate(settings)
    return dict(config, execution=dict(config.get('execution', {}), **settings))
