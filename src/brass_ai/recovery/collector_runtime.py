"""Process-local collector options, separate from frozen learning identities.

Training installs the explicit execution settings (or their historical defaults)
before scheduling work. Spawned workers receive a detached copy; their threads
share it. Environment switches remain available to standalone diagnostic tools.
"""
from contextlib import contextmanager
import os

from brass_ai.learning.contracts import require

DEFAULTS = dict(collector_validation='production_fast', replay_bindings=True)
# Operational and optional: absent keeps every historical settings shape exact.
OPTIONAL = ('native_inference',)
RUNTIME_KEYS = frozenset(DEFAULTS) | frozenset(OPTIONAL)
MODES = ('production_fast', 'strict', 'profile')
CPU_CONTRACT = 'brass-lab-cpu-inference/1.0.0'
# Quick searches label nothing, so only their K may differ from the job's 4.
TEACHER_BUDGET = dict(mode='count', full=640, full_k=16, fast=96, fast_k=8, c_scale=1.)
# The historical Gumbel constant, used by every search without a c_scale field.
DEFAULT_C_SCALE = .1
C_SCALES = (DEFAULT_C_SCALE, 1.)
_current = None


def cpu_block(*, workers=64, teacher_budget=None, pinning='one-lane-per-cpu', model_cache=8, overlap=True, readers=24):
    """Opt-in CPU inference. Absent or None keeps today's GPU path unchanged."""
    return dict(contract=CPU_CONTRACT, ordinary='cpu-ort', teacher='cpu-ort', rating='cpu-ort', pinning=pinning,
                workers=workers, model_cache=model_cache,
                teacher_budget=teacher_budget or dict(TEACHER_BUDGET), overlap=overlap, readers=readers)


# Blocks recorded before the overlapped schedule existed stay valid and sequential.
_RECORDED_BLOCK = frozenset(cpu_block()) - {'overlap', 'readers'}


def _valid_count_budget(budget):
    # Recorded configs omit K (before it was selectable) and c_scale (before it was).
    if set(budget) not in ({'mode', 'full', 'fast'}, set(TEACHER_BUDGET)-{'c_scale'}, set(TEACHER_BUDGET)):
        return False
    return (type(budget['full']) is int and 64 <= budget['full'] <= 4096 and
            type(budget['fast']) is int and 8 <= budget['fast'] <= 4096 and
            budget.get('full_k', 16) == 16 and budget.get('fast_k', 4) in (2, 4, 8, 16) and
            type(budget.get('c_scale', DEFAULT_C_SCALE)) is float and budget.get('c_scale', DEFAULT_C_SCALE) in C_SCALES)


def validate_cpu(block):
    require(type(block) is dict and set(block) in (_RECORDED_BLOCK, set(cpu_block())) and block['contract'] == CPU_CONTRACT and
            all(block[k] in ('cpu-ort', 'gpu') for k in ('ordinary', 'teacher', 'rating')) and
            block['pinning'] in ('one-lane-per-cpu', 'none') and
            type(block['workers']) is int and 1 <= block['workers'] <= 256 and
            type(block['model_cache']) is int and 4 <= block['model_cache'] <= 64 and
            type(block.get('overlap', False)) is bool and
            type(block.get('readers', 16)) is int and 1 <= block.get('readers', 16) <= 64, 'Invalid native_inference block')
    budget = block['teacher_budget']
    require(type(budget) is dict and ((budget.get('mode') == 'timed' and set(budget) == {'mode'}) or
            (budget.get('mode') == 'count' and _valid_count_budget(budget))), 'Invalid teacher search budget')


def validate(execution):
    if 'collector_validation' in execution:
        require(type(execution['collector_validation']) is str and execution['collector_validation'] in MODES,
                'Invalid collector_validation')
    if 'replay_bindings' in execution:
        require(type(execution['replay_bindings']) is bool, 'Invalid replay_bindings')
    if execution.get('native_inference') is not None:
        validate_cpu(execution['native_inference'])


def from_config(config):
    execution = config.get('execution', {})
    validate(execution)
    result = {k: execution.get(k, v) for k, v in DEFAULTS.items()}
    if execution.get('native_inference') is not None:
        result['native_inference'] = execution['native_inference']
    return result


def current():
    if _current is not None:
        return dict(_current)
    flag = os.environ.get('BRASS_REPLAY_BINDINGS', '1')
    require(flag in ('0', '1'), 'Invalid BRASS_REPLAY_BINDINGS')
    result = dict(collector_validation=os.environ.get('BRASS_COLLECTOR_VALIDATION', 'production_fast'),
                  replay_bindings=flag == '1')
    validate(result)
    return result


def _check_settings(settings):
    require(type(settings) is dict and set(DEFAULTS) <= set(settings) <= RUNTIME_KEYS, 'Invalid collector runtime settings')
    validate(settings)


def install(settings):
    global _current
    _check_settings(settings)
    _current = dict(settings)


def option(name):
    # This accessor is used per row/array. Snapshot copies belong at worker
    # launch boundaries, rather than in the replay capability hot path.
    return (_current if _current is not None else current())[name]


def native_inference():
    """The trainer's CPU-inference block, or None for today's GPU path."""
    return None if _current is None else _current.get('native_inference')


def teacher_c_scale():
    """The Gumbel c_scale of teacher searches on the path this process runs."""
    block = native_inference()
    return DEFAULT_C_SCALE if block is None else block['teacher_budget'].get('c_scale', DEFAULT_C_SCALE)


def overlap_collection():
    """True when teacher tables and ordinary games should share the CPUs at once."""
    block = native_inference()
    return bool(block and block.get('overlap', False) and block['teacher'] == 'cpu-ort' and block['ordinary'] == 'cpu-ort')


def reader_workers(default):
    """Replay readers for this process; the signed recipe's count without CPU inference."""
    block = native_inference()
    return default if block is None else block.get('readers', default)


def effective_execution(execution):
    """Worker topology for this process; the training signature never sees it."""
    block = native_inference()
    if block is None:
        return execution
    return dict(execution, processes=block['workers'], hosts_per_process=1,
                cpu_budget=block['workers'], cpu_reserve=0)


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
    if not RUNTIME_KEYS.intersection(execution):
        return config
    return dict(config, execution={k: v for k, v in execution.items() if k not in RUNTIME_KEYS})


def configured(config, settings=None):
    """Persist explicit defaults, preserving other execution and recipe fields."""
    settings = from_config(config) if settings is None else settings
    _check_settings(settings)
    kept = {k: v for k, v in config.get('execution', {}).items() if k not in OPTIONAL}
    return dict(config, execution=dict(kept, **settings))
