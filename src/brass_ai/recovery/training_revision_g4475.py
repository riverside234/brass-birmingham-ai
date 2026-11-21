"""g4475 teacher training revision (the user's request of 2026-11-21; doc ai-research/G4475_TEACHER_TRAINING.md).
Adopted once with --teacher-training-revision at a committed boundary, then kept. Trainer-only: the collection,
the CPU teacher, its tables and every recorded table output are unchanged.

rows          160 teacher label rows per PPO step (was 128), still drawn by age weight without repeating a row
              inside the PPO update and grouped by legal width. The soft head trains on the same rows. The
              adaptive teacher beta carries over and keeps the 0.1 share target from its sampled norms.
search value  the search-value head's training is paused: no post-PPG pass (no head or trunk steps, no
              guarded KL or clone probes), no held-out predictions and no stage advance. The head, its Adam,
              its stage and lambda stay as committed. The search tables still publish their search-value shards
              (roots, extra rows and era-end maps), aged-out shards still retire, and the era heads keep
              training on the search tables (search-aux 2.0.0) with the same pool and held-out set.
"""
import copy

from brass_ai.learning.contracts import keys, require

KEY = 'teacher_training_revision'
CONTRACT = 'teacher-training-g4475/1.0.0'
BASE_ROWS_PER_STEP = 128  # the g3270 revision
SETTINGS = dict(
    contract=CONTRACT,
    rows_per_step=160,
    search_value=dict(head='paused', era_heads='trained-on-search-tables'),
)


def validate(block):
    keys(block, (*SETTINGS, 'from_generation'))
    require({k: v for k, v in block.items() if k != 'from_generation'} == SETTINGS
            and type(block['from_generation']) is int and block['from_generation'] >= 0,
            'Invalid g4475 teacher training revision')
    return block


def applied(config): return KEY in (config or {})


def configure(config, generation):
    """`config` with the revision adopted at committed generation `generation`; kept unchanged afterwards."""
    if applied(config):
        validate(config[KEY])
        return copy.deepcopy(config)
    from . import training_revision_g3270 as g3270
    from .search_value_schedule import dedicated_holdout
    require(g3270.applied(config) and config['teacher_phase'].get('mode') == 'on' and
            g3270.SETTINGS['rows_per_step'] == BASE_ROWS_PER_STEP and
            dedicated_holdout(config['teacher_phase'].get('search_value', {})),
            'The g4475 teacher training revision follows the g3270 teacher revision with teaching on '
            '(128 rows per PPO step, the dedicated search-value schedule)')
    require(SETTINGS['rows_per_step'] <= config['update']['effective_batch_size'], 'Teacher rows exceed effective_batch_size')
    require(type(generation) is int and generation >= 0, 'Invalid adoption generation')
    result = copy.deepcopy(config)
    result[KEY] = validate(dict(copy.deepcopy(SETTINGS), from_generation=generation))
    from .config import validate_config
    validate_config(result)
    return result


def previous(config):
    """The recipe without the revision (recipe recognition)."""
    if not applied(config): return config
    result = copy.deepcopy(config); result.pop(KEY)
    return result


def project(old, new):
    """`old` with `new`'s revision when it is adopted or kept; None when it is removed or changed."""
    a, b = (old or {}).get(KEY), (new or {}).get(KEY)
    if a == b: return old
    if a is not None: return None
    try:
        validate(b)
    except (ValueError, KeyError, TypeError):
        return None
    result = copy.deepcopy(old); result[KEY] = copy.deepcopy(b)
    return result


# ---- settings read by the training code --------------------------------------------------------

def rows_per_step(config):
    """Teacher label rows per PPO step when adopted, or None (the g3270 revision's 128 apply)."""
    return SETTINGS['rows_per_step'] if applied(config) else None


def search_value_paused(config):
    """True when the search-value head's training is paused (the shards, their retirement and the era heads go on)."""
    return applied(config)


def tag(config):
    return 'rows160-svhead-paused' if applied(config) else 'none'


def summary(config):
    return copy.deepcopy(config.get(KEY))
