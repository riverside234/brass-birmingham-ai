"""g4551 table revision (the user's request of 2026-11-22; doc ai-research/G4551_TABLE_REVISION.md).
Adopted once with --table-revision at a committed boundary, then kept.

self-play     every PPO self-play table is a standard game. Until now the ordinary cotton recipe
              (table_plan.cotton_exploration, g3528) gave 32 of the 224 self-play tables one cotton
              exploration seat. The 70/30 self-play/historical mix, every table's seeds, policies and
              opponents, and the rest of the table plan are unchanged. The teacher's cotton tables stay as
              they are (teacher_phase, g3528 rule, 3/20 of the train tables and one of the 4 held-out).
tables        54 teacher tables every iteration: 50 train and 4 validation (the g4138 revision ran 44 + 4).
              Each table holds one of the 64 CPU lanes while it searches; ordinary games use the other 10
              and take each table's CPU as it finishes.
rows          176 teacher label rows per PPO step (the g4475 revision drew 160), drawn by age weight without
              repeating a row inside the PPO update and grouped by legal width, as before. The soft head
              trains on the same rows. The adaptive teacher beta carries over.

Kept: the label buffer and its identity, the search profile, the teacher beta, the soft head, the paused
search-value head and its stage, the era heads' search tables (now 50 train tables an iteration) and the
32 rotating held-out games. The qualified recipe (teacher_phase, table_plan) is not edited; this top-level
block overlays what the planner, the admission audit and the trainer run.
"""
import copy

from brass_ai.learning.contracts import keys, require

KEY = 'table_revision'
CONTRACT = 'table-revision-g4551/1.0.0'
BASE_TABLES = dict(train=44, held_out=4)  # the g4138 revision
BASE_ROWS_PER_STEP = 160  # the g4475 revision
MAX_TABLES = 64  # one CPU lane each (collector_runtime.MAX_WORKERS)
SETTINGS = dict(
    contract=CONTRACT,
    selfplay_cotton='off',
    tables=dict(train=50, held_out=4),
    rows_per_step=176,
)


def validate(block):
    keys(block, (*SETTINGS, 'from_generation'))
    require({k: v for k, v in block.items() if k != 'from_generation'} == SETTINGS
            and type(block['from_generation']) is int and block['from_generation'] >= 0,
            'Invalid g4551 table revision')
    return block


def applied(config): return KEY in (config or {})


def configure(config, generation):
    """`config` with the revision adopted at committed generation `generation`; kept unchanged afterwards."""
    if applied(config):
        validate(config[KEY])
        return copy.deepcopy(config)
    from . import training_revision_g4138 as g4138, training_revision_g4475 as g4475
    require(g4138.applied(config) and g4475.applied(config) and config['teacher_phase'].get('mode') == 'on' and
            g4138.SETTINGS['tables'] == BASE_TABLES and g4475.SETTINGS['rows_per_step'] == BASE_ROWS_PER_STEP,
            'The g4551 table revision follows the g4138 teacher search revision (44+4 tables) and the g4475 teacher '
            'training revision (160 rows per PPO step) with teaching on')
    t = SETTINGS['tables']
    require(t['train'] + t['held_out'] <= MAX_TABLES, 'Teacher tables exceed the CPU lanes')
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

def tables(config):
    """The adopted teacher table counts, or None (the g4138 or g3270 counts apply)."""
    return dict(SETTINGS['tables']) if applied(config) else None


def rows_per_step(config):
    """Teacher label rows per PPO step when adopted, or None (the g4475 or g3270 count applies)."""
    return SETTINGS['rows_per_step'] if applied(config) else None


def selfplay_cotton_off(config):
    """True when no PPO self-play table explores cotton (the teacher's cotton tables are unaffected)."""
    return applied(config)


def tag(config):
    return 'teacher54-rows176-selfplay-standard' if applied(config) else 'none'


def summary(config):
    return copy.deepcopy(config.get(KEY))
