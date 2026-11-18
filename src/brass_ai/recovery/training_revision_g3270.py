"""g3270 teacher revision: build steps 2 and 3 of ai-research/TRAINING_IMPROVEMENT_PLAN_G2910.md,
with the user's modifications of 2026-11-18. Adopted once with --teacher-revision at a committed
boundary, then kept.

The qualified teacher recipe in ``teacher_phase`` (32+4 tables with held-out tables every fifth
iteration, the four-iteration 4:3:2:1 label buffer, 96 rows per PPO step, the search-value
schedule) stays exactly as it was reviewed and qualified: its validators, gradient-qualification
context and recipe recognition keep reading it. This top-level block overlays what the trainer
runs:

tables        40 tables every iteration: 36 train and 4 validation. The validation tables'
              labels are measured before and after the update (teacher KL and the policy's
              movement on the same rows), then train the policy from the next iteration, their
              age counted from then. Their search-value games join the search-value held-out
              set. The held-out tables of every fifth iteration go.
buffer        labels live 8 iterations, age weights 8, 7, ..., 1 (was 4 with 4:3:2:1).
targets       long-search labels only: short-search labels stay off for policy training. Each
              use rebuilds a label's target from the current policy,
              softmax(current logits + scale * (c1 + most visits) * completed value), at the
              label scale .20 (the long searches' own C_scale), so an old label adds only the
              search's preference and never pulls the policy back to an older one.
batches       128 label rows per PPO step, drawn by age weight without repeating a row inside
              the PPO update, each step's rows of similar legal width.
staleness     the per-age staleness check reads a fixed 256-row sample of each age.
parity        the model-logit parity check of new labels runs on 8 rows of each table.
search value  a 16-iteration training buffer and at most 32 steps (two 16-step intervals). The
              early stop reads the latest 32 dedicated held-out games, 4 new each iteration
              (at least 16 to run, so it continues while the set fills from the 16 already
              held out). A game that leaves the 32 trains the head for the rest of its 16
              iterations, its age counted from its collection.

The teacher beta is re-measured once, with teaching on, at adoption: rows per step, the pool and
the target all change the teacher gradient's size. Labels, the beta's qualification, the search
head and both buffers carry over.
"""
import copy
from itertools import takewhile

from brass_ai.learning.contracts import keys, require

KEY = 'teacher_revision'
CONTRACT = 'teacher-revision-g3270/1.0.0'
BASE_TABLES = dict(train=32, held_out=4)
BASE_BUFFER = dict(iterations=4, age_weights=[4, 3, 2, 1], max_staleness_ratio=1.)
BASE_ROWS_PER_STEP = 96
SETTINGS = dict(
    contract=CONTRACT,
    tables=dict(train=36, held_out=4),
    validation='every-iteration',
    heldout_labels='train-from-next-iteration',
    buffer=dict(iterations=8, age_weights=[8, 7, 6, 5, 4, 3, 2, 1], max_staleness_ratio=1.),
    rows_per_step=128,
    batches='no-repeat-width-grouped',
    targets=dict(kind='rebuilt-from-current-policy', label_scale=.2, labels='long-only'),
    staleness_sample_rows=256,
    parity=dict(rows_per_table=8),
    search_value=dict(buffer_iterations=16, steps_per_iteration=32, interval_steps=16,
                      heldout_games=32, heldout_minimum=16, retired_heldout='train-until-buffer-age-16'),
)


def validate(block):
    keys(block, (*SETTINGS, 'from_generation'))
    require({k: v for k, v in block.items() if k != 'from_generation'} == SETTINGS
            and type(block['from_generation']) is int and block['from_generation'] >= 0,
            'Invalid g3270 teacher revision')
    return block


def applied(config): return KEY in (config or {})


def block(config):
    return validate(config[KEY]) if applied(config) else None


def configure(config, generation):
    """`config` with the revision adopted at committed generation `generation`; kept unchanged afterwards."""
    if applied(config):
        validate(config[KEY])
        return copy.deepcopy(config)
    from . import training_revision_g1491 as g1491
    from .search_value_schedule import dedicated_holdout
    phase = config.get('teacher_phase', {})
    require(phase.get('mode') == 'on' and g1491.variant_of(config) == g1491.SEARCH_REFRESH and
            phase[g1491.KEY]['labels'] == dict(searches='long-only') and
            phase['tables'] == BASE_TABLES and phase['buffer'] == BASE_BUFFER and
            phase['term']['rows_per_step'] == BASE_ROWS_PER_STEP and
            dedicated_holdout(phase.get('search_value', {})),
            'The g3270 teacher revision starts from the teacher-on search-refresh recipe: '
            '32+4 tables, the 4:3:2:1 buffer, 96 rows and dedicated search-value held-out games')
    require(type(generation) is int and generation >= 0, 'Invalid adoption generation')
    require(SETTINGS['rows_per_step'] <= config['update']['effective_batch_size'], 'Teacher rows exceed effective_batch_size')
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

def phase_options(config, options):
    """The teacher-phase options the trainer runs: the qualified ones with this revision's overlay."""
    if not applied(config): return options
    result = copy.deepcopy(options)
    result['tables'] = dict(SETTINGS['tables'])
    result['buffer'] = copy.deepcopy(SETTINGS['buffer'])
    result['term'] = dict(result['term'], rows_per_step=SETTINGS['rows_per_step'])
    return result


def tables(config):
    t = SETTINGS['tables'] if applied(config) else config['teacher_phase']['tables']
    return dict(train=t['train'], held_out=t['held_out'])


def every_iteration(config):
    """Validation tables every iteration (no held-out tables every fifth iteration)."""
    return applied(config)


def heldout_lag(config):
    """Validation labels train from this many iterations after their own; None: never."""
    return 1 if applied(config) else None


def term_settings(config):
    if not applied(config): return None
    s = SETTINGS
    return dict(label_scale=s['targets']['label_scale'], batches=s['batches'],
                staleness_sample_rows=s['staleness_sample_rows'])


def parity_rows(config):
    return SETTINGS['parity']['rows_per_table'] if applied(config) else None


def value_settings(config):
    return copy.deepcopy(SETTINGS['search_value']) if applied(config) else None


def search_value_options(config, options):
    """Search-value options with at most 32 steps, two 16-step intervals."""
    if not applied(config): return options
    v = SETTINGS['search_value']
    require(options['trunk']['interval_steps'] == v['interval_steps'], 'Search-value interval length differs')
    return dict(copy.deepcopy(options), steps_per_iteration=v['steps_per_iteration'],
                trunk=dict(options['trunk'], intervals=v['steps_per_iteration'] // v['interval_steps']))


def teacher_weight(config, state, weight):
    """Request one teacher-on beta re-measurement at adoption, including recovery before the next checkpoint."""
    from . import adaptive_teacher_beta as adaptive
    if not applied(config) or not weight:
        return weight
    generation = state['iteration']
    request = weight.get(adaptive.REMEASUREMENT)
    if request and request['requested_generation'] >= generation:
        return weight
    if applied(state['signature']['config']):
        transitions = takewhile(lambda t: t['iteration'] == generation, reversed(state.get('source_transitions', ())))
        if all(applied(t['from_signature']['config']) for t in transitions):
            return weight
    return dict(weight, **{adaptive.REMEASUREMENT: dict(requested_generation=generation, previous_beta=weight['beta'])})


def tag(config):
    return 'teacher40-rebuilt-rows128-buf8-sv16' if applied(config) else 'none'


def summary(config):
    return copy.deepcopy(config.get(KEY))
