"""One reviewed recipe revision adopted once, at a committed boundary from g1491.

The policy teacher learns from both search profiles,
``L = .6 mean(L_long) + .4 mean(L_short)``, sampled 2048 rows per PPO step from a
two-iteration label buffer with age weights 4:3. Teacher tables go from 28 to 48
(plus the four held-out tables when due) and PPO runs one epoch. Teacher games
sample the played action from the root visit counts for each seat's first seven
actions of each era, and for every long-search action.

The first two iterations re-measure a fixed beta in shadow mode: no teacher
gradient is applied, the search head waits and the PPO learning-rate controller
holds. Teaching resumes from the third iteration through the reviewed on-switch.
"""
import copy

from brass_ai.learning.contracts import digest, require

CONTRACT = 'training-revision-g1491/1.0.0'
KEY = 'revision_g1491'
FIRST_GENERATION = 1491
TABLES = dict(train=48, held_out=4)
PREVIOUS_TABLES = dict(train=28, held_out=4)
BUFFER = dict(iterations=2, age_weights=[4, 3], max_staleness_ratio=1.)
PREVIOUS_BUFFER = dict(iterations=4, age_weights=[4, 3, 2, 1], max_staleness_ratio=1.)
ROWS_PER_STEP = 2048
PREVIOUS_ROWS_PER_STEP = 64
PPO_EPOCHS = 1
PREVIOUS_PPO_EPOCHS = 2
SHADOW_ITERATIONS = 2
PREVIOUS_SHADOW_ITERATIONS = 4
LONG_WEIGHT = .6
EXPLORATION = dict(contract='root-visit-count-sampling/1.0.0', temperature=1., opening_actions_per_era=7,
                   long_search='always')


def block():
    """The reviewed marker. Workers receive ``job_block()``, derived from it."""
    return dict(contract=CONTRACT,
                labels=dict(searches='long-and-short', long_weight=LONG_WEIGHT, short_weight=1-LONG_WEIGHT),
                exploration=dict(EXPLORATION),
                calibration=dict(shadow_iterations=SHADOW_ITERATIONS, ppo_lr='hold'))


def due(state):
    return state['iteration'] >= FIRST_GENERATION


def applied(config):
    return KEY in (config or {}).get('teacher_phase', {})


def validate(config):
    """Exact reviewed values; the projected recipe is validated by the caller."""
    c = config['teacher_phase']
    require(digest(c[KEY]) == digest(block()), 'Unreviewed g1491 training revision')
    require(c.get('contract') == 'teacher-phase/2.0.0' and c.get('seating') == 'all-seat-search',
            'The g1491 revision needs all-seat teacher search')
    require(digest(c['tables']) == digest(TABLES) and digest(c['buffer']) == digest(BUFFER),
            'g1491 teacher tables/buffer differ')
    require(type(c['term']['rows_per_step']) is int and c['term']['rows_per_step'] == ROWS_PER_STEP and
            c['term']['shadow_iterations'] == SHADOW_ITERATIONS, 'g1491 teacher rows/calibration window differ')
    require(type(config['update']['epochs']) is int and config['update']['epochs'] == PPO_EPOCHS,
            'g1491 PPO epochs differ')
    return c[KEY]


def project_phase(phase):
    """The pre-revision teacher fields, for validating the unchanged rest of the phase."""
    result = {k: copy.deepcopy(v) for k, v in phase.items() if k != KEY}
    result['tables'] = dict(PREVIOUS_TABLES)
    result['buffer'] = copy.deepcopy(PREVIOUS_BUFFER)
    result['term']['rows_per_step'] = PREVIOUS_ROWS_PER_STEP
    return result


def configure(config, state):
    """Adopt the revision in shadow mode; afterwards the committed recipe is returned unchanged."""
    if applied(config):
        return copy.deepcopy(config)
    require(due(state), 'The g1491 training revision starts at g1491 or later')
    from brass_ai.rust_backend.lab_teacher_activation import is_on
    from . import training_revision_g1294 as entropy, training_revision_g1307 as lr_floor
    require(is_on(config) and entropy.applied(config) and lr_floor.applied(config) and
            'schedule' in config['teacher_phase']['search_value'],
            'The g1491 revision changes the reviewed teacher-on recipe')
    c = config['teacher_phase']
    require(digest(c['tables']) == digest(PREVIOUS_TABLES) and digest(c['buffer']) == digest(PREVIOUS_BUFFER) and
            c['term']['rows_per_step'] == PREVIOUS_ROWS_PER_STEP and
            c['term']['shadow_iterations'] == PREVIOUS_SHADOW_ITERATIONS and
            config['update']['epochs'] == PREVIOUS_PPO_EPOCHS, 'The g1491 revision starts from 28 tables, 64 rows and two epochs')
    result = copy.deepcopy(config)
    phase = result['teacher_phase']
    phase.update(mode='shadow', tables=dict(TABLES), buffer=copy.deepcopy(BUFFER))
    phase['term'].update(rows_per_step=ROWS_PER_STEP, shadow_iterations=SHADOW_ITERATIONS)
    phase[KEY] = block()
    result['update']['epochs'] = PPO_EPOCHS
    from .config import validate_config
    validate_config(result)
    return result


def previous(config):
    """The recipe before this revision, used only when comparing reviewed transitions.

    Mode, share target and the two-iteration calibration window stay: the
    transition checks compare those explicitly.
    """
    if not applied(config):
        return config
    result = copy.deepcopy(config)
    result['teacher_phase'] = project_phase(result['teacher_phase'])
    result['update']['epochs'] = PREVIOUS_PPO_EPOCHS
    return result


def compatible(before, after):
    """Adoption is one-way and its marker can't change once committed."""
    return not applied(before) or (applied(after) and
        digest(before['teacher_phase'][KEY]) == digest(after['teacher_phase'][KEY]))


def shadow_iterations(config):
    """Fixed-beta qualification window of this recipe."""
    return SHADOW_ITERATIONS if applied(config) else PREVIOUS_SHADOW_ITERATIONS


def tables(config):
    t = config['teacher_phase']['tables']
    return dict(train=t['train'], held_out=t['held_out'])


def qualification_tables(config):
    """The table counts a qualifying shadow iteration records (Counter keys)."""
    t = tables(config)
    return {'train': t['train'], 'heldout': t['held_out']}


def long_weight(config):
    """Weight of the long-search mean; None keeps the single long-only pool."""
    return config['teacher_phase'][KEY]['labels']['long_weight'] if applied(config) else None


def _job_block(exploration):
    return dict(contract=CONTRACT, short_labels=True,
                exploration=dict(contract=exploration['contract'], temperature=exploration['temperature'],
                                 opening_actions_per_era=exploration['opening_actions_per_era'],
                                 long_search=exploration['long_search'] == 'always'))


def job_block(config):
    """What a CPU teacher worker needs; absent for every earlier recipe."""
    if not applied(config):
        return None
    return _job_block(config['teacher_phase'][KEY]['exploration'])


def validate_job(job):
    """A teacher job carries either no revision or exactly the reviewed one."""
    block = job.get('teacher_revision')
    require(block is None or (digest(block) == digest(_job_block(EXPLORATION)) and
            job.get('seating') == 'all-seat-search' and not job.get('h2h') and not job.get('shadow_probe')),
            'Unreviewed g1491 teacher job revision')
    return block


def holds_ppo_lr(config):
    """The PPO LR controller holds while this recipe re-measures beta in shadow mode."""
    return applied(config) and config['teacher_phase']['mode'] == 'shadow'


def summary(config):
    c = config['teacher_phase']
    return dict(contract=CONTRACT, mode=c['mode'], tables=tables(config), buffer=copy.deepcopy(c['buffer']),
        teacher_rows_per_step=c['term']['rows_per_step'], long_weight=long_weight(config),
        short_weight=c[KEY]['labels']['short_weight'], ppo_epochs=config['update']['epochs'],
        beta_calibration_iterations=c['term']['shadow_iterations'], exploration=copy.deepcopy(c[KEY]['exploration']),
        ppo_lr_during_calibration='hold')
