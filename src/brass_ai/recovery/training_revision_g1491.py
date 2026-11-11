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

The ``g1490-branch-restart`` variant is the same exploration in a new run that
restarts from the g1490 weights: long-search labels only, 32 tables, 72 rows per
PPO step, the four-iteration 4:3:2:1 buffer, two PPO epochs and a single shadow
iteration that re-measures beta. It is adopted only by that branch restart. The ``g1190-branch-restart-256-rows``
variant is the same with 256 teacher rows per PPO step.
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
RESTART = 'g1490-branch-restart'
# The same restart with 256 teacher rows per PPO step (the g1190 restart's second attempt).
WIDE_RESTART = 'g1190-branch-restart-256-rows'
RESTARTS = (RESTART, WIDE_RESTART)
# Exact values of each reviewed variant; None is the in-run g1491 adoption.
VARIANTS = {
    None: dict(tables=TABLES, buffer=BUFFER, rows_per_step=ROWS_PER_STEP, ppo_epochs=PPO_EPOCHS,
               shadow_iterations=SHADOW_ITERATIONS, short_labels=True),
    RESTART: dict(tables=dict(train=32, held_out=4), buffer=PREVIOUS_BUFFER, rows_per_step=72,
                  ppo_epochs=PREVIOUS_PPO_EPOCHS, shadow_iterations=1, short_labels=False),
}
VARIANTS[WIDE_RESTART] = dict(VARIANTS[RESTART], rows_per_step=256)


def block(*, cotton=True, variant=None):
    """The reviewed marker. Workers receive ``job_block()``, derived from it."""
    require(variant in VARIANTS, 'Unknown g1491 revision variant')
    exploration=copy.deepcopy(EXPLORATION)
    if cotton:
        from .teacher_cotton import settings
        exploration['cotton']=settings()
    labels = (dict(searches='long-and-short', long_weight=LONG_WEIGHT, short_weight=1-LONG_WEIGHT) if variant is None
              else dict(searches='long-only'))
    result = dict(contract=CONTRACT, labels=labels, exploration=exploration,
                  calibration=dict(shadow_iterations=VARIANTS[variant]['shadow_iterations'], ppo_lr='hold'))
    if variant is not None:
        result['variant'] = variant
    return result


def due(state):
    return state['iteration'] >= FIRST_GENERATION


def applied(config):
    return KEY in (config or {}).get('teacher_phase', {})


def variant_of(config):
    """The reviewed variant of an applied revision (None: the in-run g1491 adoption)."""
    return config['teacher_phase'][KEY].get('variant') if applied(config) else None


def settings(config):
    return VARIANTS[variant_of(config)]


def validate(config):
    """Exact reviewed values; the projected recipe is validated by the caller."""
    c = config['teacher_phase']
    marker = c[KEY]
    require(isinstance(marker, dict) and marker.get('variant') in VARIANTS, 'Unreviewed g1491 training revision')
    name = marker.get('variant')
    allowed = (block(), block(cotton=False)) if name is None else (block(variant=name),)
    require(digest(marker) in tuple(digest(b) for b in allowed), 'Unreviewed g1491 training revision')
    v = VARIANTS[name]
    require(c.get('contract') == 'teacher-phase/2.0.0' and c.get('seating') == 'all-seat-search',
            'The g1491 revision needs all-seat teacher search')
    require(digest(c['tables']) == digest(v['tables']) and digest(c['buffer']) == digest(v['buffer']),
            'g1491 teacher tables/buffer differ')
    require(type(c['term']['rows_per_step']) is int and c['term']['rows_per_step'] == v['rows_per_step'] and
            c['term']['shadow_iterations'] == v['shadow_iterations'], 'g1491 teacher rows/calibration window differ')
    require(type(config['update']['epochs']) is int and config['update']['epochs'] == v['ppo_epochs'],
            'g1491 PPO epochs differ')
    return marker


def project_phase(phase):
    """The pre-revision teacher fields, for validating the unchanged rest of the phase."""
    result = {k: copy.deepcopy(v) for k, v in phase.items() if k != KEY}
    result['tables'] = dict(PREVIOUS_TABLES)
    result['buffer'] = copy.deepcopy(PREVIOUS_BUFFER)
    result['term']['rows_per_step'] = PREVIOUS_ROWS_PER_STEP
    if phase[KEY].get('variant') in RESTARTS:
        # The reviewed projection knows two- and four-iteration windows. This variant
        # always has exactly one (validate), so the mapping stays one-to-one.
        result['term']['shadow_iterations'] = 2
    return result


def configure(config, state, *, variant=None):
    """Adopt the revision in shadow mode; afterwards the committed recipe is returned unchanged."""
    if applied(config):
        require(variant is None or variant_of(config) == variant, 'Another g1491 revision variant is committed')
        return copy.deepcopy(config)
    require(variant in VARIANTS, 'Unknown g1491 revision variant')
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
    v = VARIANTS[variant]
    result = copy.deepcopy(config)
    phase = result['teacher_phase']
    phase.update(mode='shadow', tables=dict(v['tables']), buffer=copy.deepcopy(v['buffer']))
    phase['term'].update(rows_per_step=v['rows_per_step'], shadow_iterations=v['shadow_iterations'])
    phase[KEY] = block(variant=variant)
    result['update']['epochs'] = v['ppo_epochs']
    from .config import validate_config
    validate_config(result)
    return result


def previous(config):
    """The recipe before this revision, used only when comparing reviewed transitions.

    Mode, share target and the calibration window stay: the transition checks
    compare those explicitly.
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
    return settings(config)['shadow_iterations'] if applied(config) else PREVIOUS_SHADOW_ITERATIONS


def tables(config):
    t = config['teacher_phase']['tables']
    return dict(train=t['train'], held_out=t['held_out'])


def qualification_tables(config):
    """The table counts a qualifying shadow iteration records (Counter keys)."""
    t = tables(config)
    return {'train': t['train'], 'heldout': t['held_out']}


def long_weight(config):
    """Weight of the long-search mean; None keeps the single long-only pool."""
    return config['teacher_phase'][KEY]['labels'].get('long_weight') if applied(config) else None


def _job_block(exploration, *, split='train', short_labels=True):
    result=dict(contract=CONTRACT, short_labels=short_labels,
                exploration=dict(contract=exploration['contract'], temperature=exploration['temperature'],
                                 opening_actions_per_era=exploration['opening_actions_per_era'],
                                 long_search=exploration['long_search'] == 'always'))
    if split=='train' and 'cotton' in exploration:
        result['exploration']['cotton']=copy.deepcopy(exploration['cotton'])
    return result


def job_block(config, *, split='train'):
    """What a CPU teacher worker needs; absent for every earlier recipe."""
    if not applied(config):
        return None
    return _job_block(config['teacher_phase'][KEY]['exploration'], split=split,
                      short_labels=settings(config)['short_labels'])


def validate_job(job):
    """A teacher job carries either no revision or exactly a reviewed one."""
    revision = job.get('teacher_revision')
    split = job.get('split')
    allowed=(_job_block(EXPLORATION), _job_block(block()['exploration'],split=split),
             _job_block(block(variant=RESTART)['exploration'],split=split,short_labels=False))
    require(revision is None or (digest(revision) in tuple(digest(b) for b in allowed) and
            job.get('seating') == 'all-seat-search' and not job.get('h2h') and not job.get('shadow_probe')),
            'Unreviewed g1491 teacher job revision')
    return revision


def holds_ppo_lr(config):
    """The PPO LR controller holds while this recipe re-measures beta in shadow mode."""
    return applied(config) and config['teacher_phase']['mode'] == 'shadow'


def summary(config):
    c = config['teacher_phase']
    labels = c[KEY]['labels']
    return dict(contract=CONTRACT, variant=variant_of(config), mode=c['mode'], tables=tables(config),
        buffer=copy.deepcopy(c['buffer']), teacher_rows_per_step=c['term']['rows_per_step'],
        labels=labels['searches'], long_weight=long_weight(config), short_weight=labels.get('short_weight'),
        ppo_epochs=config['update']['epochs'], beta_calibration_iterations=c['term']['shadow_iterations'],
        exploration=copy.deepcopy(c[KEY]['exploration']), ppo_lr_during_calibration='hold')
