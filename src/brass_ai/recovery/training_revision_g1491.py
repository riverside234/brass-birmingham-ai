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
GAME_OPENING = dict(temperature=2., actions_per_seat=3)
RESTART = 'g1490-branch-restart'
# The same restart with 256 teacher rows per PPO step (the g1190 restart's second attempt).
WIDE_RESTART = 'g1190-branch-restart-256-rows'
SEARCH_REFRESH = 'g1190-search-head-refresh-96-rows'
RETUNED_SEARCH = dict(full_k=8, c_scale=.15, fast_c_scale=.5)
COTTON_SPLIT_SEARCH = dict(RETUNED_SEARCH, c_scale=.20, cotton_c_scale=.10)
TEACHER_SEARCH_HEAD_WEIGHT = 'teacher_search_head_weight'
RESTARTS = (RESTART, WIDE_RESTART, SEARCH_REFRESH)
# Exact values of each reviewed variant; None is the in-run g1491 adoption.
VARIANTS = {
    None: dict(tables=TABLES, buffer=BUFFER, rows_per_step=ROWS_PER_STEP, ppo_epochs=PPO_EPOCHS,
               shadow_iterations=SHADOW_ITERATIONS, short_labels=True),
    RESTART: dict(tables=dict(train=32, held_out=4), buffer=PREVIOUS_BUFFER, rows_per_step=72,
                  ppo_epochs=PREVIOUS_PPO_EPOCHS, shadow_iterations=1, short_labels=False),
}
VARIANTS[WIDE_RESTART] = dict(VARIANTS[RESTART], rows_per_step=256)
VARIANTS[SEARCH_REFRESH] = dict(VARIANTS[RESTART], rows_per_step=96)


def block(*, cotton=True, variant=None):
    """The reviewed marker. Workers receive ``job_block()``, derived from it."""
    require(variant in VARIANTS, 'Unknown g1491 revision variant')
    exploration=copy.deepcopy(EXPLORATION)
    if cotton:
        from .teacher_cotton import settings
        exploration['cotton']=settings(develop=variant==SEARCH_REFRESH)
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


def count_search(config):
    return config.get('teacher_phase', {}).get(KEY, {}).get('count_search')


def teacher_search_head_weight(config):
    return config.get('teacher_phase', {}).get(KEY, {}).get(TEACHER_SEARCH_HEAD_WEIGHT)


def seated_cotton(config):
    from .teacher_cotton import SEATED_CONTRACT
    return config.get('teacher_phase', {}).get(KEY, {}).get('exploration', {}).get('cotton', {}).get('contract') == SEATED_CONTRACT


def _mixed_tables(marker, weight, *, seated=False):
    from .teacher_cotton import MIXED_25_CONTRACT, MIXED_CONTRACT, SEATED_CONTRACT, settings
    if seated:
        contract = SEATED_CONTRACT
    else:
        contract = MIXED_25_CONTRACT if marker.get('count_search') == COTTON_SPLIT_SEARCH else MIXED_CONTRACT
    cotton = dict(settings(contract=contract), table_fraction=.5)
    return dict(marker, exploration=dict(marker['exploration'], cotton=cotton), **{TEACHER_SEARCH_HEAD_WEIGHT: weight})


def configure_teacher_games(config, weight, *, seated=False):
    """Set the teacher-only blend and split cotton exploration within each partition."""
    require(type(weight) is int and weight in (0, 1), 'Teacher search-head weight must be 0 or 1')
    require(variant_of(config) == SEARCH_REFRESH and count_search(config) in (RETUNED_SEARCH, COTTON_SPLIT_SEARCH),
            'Teacher game settings require the retuned g1190 recipe')
    phase = config['teacher_phase']
    revision = _mixed_tables(phase[KEY], weight, seated=seated or seated_cotton(config))
    result = dict(config, teacher_phase=dict(phase, **{KEY: revision}))
    validate(result)
    return result


def label_revision(marker):
    """Play sampling and the root blend vary while the sealed label format stays compatible."""
    if TEACHER_SEARCH_HEAD_WEIGHT not in marker:
        return marker
    from .teacher_cotton import settings
    result = {k: v for k, v in marker.items() if k != TEACHER_SEARCH_HEAD_WEIGHT}
    result['exploration'] = dict(marker['exploration'], cotton=settings(develop=True))
    result['exploration'].pop('game_opening', None)
    return result


def validate(config):
    """Exact reviewed values; the projected recipe is validated by the caller."""
    c = config['teacher_phase']
    marker = c[KEY]
    require(isinstance(marker, dict) and marker.get('variant') in VARIANTS, 'Unreviewed g1491 training revision')
    name = marker.get('variant')
    allowed = (block(), block(cotton=False)) if name is None else (block(variant=name),)
    if name == SEARCH_REFRESH:
        retuned = dict(allowed[0], count_search=RETUNED_SEARCH)
        split = dict(retuned, count_search=COTTON_SPLIT_SEARCH)
        allowed += (retuned, _mixed_tables(retuned, 0), _mixed_tables(retuned, 1),
                    _mixed_tables(split, 0), _mixed_tables(split, 1),
                    _mixed_tables(retuned, 0, seated=True), _mixed_tables(retuned, 1, seated=True))
        opening = dict(retuned, exploration=dict(retuned['exploration'], game_opening=GAME_OPENING))
        allowed += tuple(_mixed_tables(opening, weight, seated=True) for weight in (0, 1))
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
    """Retuned teacher games can adopt the mixed recipe and switch their blend."""
    if not applied(before):
        return True
    if not applied(after):
        return False
    old, new = before['teacher_phase'][KEY], after['teacher_phase'][KEY]
    if digest(old) == digest(new):
        return True
    if TEACHER_SEARCH_HEAD_WEIGHT not in new:
        return False
    try:
        validate(before)
        validate(after)
    except (ValueError, KeyError, TypeError):
        return False
    return digest(label_revision(old)) == digest(label_revision(new))


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


def _job_block(exploration, *, split='train', short_labels=True, search_head_weight=None, cotton_table=False, seats=None):
    result=dict(contract=CONTRACT, short_labels=short_labels,
                exploration=dict(contract=exploration['contract'], temperature=exploration['temperature'],
                                 opening_actions_per_era=exploration['opening_actions_per_era'],
                                 long_search=exploration['long_search'] == 'always'))
    if 'game_opening' in exploration:
        result['exploration']['game_opening'] = dict(exploration['game_opening'])
    if search_head_weight is not None:
        result[TEACHER_SEARCH_HEAD_WEIGHT] = search_head_weight
    if seats is not None:
        from .teacher_cotton import SEATS
        result[SEATS] = seats
    if search_head_weight is not None or seats is not None:
        if cotton_table:
            result['exploration']['cotton'] = {k: v for k, v in exploration['cotton'].items() if k != 'table_fraction'}
    elif split=='train' and 'cotton' in exploration:
        result['exploration']['cotton'] = dict(exploration['cotton'])
    return result


def job_block(config, *, split='train', cotton_table=False, stage='A', seats=None):
    """What a CPU teacher worker needs; absent for every earlier recipe."""
    if not applied(config):
        return None
    weight = teacher_search_head_weight(config)
    if seated_cotton(config):
        if stage != 'C':
            weight = None
        seats = seats or []
        cotton_table = bool(seats)
    return _job_block(config['teacher_phase'][KEY]['exploration'], split=split,
                      short_labels=settings(config)['short_labels'],
                      search_head_weight=weight, cotton_table=cotton_table, seats=seats)


def validate_job(job):
    """A teacher job carries either no revision or exactly a reviewed one."""
    revision = job.get('teacher_revision')
    if revision is None:
        return None
    split = job.get('split')
    require(isinstance(revision, dict) and job.get('seating') == 'all-seat-search' and
            not job.get('h2h') and not job.get('shadow_probe'), 'Unreviewed g1491 teacher job revision')
    from .teacher_cotton import SEATED_CONTRACT, SEATS, settings as cotton_settings
    weight = revision.get(TEACHER_SEARCH_HEAD_WEIGHT)
    if TEACHER_SEARCH_HEAD_WEIGHT in revision:
        from brass_ai.learning.search_value import binding
        require(type(weight) is int and weight in (0, 1) and split in ('train', 'heldout') and
                job.get('search_value') == binding(weight), 'Teacher job search-head weight differs')
    if SEATS in revision:
        from brass_ai.step5.policy import cotton_seats
        seats = revision[SEATS]
        require(split in ('train', 'heldout') and type(seats) is list and len(seats) <= 2 and
                all(type(seat) is int for seat in seats) and
                seats == cotton_seats(job['sampling_seed'], len(seats), SEATED_CONTRACT),
                'Teacher cotton seat assignment differs')
        exploration = dict(EXPLORATION, cotton=cotton_settings(contract=SEATED_CONTRACT))
        allowed = tuple(_job_block(e, short_labels=False, search_head_weight=weight,
                                   cotton_table=bool(seats), seats=seats) for e in
                        (exploration, dict(exploration, game_opening=GAME_OPENING)))
    elif weight is not None:
        from .teacher_cotton import MIXED_CONTRACTS
        allowed = (_job_block(EXPLORATION, short_labels=False, search_head_weight=weight), *(
            _job_block(dict(EXPLORATION, cotton=cotton_settings(contract=contract)), short_labels=False,
                       search_head_weight=weight, cotton_table=True) for contract in MIXED_CONTRACTS))
    else:
        allowed = (_job_block(EXPLORATION), _job_block(block()['exploration'],split=split),
                   _job_block(block(variant=RESTART)['exploration'],split=split,short_labels=False),
                   _job_block(block(variant=SEARCH_REFRESH)['exploration'],split=split,short_labels=False))
    require(digest(revision) in tuple(digest(b) for b in allowed),
            'Unreviewed g1491 teacher job revision')
    return revision


def holds_ppo_lr(config):
    """The PPO LR controller holds while this recipe re-measures beta in shadow mode."""
    return applied(config) and config['teacher_phase']['mode'] == 'shadow'


def summary(config):
    c = config['teacher_phase']
    labels = c[KEY]['labels']
    variant = variant_of(config)
    result = dict(contract=CONTRACT, variant=variant, mode=c['mode'], tables=tables(config),
        buffer=copy.deepcopy(c['buffer']), teacher_rows_per_step=c['term']['rows_per_step'],
        labels=labels['searches'], long_weight=long_weight(config), short_weight=labels.get('short_weight'),
        ppo_epochs=config['update']['epochs'], beta_calibration_iterations=c['term']['shadow_iterations'],
        exploration=copy.deepcopy(c[KEY]['exploration']), ppo_lr_during_calibration='normal' if variant==SEARCH_REFRESH else 'hold')
    weight = teacher_search_head_weight(config)
    if weight is not None:
        result[TEACHER_SEARCH_HEAD_WEIGHT] = weight
    return result
