"""B04.5 opt-in recipe; legacy teacher signatures remain byte-for-byte valid."""
import copy

from brass_ai.learning.contracts import digest, require
from brass_ai.learning.search_value import CONTRACT, binding

PHASE = 'teacher-phase/2.0.0'
FAST_POST_SECONDS = .5
MAX_STEPS = 64


def defaults():
    return dict(contract=CONTRACT, stage='off',
        head=dict(hidden=32, inputs='utility-head-inputs', init='zero-final-layer'),
        optimizer=dict(kind='Adam', learning_rate=1e-4, weight_decay=1e-4, grad_clip=1.),
        steps_per_iteration=32, batch_size=512,
        target=dict(kind='seat-lambda-return', lambda_candidates=[.5, .8, .95, 1.], **{'lambda': None},
                    root_value='network-plus-completed-returns'),
        buffer=dict(iterations=8, reward_bound=True, staleness='label-buffer-rule'),
        gate=dict(window_iterations=8, bootstrap=200, cluster='held-out-game', evaluation_target='final-utility-to-go'),
        blend=dict(w=0., step=.05), head_to_head=dict(games=256, tables='2v2-seat-rotations', profile='4s-k16-every-turn'),
        trunk=dict(scale=0., ramp_iterations=10, intervals=2, interval_steps=16,
                   ordinary_heldout_tolerance=.013, kl='shared-ppg-cap'))


def fast_profile(search):
    return dict(copy.deepcopy(search), k=4, allowance_seconds=.8, post_search_seconds=FAST_POST_SECONDS)


def adopt(config):
    from .teacher_phase_config import adopt as legacy
    result = legacy(config)
    c = result['teacher_phase']
    c.update(contract=PHASE, startup='immediate', seating='all-seat-search',
             fast_search=fast_profile(c['search']), search_value=defaults())
    return result


def max_steps_options(config):
    """At most 64 search-head steps per iteration: head-only, or four 16-step trunk intervals."""
    result = copy.deepcopy(config)
    options = result['teacher_phase']['search_value']
    validate(options)
    require('schedule' in options, '64 search-head steps require the reviewed stage schedule')
    options['steps_per_iteration'] = MAX_STEPS
    options['trunk']['intervals'] = MAX_STEPS//options['trunk']['interval_steps']
    validate(options)
    return result


def validate(c):
    expected = defaults()
    if 'schedule' in c:
        from .search_value_schedule import dedicated_holdout, validate as schedule_validate
        # The schedule itself is exactly one of the reviewed ones (schedule_validate).
        expected.update(schedule=copy.deepcopy(schedule_validate(c)))
        from .search_value_schedule import restart_schedule
        if set(c['schedule'])==set(restart_schedule()):
            # The g1190 restart adopts a fresh, wider head with its schedule.
            expected['head']['hidden']=256
        expected['buffer']['staleness']='none'
        expected['gate']=dict(mode='diagnostic-only',games=4,evaluation_target='final-utility-to-go')
        expected['head_to_head']['games']=0
        expected['trunk']['kl']=.003
        expected['trunk']['ordinary_heldout_tolerance']=None
        if dedicated_holdout(c) and 'max_epochs' in c:
            expected['max_epochs']=1
        if c.get('steps_per_iteration') == MAX_STEPS:
            expected['steps_per_iteration']=MAX_STEPS
            expected['trunk']['intervals']=MAX_STEPS//expected['trunk']['interval_steps']
    require(set(c) == set(expected), 'Search value fields differ')
    fixed = copy.deepcopy(c)
    fixed['stage'] = expected['stage']
    fixed['blend']['w'] = 0.
    fixed['trunk']['scale'] = 0.
    fixed['target']['lambda'] = None
    if 'schedule' in c and dedicated_holdout(c):
        require(c['target']['lambda'] == .8, 'The refreshed search-value head uses lambda .8')
    require(digest(fixed) == digest(expected), 'Unreviewed search value recipe')
    require(c['stage'] in ('off', 'A', 'B', 'head_to_head', 'C'), 'Invalid search value stage')
    binding(c['blend']['w'])
    require(type(c['trunk']['scale']) in (int, float) and 0 <= c['trunk']['scale'] <= 1, 'Invalid trunk scale')
    require(c['target']['lambda'] in (None, .5, .8, .95, 1.), 'Invalid lambda')
    require(c['stage'] not in ('off', 'A') or (c['blend']['w'] == 0 and c['trunk']['scale'] == 0), 'Premature correction')
    require(c['stage'] in ('B', 'C') or c['trunk']['scale'] == 0, 'Premature trunk pass')
    return c
