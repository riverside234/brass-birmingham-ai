"""B04.5 opt-in recipe; legacy teacher signatures remain byte-for-byte valid."""
import copy

from brass_ai.learning.contracts import digest, require
from brass_ai.learning.search_value import CONTRACT, binding

PHASE = 'teacher-phase/2.0.0'
FAST_POST_SECONDS = .5


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


def validate(c):
    expected = defaults()
    if 'schedule' in c:
        from .search_value_schedule import defaults as schedule_defaults,validate as schedule_validate
        schedule_validate(c)
        expected.update(schedule=schedule_defaults())
        expected['schedule']['validation_during_beta_qualification']=c['schedule']['validation_during_beta_qualification']
        expected['buffer']['staleness']='none'
        expected['gate']=dict(mode='diagnostic-only',games=4,evaluation_target='final-utility-to-go')
        expected['head_to_head']['games']=0
        expected['trunk']['kl']=.003
        expected['trunk']['ordinary_heldout_tolerance']=None
    require(set(c) == set(expected), 'Search value fields differ')
    fixed = copy.deepcopy(c)
    fixed['stage'] = expected['stage']
    fixed['blend']['w'] = 0.
    fixed['trunk']['scale'] = 0.
    fixed['target']['lambda'] = None
    require(digest(fixed) == digest(expected), 'Unreviewed search value recipe')
    require(c['stage'] in ('off', 'A', 'B', 'head_to_head', 'C'), 'Invalid search value stage')
    binding(c['blend']['w'])
    require(type(c['trunk']['scale']) in (int, float) and 0 <= c['trunk']['scale'] <= 1, 'Invalid trunk scale')
    require(c['target']['lambda'] in (None, .5, .8, .95, 1.), 'Invalid lambda')
    require(c['stage'] not in ('off', 'A') or (c['blend']['w'] == 0 and c['trunk']['scale'] == 0), 'Premature correction')
    require(c['stage'] in ('B', 'C') or c['trunk']['scale'] == 0, 'Premature trunk pass')
    return c
