"""Explicit committed-boundary 70/30 table mix and early PPG KL response."""
import copy

from brass_ai.learning.contracts import require
from .quality_config import TABLE_PLAN

MIGRATION='selfplay-70-30-early-ppg-and-throughput/1.0.0'
TABLE_CONTRACT='ordinary-table-mix/2.0.0'
QUOTAS=dict(selfplay=70,historical=30,legacy=0)


def mixed(config):return config.get('table_plan',{}).get('contract')==TABLE_CONTRACT


def validate_mix(config):
    from .ordinary_cotton import COTTON_PROBABILITIES, KEY, SETTINGS, enabled
    plan=config['table_plan']
    require(set(plan)==set(TABLE_PLAN)|{'contract','origin_generation','origin_games'}|({KEY} if enabled(config) else set()) and
        plan['contract']==TABLE_CONTRACT and plan['quotas']==QUOTAS and
        plan['learner_seats']==TABLE_PLAN['learner_seats'] and plan['pair_rotation']==TABLE_PLAN['pair_rotation'],
        'Invalid 70/30 table plan')
    generation,games=plan['origin_generation'],plan['origin_games']
    require(type(generation) is int and generation>=0 and type(games) is int and games>=0 and
        games==generation*320 and config['games_per_iteration']==320,'Invalid 70/30 committed origin')
    old=dict(until=games,**TABLE_PLAN['quotas'])
    require(config['curriculum']==([old] if games else [])+[dict(until=None,**QUOTAS)],
        '70/30 curriculum must preserve the previous schedule through its committed origin')
    require(config.get('v4_history') is not None,'70/30 requires v4 already moved into history')
    if enabled(config):
        require(plan[KEY] in (dict(SETTINGS, probability=rate) for rate in COTTON_PROBABILITIES)
                and config.get('branching', {}).get('rate_per_decision') == 0,
                'Ordinary cotton self-play uses the 50/20/30 recipe with branching disabled')
    return plan


def configure(config,state):
    from .ppg_g210 import early_options
    from .ppo_kl_control import configure_sparse
    from .throughput_config import configure as throughput
    from .v4_history import active
    result=throughput(configure_sparse(early_options(config)))
    require(active(state.get('quality_pool')),'70/30 continuation requires the committed v4 PFSP activation')
    if mixed(result):
        validate_mix(result)
        require(result['table_plan']['origin_generation']<=state['iteration'] and
            result['table_plan']['origin_games']<=state['games_done'],'70/30 origin is ahead of committed progress')
        return result
    require(config['table_plan']==TABLE_PLAN and config['curriculum']==[dict(until=None,**TABLE_PLAN['quotas'])],
        '70/30 adoption requires the existing 50/45/5 base schedule')
    g,games=state['iteration'],state['games_done']
    result['table_plan']=dict(copy.deepcopy(TABLE_PLAN),contract=TABLE_CONTRACT,quotas=copy.deepcopy(QUOTAS),
        origin_generation=g,origin_games=games)
    result['curriculum']=([dict(until=games,**TABLE_PLAN['quotas'])] if games else [])+[dict(until=None,**QUOTAS)]
    validate_mix(result)
    return result


def exposure(plan, *, cotton=False):
    """PFSP p is conditional on a frozen opponent seat, never a loss weight."""
    rows=[dict(model_id=r['model_id'],conditional_probability=r['p'],
        expected_opponent_seats=192*r['p'],all_seat_probability=.15*r['p']) for r in plan['members']]
    result = dict(contract=TABLE_CONTRACT,tables=dict(selfplay=224,historical=96,legacy=0),
        learner_seats=dict(selfplay=896,historical=192,legacy=0),history_opponent_seats=192,
        history_pairwise_comparisons=384,members=rows,
        scope='PFSP probabilities normalized within history; marginal exposure is 30% tables x 2/4 opponent seats. '
              'Recorded comparison counts and PPO per-row means are not multiplied by the mix again.')
    if cotton:
        from .ordinary_cotton import SETTINGS
        standard = 320 * SETTINGS['standard_selfplay'] // 100
        targeted = 320 * SETTINGS['cotton_selfplay'] // 100
        result['selfplay_exploration'] = dict(standard_tables=standard, cotton_tables=targeted,
                                            one_cotton_seat_tables=targeted//2, two_cotton_seat_tables=targeted//2)
    return result
