"""Narrow reward reduction; the existing reward transition refreshes PPG replay."""
import copy

from brass_ai.learning.contracts import require

MIGRATION='training-recipe-g1320/1.0.0'
PREVIOUS=dict(event_cost=.02,max_cost=.075)
CURRENT=dict(event_cost=.01,max_cost=.075)


def configure(config):
    from .utility import reward_spec
    result=copy.deepcopy(config)
    require(result.get('insolvency') in (PREVIOUS,CURRENT),
        'g1320 reward revision requires event_cost .02 or .01 and max_cost .075')
    require(result.get('search',{}).get('reward')==reward_spec(result),
        'Search/learner reward differs before g1320 revision')
    result['insolvency']=copy.deepcopy(CURRENT)
    result['search']['reward']=reward_spec(result)
    return result
