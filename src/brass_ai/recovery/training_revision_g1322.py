"""Explicit reward, fixed group entropy and three-decision T2 continuation."""
import copy

from brass_ai.learning.contracts import require

MIGRATION='training-recipe-g1322/1.0.0'
PREVIOUS=dict(event_cost=.01,max_cost=.075)
CURRENT=dict(event_cost=.005,max_cost=.05)


def configure(config):
    from . import exploration as E,opening_objective as O
    from .utility import reward_spec
    E.validate(config)
    require((config.get('opening'),config.get('branching'),config['update'].get('opening_objective')) in (
        (E.THREE_OPENING,E.THREE_BRANCH,O.T13),(E.THREE_T2_OPENING,E.THREE_T2_BRANCH,O.T2_THREE)),
        'g1322 revision requires the paired three-decision opening/branch/objective recipe')
    require(config.get('insolvency') in (PREVIOUS,CURRENT),
        'g1322 reward revision requires .01/.075 or .005/.05')
    require(config.get('search',{}).get('reward')==reward_spec(config),
        'Search/learner reward differs before g1322 revision')
    result=copy.deepcopy(config)
    result['insolvency']=copy.deepcopy(CURRENT)
    result['search']['reward']=reward_spec(result)
    result['opening']=copy.deepcopy(E.THREE_T2_OPENING)
    result['branching']=copy.deepcopy(E.THREE_T2_BRANCH)
    result['exploration']=.08
    result['update']['opening_objective']=copy.deepcopy(O.T2_THREE)
    result['update']['fixed_entropy']=copy.deepcopy(E.FIXED_ENTROPY)
    E.validate(result)
    return result
