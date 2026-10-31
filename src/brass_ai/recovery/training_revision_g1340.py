"""Explicit g1340 reward/exploration revision and report-only context panel."""
import copy
from brass_ai.learning.contracts import require

MIGRATION='training-recipe-g1340/1.0.0'
PREVIOUS=dict(event_cost=.005,max_cost=.05)
CURRENT=dict(event_cost=.01,max_cost=.075)


def recipe(config):
    from .utility import reward_spec
    require(config.get('insolvency') in (PREVIOUS,CURRENT),
        'g1340 requires insolvency .005/.05 or .01/.075')
    require(config.get('exploration') in (.08,.05),'g1340 requires exploration .08 or .05')
    require(config.get('search',{}).get('reward')==reward_spec(config),
        'Search/learner reward differs before g1340 revision')
    if 'teacher_phase' in config:
        from .teacher_phase_config import validate
        validate(config)
    result=copy.deepcopy(config)
    result['insolvency']=copy.deepcopy(CURRENT)
    result['search']['reward']=reward_spec(result)
    result['exploration']=.05
    if 'teacher_phase' in result:
        result['teacher_phase']['search']['reward']=reward_spec(result)
        validate(result)
    return result


def configure(config):
    from .context_evaluation_config import configure as context_config
    return context_config(recipe(config),every_iterations=config.get('context_evaluation',{}).get('every_iterations',50))
