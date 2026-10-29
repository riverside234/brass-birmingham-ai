"""Change only the fork-action mixture; retain valid critic replay and progress."""
import copy
from brass_ai.learning.contracts import require

MIGRATION='training-recipe-g1324/1.0.0'


def configure(config):
    from . import exploration as E
    E.validate(config)
    require(config.get('branching') in (E.THREE_T2_BRANCH,E.BALANCED_BRANCH),
        'g1324 revision requires the three-decision T2 branch recipe')
    result=copy.deepcopy(config)
    result['branching']=copy.deepcopy(E.BALANCED_BRANCH)
    E.validate(result)
    return result
