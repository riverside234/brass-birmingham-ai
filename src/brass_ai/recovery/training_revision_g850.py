"""Explicit g850 recipe transition; never mutate an existing run in place."""
import copy

from brass_ai.learning.contracts import require

MIGRATION='training-recipe-g850/1.0.0'


def configure(config,*,rating_workers=None):
    from . import exploration,opening_objective,ppg_g210
    from .plackett_luce import COMPUTE,validate_compute
    exploration.validate(config)
    require(config['entropy_control']['contract']==exploration.ANCHOR_ENTROPY['contract'],
        'g850 revision requires the current anchor-entropy recipe')
    result=copy.deepcopy(config)
    result['opening']=copy.deepcopy(exploration.REVISED_OPENING)
    result['entropy_control']['mode']='off'
    result['update']['entropy_weight']=.005
    result['update']['opening_objective']=copy.deepcopy(opening_objective.T15)
    result['exploration']=.05
    result['branching']=copy.deepcopy(exploration.REVISED_BRANCH)
    result=ppg_g210.cosine_options(result)
    require(result['insolvency'] in (dict(event_cost=.04,max_cost=.10),dict(event_cost=.02,max_cost=.075)),
        'g850 reward transition requires .04/.10 or .02/.075')
    require(result['search']['reward']['insolvency']==result['insolvency'],'Search/learner reward differs')
    result['insolvency']=dict(event_cost=.02,max_cost=.075)
    result['search']['reward']['insolvency']=copy.deepcopy(result['insolvency'])
    result.setdefault('rating_compute',copy.deepcopy(COMPUTE))
    if rating_workers is not None:result['rating_compute']['workers']=rating_workers
    validate_compute(result['rating_compute'])
    return result
