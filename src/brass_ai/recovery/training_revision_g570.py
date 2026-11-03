"""Reviewed continuation with distinct branch openings and a rule-bot PFSP gate."""
import copy
from brass_ai.learning.contracts import require

MIGRATION='trained-restart-g570/1.0.0'


def configure(config,*,manual_generation=570):
    from . import exploration as E,opening_objective as O,rating_coverage as R,trained_startup as T,v4_history as V
    require(T.enabled(config),'g570 revision requires the trained-restart recipe')
    T.validate(config);E.validate(config)
    result=copy.deepcopy(config)
    result['branching']=copy.deepcopy(E.FORK_T15_BRANCH)
    result['exploration']=.02
    result['update'].update(entropy_weight=.005,fixed_entropy=copy.deepcopy(E.FIXED_ENTROPY_005),
        opening_objective=copy.deepcopy(O.FORK_T15))
    result['entropy_control']['alpha_base']=.005
    from .entropy_schedule import DEFAULT,KEY
    result['update'].setdefault(KEY,copy.deepcopy(DEFAULT))
    result['rating_coverage']=copy.deepcopy(R.WITH_SEED)
    result[V.KEY]=dict(V.DEFAULT,manual_generation=manual_generation)
    T.validate(result);E.validate(result)
    return result
