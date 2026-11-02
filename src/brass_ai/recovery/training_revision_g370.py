"""Explicit continuation of the trained-restart run; no weights or progress reset."""
import copy
from brass_ai.learning.contracts import require

MIGRATION='trained-restart-g370/1.0.0'


def configure(config):
    from . import gae,exploration as E,ppg_g210 as G,trained_startup as T
    require(T.enabled(config),'g370 revision requires the trained-restart recipe')
    T.validate(config);E.validate(config)
    result=G.beta1_options(gae.direct_095_options(config))
    result['branching']=copy.deepcopy(E.T2_ONLY_BRANCH)
    T.validate(result);E.validate(result)
    return result
