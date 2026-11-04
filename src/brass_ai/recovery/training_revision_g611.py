"""Reviewed g611 continuation: prospective reward stage, zero epsilon, PPG cap."""
from brass_ai.learning.contracts import require

MIGRATION='trained-restart-g611/1.0.0'


def configure(config):
    from . import entropy_schedule as S,exploration as E,ppg_g210 as P,trained_startup as T
    T.validate(config);E.validate(config)
    require(config['branching']==E.FORK_T15_BRANCH and S.KEY in config['update'],
        'g611 revision requires the adopted g570 entropy/fork recipe')
    result=P.quarter_options(S.extend_config(config))
    result['exploration']=0.
    T.validate(result);E.validate(result)
    return result
