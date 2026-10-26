"""Explicit three-decision T1.3 openings and loss16 PPG continuation."""
import copy

from brass_ai.learning.contracts import require

MIGRATION='training-recipe-g1120/1.0.0'
KL32_MIGRATION='training-recipe-g1120/2.0.0'


def configure(config):
    from . import exploration as E,opening_objective as O,ppg_g210 as G
    E.validate(config)
    require((config['opening'],config['branching'],config['update'].get('opening_objective')) in (
        (E.REVISED_OPENING,E.REVISED_BRANCH,O.T15),(E.THREE_OPENING,E.THREE_BRANCH,O.T13)),
        'g1120 revision requires the paired g850 or g1120 opening/branch/objective recipe')
    result=G.loss16_options(config)
    result['opening']=copy.deepcopy(E.THREE_OPENING)
    result['branching']=copy.deepcopy(E.THREE_BRANCH)
    result['update']['opening_objective']=copy.deepcopy(O.T13)
    E.validate(result)
    return result


def configure_kl32(config):
    from . import exploration as E,opening_objective as O,ppg_g210 as G
    # Support a direct resume of g1120's v17 checkpoint as well as an already
    # adopted v18 checkpoint. The old explicit migration remains reproducible.
    result=copy.deepcopy(config) if config['value_phase']['contract']==G.KL32_CONTRACT else configure(config)
    E.validate(result)
    require(result['opening']==E.THREE_OPENING and result['branching']==E.THREE_BRANCH and
        result['update'].get('opening_objective')==O.T13,'KL32 recipe requires the three-decision T1.3 opening objective')
    return G.kl32_options(result)
