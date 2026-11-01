"""Explicit fresh neutral-context recipe; old signed runs remain unchanged."""
import copy
from brass_ai.learning.contracts import require

CONTEXTS=dict(contract='ordinary-neutral-contexts/1.0.0',mode='neutral',seats='all-neural')


def enabled(config):return config.get('seat_contexts')==CONTEXTS


def validate(config):
    require(enabled(config),'Invalid neutral seat-context contract')
    from .domino_config import enabled as diverse
    from .context_evaluation_config import enabled as contexts
    require(not diverse(config) and not contexts(config),'Neutral seats cannot enable DOMiNO or its context panels')


def configure(config):
    from . import exploration as E,opening_objective as O,policy_observations
    from .utility import reward_spec
    from .domino_config import defaults
    from .context_evaluation_config import defaults as evaluation_defaults
    result=copy.deepcopy(config)
    result.update(exploration=.05,insolvency=dict(event_cost=.02,max_cost=.075),
        opening=copy.deepcopy(E.THREE_OPENING),branching=copy.deepcopy(E.NEUTRAL_BRANCH),
        diversity=defaults('off'),context_evaluation=evaluation_defaults('off'),seat_contexts=copy.deepcopy(CONTEXTS),
        exploration_rollout=copy.deepcopy(E.ROLLOUT))
    result['update'].update(entropy_weight=.005,fixed_entropy=copy.deepcopy(E.FIXED_ENTROPY_005),
        opening_objective=copy.deepcopy(O.T13))
    result['entropy_control']=dict(copy.deepcopy(E.ANCHOR_ENTROPY),mode='off')
    result['search']['reward']=reward_spec(result)
    if 'teacher_phase' in result:
        teacher=result['teacher_phase'];teacher['search']['reward']=reward_spec(result)
        if 'diversity' in teacher:
            from .teacher_phase_config import CONTRACT,LABEL_SOURCE
            teacher.pop('diversity');teacher.update(contract=CONTRACT,mode='off')
            teacher['labels']['source']=LABEL_SOURCE
    # Adoption receipts belong to the old run, not to a new generation zero.
    if 'local_scoring_auxiliary' in result:result['local_scoring_auxiliary']['adoption']=None
    result=policy_observations.configure(result)
    validate(result)
    return result


def check_rows(game,proof):
    """Never relabel old focused rows as neutral, including branch replay."""
    require(proof.get('seat_contexts')==CONTEXTS,'Neutral replay context contract differs')
    require('diversity_assignment' not in proof,'Neutral replay contains DOMiNO assignments')
    for row in game['records']:
        if row['behavior']['kind']=='model':
            require(row['style']==[0.]*4 and row['intent']==[0.]*8,'Non-neutral neural seat in neutral collection')
