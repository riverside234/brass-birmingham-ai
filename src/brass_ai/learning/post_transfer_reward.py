"""Exact user-approved reward change, activated only after machine transfer."""
import copy

from .contracts import require

REVISION='post-transfer-reward/1.0.0'
VP_TIMING='canal-and-final-delta/1.0.0'
TIMED_VERSION='brass-combined-return/3.0.0'
BEFORE=dict(win=1.,placement=.1,vp=.001,vp_cap=180)
AFTER=dict(win=1.,placement=.05,vp=.001,vp_cap=180,vp_timing=VP_TIMING)
OLD_COST=dict(event_cost=.0025,max_cost=.025)
# Keep the existing positive feature-budget scale; zero charge removes the
# reward term exactly, without changing input dimensions or dividing by zero.
ZERO_COST=dict(event_cost=0.,max_cost=.025)


def eligible(config):return (config.get('reward')==BEFORE and config.get('insolvency')==OLD_COST) or is_revised(config)


def is_revised(config):return config.get('reward')==AFTER and config.get('insolvency')==ZERO_COST


def specification(revised):
    return dict(version=TIMED_VERSION if revised else 'brass-combined-return/2.0.0',
                **{k:v for k,v in (AFTER if revised else BEFORE).items() if k!='vp_timing'},
                insolvency=copy.deepcopy(ZERO_COST if revised else OLD_COST))


def _replace(config,revised):
    result=copy.deepcopy(config);result['reward']=copy.deepcopy(AFTER if revised else BEFORE)
    result['insolvency']=copy.deepcopy(ZERO_COST if revised else OLD_COST)
    spec=specification(revised)
    result['search']['reward']=copy.deepcopy(spec)
    for name in ('search','fast_search'):
        if name in result.get('teacher_phase',{}):result['teacher_phase'][name]['reward']=copy.deepcopy(spec)
    return result


def revise(config):
    if is_revised(config):return copy.deepcopy(config)
    require(config.get('reward')==BEFORE and config.get('insolvency')==OLD_COST,
        'Post-transfer reward requires the reviewed g1284 reward recipe')
    return _replace(config,True)


def previous(config):return _replace(config,False) if is_revised(config) else config


def transition(before,after):
    try:return not is_revised(before) and revise(before)==after
    except (ValueError,KeyError,TypeError):return False


def reward_upgrade(before,after):return before==specification(False) and after==specification(True)


def checkpoint_specification(state):
    config=state['signature']['config']
    # A source/config transition may have committed before its first new
    # checkpoint. The oldest transition at this same checkpoint owns its bytes.
    for transition in state.get('source_transitions',[]):
        if transition['checkpoint']==state['history'][-1]:
            config=transition['from_signature']['config'];break
    return dict(version=TIMED_VERSION if config['reward'].get('vp_timing')==VP_TIMING else 'brass-combined-return/2.0.0',
        **{k:v for k,v in config['reward'].items() if k!='vp_timing'},insolvency=copy.deepcopy(config['insolvency']))


def authorized(state,signature):
    """An exact restored checkpoint, not an arbitrary source-side config edit."""
    from .contracts import SOURCE_ROOT,project_path
    from brass_ai.recovery.transfer_resume import instruction
    value=instruction(SOURCE_ROOT,project_path(SOURCE_ROOT,state['history'][-1]['path']).parent)
    return (value is not None and value.get('reward_revision')==REVISION and
        value['origin_generation']==state['iteration'] and value['checkpoint']==state['history'][-1] and
        transition(state['signature']['config'],signature['config']))
