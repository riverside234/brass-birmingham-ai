"""Reviewed full-volume teacher on-mode, starting search-head Stage A."""
import copy
import math
import statistics

from brass_ai.learning.contracts import digest,require

CONTRACT='brass-native-teacher-activation/1.0.0'


def is_on(config):
    from brass_ai.recovery.collector_runtime import recipe
    from .lab_shadow_config import is_shadow
    try:
        config=recipe(config)
        if config['teacher_phase']['mode']!='on' or config['teacher_phase']['term']['shadow_iterations']!=4:return False
        shadow=copy.deepcopy(config);shadow['teacher_phase']['mode']='shadow'
        return is_shadow(shadow)
    except (KeyError,TypeError,ValueError):return False


def configure(before,target=None):
    from brass_ai.recovery.collector_runtime import recipe
    from brass_ai.recovery.config import validate_config
    from .lab_shadow_config import is_shadow
    before=recipe(before)
    require(is_shadow(before) or is_on(before),'Teacher activation requires the reviewed shadow/on recipe')
    require(before['teacher_phase']['term']['shadow_iterations']==4,'Finish four shadow measurements first')
    after=copy.deepcopy(before);after['teacher_phase']['mode']='on'
    if target is not None:after['teacher_phase']['term']['share_target']=target
    require(is_on(after),'Only fixed-beta teacher on / search-head Stage A is reviewed')
    validate_config(after)
    return after


def qualification(state,config):
    """Validate the measured window against its actual checkpoint lineage."""
    from brass_ai.recovery.teacher_gradient_qualification import restore
    options=config['teacher_phase']['term']
    q=restore(state.get('teacher_phase',{}).get('weight',{}).get('gradient_qualification'),options['share_target'],4)
    require(q is not None and q['passed'] and q.get('required_iterations')==4,
        'Teacher on-mode needs four qualified full shadow iterations')
    observations=q['observations'];history={e['generation']:e for e in state['history']}
    ratios=[]
    for o in observations:
        raw=o['raw_shares'];context=o['context'];entry=history.get(o['generation'])
        require(raw and len(raw)==o['valid_steps']<=o['steps'] and
            all(type(r) in (int,float) and math.isfinite(r) and r>0 for r in raw),
            'Invalid qualified teacher gradient observations')
        require(context['tables']=={'train':28,'heldout':4} and
            all(context['labels_by_phase'].get(p,0)>0 for p in ('canal_early','canal_late','rail_early','rail_late')),
            'Teacher gradient qualification lacks full tables/phases')
        require(entry is not None and context['model_id']==entry['model_id'] and context['export_hash']==entry['sha256'],
            'Teacher gradient qualification checkpoint lineage differs')
        ratios.extend(raw)
    require(q['raw_median']==statistics.median(ratios) and
        q['qualified_generation']==observations[-1]['generation']+1<=state['iteration'],
        'Teacher fixed beta is not bound to the measured four-iteration window')
    return q


def annealing(config,state):
    from brass_ai.recovery import entropy_schedule as S
    if S.KEY not in config['update']:return False
    schedule=S.restore_run(config,state)
    return schedule['config']['mode']=='on' and (schedule['progress']<schedule['config']['ramp_iterations'] or S.waiting(schedule))


def authorize(state,config):
    require(is_on(config) and state['iteration']>=1261 and state.get('native_backend',{}).get('mode')=='rust',
        'Teacher on-mode requires the adopted Rust run at g1261 or later')
    q=qualification(state,config)
    require(not annealing(config,state),'Complete entropy/reward annealing before teacher on-mode')
    if not is_on(state['signature']['config']):
        from brass_ai.recovery.utility import reward_spec
        ready=state.get('teacher_phase',{}).get('readiness')
        require(ready is not None and ready['ready'] is True and ready['reward_id']==digest(reward_spec(config)),
            'Teacher activation requires measured current-reward utility readiness')
        stage=state.get('search_value',{}).get('stage',{})
        require(stage.get('stage')=='A' and stage.get('w')==stage.get('scale')==0,
            'Teacher activation must start at search-head Stage A with w=0 and frozen trunk')
    return q


def validate_backend(state,value):
    q=authorize(state,state['signature']['config'])
    receipt=value.get('teacher',{})
    require(receipt.get('contract')==CONTRACT and receipt.get('tables')=={'train':28,'held_out':4} and
        receipt.get('gradient_qualification')==q['sha256'] and receipt.get('beta')==q['beta'] and
        receipt.get('qualified_generation')==q['qualified_generation'], 'Teacher activation receipt changed')
