"""Opt-in unconditional residual stages and sparse, diagnostic teacher games."""
import copy

from brass_ai.learning.contracts import digest,require

CONTRACT='search-value-stage-schedule/2.0.0'


def defaults():
    return dict(contract=CONTRACT,stage_a_iterations=10,stage_b_iterations=20,
        stage_c_iterations=10,validation_interval=10,
        validation_during_beta_qualification=True)


def enabled(options):return 'schedule' in options


def validate(options):
    schedule=options['schedule'];expected=defaults()
    require(type(schedule) is dict and set(schedule)==set(expected),'Invalid search-value schedule fields')
    require(type(schedule['validation_during_beta_qualification']) is bool,'Invalid beta-qualification validation switch')
    fixed=dict(schedule,validation_during_beta_qualification=True)
    require(digest(fixed)==digest(expected),'Unreviewed search-value stage schedule')
    return schedule


def revise(config):
    from .search_value_config import defaults as original
    from .ppg_g210 import teacher_floor_options
    result=teacher_floor_options(config)
    options=result['teacher_phase']['search_value']
    if enabled(options):validate(options)
    else:
        require(options==dict(original(),stage='A'),'Stage revision requires the reviewed initial head recipe')
        options['schedule']=defaults()
        options['buffer']['staleness']='none'
        options['gate']=dict(mode='diagnostic-only',games=4,evaluation_target='final-utility-to-go')
        options['head_to_head']['games']=0
        options['trunk']['kl']=.003
        options['trunk']['ordinary_heldout_tolerance']=None
    return result


def previous_recipe(config):
    """Exact projection used only for reviewed recipe-transition comparison."""
    from .search_value_config import validate as validate_options,defaults as original
    from .ppg_g210 import TEACHER_FLOOR_CONTRACT,EVERY_ITER_CONTRACT
    from .training_revision_g1294 import previous as before_revision
    from .training_revision_g1307 import previous as before_lr_floor
    result=copy.deepcopy(before_revision(before_lr_floor(config)))
    options=result.get('teacher_phase',{}).get('search_value',{})
    if not enabled(options):return result
    validate_options(options)
    require(result['value_phase']['contract']==TEACHER_FLOOR_CONTRACT and
        result['value_phase']['beta_clone_min']==.25,'Revised residual stages require PPG floor .25')
    options.pop('schedule');options['head_to_head']['games']=256;options['trunk']['kl']='shared-ppg-cap'
    options['buffer']['staleness']='label-buffer-rule';options['gate']=original()['gate']
    options['trunk']['ordinary_heldout_tolerance']=.013
    result['value_phase'].update(contract=EVERY_ITER_CONTRACT,beta_clone_min=.125)
    return result


def validation_plan(config,state):
    options=config['teacher_phase'].get('search_value',{})
    if not enabled(options):return dict(due=True,games=4,reason='legacy-every-iteration')
    schedule=validate(options)
    from .teacher_gradient_qualification import restore
    phase=config['teacher_phase'];term=phase['term']
    q=restore(state.get('teacher_phase',{}).get('weight',{}).get('gradient_qualification'),
        term['share_target'],term['shadow_iterations'])
    qualifying=phase['mode']=='shadow' and not (q and q['passed'])
    forced=qualifying and schedule['validation_during_beta_qualification']
    periodic=(state['iteration']+1)%schedule['validation_interval']==0
    due=forced or periodic
    return dict(due=due,games=4 if due else 0,qualifying=qualifying,
        forced_by_beta_qualification=forced,interval=schedule['validation_interval'],
        reason='beta-qualification' if forced else 'periodic' if due else 'not-scheduled')


def initialize(state,options):
    """Adopt counters without relabelling old gated, untrained Stage-A time."""
    validate(options)
    if state.get('schedule')==CONTRACT:
        require(type(state.get('stage_a_iterations')) is int and 0<=state['stage_a_iterations']<=10 and
            type(state.get('stage_b_iterations')) is int and 0<=state['stage_b_iterations']<=20,
            'Invalid committed residual stage counters')
        require(state['stage']!='head_to_head' and
            (state['stage'] not in ('B','C') or state['stage_a_iterations']==10) and
            (state['stage']!='B' or state['stage_b_iterations']<20 and state['w']==round(state['stage_b_iterations']/20,2)) and
            (state['stage']!='C' or state['stage_b_iterations']==20 and state['w']==1.),
            'Residual stage counters/blend differ')
        return
    require('schedule' not in state,'Unknown committed residual schedule')
    if state['stage']=='head_to_head':state.update(stage='C',scale=0.)
    state.update(schedule=CONTRACT,stage_a_iterations=0 if state['stage']=='A' else 10,
        stage_b_iterations=round(state['w']*20) if state['stage']=='B' else 20 if state['stage']=='C' else 0,
        baseline=None,h2h=None,full_scale_generation=None,g0_passed=True)


def trunk_scale(state):
    return min(1.,round(state['scale']+.1,2)) if state['stage']=='C' else 0.


def advance(state,generation,diagnostic,*,pause=False):
    require(type(generation) is int and generation>state['generation'],'Stage transition already committed/out of order')
    state['generation']=generation
    state['history']=(state['history']+[dict(generation=generation,diagnostic=copy.deepcopy(diagnostic))])[-40:]
    if pause:return copy.deepcopy(state)
    if state['stage']=='A':
        state['stage_a_iterations']+=1
        if state['stage_a_iterations']==10:state['stage']='B'
    elif state['stage']=='B':
        state['stage_b_iterations']+=1
        state['w']=round(state['stage_b_iterations']/20,2)
        if state['stage_b_iterations']==20:state.update(stage='C',scale=0.)
    elif state['stage']=='C':
        state['scale']=trunk_scale(state)
        if state['scale']==1 and state['full_scale_generation'] is None:state['full_scale_generation']=generation
    return copy.deepcopy(state)


def diagnostics(rows,main,delta):
    """Four current held-out games are monitoring only, never admission gates."""
    import numpy as np
    require(len(rows)==len(main)==len(delta),'Held-out prediction count differs')
    if not rows:return dict(recorded=False,games=0,rows=0)
    require(all(r['split']=='heldout' for r in rows) and len({r['game_id'] for r in rows})==4 and
        len(rows)==256,'Periodic residual validation needs four complete games')
    target=np.asarray([r['u'] for r in rows]);base=float(np.mean((main-target)**2))
    corrected=float(np.mean((main+delta-target)**2))
    require(np.isfinite([base,corrected]).all(),'Nonfinite held-out residual diagnostic')
    return dict(recorded=True,games=4,rows=len(rows),main_mse=base,
        corrected_mse=corrected,improvement=base-corrected,scope='diagnostic-only')
