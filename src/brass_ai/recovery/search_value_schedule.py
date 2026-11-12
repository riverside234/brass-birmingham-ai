"""Opt-in unconditional residual stages and sparse, diagnostic teacher games."""
import copy

from brass_ai.learning.contracts import digest,require

CONTRACT='search-value-stage-schedule/2.0.0'
# Stage A lasts 10 iterations, or 30 in the reviewed g1290 restart recipe.
STAGE_A_ITERATIONS=(10,30)
HELDOUT_EARLY_STOP='heldout-early-stop/1.0.0'
DEDICATED_EARLY_STOP='dedicated-heldout-early-stop/1.0.0'
HELDOUT_ITERATIONS=20
HELDOUT_GAMES=16


def defaults():
    return dict(contract=CONTRACT,stage_a_iterations=10,stage_b_iterations=20,
        stage_c_iterations=10,validation_interval=10,
        validation_during_beta_qualification=True)


def restart_schedule():
    """The g1190 branch restart: Stage A 50 head-only iterations, each fitting with the
    Stage C held-out early stop; Stage B 50 (w rises .02 a collection); the PPG beta
    floor is .5 once the head has left Stage A."""
    return dict(defaults(),stage_a_iterations=50,stage_b_iterations=50,
        stage_a_fit=HELDOUT_EARLY_STOP,ppg_beta_floor_after_stage_a=.5)


def refreshed_schedule(stage_a):
    return dict(defaults(),stage_a_iterations=stage_a,stage_b_iterations=50,validation_interval=5,
        stage_a_fit=DEDICATED_EARLY_STOP,ppg_beta_floor_after_stage_b=.5)


def dedicated_holdout(options):
    return options.get('schedule', {}).get('stage_a_fit') == DEDICATED_EARLY_STOP


def enabled(options):return 'schedule' in options


def validate(options):
    schedule=options['schedule'];expected=defaults()
    if dedicated_holdout(options):
        require(type(schedule['validation_during_beta_qualification']) is bool and
            type(schedule['stage_a_iterations']) is int and schedule['stage_a_iterations'] in (50,100) and
            dict(schedule,validation_during_beta_qualification=True)==refreshed_schedule(schedule['stage_a_iterations']),
            'Invalid refreshed search-value schedule')
        return schedule
    if type(schedule) is dict and set(schedule)==set(restart_schedule()):
        require(type(schedule['validation_during_beta_qualification']) is bool,'Invalid beta-qualification validation switch')
        require(type(schedule['stage_a_iterations']) is int and schedule['stage_a_iterations'] in (50,100) and
            digest(dict(schedule,validation_during_beta_qualification=True,stage_a_iterations=50))==digest(restart_schedule()),
            'Unreviewed search-value stage schedule')
        return schedule
    require(type(schedule) is dict and set(schedule)==set(expected),'Invalid search-value schedule fields')
    require(type(schedule['validation_during_beta_qualification']) is bool,'Invalid beta-qualification validation switch')
    require(type(schedule['stage_a_iterations']) is int and schedule['stage_a_iterations'] in STAGE_A_ITERATIONS,
        'Unreviewed Stage A length')
    fixed=dict(schedule,validation_during_beta_qualification=True,stage_a_iterations=expected['stage_a_iterations'])
    require(digest(fixed)==digest(expected),'Unreviewed search-value stage schedule')
    return schedule


def stage_a_iterations(options):
    return validate(options)['stage_a_iterations'] if enabled(options) else defaults()['stage_a_iterations']


def stage_b_iterations(options):
    return validate(options)['stage_b_iterations'] if enabled(options) else defaults()['stage_b_iterations']


def stage_a_holdout(options):
    """Stage A fits in intervals checked by the Stage C early stop."""
    return enabled(options) and validate(options).get('stage_a_fit') in (HELDOUT_EARLY_STOP, DEDICATED_EARLY_STOP)


def beta_floor(options,stage):
    """The PPG beta floor this (already validated) schedule sets for the current head stage, or None."""
    if not enabled(options):return None
    if dedicated_holdout(options):
        return options['schedule']['ppg_beta_floor_after_stage_b'] if stage=='C' else .25
    floor=options['schedule'].get('ppg_beta_floor_after_stage_a')
    return floor if floor is not None and stage in ('B','C') else None


def blend_step(options):
    """Stage B raises w by 1/stage_b_iterations per collection (.05 for twenty)."""
    return round(1/stage_b_iterations(options),2)


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
    from .training_revision_g1491 import previous as before_g1491
    result=copy.deepcopy(before_revision(before_lr_floor(before_g1491(config))))
    from .adaptive_teacher_beta import KEY as ADAPTIVE_BETA
    result.get('teacher_phase',{}).pop(ADAPTIVE_BETA,None)
    # Width is separately bound by the signed recipe and strict checkpoint loader.
    result.pop('utility_hidden',None)
    options=result.get('teacher_phase',{}).get('search_value',{})
    if not enabled(options):return result
    validate_options(options)
    require(result['value_phase']['contract']==TEACHER_FLOOR_CONTRACT and
        result['value_phase']['beta_clone_min']==.25,'Revised residual stages require PPG floor .25')
    # The restart schedule's wider head (validated together with it) projects to the reviewed width.
    if dedicated_holdout(options):options['target']['lambda']=None
    options.pop('schedule');options['head']['hidden']=32;options['head_to_head']['games']=256;options['trunk']['kl']='shared-ppg-cap'
    options['buffer']['staleness']='label-buffer-rule';options['gate']=original()['gate']
    options['trunk']['ordinary_heldout_tolerance']=.013
    result['value_phase'].update(contract=EVERY_ITER_CONTRACT,beta_clone_min=.125)
    return result


def validation_plan(config,state):
    options=config['teacher_phase'].get('search_value',{})
    if not enabled(options):return dict(due=True,games=4,reason='legacy-every-iteration')
    schedule=validate(options)
    from .teacher_gradient_qualification import restore
    from .adaptive_teacher_beta import pending
    from .teacher_phase import committed_weight
    phase=config['teacher_phase'];term=phase['term']
    weight=committed_weight(config,state)
    q=restore(weight.get('gradient_qualification'),
        term['share_target'],term['shadow_iterations'])
    qualifying=phase['mode']=='shadow' and not (q and q['passed'])
    remeasuring=phase['mode']=='on' and pending(weight)
    forced=qualifying and schedule['validation_during_beta_qualification']
    periodic=(state['iteration']+1)%schedule['validation_interval']==0
    due=remeasuring or forced or periodic
    return dict(due=due,games=4 if due else 0,qualifying=qualifying,
        forced_by_beta_qualification=forced,interval=schedule['validation_interval'],
        **({'forced_by_beta_remeasurement': True} if remeasuring else {}),
        reason='beta-remeasurement' if remeasuring else 'beta-qualification' if forced else 'periodic' if due else 'not-scheduled')


def initialize(state,options):
    """Adopt counters without relabelling old gated, untrained Stage-A time."""
    a=stage_a_iterations(options);b=stage_b_iterations(options)
    if state.get('schedule')==CONTRACT:
        require(type(state.get('stage_a_iterations')) is int and 0<=state['stage_a_iterations']<=a and
            type(state.get('stage_b_iterations')) is int and 0<=state['stage_b_iterations']<=b,
            'Invalid committed residual stage counters')
        require(state['stage']!='head_to_head' and
            (state['stage']!='A' or state['stage_a_iterations']<a) and
            (state['stage'] not in ('B','C') or state['stage_a_iterations']==a) and
            (state['stage']!='B' or state['stage_b_iterations']<b and state['w']==round(state['stage_b_iterations']/b,2)) and
            (state['stage']!='C' or state['stage_b_iterations']==b and state['w']==1.),
            'Residual stage counters/blend differ')
        return
    require('schedule' not in state,'Unknown committed residual schedule')
    if state['stage']=='head_to_head':state.update(stage='C',scale=0.)
    state.update(schedule=CONTRACT,stage_a_iterations=0 if state['stage']=='A' else a,
        stage_b_iterations=round(state['w']*b) if state['stage']=='B' else b if state['stage']=='C' else 0,
        baseline=None,h2h=None,full_scale_generation=None,g0_passed=True)


def trunk_scale(state):
    return min(1.,round(state['scale']+.1,2)) if state['stage']=='C' else 0.


def advance(state,generation,diagnostic,*,pause=False,stage_a=10,stage_b=20):
    require(type(generation) is int and generation>state['generation'],'Stage transition already committed/out of order')
    state['generation']=generation
    state['history']=(state['history']+[dict(generation=generation,diagnostic=copy.deepcopy(diagnostic))])[-40:]
    if pause:return copy.deepcopy(state)
    if state['stage']=='A':
        state['stage_a_iterations']+=1
        if state['stage_a_iterations']==stage_a:state['stage']='B'
    elif state['stage']=='B':
        state['stage_b_iterations']+=1
        state['w']=round(state['stage_b_iterations']/stage_b,2)
        if state['stage_b_iterations']==stage_b:state.update(stage='C',scale=0.)
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
