"""Explicit delayed-critic objectives and versioned auxiliary transitions."""
import copy
import math

from brass_ai.learning.contracts import require

CONTRACT='ppg-value-phase/10.0.0'
BETA_CONTRACT='ppg-value-phase/11.0.0'
SIXTEEN_CONTRACT='ppg-value-phase/12.0.0'
SIXTEEN_MIGRATION='ppg-sixteen-packed-replay/1.0.0'
SIXTEEN_CLOCK='ppg-committed-cadence/4.0.0'
FAST_CONTRACT='ppg-value-phase/13.0.0'
FAST_MIGRATION='ppg-eight-dense-latest/1.0.0'
FAST_CLOCK='ppg-committed-cadence/5.0.0'
DENSE_CACHE='ppg-eight-disk-buffer/1.0.0'
PACKED_CACHE='ppg-columnar-disk-buffer/2.0.0'
BETA_MIGRATION='ppg-clone-beta-feedback/1.0.0'
BETA_CONTROL=dict(contract=BETA_MIGRATION,decrease_factor=.5,low_kl_fraction=.5,
    signal='max-final-overall-and-age0',scope='accepted-clean-phase')
FIXED_BETA_CONTROL=dict(contract='ppg-fixed-current-beta/1.0.0',mode='fixed',
    initialization='preserve-checkpoint-or-config',scope='penalty-only-no-KL-rejection')
MIGRATION='dcpg-preservation-eight-iteration/1.0.0'
CLOCK='ppg-committed-cadence/3.0.0'
RATE='ppg-ppo-constant-rate/2.0.0'
GRADIENT=dict(contract='ppo-critic-gradient/2.0.0',mode='preserve-outputs',
    scope='outcome-utility-and-local-scoring-heads',beta=1.,
    reference='post-calibration-pre-advantages-eval',descriptors='true-no-dropout',
    losses='existing-head-forms-and-scales',fresh_targets='value-phase-only')
SCHEDULE=dict(contract='ppg-phase-rate/2.0.0',reset_adam=True,
    warmup_start=1.,decay='none',end_factor=1.)


def active(config):return config.get('contract') in (CONTRACT,BETA_CONTRACT,SIXTEEN_CONTRACT,FAST_CONTRACT)


def beta_feedback(config):return config.get('contract') in (BETA_CONTRACT,SIXTEEN_CONTRACT,FAST_CONTRACT)


def sixteen(config):return config.get('contract')==SIXTEEN_CONTRACT


def fast(config):return config.get('contract')==FAST_CONTRACT


def penalty_only(config):return sixteen(config) or fast(config)


def fast_options(config):
    """Eight dense ages, latest accepted loss, same fixed-beta penalty-only recipe."""
    from . import ppg_config
    from .ppg_selection import LATEST_DEFAULT
    require(penalty_only(config['value_phase']),'Dense eight-age transition requires penalty-only PPG')
    ppg_config.validate(config['value_phase'])
    result=copy.deepcopy(config)
    result['value_phase'].update(contract=FAST_CONTRACT,buffer_iterations=8,
        validation_iterations=8,phase_interval=8,cache_format=DENSE_CACHE,
        selection=copy.deepcopy(LATEST_DEFAULT))
    ppg_config.validate(result['value_phase']);return result


def sixteen_options(config):
    """Sixteen packed ages and penalty-only PPG; never reset effective beta."""
    from . import ppg_config
    from .ppg_selection import MINIMUM_DEFAULT
    require(beta_feedback(config['value_phase']),'Sixteen-age replay requires the adaptive-beta DCPG recipe')
    ppg_config.validate(config['value_phase'])
    result=copy.deepcopy(config)
    result['value_phase'].update(contract=SIXTEEN_CONTRACT,buffer_iterations=16,
        validation_iterations=16,phase_interval=16,cache_format=PACKED_CACHE,
        kl_cap=None,kl_control='penalty-only',beta_control=copy.deepcopy(FIXED_BETA_CONTROL),
        selection=copy.deepcopy(MINIMUM_DEFAULT))
    ppg_config.validate(result['value_phase']);return result


def beta_options(config):
    """Explicit beta-only transition; preserve the delayed-critic recipe."""
    from . import ppg_config
    require(active(config['value_phase']),'Beta feedback requires the delayed-critic recipe')
    ppg_config.validate(config['value_phase'])
    if penalty_only(config['value_phase']):return copy.deepcopy(config)
    result=copy.deepcopy(config)
    result['value_phase'].update(contract=BETA_CONTRACT,beta_clone=1.,beta_clone_max=32.,
        beta_control=copy.deepcopy(BETA_CONTROL))
    ppg_config.validate(result['value_phase']);return result


def beta_after_phase(report,config):
    """One decrease per accepted clean phase; never undo a safeguard retry."""
    before=report['beta_clone_final'];after=before;reason='not_due'
    require(type(before) in (int,float) and math.isfinite(before) and
        config['beta_clone']<=before<=config['beta_clone_max'],'Invalid phase cloning beta')
    if penalty_only(config):
        return before,dict(contract=FIXED_BETA_CONTROL['contract'],before=before,after=before,
            decision='hold',reason='fixed_current_beta',hard_cap=None,
            scope='Keep checkpoint beta unchanged; fresh runs use configured beta. No KL-triggered adjustment, rejection, rollback, retry or stopping.')
    threshold=config['beta_control']['low_kl_fraction']*config['kl_cap']
    maximum=None
    if report['scheduled']:
        if report['kl_guard_tripped'] or report['beta_retries']:
            reason='kl_guard_or_retry'
        elif not report['optimizer_steps']:reason='no_accepted_steps'
        else:
            values=[report.get('probe_kl'),report.get('age0_probe_kl')]
            require(all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=config['kl_cap'] for v in values),
                'Invalid retained PPG KL for beta feedback')
            maximum=max(values);reason='kl_not_below_half_cap'
            if maximum<threshold:
                after=max(config['beta_clone'],before*config['beta_control']['decrease_factor'])
                reason='both_KLs_below_half_cap' if after<before else 'at_minimum'
    return after,dict(contract=BETA_MIGRATION,before=before,after=after,threshold=threshold,
        maximum_retained_kl=maximum,decision='down' if after<before else 'hold',reason=reason,
        scope='Applies to the next phase only; existing within-phase KL-triggered increases remain active')


def enabled(update):return update.get('value_gradient',{}).get('mode')=='preserve-outputs'


def options(config):
    from . import ppg_config,ppg_phasic
    if active(config['value_phase']):
        ppg_config.validate(config['value_phase']);return copy.deepcopy(config)
    result=ppg_phasic.options(config)
    result['value_phase'].update(contract=CONTRACT,phase_interval=8,acceptance_steps=16,
        learning_rate_factor=.35,rate_schedule=copy.deepcopy(SCHEDULE))
    ppg_config.validate(result['value_phase']);return result


def configure(config):
    result=options(config);result['update']['value_gradient']=copy.deepcopy(GRADIENT)
    return result


def initial(generation,config=None):
    wide=sixteen(config or {})
    return dict(contract=FAST_CLOCK if fast(config or {}) else SIXTEEN_CLOCK if wide else CLOCK,origin=generation,generation=generation,last_phase=generation,phases=0,
        cadence=dict(origin=generation,previous_phases=0,previous_last_phase=generation,first_phase=generation+(16 if wide else 8)))


def migrate_sixteen_clock(previous,generation,*,fresh_replay=False):
    from . import ppg_config
    ppg_config.schedule(previous,generation)
    if previous['contract']==SIXTEEN_CLOCK and not fresh_replay:return copy.deepcopy(previous)
    result=dict(copy.deepcopy(previous),contract=SIXTEEN_CLOCK,cadence=dict(origin=generation,
        previous_phases=previous['phases'],previous_last_phase=previous['last_phase'],first_phase=generation+16))
    return schedule(result,generation)


def migrate_fast_clock(previous,generation):
    from . import ppg_config
    ppg_config.schedule(previous,generation)
    if previous['contract']==FAST_CLOCK:return copy.deepcopy(previous)
    result=dict(copy.deepcopy(previous),contract=FAST_CLOCK,cadence=dict(origin=generation,
        previous_phases=previous['phases'],previous_last_phase=previous['last_phase'],first_phase=generation+8))
    return schedule(result,generation)


def migrate_clock(previous,generation):
    from . import ppg_config
    ppg_config.schedule(previous,generation)
    if previous['contract']==CLOCK:return copy.deepcopy(previous)
    result=dict(copy.deepcopy(previous),contract=CLOCK,cadence=dict(origin=generation,
        previous_phases=previous['phases'],previous_last_phase=previous['last_phase'],
        first_phase=max(generation+1,previous['last_phase']+8,previous['origin']+8)))
    return schedule(result,generation)


def next_phase(state):
    interval=16 if state['contract']==SIXTEEN_CLOCK else 8
    return state['cadence']['first_phase']+interval*(state['phases']-state['cadence']['previous_phases'])


def schedule(state,generation):
    require(type(state) is dict and set(state)==set(initial(0)) and state['contract'] in (CLOCK,SIXTEEN_CLOCK,FAST_CLOCK),
        'Invalid delayed-critic schedule fields')
    c=state['cadence']
    require(type(c) is dict and set(c)==set(initial(0)['cadence'])
        and all(type(v) is int for v in c.values())
        and all(type(state[k]) is int for k in ('origin','generation','last_phase','phases')),
        'Invalid delayed-critic schedule counters')
    done=state['phases']-c['previous_phases']
    interval=16 if state['contract']==SIXTEEN_CLOCK else 8
    first=c['origin']+interval if state['contract'] in (SIXTEEN_CLOCK,FAST_CLOCK) else max(c['origin']+1,c['previous_last_phase']+8,state['origin']+8)
    require(0<=state['origin']<=c['previous_last_phase']<=c['origin']<=generation==state['generation']
        and 0<=c['previous_phases']<=state['phases']
        and c['first_phase']==first
        and state['last_phase']==(c['first_phase']+interval*(done-1) if done else c['previous_last_phase'])
        and state['last_phase']<=generation<next_phase(state),'Invalid delayed-critic committed schedule')
    return state


def rate_receipt(ppo_lr,generation,steps):
    result=dict(contract=RATE,peak_lr=ppo_lr*.35,ppo_lr=ppo_lr,factor=.35,
        generation=generation,source='ppo-last-applied',ppo_steps=steps)
    return validate_rate(result,generation)


def validate_rate(r,generation):
    require(type(r) is dict and set(r)=={'contract','peak_lr','ppo_lr','factor','generation','source','ppo_steps'}
        and r['contract']==RATE and r['source']=='ppo-last-applied'
        and type(r['generation']) is int and 0<=r['generation']<=generation
        and type(r['ppo_steps']) is int and r['ppo_steps']>0
        and all(type(r[k]) in (int,float) and math.isfinite(r[k]) and r[k]>0 for k in ('peak_lr','ppo_lr','factor'))
        and r['factor']==.35 and r['peak_lr']==r['ppo_lr']*.35,'Invalid constant PPG/PPO LR receipt')
    return r
