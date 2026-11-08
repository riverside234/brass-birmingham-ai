"""Explicit g210 auxiliary recipe, with reviewed beta/LR/epoch adjustments.

PPO's objective is deliberately outside this configuration migration. The
versioned KL guard and monotone loss acceptance live in value_phase.fit.
"""
import copy
import math
import time

import numpy as np
from brass_ai.learning.search_value import ordinary_parameters
import torch

from brass_ai.learning.contracts import require

CONTRACT='ppg-value-phase/14.0.0'
MIGRATION='ppg-g210-two-age-ppo-rate/1.0.0'
CHECKPOINT='actor-value-phases/11.0.0'
RATE='ppg-actual-ppo-rate/2.0.0'
DEFAULT=dict(contract=CONTRACT,enabled=True,buffer_iterations=2,max_epochs=3,
    batch_size=4096,clone_rows=8192,clone_batch_size=512,beta_clone=4.,
    beta_clone_min=1.,beta_clone_max=32.,kl_cap=.002,learning_rate=None,
    learning_rate_source='ppo-last-applied',learning_rate_factor=.65,
    holdout_fraction=.05,holdout=dict(contract='ppg-critic-holdout/2.0.0',minimum_games_per_table=2),
    acceptance_steps=16,kl_check_steps=32,phase_interval=1,validation_iterations=1)
COSINE_CONTRACT='ppg-value-phase/15.0.0'
COSINE=dict(contract='ppg-cosine-warmup-decay/1.0.0',warmup_steps=16,start_factor=.1,end_factor=.3)
COSINE_DEFAULT=dict(DEFAULT,contract=COSINE_CONTRACT,max_epochs=2,beta_clone_min=.5,
    kl_check_steps=16,validation_iterations=2,rate_schedule=COSINE)
LINEAR_CONTRACT='ppg-value-phase/16.0.0'
LINEAR_MIGRATION='ppg-linear-warmup-near-zero-decay/1.0.0'
LINEAR=dict(contract='ppg-linear-warmup-cosine-decay/1.0.0',warmup_steps=16,start_factor=.1,end_factor=.001)
LINEAR_DEFAULT=dict(COSINE_DEFAULT,contract=LINEAR_CONTRACT,beta_clone_min=1.,rate_schedule=LINEAR)
SHORT_CONTRACT='ppg-value-phase/17.0.0'
SHORT_MIGRATION='ppg-eight-step-loss-half-beta/1.0.0'
SHORT_DEFAULT=dict(LINEAR_DEFAULT,contract=SHORT_CONTRACT,beta_clone_min=.5,acceptance_steps=8)
LOSS16_CONTRACT='ppg-value-phase/18.0.0'
LOSS16_MIGRATION='ppg-sixteen-step-loss-half-beta/1.0.0'
LOSS16_DEFAULT=dict(SHORT_DEFAULT,contract=LOSS16_CONTRACT,acceptance_steps=16)
KL32_CONTRACT='ppg-value-phase/19.0.0'
KL32_MIGRATION='ppg-loss16-kl32-half-beta/1.0.0'
KL32_DEFAULT=dict(LOSS16_DEFAULT,contract=KL32_CONTRACT,kl_check_steps=32)
BETA1_CONTRACT='ppg-value-phase/20.0.0'
BETA1_MIGRATION='ppg-loss16-kl32-beta-floor-one/1.0.0'
BETA1_DEFAULT=dict(KL32_DEFAULT,contract=BETA1_CONTRACT,beta_clone_min=1.)
QUARTER_CONTRACT='ppg-value-phase/21.0.0'
QUARTER_MIGRATION='ppg-loss16-kl32-quarter-beta-cap003/1.0.0'
QUARTER_DEFAULT=dict(BETA1_DEFAULT,contract=QUARTER_CONTRACT,beta_clone_min=.25,kl_cap=.003)
EARLY_CONTRACT='ppg-value-phase/22.0.0'
EARLY_MIGRATION='ppg-early-retained-kl-beta/1.0.0'
EARLY_DEFAULT=dict(QUARTER_DEFAULT,contract=EARLY_CONTRACT,
    early_kl_beta=dict(retained_steps_at_most=16,multiplier=1.5))
TWO_ITER_CONTRACT='ppg-value-phase/23.0.0'
TWO_ITER_MIGRATION='ppg-two-iteration-half-rate-eighth-beta/1.0.0'
TWO_ITER_CHECKPOINT='actor-value-phases/12.0.0'
TWO_ITER_DEFAULT=dict(EARLY_DEFAULT,contract=TWO_ITER_CONTRACT,
    beta_clone_min=.125,learning_rate_factor=.5,phase_interval=2)
EVERY_ITER_CONTRACT='ppg-value-phase/24.0.0'
EVERY_ITER_MIGRATION='ppg-every-iteration-beta-eighth-hotfix/1.0.0'
EVERY_ITER_DEFAULT=dict(TWO_ITER_DEFAULT,contract=EVERY_ITER_CONTRACT,
    learning_rate_factor=.65,phase_interval=1)
TEACHER_FLOOR_CONTRACT='ppg-value-phase/25.0.0'
TEACHER_FLOOR_MIGRATION='ppg-teacher-quarter-beta-floor/1.0.0'
TEACHER_FLOOR_DEFAULT=dict(EVERY_ITER_DEFAULT,contract=TEACHER_FLOOR_CONTRACT,beta_clone_min=.25)


def active(c):return c.get('contract') in (CONTRACT,COSINE_CONTRACT,LINEAR_CONTRACT,SHORT_CONTRACT,LOSS16_CONTRACT,KL32_CONTRACT,BETA1_CONTRACT,QUARTER_CONTRACT,EARLY_CONTRACT,TWO_ITER_CONTRACT,EVERY_ITER_CONTRACT,TEACHER_FLOOR_CONTRACT)


def cosine(c):return c.get('contract')==COSINE_CONTRACT


def linear(c):return c.get('contract') in (LINEAR_CONTRACT,SHORT_CONTRACT,LOSS16_CONTRACT,KL32_CONTRACT,BETA1_CONTRACT,QUARTER_CONTRACT,EARLY_CONTRACT,TWO_ITER_CONTRACT,EVERY_ITER_CONTRACT,TEACHER_FLOOR_CONTRACT)


def short(c):return c.get('contract')==SHORT_CONTRACT


def scheduled(c):return cosine(c) or linear(c)


def two_iteration(c):return c.get('contract')==TWO_ITER_CONTRACT


def teacher_floor_options(config):
    validate(config['value_phase'])
    require(config['value_phase']['contract'] in (EVERY_ITER_CONTRACT,TEACHER_FLOOR_CONTRACT),
        'Teacher stage revision requires the every-iteration PPG recipe')
    result=copy.deepcopy(config)
    result['value_phase'].update(contract=TEACHER_FLOOR_CONTRACT,beta_clone_min=.25)
    validate(result['value_phase']);return result


def every_iteration_options(config):
    validate(config['value_phase'])
    require(config['value_phase']['contract'] in (TWO_ITER_CONTRACT,EVERY_ITER_CONTRACT),
        'Every-iteration beta hotfix requires the two-iteration PPG recipe')
    result=copy.deepcopy(config)
    result['value_phase'].update(contract=EVERY_ITER_CONTRACT,learning_rate_factor=.65,phase_interval=1)
    validate(result['value_phase']);return result


def two_iteration_options(config):
    """Change only future PPG cadence, LR factor and allowed beta floor."""
    validate(config['value_phase'])
    require(config['value_phase']['contract'] in (EARLY_CONTRACT,TWO_ITER_CONTRACT),
        'Two-iteration PPG requires the reviewed early-KL beta recipe')
    result=copy.deepcopy(config)
    result['value_phase'].update(contract=TWO_ITER_CONTRACT,beta_clone_min=.125,
        learning_rate_factor=.5,phase_interval=2)
    validate(result['value_phase']);return result


def phase_clock(origin,generation,config):
    require(type(origin) is int and type(generation) is int and 0<=origin<=generation,
        'Invalid committed PPG phase origin')
    interval=config['phase_interval']
    completed=(generation-origin)//interval
    last=origin+completed*interval
    return dict(origin=origin,generation=generation,phase_interval=interval,
        scheduled_phases=completed,last_scheduled_phase=last if completed else None,
        next_phase=last+interval)


def early_options(config):
    validate(config['value_phase'])
    require(config['value_phase']['contract'] in (QUARTER_CONTRACT,EARLY_CONTRACT),
        'Early-KL beta response requires quarter-floor/cap .003 PPG')
    result=copy.deepcopy(config)
    result['value_phase'].update(contract=EARLY_CONTRACT,early_kl_beta=copy.deepcopy(EARLY_DEFAULT['early_kl_beta']))
    validate(result['value_phase']);return result


def quarter_options(config):
    validate(config['value_phase'])
    require(config['value_phase']['contract'] in (BETA1_CONTRACT,QUARTER_CONTRACT),
        'Quarter-beta continuation requires loss16/KL32 beta-floor-one PPG')
    result=copy.deepcopy(config)
    result['value_phase'].update(contract=QUARTER_CONTRACT,beta_clone_min=.25,kl_cap=.003)
    validate(result['value_phase']);return result


def beta1_options(config):
    validate(config['value_phase'])
    require(config['value_phase']['contract'] in (KL32_CONTRACT,BETA1_CONTRACT),'Beta floor one requires loss16/KL32 PPG')
    result=copy.deepcopy(config);result['value_phase'].update(contract=BETA1_CONTRACT,beta_clone_min=1.)
    validate(result['value_phase']);return result


def kl32_options(config):
    """Loss16/KL32, preserving saved beta and compatible two-age replay."""
    validate(config['value_phase'])
    require(config['value_phase']['contract'] in (SHORT_CONTRACT,LOSS16_CONTRACT,KL32_CONTRACT),
        'KL32 continuation requires the g1050 or g1120 PPG recipe')
    result=copy.deepcopy(config)
    result['value_phase'].update(contract=KL32_CONTRACT,acceptance_steps=16,kl_check_steps=32)
    return result


def loss16_options(config):
    """Restore loss16 checks without repeating the earlier beta hotfix."""
    validate(config['value_phase'])
    require(config['value_phase']['contract'] in (SHORT_CONTRACT,LOSS16_CONTRACT),
        'Sixteen-step half-beta loss checks require the g1050 PPG recipe')
    result=copy.deepcopy(config)
    result['value_phase'].update(contract=LOSS16_CONTRACT,acceptance_steps=16)
    return result


def short_options(config):
    """Exact g1050 PPG-only transition; preserve timing, targets and cache ages."""
    validate(config['value_phase'])
    require(linear(config['value_phase']),'Eight-step loss checks require the g950 PPG recipe')
    result=copy.deepcopy(config)
    result['value_phase'].update(contract=SHORT_CONTRACT,beta_clone_min=.5,acceptance_steps=8)
    return result


def linear_options(config):
    """Exact g950 PPG-only transition; compatible critic caches stay valid."""
    validate(config['value_phase'])
    require(scheduled(config['value_phase']),'Linear warmup requires the two-age g850 PPG recipe')
    result=copy.deepcopy(config)
    result['value_phase'].update(contract=LINEAR_CONTRACT,beta_clone_min=1.,rate_schedule=copy.deepcopy(LINEAR))
    return result


def cosine_options(config):
    validate(config['value_phase'])
    result=copy.deepcopy(config)
    result['value_phase'].update(contract=COSINE_CONTRACT,max_epochs=2,beta_clone_min=.5,
        kl_check_steps=16,validation_iterations=2,rate_schedule=copy.deepcopy(COSINE))
    return result


def rate_factor(step,total,config):
    """One-based attempted step; full-cap schedule, never compressed by rejection."""
    require(type(step) is int and type(total) is int and 1<=step<=total,'Invalid PPG rate step')
    schedule=config['rate_schedule'];warmup=schedule['warmup_steps']
    if step<=warmup:
        fraction=(step-1)/(warmup-1)
        if not linear(config):fraction=.5*(1-math.cos(math.pi*fraction))
        return schedule['start_factor']+(1-schedule['start_factor'])*fraction
    fraction=(step-warmup)/(total-warmup)
    return schedule['end_factor']+(1-schedule['end_factor'])*.5*(1+math.cos(math.pi*fraction))


def validate(c):
    default=TEACHER_FLOOR_DEFAULT if c.get('contract')==TEACHER_FLOOR_CONTRACT else EVERY_ITER_DEFAULT if c.get('contract')==EVERY_ITER_CONTRACT else TWO_ITER_DEFAULT if two_iteration(c) else EARLY_DEFAULT if c.get('contract')==EARLY_CONTRACT else QUARTER_DEFAULT if c.get('contract')==QUARTER_CONTRACT else BETA1_DEFAULT if c.get('contract')==BETA1_CONTRACT else KL32_DEFAULT if c.get('contract')==KL32_CONTRACT else LOSS16_DEFAULT if c.get('contract')==LOSS16_CONTRACT else SHORT_DEFAULT if short(c) else LINEAR_DEFAULT if linear(c) else COSINE_DEFAULT if cosine(c) else DEFAULT
    require(type(c) is dict and set(c)==set(default) and active(c),'Invalid g210 PPG fields/contract')
    # Smaller batches are useful for bounded regression/smoke fixtures only.
    for k,v in default.items():
        if k in ('batch_size','clone_rows','clone_batch_size'):continue
        require(c[k]==v and type(c[k]) is type(v),'Invalid g210 PPG option: '+k)
    if scheduled(c):
        require(all(type(c['rate_schedule'][k]) is type(v) for k,v in default['rate_schedule'].items()),
            'Invalid PPG rate schedule field type')
    if c['contract'] in (EARLY_CONTRACT,TWO_ITER_CONTRACT,EVERY_ITER_CONTRACT,TEACHER_FLOOR_CONTRACT):
        require(all(type(c['early_kl_beta'][k]) is type(v) for k,v in EARLY_DEFAULT['early_kl_beta'].items()),
            'Invalid early-KL beta field type')
    require(all(type(c[k]) is int and 1<=c[k]<=DEFAULT[k] for k in ('batch_size','clone_rows','clone_batch_size'))
        and c['clone_batch_size']<=c['clone_rows'],'Invalid g210 PPG batch bounds')
    return c


def options(config):
    from .value_phase import validate as validate_value
    validate_value(config['value_phase'])
    result=copy.deepcopy(config)
    if not active(result['value_phase']):result['value_phase']=copy.deepcopy(DEFAULT)
    validate(result['value_phase']);return result


def rate_receipt(rate,generation,steps,config=None):
    require(type(rate) in (int,float) and math.isfinite(rate) and rate>0
        and type(steps) is int and steps>0,'PPG needs an actual applied PPO LR')
    factor=(config or DEFAULT)['learning_rate_factor']
    return dict(contract=RATE,source='ppo-last-applied',generation=generation,
        ppo_steps=steps,ppo_lr=rate,factor=factor,peak_lr=rate*factor)


def beta_after_phase(beta,trips,report,config):
    """g210's new-trip/3-of-5 escalation; clean accepted low-KL phases halve."""
    require(math.isfinite(beta) and config['beta_clone_min']<=beta<=32 and type(trips) is list
        and len(trips)<=5 and all(type(x) is bool for x in trips),'Invalid g210 beta state')
    if report.get('scheduled') is False:
        require(two_iteration(config) and report['optimizer_steps']==0 and not report['kl_guard_tripped'],
            'Unscheduled PPG phase cannot have optimizer work or a KL trip')
        return beta,list(trips),dict(before=beta,after=beta,reason='not_due',decision='hold',
            minimum=config['beta_clone_min'],maximum=config['beta_clone_max'],kl_cap=config['kl_cap'],
            scope='No phase: preserve beta and the last five actual phase outcomes.')
    trips=(trips+[bool(report['kl_guard_tripped'])])[-5:]
    retained=report['optimizer_steps']
    require(type(retained) is int and retained>=0,'Invalid retained PPG step count')
    early=config.get('early_kl_beta')
    early_hit=bool(early and trips[-1] and retained<=early['retained_steps_at_most'])
    after=beta;reason='fewer_than_three_trips' if trips[-1] else 'no_new_kl_trip'
    if trips[-1] and sum(trips)>=3:
        after=min(config['beta_clone_max'],beta*2)
        reason='new_trip_in_three_of_five' if after>beta else 'at_maximum'
    elif early_hit:
        after=min(config['beta_clone_max'],beta*early['multiplier'])
        reason='kl_trip_with_at_most_16_retained_steps' if after>beta else 'at_maximum'
    elif not trips[-1] and report['optimizer_steps']>0:
        kl=report.get('probe_kl')
        require(type(kl) in (int,float) and math.isfinite(kl) and 0<=kl<=config['kl_cap']+1e-7,
            'Invalid retained PPG KL')
        if kl<.5*config['kl_cap']:
            after=max(config['beta_clone_min'],beta/2)
            reason='retained_KL_below_half_cap' if after<beta else 'at_minimum'
    elif not trips[-1]:reason='no_accepted_steps'
    return after,trips,dict(before=beta,after=after,reason=reason,
        retained_steps=retained,early_kl_trigger=early_hit,
        decision='up' if after>beta else 'down' if after<beta else 'hold',
        threshold=.5*config['kl_cap'],minimum=config['beta_clone_min'],maximum=config['beta_clone_max'],kl_cap=config['kl_cap'],
        scope=('Immediately persist beta for the next phase: KL hit with <=16 retained steps raises x1.5; '
               'the existing 3/5 rule takes precedence (x2, never stacked). ' if early else 'Next phase only; ')+
              'A new hard-cap trip with 3/5 hits doubles beta. Clean accepted phases below half cap halve beta. No within-phase retries.')


def initialize(phases,model,config,*,raw,generation,root,run):
    from . import actor_value,gae,critic_data,ppg_config,ppg_epochs,ppg_replay
    validate(config['value_phase'])
    require(root is not None and run is not None,'g210 PPG requires an owned disk cache')
    phases.config=config;phases.root=root;phases.run=run;phases.rolling=False
    phases.options={k:copy.deepcopy(config[k]) for k in ('advantage','value_phase','critic_data','seat_descriptors') if k in config}
    phases.contract=TWO_ITER_CHECKPOINT if two_iteration(config['value_phase']) else CHECKPOINT
    phases.snapshot_data=[];phases.fingerprint=None;phases.phase_origin=None
    phases.dataset=None;phases.probe_ids=None;phases.collection_source=None;phases.ppo_work=None
    for name in ('critic_evidence','disable_migration','holdout_migration','refinement_migration','window_migration',
                 'evidence_migration','critic_migration','snapshot_r2_migration','disabled_startup_cleanup','g210_migration',
                 'ppg_schedule_migration'):
        setattr(phases,name,None)
    phases.optimizer=torch.optim.Adam(ordinary_parameters(model),lr=config['learning_rate']*config['value_phase']['learning_rate_factor'])
    saved=(raw or {}).get('training',{}).get(actor_value.KEY)
    if saved is None:
        require(generation==0,'Missing g210 continuation checkpoint')
        phases.state=gae.initial(config['advantage'])
        phases.critic_state=critic_data.initial(config['critic_data']) if 'critic_data' in config else None
        phases.beta=4.;phases.trips=[];phases.cache_origin=0;phases.adoption=None;receipts=[]
        if two_iteration(config['value_phase']):phases.phase_origin=0
    else:
        saved=copy.deepcopy(saved)
        migrating=not active(saved.get('options',{}).get('value_phase',{}))
        if migrating:
            require(saved.get('contract')=='actor-value-phases/10.0.0' and
                options(saved['options'])==phases.options,'Unreviewed g210 PPG continuation')
            clock=ppg_config.schedule(saved['schedule'],generation)
            ppg_epochs.validate(saved['epoch_control'],saved['options']['value_phase'],
                phases=clock['phases'],last_phase=clock['last_phase'])
            phases.adoption=dict(contract=MIGRATION,origin=generation,beta_before=saved['beta_clone'],beta_after=4.,
                previous_schedule=copy.deepcopy(clock),invalidated_generations=[r['generation'] for r in saved['rolling_buffer']],
                scope='Only PPG changes. Keep model, PPO/value Adam moments, RNG, GAE, snapshot state and ordinary progress. New holdout protocol starts fresh auxiliary caches; first phase uses current data, then current plus previous.')
            phases.g210_migration=copy.deepcopy(phases.adoption)
            phases.beta=4.;phases.trips=[];phases.cache_origin=generation;receipts=[]
        else:
            fields={'contract','options','gae','beta_clone','kl_trips','optimizer','rolling_buffer','cache_origin','adoption'}
            if 'critic_data' in config:fields.add('critic_data')
            options_match=saved['options']==phases.options
            if cosine(config['value_phase']) and saved['options']['value_phase']['contract']==CONTRACT:
                options_match=cosine_options(saved['options'])==phases.options
            adopting_linear=config['value_phase']['contract']==LINEAR_CONTRACT and cosine(saved['options']['value_phase'])
            if adopting_linear:options_match=linear_options(saved['options'])==phases.options
            adopting_short=short(config['value_phase']) and saved['options']['value_phase']['contract']==LINEAR_CONTRACT
            if adopting_short:options_match=short_options(saved['options'])==phases.options
            adopting_loss16=config['value_phase']['contract']==LOSS16_CONTRACT and short(saved['options']['value_phase'])
            if adopting_loss16:options_match=loss16_options(saved['options'])==phases.options
            adopting_kl32=config['value_phase']['contract']==KL32_CONTRACT and saved['options']['value_phase']['contract'] in (SHORT_CONTRACT,LOSS16_CONTRACT)
            if adopting_kl32:options_match=kl32_options(saved['options'])==phases.options
            adopting_beta1=config['value_phase']['contract']==BETA1_CONTRACT and saved['options']['value_phase']['contract']==KL32_CONTRACT
            if adopting_beta1:options_match=beta1_options(saved['options'])==phases.options
            adopting_quarter=config['value_phase']['contract']==QUARTER_CONTRACT and saved['options']['value_phase']['contract']==BETA1_CONTRACT
            if adopting_quarter:options_match=quarter_options(saved['options'])==phases.options
            adopting_early=config['value_phase']['contract']==EARLY_CONTRACT and saved['options']['value_phase']['contract']==QUARTER_CONTRACT
            if adopting_early:options_match=early_options(saved['options'])==phases.options
            adopting_two_iteration=two_iteration(config['value_phase']) and saved['options']['value_phase']['contract']==EARLY_CONTRACT
            if adopting_two_iteration:options_match=two_iteration_options(saved['options'])==phases.options
            adopting_every_iteration=config['value_phase']['contract']==EVERY_ITER_CONTRACT and two_iteration(saved['options']['value_phase'])
            if adopting_every_iteration:options_match=every_iteration_options(saved['options'])==phases.options
            adopting_teacher_floor=config['value_phase']['contract']==TEACHER_FLOOR_CONTRACT and saved['options']['value_phase']['contract']==EVERY_ITER_CONTRACT
            if adopting_teacher_floor:options_match=teacher_floor_options(saved['options'])==phases.options
            if two_iteration(saved['options']['value_phase']):fields.add('phase_origin')
            expected_contract=TWO_ITER_CHECKPOINT if two_iteration(saved['options']['value_phase']) else CHECKPOINT
            require(saved.get('contract')==expected_contract and set(saved)==fields and options_match,
                'g210 checkpoint configuration differs')
            phases.beta=saved['beta_clone'];phases.trips=saved['kl_trips']
            phases.cache_origin=saved['cache_origin'];receipts=saved['rolling_buffer'];phases.adoption=saved['adoption']
            if adopting_teacher_floor:
                require(type(phases.beta) in (int,float) and math.isfinite(phases.beta) and
                    saved['options']['value_phase']['beta_clone_min']<=phases.beta<=32,'Invalid beta before teacher-floor adoption')
                phases.ppg_schedule_migration=dict(contract=TEACHER_FLOOR_MIGRATION,origin=generation,
                    beta_before=phases.beta,beta_after=max(.25,phases.beta),previous_adoption=copy.deepcopy(phases.adoption),
                    scope='Raise the floor to .25 without resetting beta above it, trip history, cached replay, model/Adam/RNG or PPG cadence.')
                phases.adoption=copy.deepcopy(phases.ppg_schedule_migration);phases.beta=max(.25,phases.beta)
            if adopting_every_iteration:
                phase_clock(saved['phase_origin'],generation,saved['options']['value_phase'])
                require(type(phases.beta) in (int,float) and math.isfinite(phases.beta) and
                    saved['options']['value_phase']['beta_clone_min']<=phases.beta<=32,'Invalid beta before one-time hotfix')
                phases.ppg_schedule_migration=dict(contract=EVERY_ITER_MIGRATION,origin=generation,
                    beta_before=phases.beta,beta_after=.125,previous_adoption=copy.deepcopy(phases.adoption),
                    previous_phase_origin=saved['phase_origin'],next_phase=generation+1,
                    scope='One-time beta .125 reset; every iteration, PPO peak factor .65. Preserve trip history, replay, both Adams/RNG, weights and gate evidence.')
                phases.adoption=copy.deepcopy(phases.ppg_schedule_migration);phases.beta=.125
            if two_iteration(config['value_phase']):
                phases.phase_origin=generation if adopting_two_iteration else saved['phase_origin']
            if adopting_two_iteration:
                phases.ppg_schedule_migration=dict(contract=TWO_ITER_MIGRATION,origin=generation,
                    beta_before=phases.beta,beta_after=phases.beta,previous_adoption=copy.deepcopy(phases.adoption),
                    next_phase=generation+2,
                    scope='Preserve beta/trips, both Adams/RNG, replay, critic/GAE evidence; floor .125, PPO peak factor .5. First phase after two new committed collections.')
                phases.adoption=copy.deepcopy(phases.ppg_schedule_migration)
            if adopting_early:
                phases.ppg_schedule_migration=dict(contract=EARLY_MIGRATION,origin=generation,
                    beta_before=phases.beta,beta_after=phases.beta,previous_adoption=copy.deepcopy(phases.adoption),
                    scope='Preserve beta/trips, optimizers/RNG, compatible replay, cadence and evidence; add early retained-step KL response.')
                phases.adoption=copy.deepcopy(phases.ppg_schedule_migration)
            if adopting_quarter:
                require(type(phases.beta) in (int,float) and math.isfinite(phases.beta)
                    and saved['options']['value_phase']['beta_clone_min']<=phases.beta<=32,'Invalid beta before quarter-floor adoption')
                phases.ppg_schedule_migration=dict(contract=QUARTER_MIGRATION,origin=generation,
                    beta_before=phases.beta,beta_after=phases.beta,previous_adoption=copy.deepcopy(phases.adoption),
                    scope='Preserve saved beta/trips, Adam/RNG, compatible cache and cadence. New floor .25; cap .003; clean accepted phases below .0015 halve beta for the next phase.')
                phases.adoption=copy.deepcopy(phases.ppg_schedule_migration)
            if adopting_beta1:
                require(type(phases.beta) in (int,float) and math.isfinite(phases.beta)
                    and saved['options']['value_phase']['beta_clone_min']<=phases.beta<=32,'Invalid beta before floor adoption')
                phases.ppg_schedule_migration=dict(contract=BETA1_MIGRATION,origin=generation,
                    beta_before=phases.beta,beta_after=max(1.,phases.beta),previous_adoption=copy.deepcopy(phases.adoption),
                    scope='Raise beta only if below the new floor; preserve trip history, cache, Adam/RNG, schedule and progress.')
                phases.adoption=copy.deepcopy(phases.ppg_schedule_migration);phases.beta=max(1.,phases.beta)
            if adopting_linear or adopting_short:
                require(type(phases.beta) in (int,float) and math.isfinite(phases.beta) and
                    saved['options']['value_phase']['beta_clone_min']<=phases.beta<=32,
                    'Invalid beta before PPG schedule adoption')
                beta_after=.5 if adopting_short else 1.
                phases.ppg_schedule_migration=dict(contract=SHORT_MIGRATION if adopting_short else LINEAR_MIGRATION,origin=generation,
                    beta_before=phases.beta,beta_after=beta_after,previous_adoption=copy.deepcopy(phases.adoption),
                    scope='One-time beta hotfix; preserve trip history, model/Adam/RNG, cached targets, progress and gate evidence.')
                phases.adoption=copy.deepcopy(phases.ppg_schedule_migration);phases.beta=beta_after
            if adopting_loss16 or adopting_kl32:
                phases.ppg_schedule_migration=dict(contract=KL32_MIGRATION if adopting_kl32 else LOSS16_MIGRATION,origin=generation,
                    beta_before=phases.beta,beta_after=phases.beta,previous_adoption=copy.deepcopy(phases.adoption),
                    scope=f'Loss checks every 16 steps, KL every {config["value_phase"]["kl_check_steps"]}; preserve beta/trips, model/Adam/RNG, cached targets, progress and gate evidence.')
                phases.adoption=copy.deepcopy(phases.ppg_schedule_migration)
        phases.state=copy.deepcopy(gae.validate_state(saved['gae'],config['advantage'],generation))
        phases.critic_state=copy.deepcopy(critic_data.validate_state(saved['critic_data'],config['critic_data'],generation)) if 'critic_data' in config else None
        phases.optimizer.load_state_dict(saved['optimizer'])
    require(type(phases.beta) in (int,float) and math.isfinite(phases.beta) and config['value_phase']['beta_clone_min']<=phases.beta<=32
        and type(phases.trips) is list and len(phases.trips)<=5 and all(type(x) is bool for x in phases.trips),
        'Invalid persisted g210 beta')
    group=phases.optimizer.param_groups[0]
    require(len(phases.optimizer.param_groups)==1 and math.isfinite(group['lr']) and group['lr']>0
        and group['weight_decay']==0,'Invalid g210 value Adam')
    require(type(phases.cache_origin) is int and 0<=phases.cache_origin<=generation,'Invalid g210 replay origin')
    from .utility import reward_spec,reviewed_reward_upgrade
    previous_reward=(raw or {}).get('model_config',{}).get('utility')
    current_reward=reward_spec(config)
    if previous_reward is not None and previous_reward!=current_reward:
        require(reviewed_reward_upgrade(previous_reward,current_reward),'Unreviewed g210 reward-cache transition')
        phases.g210_migration=dict(contract='ppg-g210-reward-cache/1.0.0',origin=generation,
            invalidated_generations=[r['generation'] for r in receipts],previous_reward=previous_reward,
            current_reward=current_reward,beta_before=phases.beta,beta_after=phases.beta,
            scope='Discard incompatible reward targets after reviewed adoption; preserve beta/trips, Adam/RNG and all gate evidence.')
        phases.adoption=copy.deepcopy(phases.g210_migration)
        receipts=[];phases.cache_origin=generation
    phases.buffer=ppg_replay.Buffer(root,run,config,generation=generation,receipts=receipts,origin=phases.cache_origin)
    if two_iteration(config['value_phase']):phase_clock(phases.phase_origin,generation,config['value_phase'])


def startup(phases,generation):
    if phases.g210_migration is None:return None
    from .value_buffer import retire_fresh
    from brass_ai.step5.league import write_json
    require(phases.cache_origin==generation and not phases.buffer.receipts,'Invalid g210 startup boundary')
    result=retire_fresh(phases.root,phases.run,generation,reward_id=phases.buffer.reward_id)
    write_json(phases.run/'ppg-fresh-cache-reset.json',dict(result,generation=generation,phase='startup',next_phase=generation+1))
    return dict(result,fresh_ppg=True,next_phase=generation+1)


def fit(phases,model,preparation_pool):
    from . import value_phase
    from .ppg_phasic import RATE_KEY
    from .ppo_preparation import phase_preparation
    require(phases.collection_source is not None,'g210 PPG requires sealed collection provenance')
    began=time.perf_counter();c=phases.options['value_phase'];rate=getattr(phases,'ppo_peak',None)
    cache=phases.buffer.stage(phases.critic_dataset,phases.train_ids,phases.heldout_ids,phases.actor_ids,
        phases.collection_source,model,phases.split['selection'])
    staged=time.perf_counter()
    clock=None
    if two_iteration(c):
        generation=phases.state['generation']+1
        clock=phase_clock(phases.phase_origin,phases.state['generation'],c)
        ran=generation==clock['next_phase']
        clock=phase_clock(phases.phase_origin,generation,c)
        if not ran:
            # Cache every collection, but avoid opening the old cache, creating
            # policy references, shuffling rows or changing either Adam off-cycle.
            effective=dict(c,learning_rate=0.)
            report=value_phase.fit(model,phases.optimizer,phases.dataset,phases.train_ids,
                phases.heldout_ids,[],phases.probe_ids,effective,phases.config['update'],beta=phases.beta,
                value_dataset=phases.critic_dataset,validation_dataset=phases.critic_dataset)
            report.update(scheduled=False,early_stop_reason='not_due',buffer=cache,split=phases.split,
                phase_interval=c['phase_interval'],buffer_iterations=c['buffer_iterations'],schedule=clock,
                validation_iterations=c['validation_iterations'],gate_critic_training=dict(source='ppg',optimizer_steps=0),
                learning_rate_source=copy.deepcopy(rate),peak_learning_rate=rate['peak_lr'] if rate else None,
                beta_clone_final=phases.beta,beta_mode='adaptive-up-down',
                epoch_controller=dict(cap_used=c['max_epochs'],cap_next=c['max_epochs'],decision='fixed'),
                schedule_migration=copy.deepcopy(phases.ppg_schedule_migration),
                scope='Cache current critic rows; current whole-game holdout evidence only. PPG not due; no fitting, cloning, beta feedback or optimizer/RNG movement.')
            phases.value_report=report;return report
    # An empty PPO update supplies no actual LR; do not invent one or change beta.
    effective=copy.deepcopy(c);effective['learning_rate']=rate['peak_lr'] if rate is not None else 0.
    if rate is not None:
        require(rate['generation']==phases.state['generation']+1 and rate['contract']==RATE and
            rate['factor']==c['learning_rate_factor'] and rate['peak_lr']==rate['ppo_lr']*c['learning_rate_factor'],
            'Stale or incompatible g210 PPO rate')
        phases.optimizer.param_groups[0].update(lr=rate['peak_lr'],**{RATE_KEY:copy.deepcopy(rate)})
    prepare=phase_preparation(preparation_pool,phases.config['update'].get('preparation_workers',0),
        directory=phases.root/'artifacts',enabled=True)
    with phases.buffer.datasets(phases.dataset.device,policy=False) as (ds,_,receipts), prepare as preparation:
        indices=[];validation=[];validation_support=[];offset=0
        for r,part in zip(receipts,ds.data.parts):
            held=set(r['holdout_selection']['heldout_games'])
            membership=np.array([g not in held for g in r['game_ids']],dtype=bool)
            indices.extend((np.flatnonzero(membership[part.array['game']])+offset).tolist())
            selected=np.flatnonzero(~membership[part.array['game']])
            retired=getattr(phases,'retired_tables',())
            if retired:
                eligible=np.array([r['holdout_selection']['games'][g] not in retired for g in r['game_ids']])
                selected=selected[eligible[part.array['game'][selected]]]
            validation.extend((selected+offset).tolist())
            validation_support.append(dict(generation=r['generation'],rows=len(selected),
                games=len(set(part.array['game'][selected].tolist()))))
            offset+=r['rows']
        snapshot_weight=phases.critic_state['snapshot_weight'] if phases.critic_state else 1.
        ds.critic_weights=torch.tensor([snapshot_weight if r=='snapshot' else 1. for r in ds.roles],device=ds.device)
        companion=getattr(phases,'local_scoring',None)
        if getattr(phases,'diversity',None) is not None:
            from .domino_auxiliary import PPGCompanions
            companion=PPGCompanions(companion,phases.diversity,phases.optimizer,phases.config['update']['value_weight'])
        if companion is not None and companion.active:companion.inventory(ds,indices)
        pooled=scheduled(c)
        indexed=time.perf_counter()
        report=value_phase.fit(model,phases.optimizer,phases.dataset,phases.train_ids,validation if pooled else phases.heldout_ids,
            phases.actor_ids if rate is not None else [],phases.probe_ids,effective,phases.config['update'],beta=phases.beta,
            value_dataset=ds,value_train_ids=indices,validation_dataset=ds if pooled else phases.critic_dataset,
            **(dict(gate_dataset=phases.critic_dataset,gate_ids=phases.heldout_ids) if pooled else {}),
            current_count=receipts[0]['rows'],dropout=(phases.state['generation'],phases.config['seat_descriptors']['dropout_games']) if phases.critic_state else None,
            companion=companion,preparation=preparation,post_fit=getattr(phases,'post_value_fit',None))
        if getattr(phases,'diversity',None) is not None:
            phases.diversity.report['intrinsic_fit']=companion.report()
        report['cpu_preparation']=preparation.report() if preparation is not None else dict(workers=0)
        report['value_work']=dict(cache_stage_seconds=staged-began,replay_open_and_index_seconds=indexed-staged,
            fit_with_auxiliary_seconds=time.perf_counter()-indexed,
            scope='Wall attribution; fit includes its existing synchronous metrics and post-PPG auxiliary callback')
        if pooled:report['acceptance_support']=validation_support
        if getattr(phases,'retired_tables',()):report['retired_table_checks']=list(phases.retired_tables)
    if rate is None:report['early_stop_reason']='no_applied_ppo_rate'
    report.update(buffer=cache,split=phases.split,migration=copy.deepcopy(phases.g210_migration),
        phase_interval=c['phase_interval'],buffer_iterations=2,validation_iterations=c['validation_iterations'],scheduled=True,
        epoch_cap_used=c['max_epochs'],kl_check_steps=c['kl_check_steps'],policy_guard_enabled=True,kl_control='hard-cap-and-penalty',
        learning_rate_source=copy.deepcopy(rate),peak_learning_rate=effective['learning_rate'],beta_clone_final=phases.beta,
        beta_mode='adaptive-up-down',
        acceptance_generations=sorted(r['generation'] for r in receipts) if scheduled(c) else [phases.state['generation']+1],beta_retries=0,
        gate_critic_training=dict(source='ppg',optimizer_steps=report['optimizer_steps']),
        rate_schedule=(dict(copy.deepcopy(c['rate_schedule']),reset_adam=False,warmup=True,decay=True,factor=c['learning_rate_factor'])
            if scheduled(c) else dict(reset_adam=False,warmup=False,decay=False,factor=c['learning_rate_factor'])))
    if clock is not None:
        report.update(schedule=clock,
            epoch_controller=dict(cap_used=c['max_epochs'],cap_next=c['max_epochs'],decision='fixed'))
    if linear(c):report['schedule_migration']=copy.deepcopy(phases.ppg_schedule_migration)
    if getattr(phases,'diversity',None) is not None:
        phases.diversity.diagnostics(model,phases.critic_dataset,phases.heldout_ids,phases.config['update'])
        report['diversity']=copy.deepcopy(phases.diversity.report)
    report['value_work']['total_seconds']=time.perf_counter()-began
    phases.value_report=report;return report


def payload(phases):
    require(phases.buffer.pending is not None,'g210 checkpoint needs a staged cache')
    return dict(contract=phases.contract,options=copy.deepcopy(phases.options),gae=copy.deepcopy(phases.state),
        **({'phase_origin':phases.phase_origin} if two_iteration(phases.options['value_phase']) else {}),
        **({'critic_data':copy.deepcopy(phases.critic_state)} if phases.critic_state is not None else {}),
        beta_clone=phases.beta,kl_trips=copy.deepcopy(phases.trips),optimizer=phases.optimizer.state_dict(),
        rolling_buffer=copy.deepcopy(phases.buffer.pending),cache_origin=phases.cache_origin,adoption=copy.deepcopy(phases.adoption))
