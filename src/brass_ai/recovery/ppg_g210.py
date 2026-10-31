"""Explicit g210 auxiliary recipe, with reviewed beta/LR/epoch adjustments.

PPO's objective is deliberately outside this configuration migration. The
versioned KL guard and monotone loss acceptance live in value_phase.fit.
"""
import copy
import math

import numpy as np
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


def active(c):return c.get('contract') in (CONTRACT,COSINE_CONTRACT,LINEAR_CONTRACT,SHORT_CONTRACT,LOSS16_CONTRACT,KL32_CONTRACT)


def cosine(c):return c.get('contract')==COSINE_CONTRACT


def linear(c):return c.get('contract') in (LINEAR_CONTRACT,SHORT_CONTRACT,LOSS16_CONTRACT,KL32_CONTRACT)


def short(c):return c.get('contract')==SHORT_CONTRACT


def scheduled(c):return cosine(c) or linear(c)


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
    default=KL32_DEFAULT if c.get('contract')==KL32_CONTRACT else LOSS16_DEFAULT if c.get('contract')==LOSS16_CONTRACT else SHORT_DEFAULT if short(c) else LINEAR_DEFAULT if linear(c) else COSINE_DEFAULT if cosine(c) else DEFAULT
    require(type(c) is dict and set(c)==set(default) and active(c),'Invalid g210 PPG fields/contract')
    # Smaller batches are useful for bounded regression/smoke fixtures only.
    for k,v in default.items():
        if k in ('batch_size','clone_rows','clone_batch_size'):continue
        require(c[k]==v and type(c[k]) is type(v),'Invalid g210 PPG option: '+k)
    if scheduled(c):
        require(all(type(c['rate_schedule'][k]) is type(v) for k,v in default['rate_schedule'].items()),
            'Invalid PPG rate schedule field type')
    require(all(type(c[k]) is int and 1<=c[k]<=DEFAULT[k] for k in ('batch_size','clone_rows','clone_batch_size'))
        and c['clone_batch_size']<=c['clone_rows'],'Invalid g210 PPG batch bounds')
    return c


def options(config):
    from .value_phase import validate as validate_value
    validate_value(config['value_phase'])
    result=copy.deepcopy(config)
    if not active(result['value_phase']):result['value_phase']=copy.deepcopy(DEFAULT)
    validate(result['value_phase']);return result


def rate_receipt(rate,generation,steps):
    require(type(rate) in (int,float) and math.isfinite(rate) and rate>0
        and type(steps) is int and steps>0,'PPG needs an actual applied PPO LR')
    return dict(contract=RATE,source='ppo-last-applied',generation=generation,
        ppo_steps=steps,ppo_lr=rate,factor=.65,peak_lr=rate*.65)


def beta_after_phase(beta,trips,report,config):
    """g210's new-trip/3-of-5 escalation; clean accepted low-KL phases halve."""
    require(math.isfinite(beta) and config['beta_clone_min']<=beta<=32 and type(trips) is list
        and len(trips)<=5 and all(type(x) is bool for x in trips),'Invalid g210 beta state')
    trips=(trips+[bool(report['kl_guard_tripped'])])[-5:]
    after=beta;reason='fewer_than_three_trips' if trips[-1] else 'no_new_kl_trip'
    if trips[-1] and sum(trips)>=3:
        after=min(config['beta_clone_max'],beta*2)
        reason='new_trip_in_three_of_five' if after>beta else 'at_maximum'
    elif not trips[-1] and report['optimizer_steps']>0:
        kl=report.get('probe_kl')
        require(type(kl) in (int,float) and math.isfinite(kl) and 0<=kl<=config['kl_cap']+1e-7,
            'Invalid retained PPG KL')
        if kl<.5*config['kl_cap']:
            after=max(config['beta_clone_min'],beta/2)
            reason='retained_KL_below_half_cap' if after<beta else 'at_minimum'
    elif not trips[-1]:reason='no_accepted_steps'
    return after,trips,dict(before=beta,after=after,reason=reason,
        decision='up' if after>beta else 'down' if after<beta else 'hold',
        threshold=.5*config['kl_cap'],scope='Next phase only; a new hard-cap trip with 3/5 hits doubles beta. Clean accepted phases below half cap halve beta. No within-phase retries.')


def initialize(phases,model,config,*,raw,generation,root,run):
    from . import actor_value,gae,critic_data,ppg_config,ppg_epochs,ppg_replay
    validate(config['value_phase'])
    require(root is not None and run is not None,'g210 PPG requires an owned disk cache')
    phases.config=config;phases.root=root;phases.run=run;phases.rolling=False
    phases.options={k:copy.deepcopy(config[k]) for k in ('advantage','value_phase','critic_data','seat_descriptors') if k in config}
    phases.contract=CHECKPOINT;phases.snapshot_data=[];phases.fingerprint=None
    phases.dataset=None;phases.probe_ids=None;phases.collection_source=None;phases.ppo_work=None
    for name in ('critic_evidence','disable_migration','holdout_migration','refinement_migration','window_migration',
                 'evidence_migration','critic_migration','snapshot_r2_migration','disabled_startup_cleanup','g210_migration',
                 'ppg_schedule_migration'):
        setattr(phases,name,None)
    phases.optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate']*.65)
    saved=(raw or {}).get('training',{}).get(actor_value.KEY)
    if saved is None:
        require(generation==0,'Missing g210 continuation checkpoint')
        phases.state=gae.initial(config['advantage'])
        phases.critic_state=critic_data.initial(config['critic_data']) if 'critic_data' in config else None
        phases.beta=4.;phases.trips=[];phases.cache_origin=0;phases.adoption=None;receipts=[]
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
            require(saved.get('contract')==CHECKPOINT and set(saved)==fields and options_match,
                'g210 checkpoint configuration differs')
            phases.beta=saved['beta_clone'];phases.trips=saved['kl_trips']
            phases.cache_origin=saved['cache_origin'];receipts=saved['rolling_buffer'];phases.adoption=saved['adoption']
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
    c=phases.options['value_phase'];rate=getattr(phases,'ppo_peak',None)
    cache=phases.buffer.stage(phases.critic_dataset,phases.train_ids,phases.heldout_ids,phases.actor_ids,
        phases.collection_source,model,phases.split['selection'])
    # An empty PPO update supplies no actual LR; do not invent one or change beta.
    effective=copy.deepcopy(c);effective['learning_rate']=rate['peak_lr'] if rate is not None else 0.
    if rate is not None:
        require(rate['generation']==phases.state['generation']+1 and rate['contract']==RATE,'Stale g210 PPO rate')
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
        report=value_phase.fit(model,phases.optimizer,phases.dataset,phases.train_ids,validation if pooled else phases.heldout_ids,
            phases.actor_ids if rate is not None else [],phases.probe_ids,effective,phases.config['update'],beta=phases.beta,
            value_dataset=ds,value_train_ids=indices,validation_dataset=ds if pooled else phases.critic_dataset,
            **(dict(gate_dataset=phases.critic_dataset,gate_ids=phases.heldout_ids) if pooled else {}),
            current_count=receipts[0]['rows'],dropout=(phases.state['generation'],phases.config['seat_descriptors']['dropout_games']) if phases.critic_state else None,
            companion=companion,preparation=preparation)
        if getattr(phases,'diversity',None) is not None:
            phases.diversity.report['intrinsic_fit']=companion.report()
            from .domino_search_fit import retain
            retain(phases.diversity,model,ds,companion.retained_rows,companion.retained_steps)
        report['cpu_preparation']=preparation.report() if preparation is not None else dict(workers=0)
        if pooled:report['acceptance_support']=validation_support
    if rate is None:report['early_stop_reason']='no_applied_ppo_rate'
    report.update(buffer=cache,split=phases.split,migration=copy.deepcopy(phases.g210_migration),
        phase_interval=1,buffer_iterations=2,validation_iterations=c['validation_iterations'],scheduled=True,
        epoch_cap_used=c['max_epochs'],kl_check_steps=c['kl_check_steps'],policy_guard_enabled=True,kl_control='hard-cap-and-penalty',
        learning_rate_source=copy.deepcopy(rate),peak_learning_rate=effective['learning_rate'],beta_clone_final=phases.beta,
        beta_mode='adaptive-up-down',
        acceptance_generations=sorted(r['generation'] for r in receipts) if scheduled(c) else [phases.state['generation']+1],beta_retries=0,
        gate_critic_training=dict(source='ppg',optimizer_steps=report['optimizer_steps']),
        rate_schedule=(dict(copy.deepcopy(c['rate_schedule']),reset_adam=False,warmup=True,decay=True,factor=.65)
            if scheduled(c) else dict(reset_adam=False,warmup=False,decay=False,factor=.65)))
    if linear(c):report['schedule_migration']=copy.deepcopy(phases.ppg_schedule_migration)
    if getattr(phases,'diversity',None) is not None:
        phases.diversity.diagnostics(model,phases.critic_dataset,phases.heldout_ids,phases.config['update'])
        report['diversity']=copy.deepcopy(phases.diversity.report)
    phases.value_report=report;return report


def payload(phases):
    require(phases.buffer.pending is not None,'g210 checkpoint needs a staged cache')
    return dict(contract=CHECKPOINT,options=copy.deepcopy(phases.options),gae=copy.deepcopy(phases.state),
        **({'critic_data':copy.deepcopy(phases.critic_state)} if phases.critic_state is not None else {}),
        beta_clone=phases.beta,kl_trips=copy.deepcopy(phases.trips),optimizer=phases.optimizer.state_dict(),
        rolling_buffer=copy.deepcopy(phases.buffer.pending),cache_origin=phases.cache_origin,adoption=copy.deepcopy(phases.adoption))
