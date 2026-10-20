"""Deterministic, checkpointed PPO-only learning-rate feedback.

The post-PPO categorical measurement shares its fixed old policy/probe with
the per-step early-stop guard. No RNG calls, model updates, or teacher targets
live here.
"""
import copy
import math
from statistics import median

from brass_ai.learning.contracts import digest,require

KEY='brass_ppo_lr_controller'
LEGACY_CONTRACT='ppo-categorical-kl-lr/1.0.0'
CONTRACT='ppo-categorical-kl-lr/1.1.0'
AUTO_CONTRACT='ppo-categorical-kl-lr/1.2.0'
GROUP_FIELDS=frozenset(('anchor_group_kl_budget','anchor_group_min_positions'))
DEFAULT_CONTROLLER=dict(contract=CONTRACT,mode='apply',target_kl=.01,
    raise_factor=1.25,cut_divisor=1.5,low_ratio=.45,high_ratio=1.5,
    window_iterations=3,cooldown_iterations=3,min_multiplier=.5,max_multiplier=3.,
    anchor_kl_budget=.15,anchor_group_kl_budget=.30,anchor_group_min_positions=64,
    vp_max_drop=.10,zero_vp_max_fraction=.05)
AUTO_CONTROLLER=dict(DEFAULT_CONTROLLER,contract=AUTO_CONTRACT)


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def validate_config(config,update=None):
    require(type(config) is dict and config.get('contract') in (LEGACY_CONTRACT,CONTRACT,AUTO_CONTRACT),
            'Unsupported PPO LR controller contract')
    expected=set(DEFAULT_CONTROLLER)-(GROUP_FIELDS if config['contract']==LEGACY_CONTRACT else set())
    require(set(config)==expected,
            'Invalid PPO LR controller contract/fields')
    require(config['mode'] in ('off','shadow','apply'),'Invalid PPO LR controller mode')
    for key in ('raise_factor','cut_divisor','low_ratio','high_ratio','min_multiplier','max_multiplier',
                'vp_max_drop','zero_vp_max_fraction','target_kl'):
        require(finite(config[key]) and config[key]>0,'Invalid PPO LR controller '+key)
    require(config['raise_factor']>1 and config['cut_divisor']>1 and 0<config['low_ratio']<1<config['high_ratio']
            and 0<config['min_multiplier']<=1<=config['max_multiplier']
            and config['vp_max_drop']<1 and config['zero_vp_max_fraction']<1 and config['target_kl']<=.1,
            'Invalid PPO LR controller bounds')
    for key in ('window_iterations','cooldown_iterations'):
        require(type(config[key]) is int and config[key]==3,'PPO LR controller requires three-iteration windows/cooldown')
    require(config['anchor_kl_budget'] is None or (finite(config['anchor_kl_budget']) and config['anchor_kl_budget']>0),
            'Invalid anchor KL budget; null blocks LR increases')
    if config['contract'] in (CONTRACT,AUTO_CONTRACT):
        require(finite(config['anchor_group_kl_budget']) and config['anchor_group_kl_budget']>0,
                'Invalid per-phase anchor KL budget')
        require(type(config['anchor_group_min_positions']) is int and 1<=config['anchor_group_min_positions']<=512,
                'Invalid anchor group minimum; four groups must fit the 2048-row probe')
    if update is not None:
        require(config['target_kl']==update['target_kl'],'Controller and PPO target_kl must match')
        require(config['mode']=='off' or (update.get('actor_nonforced') and update.get('kl_probe_rows')==2048),
                'Active PPO LR controller requires the 2048-row nonforced categorical probe')
    return config


def initial_state(config):
    validate_config(config)
    return dict(config=copy.deepcopy(config),multiplier=1.,window=[],cooldown=0,
                health_window=[],data_signature=None)


def validate_state(state):
    require(type(state) is dict and set(state)==set(initial_state(DEFAULT_CONTROLLER)),
            'Invalid PPO LR controller checkpoint state')
    config=validate_config(state['config'])
    require(finite(state['multiplier']) and config['min_multiplier']<=state['multiplier']<=config['max_multiplier'],
            'Invalid persisted PPO LR multiplier')
    require(type(state['cooldown']) is int and 0<=state['cooldown']<=3,'Invalid persisted PPO LR cooldown')
    require(type(state['window']) is list and len(state['window'])<=3 and all(
        type(row) is dict and set(row)=={'kl','early_stop'} and finite(row['kl']) and
        type(row['early_stop']) is bool for row in state['window']),'Invalid PPO LR measurement window')
    require(type(state['health_window']) is list and len(state['health_window'])<=5 and all(
        row is None or (finite(row) and row>=0) for row in state['health_window']),'Invalid PPO LR health window')
    require(state['data_signature'] is None or (type(state['data_signature']) is str and len(state['data_signature'])==64),
            'Invalid PPO LR data signature')
    return state


def configure(optimizer,config):
    """Called only after the run signature and Adam state have been admitted."""
    validate_config(config)
    require(len(optimizer.param_groups)==1,'PPO LR controller requires one optimizer group')
    group=optimizer.param_groups[0]
    from .learning_rate import KEY as WARMUP_KEY
    require(group.get(WARMUP_KEY,{}).get('kind')=='selfplay','PPO LR controller requires self-play warmup state')
    if KEY not in group:
        require(not optimizer.state,'Missing PPO LR controller state in a trained optimizer')
        group[KEY]=initial_state(config)
    state=validate_state(group[KEY])
    # A config edit still needs the existing explicit signature/source review.
    # Changing mode must not reset a streak or multiplier on resume.
    require(config['min_multiplier']<=state['multiplier']<=config['max_multiplier'],
            'Persisted PPO LR multiplier outside requested bounds')
    state['config']=copy.deepcopy(config)
    return copy.deepcopy(state)


def multiplier(group):
    state=group.get(KEY)
    if state is None or state['config']['mode']!='apply':return 1.
    from .learning_rate import KEY as WARMUP_KEY
    warmup=group[WARMUP_KEY]
    return state['multiplier'] if warmup['completed_steps']>=warmup['steps'] else 1.


def anchor_group_health(measurements,config):
    """Insufficient evidence blocks this raise only; no latch or LR cut."""
    from .teacher_schedule import PHASES
    groups=measurements.get('anchor_groups');evidence={};reasons=[];supported=[]
    minimum=config['anchor_group_min_positions'];budget=config['anchor_group_kl_budget']
    for name,_,_,_ in PHASES:
        row=groups.get(name) if type(groups) is dict else None
        count=row.get('positions') if type(row) is dict else None
        distance=row.get('anchor_distance') if type(row) is dict else None
        count=count if type(count) is int and count>=0 else None
        distance=distance if finite(distance) else None
        if count is None:status='unavailable'
        elif count<minimum:status='insufficient_samples'
        elif distance is None:status='unavailable'
        else:
            supported.append(distance)
            status='budget_exceeded' if distance>budget else 'ok'
        if status!='ok':reasons.append('anchor_group_'+status+':'+name)
        evidence[name]=dict(positions=count,anchor_distance=distance,status=status)
    unassigned=measurements.get('anchor_unassigned_positions')
    unassigned=unassigned if type(unassigned) is int and unassigned>=0 else None
    if unassigned is None:reasons.append('anchor_group_coverage_unavailable')
    elif unassigned:reasons.append('anchor_group_unassigned_positions')
    return reasons,dict(anchor_groups=evidence,anchor_group_budget=budget,anchor_group_min_positions=minimum,
        anchor_group_max=max(supported) if supported else None,anchor_unassigned_positions=unassigned)


def health(state,measurements,config):
    if config['contract']==AUTO_CONTRACT:
        evidence=measurements.get('calibrated_anchor')
        require(type(evidence) is dict and type(evidence.get('ok')) is bool and
                'anchor_distance' in evidence,'Missing calibrated IL-KL gate evidence')
        return copy.deepcopy(evidence)
    distance=measurements.get('anchor_distance');budget=config['anchor_kl_budget']
    current=measurements.get('selfplay_vp_per_seat');zeros=measurements.get('zero_vp_fraction')
    distance=distance if finite(distance) else None
    current=current if finite(current) else None
    zeros=zeros if finite(zeros) else None
    old=state['health_window'][0] if len(state['health_window'])==5 else None
    reasons=[]
    if budget is None:reasons.append('anchor_budget_unset')
    elif not finite(distance):reasons.append('anchor_distance_unavailable')
    elif distance>budget:reasons.append('anchor_budget_exceeded')
    groups={}
    if config['contract']==CONTRACT:
        group_reasons,groups=anchor_group_health(measurements,config)
        reasons.extend(group_reasons)
    if not finite(current) or not finite(old):reasons.append('five_iteration_vp_history_unavailable')
    elif current<(1-config['vp_max_drop'])*old:reasons.append('selfplay_vp_declined')
    if not finite(zeros):reasons.append('zero_vp_fraction_unavailable')
    elif zeros>=config['zero_vp_max_fraction']:reasons.append('too_many_zero_vp_seats')
    return dict(ok=not reasons,reasons=reasons,anchor_distance=distance,anchor_budget=budget,**groups,
        selfplay_vp_per_seat=current,selfplay_vp_five_iterations_ago=old,zero_vp_fraction=zeros,
        scope='VP trend: self-play seats, current vs five iterations earlier; zero VP: all ordinary learner seats')


def decide(state,measurements):
    """Pure transition: return a new state and JSON-safe decision evidence."""
    state=copy.deepcopy(validate_state(state));cfg=state['config'];before=state['multiplier']
    kl=measurements.get('kl');done=measurements['steps_done'];planned=measurements['steps_planned']
    require(type(done) is int and type(planned) is int and 0<=done<=planned,'Invalid PPO completed/planned steps')
    require(type(measurements['early_stop']) is bool and type(measurements['warmup_done']) is bool,
            'Invalid PPO controller measurement flags')
    evidence=health(state,measurements,cfg)
    decision,reason='hold','in_target_band';window_median=None;window_size=0;reset=False
    valid=done>0 and planned>0 and finite(kl)
    if cfg['mode']=='off':reason='off'
    elif not valid:reason='empty_or_nonfinite_update'
    else:
        signature=measurements['data_signature']
        require(type(signature) is str and len(signature)==64,'Missing PPO data signature')
        reset=state['data_signature'] is not None and state['data_signature']!=signature
        if reset:state['window']=[]
        state['data_signature']=signature
        vp=measurements.get('selfplay_vp_per_seat')
        state['health_window']=(state['health_window']+[vp if finite(vp) and vp>=0 else None])[-5:]
        if not measurements['warmup_done']:reason='warmup'
        else:
            state['window']=(state['window']+[dict(kl=kl,early_stop=measurements['early_stop'])])[-3:]
            window_size=len(state['window'])
            window_median=median(row['kl'] for row in state['window']) if window_size==3 else None
            stops=sum(row['early_stop'] for row in state['window'])
            cut_reasons=[]
            if measurements['early_stop'] and done<.5*planned:cut_reasons.append('early_stop_before_half')
            if stops>=2:cut_reasons.append('two_early_stops_in_three')
            if kl>cfg['high_ratio']*cfg['target_kl']:cut_reasons.append('kl_above_upper_band')
            if cut_reasons:
                state['multiplier']=max(cfg['min_multiplier'],before/cfg['cut_divisor'])
                state['cooldown']=cfg['cooldown_iterations'];state['window']=[]
                decision,reason='down',','.join(cut_reasons)
            elif state['cooldown']:
                state['cooldown']-=1;decision,reason='cooldown','after_cut'
            elif len(state['window'])<3:reason='collecting_three_iterations'
            elif stops:reason='early_stop_in_window'
            elif window_median<cfg['low_ratio']*cfg['target_kl']:
                if evidence['ok']:
                    state['multiplier']=min(cfg['max_multiplier'],before*cfg['raise_factor'])
                    state['window']=[];decision,reason='up','low_median_and_healthy'
                else:reason='health_check_failed'
    report=dict(contract=cfg['contract'],mode=cfg['mode'],kl=kl if finite(kl) else None,
        median_kl=window_median,window_size=window_size,early_stop=measurements['early_stop'],steps_done=done,steps_planned=planned,
        steps_fraction=done/planned if planned else None,warmup_done=measurements['warmup_done'],
        multiplier_before=before,multiplier_after=state['multiplier'],decision=decision,reason=reason,
        at_min=state['multiplier']==cfg['min_multiplier'],at_max=state['multiplier']==cfg['max_multiplier'],
        cooldown=state['cooldown'],window=copy.deepcopy(state['window']),window_reset=reset,health=evidence,
        anchor_distance=evidence['anchor_distance'],anchor_identity=measurements.get('anchor_identity'),
        scope='Exact post-PPO/pre-teaching KL; decision affects subsequent PPO steps only')
    return state,report


def collection_signature(config,sampling):
    """Reset for changed collection volume/mix, not random decision lengths."""
    return digest(dict(recipe=config.get('selfplay_recipe'),games=config['games_per_iteration'],
        tables={k:{f:v[f] for f in ('tables','learner_trajectories')} for k,v in sampling['by_composition'].items()},
        exploration=config['exploration'],update={k:config['update'][k] for k in
            ('epochs','effective_batch_size','critic_batch_size','kl_probe_rows')},
        **({'exploration_regime':sampling['exploration_regime']} if 'exploration_regime' in sampling else {})))


def game_health(games,policy_id,kinds):
    from brass_ai.learning.replay import terminal_targets
    selfplay=[];learner=[]
    for game in games:
        vp=terminal_targets(game['result'])['vp']
        seats=[s for s,p in enumerate(game['opponents']) if p==policy_id]
        learner.extend(vp[s] for s in seats)
        if kinds[game['setup_group']]=='selfplay':selfplay.extend(vp[s] for s in seats)
    return dict(selfplay_vp_per_seat=sum(selfplay)/len(selfplay) if selfplay else None,
                zero_vp_fraction=sum(v==0 for v in learner)/len(learner) if learner else None)


def finish_iteration(optimizer,update,config,sampling,games,policy_id,kinds,anchor,calibrated_anchor=None):
    group=optimizer.param_groups[0]
    if KEY not in group:return None
    probe=update['categorical_probe']
    measurements=dict(kl=probe['categorical_kl'] if probe['positions'] else None,
        steps_done=update['optimizer_steps'],steps_planned=update['optimizer_steps_planned'],
        early_stop=update['kl_early_stop'],warmup_done=update['controller_warmup_done'],
        data_signature=collection_signature(config,sampling),anchor_distance=probe.get('anchor_distance'),
        anchor_groups=probe.get('anchor_groups'),anchor_unassigned_positions=probe.get('anchor_unassigned_positions'),
        anchor_identity=anchor,calibrated_anchor=calibrated_anchor,**game_health(games,policy_id,kinds))
    group[KEY],report=decide(group[KEY],measurements)
    report['early_stop_reason']=update.get('kl_guard',{}).get('stop_reason')
    from .learning_rate import KEY as WARMUP_KEY,rate
    warmup=group[WARMUP_KEY]
    report.update(effective_lr=update['learning_rates'][0],
        next_ppo_lr=rate(warmup,warmup['completed_steps']+1)*multiplier(group),
        applied_multiplier=report['multiplier_before'] if config['ppo_lr_controller']['mode']=='apply' and
            measurements['warmup_done'] else 1.)
    return report
