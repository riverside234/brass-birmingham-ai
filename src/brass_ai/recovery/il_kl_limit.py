"""Pure, committed health and adaptive fixed-IL KL limit transitions."""
import copy
import math
from brass_ai.learning.contracts import require

LEGACY_CONTRACT='auto-il-kl-limit/1.0.0'
CONTRACT='auto-il-kl-limit/1.2.0'
GROUPS=('canal_action','canal_other','rail_action','rail_other')
DEFAULT=dict(contract=CONTRACT,probe_positions=2048,probe_seed=710193,probability_floor=1e-6,
    initial_limit=.15,minimum_limit=.05,maximum_limit=1.,group_factor=2.,
    growth_factor=1.25,recovery_factor=.8,alarm_factor=2.,update_interval=10,
    baseline_iterations=10,rolling_iterations=5,group_min_positions=64,
    vp_ratio=.85,pass_factor=2.,pass_allowance=.5,
    zero_vp_floor=.05,zero_vp_factor=2.,insolvency_allowance=.10)
COUNTS=('seat_games','vp_sum','passes_with_alternative','zero_vp_seats','insolvent_seats','insolvency_events')
METRICS=('vp_per_seat','passes_per_seat','zero_vp_fraction','insolvency_fraction')


def finite(x):return type(x) in (int,float) and math.isfinite(x)


def validate(config):
    require(type(config) is dict and config.get('contract') in (LEGACY_CONTRACT,'auto-il-kl-limit/1.1.0',CONTRACT),
            'Invalid auto IL-KL limit contract/fields')
    legacy=config['contract']==LEGACY_CONTRACT
    require(set(config)==set(DEFAULT)|({'sells_ratio'} if legacy else set()),
            'Invalid auto IL-KL limit contract/fields')
    # Read old signed configs/probe manifests without rewriting their identity.
    # This field is compatibility metadata only; no decision reads it.
    if legacy:
        require(finite(config['sells_ratio']) and 0<config['sells_ratio']<=1,
                'Invalid retired auto IL-KL sells_ratio metadata')
    for key,default in DEFAULT.items():
        if key=='contract':continue
        if type(default) is int:
            require(type(config[key]) is int and config[key]>0,'Invalid auto IL-KL '+key)
        else:require(finite(config[key]) and config[key]>0,'Invalid auto IL-KL '+key)
    require(config['probe_positions']==2048 and config['probability_floor']==1e-6 and
            config['group_factor']==2. and config['update_interval']==10 and
            config['baseline_iterations']==10 and config['rolling_iterations']==5,
            'Auto IL-KL requires the fixed probe, doubled group limits and 10/5 iteration cadence')
    require(0<config['minimum_limit']<=config['initial_limit']<=config['maximum_limit']<=1 and
            1<config['growth_factor']<=1.25 and 0<config['recovery_factor']<1 and
            config['alarm_factor']>=1 and 4*config['group_min_positions']<=config['probe_positions'],
            'Invalid auto IL-KL bounds')
    require(config['vp_ratio']<=1 and config['zero_vp_floor']<1 and
            config['insolvency_allowance']<1,'Invalid auto IL-KL health thresholds')
    return config


def current_config(config):
    """Fresh generation only; existing run/probe signatures stay byte-bound."""
    result=copy.deepcopy(validate(config))
    result.pop('sells_ratio',None);result['contract']=CONTRACT
    return result


def baseline_evidence(baseline):
    """Ignore only the retired sales metric in an otherwise exact old baseline."""
    if baseline is None:return None
    result=copy.deepcopy(baseline)
    for key in ('metrics','thresholds'):
        if isinstance(result.get(key),dict):result[key].pop('sells_per_seat',None)
    return result


def initial_state(config,probe):
    validate(config)
    return dict(contract=config['contract'],config=copy.deepcopy(config),limit=config['initial_limit'],
        kl_last_healthy=0.,last_healthy_generation=0,last_update=0,last_generation=0,
        baseline=None,baseline_rows=[],rolling=[],healthy_window=[],blocked=False,probe=copy.deepcopy(probe))


def validate_state(state,config,generation=None):
    require(type(state) is dict and set(state)==set(initial_state(config,{})) and
            state['contract']==config['contract'] and state['config']==config,'Auto IL-KL checkpoint config differs')
    require(finite(state['limit']) and config['minimum_limit']<=state['limit']<=config['maximum_limit'],
            'Invalid persisted auto IL-KL limit')
    for key in ('last_update','last_healthy_generation','last_generation'):
        require(type(state[key]) is int and 0<=state[key]<=state['last_generation'],'Invalid auto IL-KL progress')
    require(finite(state['kl_last_healthy']) and state['kl_last_healthy']>=0 and type(state['blocked']) is bool,
            'Invalid auto IL-KL healthy checkpoint')
    require(generation is None or generation==state['last_generation'],'Auto IL-KL generation differs')
    for key,limit in (('baseline_rows',10),('rolling',5),('healthy_window',10)):
        require(type(state[key]) is list and len(state[key])<=limit,'Invalid auto IL-KL '+key)
    require(all(finite(x) and x>=0 for x in state['healthy_window']),'Invalid auto IL-KL distance window')
    for key in ('baseline_rows','rolling'):
        require(all(row is None or rates([row]) is not None for row in state[key]),'Invalid auto IL-KL health counts')
    expected=baseline_record(state['baseline_rows'],config) if len(state['baseline_rows'])==10 else None
    require(baseline_evidence(state['baseline'])==expected,
            'Auto IL-KL health baseline differs from committed evidence')
    return state


def rates(rows):
    if not rows or any(type(r) is not dict or any(not finite(r.get(k)) or r[k]<0 for k in COUNTS) or
                       r['seat_games']<=0 or r['zero_vp_seats']>r['seat_games'] or
                       r['insolvent_seats']>r['seat_games'] for r in rows):return None
    totals={k:sum(r[k] for r in rows) for k in COUNTS};seats=totals['seat_games']
    return dict(zip(METRICS,(totals[k]/seats for k in
        ('vp_sum','passes_with_alternative','zero_vp_seats','insolvent_seats'))))


def health(baseline,current,behavior=None):
    # Recompute shared health from the active metrics, never an archived flag.
    # The optional legacy behavior argument is ignored. Neither IL sales nor
    # IL pass probabilities may affect any consumer of shared economic health.
    reasons=[];checks={}
    if baseline is None:reasons.append('baseline_unavailable')
    if current is None:reasons.append('rolling_health_unavailable')
    if baseline is not None and current is not None:
        # Thresholds are supplied by the caller in the baseline record.
        limits=baseline['thresholds']
        for key in METRICS:
            okay=current[key]>=limits[key] if key=='vp_per_seat' else current[key]<=limits[key]
            checks[key]=dict(value=current[key],threshold=limits[key],ok=okay)
            if not okay:reasons.append(key)
    return dict(contract='selfplay-economic-health/3.0.0',ok=not reasons,reasons=reasons,
                metrics={key:current[key] for key in METRICS} if current is not None else None,
                checks=checks,
                scope='All-learner self-play: seat-weighted five-iteration VP, ordinary passes, zero-VP and insolvency rates. No IL action-probability checks; sales are reporting only.')


def baseline_record(rows,config):
    measured=rates(rows)
    if measured is None:return None
    b=measured;c=config
    return dict(metrics=b,thresholds=dict(vp_per_seat=c['vp_ratio']*b['vp_per_seat'],
        passes_per_seat=c['pass_factor']*b['passes_per_seat']+c['pass_allowance'],
        zero_vp_fraction=max(c['zero_vp_floor'],c['zero_vp_factor']*b['zero_vp_fraction']),
        insolvency_fraction=b['insolvency_fraction']+c['insolvency_allowance']))


def measurement_supported(measurement,c):
    distance=measurement.get('kl_mean');groups=measurement.get('groups',{})
    return (measurement.get('positions')==c['probe_positions'] and finite(distance) and distance>=0 and all(
        type(groups.get(k)) is dict and type(groups[k].get('positions')) is int and
        groups[k]['positions']>=c['group_min_positions'] and finite(groups[k].get('kl_mean')) and
        groups[k]['kl_mean']>=0 for k in GROUPS) and sum(groups[k]['positions'] for k in GROUPS)==c['probe_positions']
        and measurement.get('unassigned_positions')==0)


def reset_reference(state,measurement,*,generation):
    """Reset reference-dependent calibration only, at committed progress."""
    out=copy.deepcopy(validate_state(state,state['config'],generation));c=out['config']
    require(c['initial_limit']==.15 and c['group_factor']==2.,'Qualified anchor starts at .15/.30 nats')
    supported=measurement_supported(measurement,c)
    current=rates(out['rolling']) if len(out['rolling'])==c['rolling_iterations'] else None
    evidence=health(out['baseline'],current)
    distance=measurement.get('kl_mean');groups=measurement.get('groups',{})
    over=not supported or distance>.15 or any(groups[k]['kl_mean']>.30 for k in GROUPS)
    out.update(limit=.15,last_update=generation,healthy_window=[],blocked=bool(over),
        kl_last_healthy=distance if supported and evidence['ok'] else 0.,
        last_healthy_generation=generation if supported and evidence['ok'] else 0)
    for k in ('baseline','baseline_rows','rolling','probe','last_generation','config'):
        require(out[k]==state[k],'Anchor promotion changed independent health/progress')
    validate_state(out,c,generation)
    return out,dict(generation=generation,measurement=copy.deepcopy(measurement),health=evidence,
        limit_before=state['limit'],limit_after=.15,group_limit_before=2*state['limit'],group_limit_after=.30,
        blocked=out['blocked'],lr_anchor_health_permits=bool(supported and evidence['ok'] and not over),
        reason='new_reference_recalibration',scope='Reference KL only; economic baseline/rolling evidence and all optimizer state preserved')


def advance(state,measurement,counts,*,generation,warmup_done):
    state=copy.deepcopy(validate_state(state,state['config']));c=state['config'];before=state['limit']
    # Preserve all active thresholds, counters, windows, limits and latch. Do
    # not rebaseline or retroactively award healthy iterations on continuation.
    state['baseline']=baseline_evidence(state['baseline'])
    decision_counts=lambda row:{key:row[key] for key in COUNTS} if rates([row]) is not None else None
    for key in ('baseline_rows','rolling'):
        state[key]=[decision_counts(row) for row in state[key]]
    measurement=copy.deepcopy(measurement)
    measurement.pop('behavior',None)
    require(generation==state['last_generation']+1,'Auto IL-KL must advance once per committed iteration')
    state['last_generation']=generation
    counts=decision_counts(counts)
    state['rolling']=(state['rolling']+[copy.deepcopy(counts)])[-c['rolling_iterations']:]
    just_ready=False
    if warmup_done and len(state['baseline_rows'])<c['baseline_iterations']:
        state['baseline_rows'].append(copy.deepcopy(counts))
        if len(state['baseline_rows'])==c['baseline_iterations']:
            state['baseline']=baseline_record(state['baseline_rows'],c)
            state['last_update']=generation;just_ready=True
    current=rates(state['rolling']) if len(state['rolling'])==c['rolling_iterations'] else None
    evidence=health(state['baseline'],current)
    distance=measurement.get('kl_mean');groups=measurement.get('groups',{})
    supported=measurement_supported(measurement,c)
    if not supported:
        evidence['ok']=False;evidence['reasons'].append('probe_evidence_unavailable')
    action='hold';reason='warmup' if not warmup_done else 'collecting_baseline'
    if state['baseline'] is not None:
        if not evidence['ok']:
            state['limit']=min(before,max(c['minimum_limit'],state['kl_last_healthy']))
            state['last_update']=generation;state['healthy_window']=[]
            action='tighten';reason='health_failed'
        else:
            state['kl_last_healthy']=distance;state['last_healthy_generation']=generation
            if not just_ready:
                state['healthy_window']=(state['healthy_window']+[distance])[-c['update_interval']:]
            reason='healthy_wait'
            if generation-state['last_update']>=c['update_interval'] and len(state['healthy_window'])==c['update_interval']:
                maximum=max(state['healthy_window'])
                state['limit']=min(c['maximum_limit'],c['growth_factor']*before,max(before,c['growth_factor']*maximum))
                state['last_update']=generation
                action='raise' if state['limit']>before else 'hold';reason='healthy_window'
    limit=state['limit'];group_limit=c['group_factor']*limit
    over=supported and (distance>limit or any(groups[k]['kl_mean']>group_limit for k in GROUPS))
    if over:state['blocked']=True
    elif state['blocked'] and supported and evidence['ok'] and distance<c['recovery_factor']*limit and all(
            groups[k]['kl_mean']<c['recovery_factor']*group_limit for k in GROUPS):state['blocked']=False
    gate_reasons=list(evidence['reasons'])
    if over:gate_reasons.append('anchor_budget_exceeded')
    if state['blocked']:gate_reasons.append('anchor_recovery_latch')
    group_evidence={k:dict(positions=groups.get(k,{}).get('positions'),anchor_distance=groups.get(k,{}).get('kl_mean'),
        status=('insufficient_samples' if groups.get(k,{}).get('positions',0)<c['group_min_positions'] else
                'unavailable' if not finite(groups.get(k,{}).get('kl_mean')) else
                'budget_exceeded' if groups[k]['kl_mean']>group_limit else 'ok')) for k in GROUPS}
    gate=dict(ok=evidence['ok'] and supported and not over and not state['blocked'],reasons=gate_reasons,
        anchor_distance=distance if finite(distance) else None,anchor_budget=limit,anchor_group_budget=group_limit,
        anchor_group_min_positions=c['group_min_positions'],anchor_groups=group_evidence,
        anchor_unassigned_positions=measurement.get('unassigned_positions'),
        anchor_group_max=max(groups[k]['kl_mean'] for k in GROUPS) if supported else None,
        health=evidence,scope='Fixed raw IL validation probe and calibrated all-learner health')
    report=dict(contract=CONTRACT,generation=generation,measurement=measurement,health=evidence,
        limit_before=before,limit_after=limit,group_limit_before=c['group_factor']*before,group_limit_after=group_limit,
        decision=action,reason=reason,baseline=state['baseline'],baseline_iterations=len(state['baseline_rows']),
        last_update=state['last_update'],last_healthy_generation=state['last_healthy_generation'],
        kl_last_healthy=state['kl_last_healthy'],healthy_window_length=len(state['healthy_window']),
        blocked=state['blocked'],recovery_limit=c['recovery_factor']*limit,
        alarm=bool(supported and distance>c['alarm_factor']*limit),lr_gate=gate,
        probe=copy.deepcopy(state['probe']))
    return state,report
