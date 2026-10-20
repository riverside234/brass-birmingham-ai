"""Undiscounted, decision-chain GAE and committed, deterministic adoption gates."""
import copy
import math
from collections import defaultdict

import numpy as np
import torch

from brass_ai.learning.contracts import require
from . import ev_evidence,gae_health

CONTRACT='gae-advantage/2.0.0'
LEGACY_GATE=dict(contract='staged-gae-gate/1.0.0',mode='auto',minimum_value_iterations=10,
    heldout_ev_min=.5,ev_streak=5,fallback_value_iterations=30,healthy_iterations=20,
    rating_slots=3,rating_min_slope=0.)
GATE=dict(LEGACY_GATE,contract='staged-gae-gate/2.0.0',window_iterations=20,
    minimum_accepted_steps=16,minimum_value_iterations_in_window=3)
ARCHIVED_TABLE_GATE=dict(contract='table-gae-ev-gate/1.0.0',mode='auto',ev_tolerance=.05,consecutive_iterations=10)
TABLE_GATE=ev_evidence.DEFAULT
CONFIG_MIGRATION='table-gae-ev-window/1.0.0'
HEALTH_MIGRATION='staged-gae-health-window/1.0.0'
DEFAULT=dict(contract=CONTRACT,**{'lambda':1.0},shadow_lambdas=[.98,.95,.9],chain='nonforced',gate=GATE,table_gate=TABLE_GATE)


def finite(x):return type(x) in (int,float) and math.isfinite(x)


def validate(config,*,combined=True):
    legacy=config.get('contract')=='gae-advantage/1.0.0'
    require(type(config) is dict and set(config)==(set(DEFAULT)-{'table_gate'} if legacy else set(DEFAULT)) and
            config['contract'] in (CONTRACT,'gae-advantage/1.0.0')
            and config['chain']=='nonforced','Invalid GAE contract/fields/chain')
    if not legacy:
        t=config['table_gate']
        require(type(t) is dict,'Invalid table GAE gate')
        if t.get('contract')==ARCHIVED_TABLE_GATE['contract']:
            require(type(t) is dict and set(t)==set(ARCHIVED_TABLE_GATE)
                and t['mode'] in ('auto','selfplay_only') and t['ev_tolerance']==.05
                and type(t['consecutive_iterations']) is int and t['consecutive_iterations']==10,'Invalid table GAE gate')
        else:ev_evidence.validate_config(t)
    require(finite(config['lambda']) and 0<config['lambda']<=1,'Invalid GAE lambda')
    require(type(config['shadow_lambdas']) is list and config['shadow_lambdas']==[.98,.95,.9],
            'GAE shadows must be 0.98, 0.95 and 0.9')
    gate=config['gate']
    current=stage_windowed(config)
    expected=GATE if current else LEGACY_GATE
    require(type(gate) is dict and set(gate)==set(expected) and gate['contract']==expected['contract']
            and gate['mode'] in ('fixed','auto'),'Invalid GAE gate contract/fields/mode')
    for key in ('minimum_value_iterations','ev_streak','fallback_value_iterations','healthy_iterations','rating_slots'):
        require(type(gate[key]) is int and gate[key]>0,'Invalid GAE gate '+key)
    require(gate['minimum_value_iterations']<=gate['fallback_value_iterations'] and gate['rating_slots']>=3
            and finite(gate['heldout_ev_min']) and 0<=gate['heldout_ev_min']<=1
            and finite(gate['rating_min_slope']) and gate['rating_min_slope']>=0,'Invalid GAE gate bounds')
    if current:
        require(type(gate['window_iterations']) is int and gate['window_iterations']==gate['healthy_iterations']==20,
            'Windowed GAE requires a 20-iteration observation period')
        require(type(gate['minimum_accepted_steps']) is int and gate['minimum_accepted_steps']>0
            and type(gate['minimum_value_iterations_in_window']) is int
            and 1<=gate['minimum_value_iterations_in_window']<=20,'Invalid GAE accepted-work support')
    require(gate['mode']!='auto' or config['lambda']==1.,'Automatic GAE starts at lambda 1')
    require(combined or (config['lambda']==1. and gate['mode']=='fixed'),
            'GAE lambda below 1 / automatic gate requires combined reward')
    return config


def windowed(config):
    return config.get('table_gate',{}).get('contract')==ev_evidence.CONTRACT


def stage_windowed(config):
    return config.get('gate',{}).get('contract')==GATE['contract']


def health_options(options):
    result=copy.deepcopy(options);advantage=result.get('advantage',{})
    gate=advantage.get('gate',{})
    if gate.get('contract')==LEGACY_GATE['contract']:
        validate(advantage)
        require(gate['healthy_iterations']==20,'GAE health upgrade requires the existing 20-iteration period')
        gate.update({k:copy.deepcopy(v) for k,v in GATE.items() if k not in LEGACY_GATE or k=='contract'})
    return result


def migrate_health(state,old,new,generation):
    validate_state(state,old,generation)
    require(health_options(dict(advantage=old))['advantage']==new
        and not stage_windowed(old) and stage_windowed(new),'Unreviewed GAE health migration')
    state=copy.deepcopy(state);state['stage_evidence']=gae_health.initial(generation)
    return validate_state(state,new,generation)


def evidence_config(config):
    """Only the recognized prior table gate upgrades, under signed source review."""
    result=copy.deepcopy(config);advantage=result.get('advantage',{})
    old=advantage.get('table_gate',{})
    if old.get('contract')==ARCHIVED_TABLE_GATE['contract']:
        validate(advantage,combined='reward' in config)
        advantage['table_gate']=dict(copy.deepcopy(TABLE_GATE),mode=old['mode'])
    return result


def migrate_state(state,old_config,new_config,generation):
    validate_state(state,old_config,generation)
    require(evidence_config(dict(advantage=old_config,reward={}))['advantage']==new_config
        and not windowed(old_config) and windowed(new_config),'Unreviewed table EV state migration')
    state=copy.deepcopy(state)
    state['table_evidence']=ev_evidence.initial(generation)
    # Old per-iteration EV streaks have no pooled moments or game support.
    # Keep already enabled tables and origins; do not backdate new evidence.
    state['table_streaks']=dict(historical=0,legacy=0)
    return validate_state(state,new_config,generation)


def compute(data,actor_ids,mc_advantages,returns,lam,shadow_lambdas,lambda_by_table=None):
    """No RNG, model writes or critic-target changes; each seat is its own chain.

    Ingress retains game-record order even though seats interleave. The next
    nonforced return includes all intervening forced decisions' causal costs.
    Preserve the original float32 subtraction at lambda=1 for exact PPO parity.
    """
    ids=list(actor_ids);mc=mc_advantages.detach().double().cpu().numpy()
    target=returns.detach().double().cpu().numpy();values=target-mc
    chains=defaultdict(list)
    for i in ids:chains[data[i][2]].append(i)
    delta=np.zeros(len(data),dtype=np.float64)
    for chain in chains.values():
        next_g=next_v=0.
        for i in reversed(chain):
            delta[i]=target[i]-next_g+next_v-values[i]
            next_g,next_v=target[i],values[i]
    computed={1.:mc}
    for coefficient in dict.fromkeys([lam,*shadow_lambdas,*(lambda_by_table or {}).values()]):
        if coefficient==1.:continue
        out=np.zeros(len(data),dtype=np.float64)
        for chain in chains.values():
            following=0.
            for i in reversed(chain):
                following=delta[i]+coefficient*following;out[i]=following
        computed[coefficient]=out
    base=mc[ids];base_std=float(base.std()) if ids else None
    from .teacher_schedule import phase_for
    groups=defaultdict(list)
    for i in ids:
        context=data[i][0]['view']['context'];era=context['era']
        groups['era:'+era].append(i)
        groups['phase:'+era+'/'+context['phase']].append(i)
        name=phase_for(context) if type(context.get('round')) is int else None
        groups['period:'+(name or 'unassigned')].append(i)
    diagnostics={}
    for coefficient,out in computed.items():
        a=out[ids];std=float(a.std()) if ids else None
        correlation=(float(np.corrcoef(base,a)[0,1]) if ids and base_std>1e-12 and std>1e-12 else None)
        diagnostics[str(coefficient)]=dict(rows=len(ids),std=std,correlation_with_mc=correlation,
            same_sign_fraction=float((np.sign(a)==np.sign(base)).mean()) if ids else None,
            means={key:dict(rows=len(group),mean=float(out[group].mean())) for key,group in sorted(groups.items())})
    require(all(np.isfinite(out[ids]).all() for out in computed.values()),'Nonfinite GAE advantages')
    actual=mc_advantages if lam==1. else mc_advantages.clone()
    if lam!=1. and ids:actual[ids]=torch.as_tensor(computed[lam][ids],device=actual.device,dtype=actual.dtype)
    if lambda_by_table is not None:
        actual=mc_advantages.clone()
        for kind,coefficient in lambda_by_table.items():
            chosen=[i for i in ids if data[i][0].get('_table_kind','selfplay')==kind]
            if chosen:actual[chosen]=torch.as_tensor(computed[coefficient][chosen],device=actual.device,dtype=actual.dtype)
    actual_cpu=actual.detach().double().cpu().numpy();table_stats={}
    for kind in ('selfplay','historical','legacy'):
        chosen=[i for i in ids if data[i][0].get('_table_kind','selfplay')==kind]
        table_stats[kind]=dict(rows=len(chosen),actual_std=float(actual_cpu[chosen].std()) if chosen else None,
            mc_std=float(mc[chosen].std()) if chosen else None)
    return actual,dict(contract=CONTRACT,**{'lambda':lam},gamma=1.,chain='nonforced',
        lambda_by_table=lambda_by_table,by_table=table_stats,
        trajectories=len(chains),td_error_std=float(delta[ids].std()) if ids else None,shadows=diagnostics,
        scope='Raw pre-normalization nonforced learner decisions; critic targets remain Monte Carlo')


def initial(config,generation=0):
    validate(config)
    return dict(generation=generation,active_lambda=config['lambda'],value_iterations=0,
        ev_streak=0,healthy_streak=0,stage_origin=generation,
        **(dict(table_streaks=dict(historical=0,legacy=0),table_enabled=dict(historical=False,legacy=False),
                table_origins=dict(historical=None,legacy=None)) if 'table_gate' in config else {}),
        **(dict(table_evidence=ev_evidence.initial(generation)) if windowed(config) else {}),
        **(dict(stage_evidence=gae_health.initial(generation)) if stage_windowed(config) else {}))


def validate_state(state,config,generation):
    require(type(state) is dict and set(state)==set(initial(config)),'Invalid GAE checkpoint fields')
    require(state['generation']==generation,'GAE checkpoint progress differs')
    for key in ('generation','value_iterations','ev_streak','healthy_streak','stage_origin'):
        require(type(state[key]) is int and 0<=state[key]<=generation,'Invalid GAE checkpoint '+key)
    allowed=(1.,.98,.95) if config['gate']['mode']=='auto' else (config['lambda'],)
    require(finite(state['active_lambda']) and state['active_lambda'] in allowed,'Invalid persisted GAE lambda')
    require(state['ev_streak']<=state['value_iterations'] and
            state['healthy_streak']<=generation-state['stage_origin'],'Invalid GAE checkpoint streak')
    if 'table_gate' in config:
        for key in ('table_streaks','table_enabled','table_origins'):
            require(type(state[key]) is dict and set(state[key])=={'historical','legacy'},'Invalid table GAE state')
        require(all(type(v) is int and 0<=v<=generation for v in state['table_streaks'].values()) and
                all(type(v) is bool for v in state['table_enabled'].values()),'Invalid table GAE streaks')
        for k,origin in state['table_origins'].items():
            require((origin is None and not state['table_enabled'][k]) or
                    (type(origin) is int and 1<=origin<=generation and state['table_enabled'][k]),'Invalid table GAE origin')
    if windowed(config):ev_evidence.validate_state(state['table_evidence'],config['table_gate'],generation)
    if stage_windowed(config):gae_health.validate(state['stage_evidence'],config['gate'],generation)
    return state


def advance(state,config,*,generation,value_report,health_ok,rating):
    """Decisions apply to the NEXT collection; no backdating or automatic rollback."""
    state=copy.deepcopy(validate_state(state,config,generation-1));gate=config['gate'];used=state['active_lambda']
    state['generation']=generation
    from .value_phase import critic_work
    work=critic_work(value_report);ran=work['optimizer_steps']>0
    metrics=value_report.get('heldout_after',{})
    ev=(metrics.get('groups',{}).get('table:selfplay/role:learner',{}).get('explained_variance')
        if 'table_gate' in config else metrics.get('explained_variance'))
    healthy=bool(health_ok and (ran or (stage_windowed(config) and used!=1.))
        and finite(ev) and not value_report.get('kl_guard_tripped',False))
    stage_evidence=None
    if stage_windowed(config):
        state['stage_evidence'],stage_evidence=gae_health.advance(state['stage_evidence'],gate,
            generation=generation,stage_origin=state['stage_origin'],healthy=healthy,
            accepted_steps=work['optimizer_steps'],source=work['source'])
    state['value_iterations']+=int(ran)
    state['ev_streak']=(state['ev_streak']+1 if ran and finite(ev) and ev>=gate['heldout_ev_min'] else 0)
    state['healthy_streak']=(state['healthy_streak']+1 if used==.98 and healthy else 0)
    ready=(state['value_iterations']>=gate['minimum_value_iterations'] and state['ev_streak']>=gate['ev_streak'])
    fallback=state['value_iterations']>=gate['fallback_value_iterations']
    reason='fixed' if gate['mode']=='fixed' else 'improving_critic';decision='hold'
    if gate['mode']=='auto':
        if used==1. and (ready or fallback):
            if healthy:
                state['active_lambda']=.98;decision='advance';reason='critic_ev_gate' if ready else 'value_iteration_fallback'
            else:reason='health_not_ready'
        elif used==.98:
            reason='collecting_healthy_iterations'
            if state['healthy_streak']>=gate['healthy_iterations']:
                if stage_evidence is not None and not stage_evidence['ready']:reason='accepted_value_window_not_ready'
                elif rating.get('ok') is True:state['active_lambda']=.95;decision='advance';reason='healthy_streak_and_rating'
                else:reason='rating_evidence_not_ready'
        elif used==.95:reason='final_stage'
    if decision=='advance':state['stage_origin']=generation;state['healthy_streak']=0
    table_report={}
    if 'table_gate' in config:
        tg=config['table_gate']
        pooled={}
        if windowed(config):
            # A restored/no-improvement PPG phase still yields valid held-out
            # predictions. Require recent accepted value work in the window,
            # rather than forcing one accepted step on every sparse sample.
            table_healthy=bool(health_ok and not value_report.get('kl_guard_tripped',False))
            state['table_evidence'],pooled=ev_evidence.advance(state['table_evidence'],tg,
                generation=generation,metrics=metrics,healthy=table_healthy,value_updated=ran,source=work['source'])
        for kind in ('historical','legacy'):
            table_ev=metrics.get('groups',{}).get('table:'+kind+'/role:learner',{}).get('explained_variance')
            okay=table_healthy if pooled else healthy and finite(ev) and finite(table_ev) and table_ev>=ev-tg['ev_tolerance']
            state['table_streaks'][kind]=state['table_streaks'][kind]+1 if okay else 0
            before=state['table_enabled'][kind]
            if (not before and tg['mode']=='auto' and state['table_streaks'][kind]>=tg['consecutive_iterations']
                    and (not pooled or pooled[kind]['ready'])):
                state['table_enabled'][kind]=True
                state['table_origins'][kind]=generation
            origin=state['table_origins'][kind]
            # Qualification can precede the first non-MC collection. Review the
            # actual lambda stage, not time spent qualified at lambda=1. Both
            # origins already live in the checkpoint; no progress is rewritten.
            effective_origin=(max(origin,state['stage_origin'])
                if origin is not None and state['active_lambda']<1. else None)
            elapsed=generation-effective_origin if effective_origin is not None else 0
            table_report[kind]=dict(learner_heldout_ev=table_ev,selfplay_learner_ev=ev,
                streak=state['table_streaks'][kind],enabled=state['table_enabled'][kind],
                origin=origin,iterations_since_qualification=generation-origin if origin is not None else 0,
                effective_stage_origin=effective_origin,iterations_since_extension=elapsed,
                extension_age_scope='Committed collections using this lambda stage; qualification at lambda=1 does not count',
                thirty_iteration_review_due=effective_origin is not None and elapsed>=30,
                review='Inspect common PL rating slope/uncertainty and by-table advantage std; no automatic rollback',
                decision='enable_next_iteration' if not before and state['table_enabled'][kind] else 'hold',
                lambda_used=used if before else 1.,lambda_next=state['active_lambda'] if state['table_enabled'][kind] else 1.)
            if pooled:
                table_report[kind].update(evidence=pooled[kind],required_healthy_iterations=tg['consecutive_iterations'],
                    reason=('already_enabled' if before else 'selfplay_only' if tg['mode']=='selfplay_only' else
                            'healthy_streak_not_ready' if state['table_streaks'][kind]<tg['consecutive_iterations'] else
                            ','.join(pooled[kind]['reasons']) if pooled[kind]['reasons'] else 'window_ev_and_health_ready'),
                    streak_scope='Consecutive PPO/economic health and applicable value KL guard; critic work supported across the mixed-table EV window')
    return state,dict(contract=gate['contract'],mode=gate['mode'],lambda_used=used,
        table_gate=table_report,
        lambda_next=state['active_lambda'],decision=decision,reason=reason,generation=generation,
        value_iterations=state['value_iterations'],heldout_ev=ev,ev_streak=state['ev_streak'],
        healthy=healthy,healthy_streak=state['healthy_streak'],stage_origin=state['stage_origin'],rating=rating,
        stage_evidence=stage_evidence,critic_training=work,
        scope='Committed iterations only; no forks, extra ratings, or automatic rollback')


def rating_evidence(root,state,gate):
    """Read only the latest committed common PL fit, never compare separate fits."""
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import project_path
    reference=state.get('ratings',{}).get('latest_report')
    if reference is None:return dict(ok=False,reason='rating_unavailable',slope=None,points=[])
    path=project_path(root,reference['path']);require(sha(path)==reference['sha256'],'GAE rating evidence changed')
    report=read(path);points=[]
    for name,row in report['ratings'].items():
        if name.startswith('generation-'):
            generation=int(name.removeprefix('generation-'))
            require(generation<=state['iteration'],'GAE cannot use uncommitted ratings')
            if finite(row.get('points')):points.append((generation,row['points']))
    points=sorted(points)[-gate['rating_slots']:]
    slope=None
    if len(points)==gate['rating_slots']:
        x=np.asarray([g for g,_ in points],dtype=float);y=np.asarray([p for _,p in points],dtype=float)
        slope=float(((x-x.mean())*(y-y.mean())).sum()/((x-x.mean())**2).sum())
    alerts=report.get('alerts',{});warning=bool(alerts.get('regression') or alerts.get('v4_watch'))
    okay=slope is not None and slope>=gate['rating_min_slope'] and not warning
    return dict(ok=okay,reason='ok' if okay else 'rating_watch' if warning else
        'insufficient_rating_slots' if slope is None else 'negative_rating_slope',
        slope=slope,units='PL points per committed iteration',points=points,report=reference,
        warning=warning,scope='Latest common fit point-estimate slope; uncertainty remains, not proof of strength')
