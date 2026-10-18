"""Undiscounted, decision-chain GAE and committed, deterministic adoption gates."""
import copy
import math
from collections import defaultdict

import numpy as np
import torch

from brass_ai.learning.contracts import require

CONTRACT='gae-advantage/2.0.0'
GATE=dict(contract='staged-gae-gate/1.0.0',mode='auto',minimum_value_iterations=10,
    heldout_ev_min=.5,ev_streak=5,fallback_value_iterations=30,healthy_iterations=20,
    rating_slots=3,rating_min_slope=0.)
TABLE_GATE=dict(contract='table-gae-ev-gate/1.0.0',mode='auto',ev_tolerance=.05,consecutive_iterations=10)
DEFAULT=dict(contract=CONTRACT,**{'lambda':1.0},shadow_lambdas=[.98,.95,.9],chain='nonforced',gate=GATE,table_gate=TABLE_GATE)


def finite(x):return type(x) in (int,float) and math.isfinite(x)


def validate(config,*,combined=True):
    legacy=config.get('contract')=='gae-advantage/1.0.0'
    require(type(config) is dict and set(config)==(set(DEFAULT)-{'table_gate'} if legacy else set(DEFAULT)) and
            config['contract'] in (CONTRACT,'gae-advantage/1.0.0')
            and config['chain']=='nonforced','Invalid GAE contract/fields/chain')
    if not legacy:
        t=config['table_gate']
        require(type(t) is dict and set(t)==set(TABLE_GATE) and t['contract']==TABLE_GATE['contract']
            and t['mode'] in ('auto','selfplay_only') and t['ev_tolerance']==.05
            and type(t['consecutive_iterations']) is int and t['consecutive_iterations']==10,'Invalid table GAE gate')
    require(finite(config['lambda']) and 0<config['lambda']<=1,'Invalid GAE lambda')
    require(type(config['shadow_lambdas']) is list and config['shadow_lambdas']==[.98,.95,.9],
            'GAE shadows must be 0.98, 0.95 and 0.9')
    gate=config['gate']
    require(type(gate) is dict and set(gate)==set(GATE) and gate['contract']==GATE['contract']
            and gate['mode'] in ('fixed','auto'),'Invalid GAE gate contract/fields/mode')
    for key in ('minimum_value_iterations','ev_streak','fallback_value_iterations','healthy_iterations','rating_slots'):
        require(type(gate[key]) is int and gate[key]>0,'Invalid GAE gate '+key)
    require(gate['minimum_value_iterations']<=gate['fallback_value_iterations'] and gate['rating_slots']>=3
            and finite(gate['heldout_ev_min']) and 0<=gate['heldout_ev_min']<=1
            and finite(gate['rating_min_slope']) and gate['rating_min_slope']>=0,'Invalid GAE gate bounds')
    require(gate['mode']!='auto' or config['lambda']==1.,'Automatic GAE starts at lambda 1')
    require(combined or (config['lambda']==1. and gate['mode']=='fixed'),
            'GAE lambda below 1 / automatic gate requires combined reward')
    return config


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
                table_origins=dict(historical=None,legacy=None)) if 'table_gate' in config else {}))


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
    return state


def advance(state,config,*,generation,value_report,health_ok,rating):
    """Decisions apply to the NEXT collection; no backdating or automatic rollback."""
    state=copy.deepcopy(validate_state(state,config,generation-1));gate=config['gate'];used=state['active_lambda']
    state['generation']=generation
    ran=value_report.get('optimizer_steps',0)>0
    metrics=value_report.get('heldout_after',{})
    ev=(metrics.get('groups',{}).get('table:selfplay/role:learner',{}).get('explained_variance')
        if 'table_gate' in config else metrics.get('explained_variance'))
    healthy=bool(health_ok and ran and finite(ev) and not value_report.get('kl_guard_tripped',False))
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
                if rating.get('ok') is True:state['active_lambda']=.95;decision='advance';reason='healthy_streak_and_rating'
                else:reason='rating_evidence_not_ready'
        elif used==.95:reason='final_stage'
    if decision=='advance':state['stage_origin']=generation;state['healthy_streak']=0
    table_report={}
    if 'table_gate' in config:
        tg=config['table_gate']
        for kind in ('historical','legacy'):
            table_ev=metrics.get('groups',{}).get('table:'+kind+'/role:learner',{}).get('explained_variance')
            okay=healthy and finite(ev) and finite(table_ev) and table_ev>=ev-tg['ev_tolerance']
            state['table_streaks'][kind]=state['table_streaks'][kind]+1 if okay else 0
            before=state['table_enabled'][kind]
            if not before and tg['mode']=='auto' and state['table_streaks'][kind]>=tg['consecutive_iterations']:
                state['table_enabled'][kind]=True
                state['table_origins'][kind]=generation
            elapsed=generation-state['table_origins'][kind] if state['table_origins'][kind] is not None else 0
            table_report[kind]=dict(learner_heldout_ev=table_ev,selfplay_learner_ev=ev,
                streak=state['table_streaks'][kind],enabled=state['table_enabled'][kind],
                origin=state['table_origins'][kind],iterations_since_extension=elapsed,
                thirty_iteration_review_due=elapsed>=30,
                review='Inspect common PL rating slope/uncertainty and by-table advantage std; no automatic rollback',
                decision='enable_next_iteration' if not before and state['table_enabled'][kind] else 'hold',
                lambda_used=used if before else 1.,lambda_next=state['active_lambda'] if state['table_enabled'][kind] else 1.)
    return state,dict(contract=gate['contract'],mode=gate['mode'],lambda_used=used,
        table_gate=table_report,
        lambda_next=state['active_lambda'],decision=decision,reason=reason,generation=generation,
        value_iterations=state['value_iterations'],heldout_ev=ev,ev_streak=state['ev_streak'],
        healthy=healthy,healthy_streak=state['healthy_streak'],stage_origin=state['stage_origin'],rating=rating,
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
