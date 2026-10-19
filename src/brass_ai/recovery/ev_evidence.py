"""Bounded online held-out learner EV evidence; no training rows or predictions cached."""
import copy
import math

import numpy as np

from brass_ai.learning.contracts import require

CONTRACT='table-gae-ev-gate/2.0.0'
DEFAULT=dict(contract=CONTRACT,mode='auto',ev_tolerance=.05,consecutive_iterations=10,
    window_iterations=20,minimum_games=8,minimum_iterations=3,minimum_value_iterations=3)
TABLES=('selfplay','historical','legacy')
MOMENTS=('target_mean','target_m2','error_mean','error_m2')


def finite(x):return type(x) in (int,float) and math.isfinite(x)


def validate_config(config):
    require(type(config) is dict and set(config)==set(DEFAULT) and config['contract']==CONTRACT
        and config['mode'] in ('auto','selfplay_only') and config['ev_tolerance']==.05
        and type(config['consecutive_iterations']) is int and config['consecutive_iterations']==10,
        'Invalid windowed table GAE gate')
    for key in ('window_iterations','minimum_games','minimum_iterations','minimum_value_iterations'):
        require(type(config[key]) is int and config[key]>0,'Invalid table EV '+key)
    require(config['consecutive_iterations']<=config['window_iterations']<=64
        and 2<=config['minimum_games']<=1024
        and config['minimum_iterations']<=config['window_iterations']
        and config['minimum_value_iterations']<=config['window_iterations'],'Invalid table EV window/support bounds')
    return config


def statistics(target,prediction,game_ids):
    target=np.asarray(target,dtype=np.float64);prediction=np.asarray(prediction,dtype=np.float64)
    require(target.ndim==prediction.ndim==1 and target.shape==prediction.shape
        and len(target)==len(game_ids)>0 and np.isfinite(target).all() and np.isfinite(prediction).all(),
        'Invalid held-out EV observations')
    error=prediction-target
    result=dict(rows=len(target),game_ids=sorted(set(game_ids)),
        target_mean=float(target.mean()),target_m2=float(np.square(target-target.mean()).sum()),
        error_mean=float(error.mean()),error_m2=float(np.square(error-error.mean()).sum()))
    return validate_statistics(result)


def validate_statistics(s):
    require(type(s) is dict and set(s)=={'rows','game_ids',*MOMENTS}
        and type(s['rows']) is int and s['rows']>0,'Invalid table EV statistics')
    ids=s['game_ids']
    require(type(ids) is list and ids and all(type(g) is str and g for g in ids)
        and ids==sorted(set(ids)) and len(ids)<=s['rows'],'Invalid distinct EV game support')
    require(all(finite(s[k]) for k in MOMENTS) and s['target_m2']>=0 and s['error_m2']>=0,
        'Invalid table EV moments')
    return s


def pool(parts):
    """Parallel-variance merge; averaging per-iteration EV is not equivalent."""
    n=0;means=dict(target=0.,error=0.);m2=dict(target=0.,error=0.);ids=set()
    for s in parts:
        validate_statistics(s)
        require(ids.isdisjoint(s['game_ids']),'Repeated game in table EV evidence')
        ids.update(s['game_ids']);count=s['rows'];total=n+count
        for key in means:
            delta=s[key+'_mean']-means[key]
            m2[key]+=s[key+'_m2']+delta*delta*n*count/total
            means[key]+=delta*count/total
        n=total
    ev=1-m2['error']/m2['target'] if n and m2['target']/n>1e-12 else None
    require(ev is None or finite(ev),'Nonfinite pooled table EV')
    return dict(rows=n,games=len(ids),explained_variance=ev,
        target_variance=m2['target']/n if n else None,error_variance=m2['error']/n if n else None)


def initial(generation):return dict(contract=CONTRACT,origin=generation,entries=[])


def validate_state(state,config,generation):
    validate_config(config)
    require(type(state) is dict and set(state)=={'contract','origin','entries'} and state['contract']==CONTRACT
        and type(state['origin']) is int and 0<=state['origin']<=generation,'Invalid table EV checkpoint')
    entries=state['entries'];start=max(state['origin']+1,generation-config['window_iterations']+1)
    require(type(entries) is list and len(entries)<=config['window_iterations']
        and [e.get('generation') for e in entries]==list(range(start,generation+1)),
        'Table EV evidence progress differs')
    seen=set()
    for e in entries:
        require(set(e)=={'generation','healthy','value_updated','tables'} and type(e['generation']) is int and type(e['healthy']) is bool
            and type(e['value_updated']) is bool and type(e['tables']) is dict and set(e['tables'])==set(TABLES),
            'Invalid table EV evidence entry')
        for s in e['tables'].values():
            if s is None:continue
            validate_statistics(s)
            require(seen.isdisjoint(s['game_ids']),'Repeated game in table EV evidence')
            seen.update(s['game_ids'])
    return state


def advance(state,config,*,generation,metrics,healthy,value_updated):
    state=copy.deepcopy(validate_state(state,config,generation-1));groups=metrics.get('groups',{})
    tables={kind:copy.deepcopy(groups.get('table:'+kind+'/role:learner',{}).get('ev_evidence')) for kind in TABLES}
    state['entries']=(state['entries']+[dict(generation=generation,healthy=bool(healthy),
        value_updated=bool(value_updated),tables=tables)])[-config['window_iterations']:]
    validate_state(state,config,generation)
    reports={}
    for kind in TABLES[1:]:
        paired=[e for e in state['entries'] if e['healthy'] and e['tables'][kind] is not None
                and e['tables']['selfplay'] is not None]
        mixed=pool([e['tables'][kind] for e in paired]);reference=pool([e['tables']['selfplay'] for e in paired])
        # Value updates can occur between sparse mixed-table samples. They need
        # not coincide with every sample, but must be recent and healthy.
        updates=sum(e['healthy'] and e['value_updated'] for e in state['entries'])
        fresh=bool(paired and paired[-1]['generation']==generation)
        reasons=[]
        if len(paired)<config['minimum_iterations']:reasons.append('insufficient_iterations')
        if min(mixed['games'],reference['games'])<config['minimum_games']:reasons.append('insufficient_games')
        if updates<config['minimum_value_iterations']:reasons.append('insufficient_value_updates')
        if not finite(mixed['explained_variance']) or not finite(reference['explained_variance']):reasons.append('ev_unavailable')
        supported=not reasons
        ev_ok=supported and mixed['explained_variance']>=reference['explained_variance']-config['ev_tolerance']
        if supported and not ev_ok:reasons.append('ev_below_selfplay_tolerance')
        if not fresh:reasons.append('no_fresh_mixed_sample')
        if not healthy:reasons.append('health_not_ready')
        reports[kind]=dict(contract=CONTRACT,origin=state['origin'],window_iterations=config['window_iterations'],
            window_start=state['entries'][0]['generation'],window_end=generation,
            contributing_iterations=len(paired),contributing_generations=[e['generation'] for e in paired],
            value_iterations=updates,minimum_games=config['minimum_games'],minimum_iterations=config['minimum_iterations'],
            minimum_value_iterations=config['minimum_value_iterations'],mixed=mixed,selfplay=reference,
            fresh=fresh,current_healthy=bool(healthy),supported=supported,ev_ok=ev_ok,ready=bool(healthy and fresh and ev_ok),reasons=reasons,
            scope='Pooled row-weighted online EV at each iteration\'s accepted post-value weights; same contributing iterations for self-play; distinct whole-game support, learner seats only; not current-model re-evaluation or independent confidence bounds')
    return state,reports
