"""Decayed pairwise training outranks; frozen PFSP plans, not strength ratings."""
import copy
import math

from brass_ai.learning.contracts import require
from .quality_config import OPPONENT_POOL,LEGACY_POOL,pfsp,validate_pool


def capacity(config,games):
    require(type(games) is int and games>=0,'Invalid PFSP committed ordinary games')
    return next(r['max'] for r in config['max_scored_members'] if r['until_games'] is None or games<r['until_games'])


def prior(config):
    c=config['outrank_rate']
    return dict(S=float(c['prior_rate']*c['prior_comparisons']),C=float(c['prior_comparisons']),alarm_streak=0)


def rate(member):return member['S']/member['C']


def weight(member,config):
    x=rate(member);c=config['weighting']
    return (1-x)**c['power'] if c['function']=='hard' else (x*(1-x))**c['power']


def _distribution(pool):
    members=[m for m in pool['members'] if m['generation']!=0];n=len(members);k=n+1;c=pool['config']
    rule=c['pinned_seat_share'];share=1. if n==0 else min(1/(k+rule['offset']),rule['max'] or 1.)
    weights=[weight(m,c) for m in members];total=math.fsum(weights);floor=c['uniform_floor']
    rows=[dict(model_id=m['model_id'],generation=m['generation'],x_hat=rate(m),C=m['C'],w=w,
        p=(1-floor)*w/total+floor/n if total else 1/n) for m,w in zip(members,weights)]
    return dict(generation=pool['generation'],games_done=pool['games_done'],k=k,N=n,il_share=share,members=rows,
        effective_members=1/math.fsum(r['p']**2 for r in rows) if n else 0.,
        largest_share=max((r['p'] for r in rows),default=0.),
        recent_share=math.fsum(r['p'] for m,r in zip(members,rows) if pool['generation']-m['admitted_iteration']<20),
        weighted_outrank_rate=math.fsum(r['p']*r['x_hat'] for r in rows) if n else None,
        all_zero_weights=bool(n and not total))


def seal(pool):
    pool['next_planning']=_distribution(pool)
    return pool


def initial(anchor,config,probe):
    from .quality_pool import reference
    validate_pool(config)
    member=dict(reference(anchor),admitted_iteration=0,probe=copy.deepcopy(probe),**prior(config),
        admission_health=dict(ok=True,scope='Pinned completed IL reference'))
    return seal(dict(contract=config['contract'],config=copy.deepcopy(config),generation=0,games_done=0,
        members=[member],health_window=[],planning=None,events=[]))


def validate_state(pool,config,generation=None):
    validate_pool(config)
    require(type(pool) is dict and set(pool)=={'contract','config','generation','games_done','members',
        'health_window','planning','next_planning','events'},'Invalid PFSP state fields')
    require(pool['contract']==config['contract'] and pool['config']==config,'PFSP pool config differs')
    g=pool['generation'];require(type(g) is int and g>=0 and (generation is None or generation==g),'PFSP generation differs')
    members=pool['members'];cap=capacity(config,pool['games_done'])
    require(type(members) is list and 1<=len(members)<=1+cap,'PFSP pool exceeds its committed-game capacity')
    require(len({m['model_id'] for m in members})==len(members) and sum(m['generation']==0 for m in members)==1,
        'PFSP membership duplicate or missing pinned IL')
    for m in members:
        require(type(m['generation']) is int and 0<=m['generation']<=g and
            type(m['admitted_iteration']) is int and m['generation']<=m['admitted_iteration']<=g,
            'Invalid PFSP admission iteration')
        require(not {'q','eta','recent_rates'}&set(m),'Quality-score fields are forbidden in PFSP state')
        require(all(type(m[k]) in (int,float) and math.isfinite(m[k]) for k in ('S','C')) and
            0<=m['S']<=m['C'] and m['C']>0,'Invalid PFSP decayed sums')
        require(type(m['alarm_streak']) is int and 0<=m['alarm_streak']<=config['alarm']['iterations'],
            'Invalid PFSP alarm streak')
    require(type(pool['health_window']) is list and len(pool['health_window'])<=config['admit_health_iterations']
        and all(type(v) is bool for v in pool['health_window']),'Invalid PFSP health window')
    require(type(pool['events']) is list and pool['next_planning']==_distribution(pool),'PFSP stored next plan differs')
    return pool


def distribution(pool):return copy.deepcopy(pool['next_planning'])


def draw(pool,planned,rng):
    anchor=next(m for m in pool['members'] if m['generation']==0);share=planned['il_share']
    if not planned['N'] or rng.random()<share:
        return anchor,dict(model_id=anchor['model_id'],source='IL',N=planned['N'],p=None,il_share=share,seat_probability=share)
    ticket=rng.random();cumulative=0.;selected=planned['members'][-1]
    for row in planned['members']:
        cumulative+=row['p']
        if ticket<cumulative:selected=row;break
    member=next(m for m in pool['members'] if m['model_id']==selected['model_id'])
    return member,dict(model_id=member['model_id'],source='scored',N=planned['N'],p=selected['p'],
        il_share=share,seat_probability=(1-share)*selected['p'])


def comparisons(pool,games,tables,planned):
    from brass_ai.learning.replay import terminal_targets
    rows={m['model_id']:dict(seats=0,outcomes=[]) for m in pool['members']}
    lookup={g['setup_group']:g for g in games}
    require(len(lookup)==len(games) and set(tables)<=set(lookup),'Missing/duplicate PFSP history games')
    expected={r['model_id']:r['p'] for r in planned['members']}
    anchor=next(m['model_id'] for m in pool['members'] if m['generation']==0)
    for setup,table in sorted(tables.items(),key=lambda item:item[1]['index']):
        game=lookup[setup];require(game['opponents']==table['policy_ids'],'PFSP seat identities differ from plan')
        learner=table['learner_seats'];opponents=table['opponents'];seats=[o['seat'] for o in opponents]
        require(len(learner)==len(opponents)==2 and len(set(learner))==len(set(seats))==2 and
            set(learner).isdisjoint(seats) and set(learner+seats)==set(range(4)),
            'PFSP requires two learner and two separate snapshot seats')
        ranks=terminal_targets(game['result'])['placement']
        for o in opponents:
            is_il=o['model_id']==anchor;p=expected.get(o['model_id'])
            require(o['model_id'] in rows and o['source']==('IL' if is_il else 'scored') and o['N']==planned['N']
                and o['p']==p and o['il_share']==planned['il_share'] and
                o['seat_probability']==(planned['il_share'] if is_il else (1-planned['il_share'])*p),
                'PFSP update requires the original planning probability')
            row=rows[o['model_id']];row['seats']+=1
            for seat in learner:
                a,b=ranks[seat],ranks[o['seat']]
                row['outcomes'].append(1. if a<b else .5 if a==b else 0.)
    return rows


def update_scores(pool,games,tables,planned,*,generation,games_done):
    validate_state(pool,pool['config'])
    require(generation==pool['generation']+1,'PFSP must advance once per committed iteration')
    require(type(games_done) is int and games_done>pool['games_done'],'PFSP requires advancing ordinary-game progress')
    require(planned==distribution(pool),'PFSP planning differs from committed estimates')
    rows=comparisons(pool,games,tables,planned)
    return update_counts(pool,rows,planned,generation=generation,games_done=games_done)


def update_counts(pool,rows,planned,*,generation,games_done):
    """Pure float64 estimator; callers provide only this collection's comparisons."""
    pool=copy.deepcopy(pool);c=pool['config'];decay=.5**(1/c['outrank_rate']['half_life_iterations'])
    report=[];alarms=[];old={r['model_id']:r for r in planned['members']}
    for m in pool['members']:
        row=rows[m['model_id']];n=len(row['outcomes']);wins=math.fsum(row['outcomes'])
        require(type(row['seats']) is int and row['seats']>=0 and n==2*row['seats'] and
            all(v in (0.,.5,1.) for v in row['outcomes']),'Invalid PFSP comparison counts')
        before=dict(S=m['S'],C=m['C'],x_hat=rate(m))
        m['S']=decay*m['S']+wins;m['C']=decay*m['C']+n
        alarm=c['alarm'];bad=rate(m)<alarm['rate_below'] and m['C']>=alarm['min_decayed_comparisons']
        m['alarm_streak']=min(alarm['iterations'],m['alarm_streak']+1) if bad else 0
        if m['alarm_streak']==alarm['iterations']:
            alarms.append(dict(model_id=m['model_id'],generation=m['generation'],x_hat=rate(m),C=m['C'],
                iterations=m['alarm_streak'],reason='pool_regression',action='review_only'))
        used=old.get(m['model_id']);is_il=m['generation']==0
        report.append(dict(model_id=m['model_id'],generation=m['generation'],age=generation-m['admitted_iteration'],
            before=before,S=m['S'],C=m['C'],x_hat=rate(m),w=weight(m,c),
            p=used['p'] if used else None,w_used=used['w'] if used else None,
            seat_probability=planned['il_share'] if is_il else (1-planned['il_share'])*used['p'],
            seats_played=row['seats'],comparisons=n,outrank_rate=wins/n if n else None,alarm_streak=m['alarm_streak']))
    pool.update(generation=generation,games_done=games_done,planning=copy.deepcopy(planned));seal(pool)
    validate_state(pool,c,generation)
    il=next(r for r in report if r['generation']==0)
    return pool,dict(contract=c['contract'],planning=copy.deepcopy(planned),members=report,alarms=alarms,
        il=dict(share=planned['il_share'],seats=il['seats_played'],comparisons=il['comparisons'],x_hat=il['x_hat'],C=il['C']),
        decay=decay,scope='Unweighted learner/snapshot pairs in completed history tables only; p/IL share used are frozen before collection. Decayed rates are training diagnostics, not PL strength.')


def admission(pool,entry,health,distances,pair_distance):
    from .quality_pool import reference
    pool=copy.deepcopy(pool);c=pool['config'];g=pool['generation'];cap=capacity(c,pool['games_done'])
    event=dict(generation=g,candidate=entry['model_id'],admitted=False,evicted=None,reason='not_due',max_scored_members=cap)
    if g%c['admit_every_iterations']:return pool,event
    if len(pool['health_window'])<c['admit_health_iterations'] or not all(pool['health_window']):
        event['reason']='five_healthy_iterations_required'
    elif health.get('ok') is not True:event['reason']='health_unavailable'
    elif entry['model_id'] in {m['model_id'] for m in pool['members']}:event['reason']='already_present'
    elif distances is None or set(distances)!={m['model_id'] for m in pool['members']}:event['reason']='probe_unavailable'
    else:
        require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in distances.values()),'Invalid PFSP admission KL')
        event['minimum_kl']=min(distances.values())
        if event['minimum_kl']<c['admit_min_kl']:event['reason']='insufficient_distinctness'
        else:
            scored=[m for m in pool['members'] if m['generation']!=0];e=c['evict']
            newest={m['model_id'] for m in sorted(scored,key=lambda m:(m['admitted_iteration'],m['model_id']))[-e['protect_newest']:]}
            eligible=[m for m in scored if m['model_id'] not in newest and g-m['admitted_iteration']>=e['min_age_iterations']
                and m['C']>=e['min_decayed_comparisons']]
            event['eligible_members']=[m['model_id'] for m in eligible]
            if len(scored)>=cap and not eligible:event['reason']='no_eligible_eviction'
            else:
                if len(scored)>=cap:
                    high=max(rate(m) for m in eligible);shortlist=[m for m in eligible if rate(m)>=high-e['redundancy_band']]
                    def redundancy(m):
                        values=[pair_distance(m,other) for other in pool['members'] if other['model_id']!=m['model_id']]
                        require(all(math.isfinite(v) and v>=0 for v in values),'Invalid PFSP redundancy KL')
                        return min(values)
                    distances_to_nearest={m['model_id']:redundancy(m) for m in shortlist}
                    victim=min(shortlist,key=lambda m:(distances_to_nearest[m['model_id']],-rate(m),m['generation'],m['model_id']))
                    event.update(evicted=reference(victim),eviction=dict(reason='easy_and_redundant',x_hat=rate(victim),C=victim['C'],
                        highest_eligible_rate=high,shortlist=distances_to_nearest,nearest_kl=distances_to_nearest[victim['model_id']]))
                    pool['members'].remove(victim)
                pool['members'].append(dict(entry,admitted_iteration=g,admission_health=copy.deepcopy(health),**prior(c)))
                event.update(admitted=True,reason='healthy_and_distinct')
    pool['events'].append(copy.deepcopy(event));seal(pool)
    return pool,event


def migrate(pool,config,generation,games_done):
    """No reinterpretation of q: seed from logged outcomes at an admission boundary."""
    from .quality_pool import validate_state as validate_old,state_digest
    validate_old(pool,pool['config'],generation);validate_pool(config)
    require(pool['config']==LEGACY_POOL and config==OPPONENT_POOL,'Unreviewed PFSP pool migration settings')
    require(generation%config['admit_every_iterations']==0,
        'PFSP switch requires an admission boundary (committed generation divisible by 5); finish with the previous config first')
    require(len(pool['members'])<=1+capacity(config,games_done),'Existing pool exceeds the PFSP stage cap; explicit membership review required')
    old_digest=state_digest(pool);result=copy.deepcopy(pool);seeds=[]
    for m in result['members']:
        rates=m.pop('recent_rates');m.pop('q');n=sum(r['comparisons'] for r in rates)
        wins=math.fsum(r['outcomes'] for r in rates)
        require(all(type(r['comparisons']) is int and r['comparisons']>=0 and
            type(r['outcomes']) in (int,float) and math.isfinite(r['outcomes']) and 0<=r['outcomes']<=r['comparisons']
            for r in rates),'Invalid logged quality-pool outrank rates')
        m.update(prior(config))
        if n:m['S']=m['C']*wins/n
        seeds.append(dict(model_id=m['model_id'],x_hat=rate(m),S=m['S'],C=m['C'],
            source='logged_five_iteration_outrank' if n else 'prior',logged_comparisons=n))
    event=dict(kind='quality_to_pfsp',generation=generation,games_done=games_done,seeds=seeds,
        from_digest=old_digest,scope='Preserve membership, checkpoint/probe/fingerprint identities, admission ages and health window; no old q conversion')
    result.update(contract=config['contract'],config=copy.deepcopy(config),games_done=games_done,planning=None)
    result['events'].append(copy.deepcopy(event));seal(result);validate_state(result,config,generation)
    return result,dict(event,to_digest=state_digest(result))


def report_finish(report,pool,event):
    plan=distribution(pool)
    report.update(admission=event,next_distribution=plan,size=len(pool['members']),
        max_members=1+capacity(pool['config'],pool['games_done']),uniform_floor=pool['config']['uniform_floor'],
        pinned_seat_share=plan['il_share'],summary={k:plan[k] for k in
            ('k','effective_members','largest_share','recent_share','weighted_outrank_rate')})
    return report


def console(report):
    if not report or not pfsp(report):return ''
    p=report['next_distribution']
    return f' PFSP=IL:{p["il_share"]:.3f} effective:{p["effective_members"]:.1f} alarms:{len(report["alarms"])}'
