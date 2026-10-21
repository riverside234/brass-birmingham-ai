"""Versioned history sampling, probe admission and checkpoint references."""
import copy
import math
from itertools import combinations
from pathlib import Path
import numpy as np

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest,project_path,require
from brass_ai.learning.replay import terminal_targets
from .quality_config import validate_pool,pfsp
from . import pfsp_pool

PAIRS=tuple(combinations(range(4),2))
REF_FIELDS=('path','sha256','model_id','generation','weights_only')


def reference(member):return {k:member[k] for k in REF_FIELDS if k in member}


def state_digest(pool):
    # Derived opponent caches change path/hash without changing the committed
    # state. Excluding these fields also avoids a checkpoint hashing itself.
    out=copy.deepcopy(pool)
    for member in out['members']:
        for key in ('path','sha256','weights_only'):member.pop(key,None)
    return digest(out)


def initial(anchor,config,probe):
    if pfsp(config):return pfsp_pool.initial(anchor,config,probe)
    validate_pool(config)
    entry=dict(reference(anchor),admitted_iteration=0,q=None,probe=copy.deepcopy(probe),
               admission_health=dict(ok=True,scope='Pinned completed IL reference'),recent_rates=[])
    return dict(contract=config['contract'],config=copy.deepcopy(config),generation=0,
                members=[entry],health_window=[],planning=None,events=[])


def validate_state(pool,config,generation=None):
    if pfsp(config):return pfsp_pool.validate_state(pool,config,generation)
    validate_pool(config)
    require(pool['contract']==config['contract'] and pool['config']==config,'Quality pool config differs')
    require(type(pool['generation']) is int and pool['generation']>=0 and
            (generation is None or pool['generation']==generation),'Quality pool generation differs')
    members=pool['members'];ids=[e['model_id'] for e in members]
    require(1<=len(members)<=config['max_members'] and len(set(ids))==len(ids),'Invalid quality pool membership')
    require(sum(e['generation']==0 for e in members)==1,'Quality pool lost the pinned IL member')
    for member in members:
        require(type(member['admitted_iteration']) is int and 0<=member['admitted_iteration']<=pool['generation'],
                'Invalid quality pool admission age')
        require(member['q'] is None if member['generation']==0 else
                type(member['q']) in (int,float) and math.isfinite(member['q']), 'Invalid quality score')
    require(len(pool['health_window'])<=config['admit_health_iterations'] and
            all(type(v) is bool for v in pool['health_window']),'Invalid quality-pool health window')
    return pool


def distribution(pool,il_gate=None):
    if pfsp(pool):return pfsp_pool.distribution(pool,il_gate=il_gate)
    members=[m for m in pool['members'] if m['generation']!=0]
    if not members:return dict(N=0,members=[])
    top=max(m['q'] for m in members);weights=[math.exp(m['q']-top) for m in members]
    total=math.fsum(weights);floor=pool['config']['uniform_floor'];n=len(members)
    return dict(N=n,members=[dict(model_id=m['model_id'],generation=m['generation'],q=m['q'],
        p=(1-floor)*w/total+floor/n) for m,w in zip(members,weights)])


def draw(pool,planned,rng):
    """Independent seat draw; p is conditional on drawing the scored pool."""
    if pfsp(pool):return pfsp_pool.draw(pool,planned,rng)
    anchor=next(m for m in pool['members'] if m['generation']==0)
    share=pool['config']['pinned_seat_share']
    if not planned['N'] or rng.random()<share:
        return anchor,dict(model_id=anchor['model_id'],source='IL',N=planned['N'],p=None,
                           seat_probability=share if planned['N'] else 1.)
    ticket=rng.random();cumulative=0.;selected=planned['members'][-1]
    for row in planned['members']:
        cumulative+=row['p']
        if ticket<cumulative:selected=row;break
    member=next(m for m in pool['members'] if m['model_id']==selected['model_id'])
    return member,dict(model_id=member['model_id'],source='scored',N=planned['N'],p=selected['p'],
                       seat_probability=(1-share)*selected['p'])


def learner_pair(index,kind,quotas):
    """Cycle each mixed-table type through six pairs using absolute progress."""
    from .config import table_kind
    cycle=[table_kind(i,[dict(until=None,**quotas)]) for i in range(20)]
    require(kind in ('historical','legacy') and cycle[index%20]==kind,'Invalid mixed-table index')
    ordinal=(index//20)*cycle.count(kind)+cycle[:index%20].count(kind)
    return PAIRS[ordinal%len(PAIRS)]


def update_scores(pool,games,tables,planned,*,generation,games_done=None):
    """No mutation before a generation commits; each learner/opponent pair counts."""
    if pfsp(pool):return pfsp_pool.update_scores(pool,games,tables,planned,generation=generation,games_done=games_done)
    pool=copy.deepcopy(validate_state(pool,pool['config']))
    require(generation==pool['generation']+1,'Quality scores must advance once per generation')
    require(planned==distribution(pool),'Quality planning probabilities differ from committed scores')
    rows={m['model_id']:dict(seats=0,outcomes=[]) for m in pool['members']}
    lookup={g['setup_group']:g for g in games}
    require(len(lookup)==len(games) and set(tables)<=set(lookup),'Missing/duplicate history games')
    expected={m['model_id']:m['p'] for m in planned['members']}
    for setup,table in sorted(tables.items(),key=lambda item:item[1]['index']):
        game=lookup[setup];require(game['opponents']==table['policy_ids'],'History seat identities differ from plan')
        ranks=terminal_targets(game['result'])['placement']
        require(len(table['learner_seats'])==2 and len(table['opponents'])==2,'History requires two independent opponent seats')
        for opponent in table['opponents']:
            row=rows[opponent['model_id']];row['seats']+=1
            if opponent['source']=='IL':
                require(opponent['model_id'] not in expected,'Scored opponent relabelled IL')
            else:
                require(opponent['source']=='scored' and opponent['N']==planned['N'] and
                        opponent['p']==expected[opponent['model_id']],
                        'Quality update must use the original conditional planning probability')
            for learner in table['learner_seats']:
                a,b=ranks[learner],ranks[opponent['seat']]
                row['outcomes'].append(1. if a<b else .5 if a==b else 0.)
    report=[]
    for member in pool['members']:
        row=rows[member['model_id']];wins=math.fsum(row['outcomes']);n=len(row['outcomes']);before=member['q']
        if member['generation']!=0:
            member['q']-=pool['config']['eta']*wins/(planned['N']*expected[member['model_id']])
        member['recent_rates']=(member['recent_rates']+[dict(outcomes=wins,comparisons=n)])[-5:]
        samples=sum(r['comparisons'] for r in member['recent_rates'])
        report.append(dict(model_id=member['model_id'],generation=member['generation'],q_before=before,
            p=expected.get(member['model_id']),seats_played=row['seats'],comparisons=n,
            outrank_rate=wins/n if n else None,moving_outrank_rate=(
                sum(r['outcomes'] for r in member['recent_rates'])/samples if samples else None)))
    scored=[m for m in pool['members'] if m['generation']!=0]
    if scored:
        top=max(m['q'] for m in scored)
        for member in scored:member['q']-=top
    for row,member in zip(report,pool['members']):row['q_after']=member['q']
    pool.update(generation=generation,planning=copy.deepcopy(planned))
    return pool,dict(contract=pool['contract'],planning=planned,members=report,
        scope='Conditional scored-pool p frozen at planning; two learner comparisons per opponent seat; IL outcomes are diagnostic only, never scored')


def mean_kl(reference,current,offsets,floor=1e-6):
    require(reference.shape==current.shape and len(reference)==int(offsets[-1]),'Different opponent probe shapes')
    p=np.maximum(reference.astype(np.float64),floor);q=np.maximum(current.astype(np.float64),floor)
    for start,end in zip(offsets[:-1],offsets[1:]):
        p[start:end]/=p[start:end].sum();q[start:end]/=q[start:end].sum()
    values=np.add.reduceat(p*(np.log(p)-np.log(q)),offsets[:-1])
    return float(np.maximum(values,0).mean())


def cached(root,member,probe):
    receipt=member['probe'];path=project_path(root,receipt['path'])
    require(receipt['probe_hash']==probe.metadata['probe_hash'] and sha(path)==receipt['sha256'],
            'Quality member probe changed')
    with np.load(path,allow_pickle=False) as data:values=data['reference'].copy()
    require(values.shape==probe.arrays['reference'].shape and np.isfinite(values).all() and (values>=0).all(),
            'Invalid quality member probabilities')
    for a,b in zip(probe.arrays['offsets'][:-1],probe.arrays['offsets'][1:]):
        require(abs(float(values[a:b].sum())-1)<1e-5,'Quality member probabilities are not normalized')
    return values


def cache_candidate(root,run,entry,values,probe):
    path=run/'opponent-pool'/'probes'/(entry['model_id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as old:require(np.array_equal(old['reference'],values),'Candidate probe cache differs')
    else:
        temp=path.with_suffix('.tmp')
        with temp.open('wb') as handle:np.savez(handle,reference=values)
        temp.replace(path)
    return dict(path=path.relative_to(root).as_posix(),sha256=sha(path),probe_hash=probe.metadata['probe_hash'])


def admission(pool,entry,health,distances,pair_distance):
    """Pure membership decision; caller supplies cached probe comparisons."""
    if pfsp(pool):return pfsp_pool.admission(pool,entry,health,distances,pair_distance)
    pool=copy.deepcopy(pool);c=pool['config'];generation=pool['generation']
    outcome=dict(generation=generation,candidate=entry['model_id'],admitted=False,evicted=None,reason='not_due')
    if generation%c['admit_every_iterations']:return pool,outcome
    if len(pool['health_window'])<c['admit_health_iterations'] or not all(pool['health_window']):
        outcome['reason']='five_healthy_iterations_required'
    elif health.get('ok') is not True:outcome['reason']='health_unavailable'
    elif entry['model_id'] in {m['model_id'] for m in pool['members']}:outcome['reason']='already_present'
    elif distances is None or set(distances)!={m['model_id'] for m in pool['members']}:
        outcome['reason']='probe_unavailable'
    else:
        require(all(math.isfinite(v) and v>=0 for v in distances.values()),'Invalid admission KL')
        outcome['minimum_kl']=min(distances.values())
        if outcome['minimum_kl']<c['admit_min_kl']:outcome['reason']='insufficient_distinctness'
        else:
            scored=[m for m in pool['members'] if m['generation']!=0]
            newest={m['model_id'] for m in sorted(scored,key=lambda m:(m['admitted_iteration'],m['model_id']))[-c['protect_newest']: ]}
            eligible=[m for m in scored if m['model_id'] not in newest and
                      generation-m['admitted_iteration']>=c['evict_min_age_iterations']]
            if len(pool['members'])>=c['max_members'] and not eligible:outcome['reason']='all_members_protected'
            else:
                if len(pool['members'])>=c['max_members']:
                    low=min(m['q'] for m in eligible);tied=[m for m in eligible if m['q']==low]
                    def redundancy(m):
                        return min(pair_distance(m,other) for other in pool['members'] if other['model_id']!=m['model_id'])
                    victim=min(tied,key=lambda m:(redundancy(m) if len(tied)>1 else 0.,m['generation'],m['model_id']))
                    pool['members'].remove(victim);outcome['evicted']=reference(victim)
                top=max((m['q'] for m in pool['members'] if m['q'] is not None),default=0.)
                pool['members'].append(dict(entry,admitted_iteration=generation,q=top,
                    admission_health=copy.deepcopy(health),recent_rates=[]))
                outcome.update(admitted=True,reason='healthy_and_distinct')
    pool['events'].append(copy.deepcopy(outcome))
    return pool,outcome


def finish(root,run,pool,model,entry,probe,update,limit_report):
    pool=copy.deepcopy(pool);c=pool['config'];health=limit_report['health']
    pool['health_window']=(pool['health_window']+[health['ok'] is True])[-c['admit_health_iterations']:]
    distances=None;values=None;cache={}
    ready=(pool['generation']%c['admit_every_iterations']==0 and len(pool['health_window'])==c['admit_health_iterations']
           and all(pool['health_window']) and entry['model_id'] not in {m['model_id'] for m in pool['members']})
    if ready:
        values=probe.probabilities(model,update)
        cache={m['model_id']:cached(root,m,probe) for m in pool['members']}
        distances={identity:mean_kl(old,values,probe.arrays['offsets'],probe.config['probability_floor']) for identity,old in cache.items()}
    pair=lambda a,b:mean_kl(cache[a['model_id']],cache[b['model_id']],probe.arrays['offsets'],probe.config['probability_floor'])
    result,event=admission(pool,entry,health,distances,pair)
    if event['admitted']:
        member=result['members'][-1];member['probe']=cache_candidate(root,run,entry,values,probe)
        if 'fingerprint' in pool['members'][0]:
            from .critic_data import fingerprint
            member['fingerprint']=fingerprint(probe,values)
    return result,event


def finish_report(report,pool,event):
    if pfsp(pool):return pfsp_pool.report_finish(report,pool,event)
    c=pool['config']
    report.update(admission=event,next_distribution=distribution(pool),size=len(pool['members']),
        max_members=c['max_members'],eta=c['eta'],uniform_floor=c['uniform_floor'],pinned_seat_share=c['pinned_seat_share'])
    return report


def sampling_description(pool):
    if pfsp(pool) and (pool.get('planning') or {}).get('il_gate'):
        return 'Independent PFSP snapshot seats including IL, with the same uniform floor; IL checkpoint retained as evaluation/calibration anchor'
    if pfsp(pool):return 'Independent snapshot seats: stored reciprocal IL share; remainder PFSP with uniform floor'
    return 'Independent seats: 20% IL, 80% quality distribution (80% softmax + 20% uniform)'
