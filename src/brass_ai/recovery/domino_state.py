"""Float64 committed evidence and deterministic, separate six-logit Adam."""
import copy
from collections import Counter,defaultdict
import numpy as np
from brass_ai.learning.contracts import digest,require
from .domino_config import CONTRACT,mapping
from .domino_features import width,occupancy


def initial(config,generation,regime):
    return dict(contract=CONTRACT,generation=generation,origin=generation,regime=regime,
        descriptor=config['descriptor_manifest_sha256'],contexts=mapping(),centroids=[None]*7,quality=[None]*7,
        seat_games=[0]*7,distinct_games=[0]*7,logits=[0.]*6,adam=dict(step=0,m=[0.]*6,v=[0.]*6),archive=[])


def validate(s,c,generation):
    require(s['contract']==CONTRACT and s['generation']==generation and s['origin']<=generation and
        s['descriptor']==c['descriptor_manifest_sha256'] and s['contexts']==mapping(),'DOMiNO state identity/progress differs')
    for k in ('centroids','quality','seat_games','distinct_games'):require(len(s[k])==7,'Missing DOMiNO strategy state')
    for k in ('seat_games','distinct_games'):require(all(type(n) is int and n>=0 for n in s[k]),'Invalid DOMiNO support')
    require(all(n<=s['seat_games'][i] for i,n in enumerate(s['distinct_games'])),'DOMiNO game support exceeds seats')
    for x in s['centroids']:require(x is None or (np.asarray(x).shape==(4*width(),) and np.isfinite(x).all()),'Invalid centroid')
    require(all(x is None or np.isfinite(x) for x in s['quality']),'Nonfinite strategy quality')
    require(np.asarray(s['logits']).shape==(6,) and np.isfinite(s['logits']).all(),'Invalid dual logits')
    a=s['adam'];require(type(a['step']) is int and 0<=a['step']<=generation-s['origin'] and
        all(np.asarray(a[k]).shape==(6,) and np.isfinite(a[k]).all() for k in ('m','v')) and min(a['v'])>=0,'Invalid dual Adam')
    return s


def readiness(s,c):
    if any(x is None for x in s['centroids']):return 'centroids_incomplete'
    if any(n<c['minimum_seat_games_per_strategy'] for n in s['seat_games']):return 'seat_game_support'
    if any(n<c['minimum_distinct_games_per_strategy'] for n in s['distinct_games']):return 'whole_game_support'
    if any(x is None or not np.isfinite(x) for x in s['quality']):return 'quality_unavailable'
    if s['quality'][0]<=0:return 'neutral_quality_not_positive'
    return 'ready'


def directions(centroids):
    x=np.asarray(centroids,dtype=np.float64);require(x.shape[0]==7 and np.isfinite(x).all(),'Invalid centroid geometry')
    distances=((x[:,None]-x[None,:])**2).sum(-1);np.fill_diagonal(distances,np.inf)
    near=np.argmin(distances,axis=1);d=x-x[near];d[0]=0.
    return d,near,distances


def coefficients(s,c):
    if c['mode']!='apply' or readiness(s,c)!='ready':return np.ones(7,dtype=np.float64)
    m=np.asarray(s['logits']);return np.r_[1.,np.exp(-np.logaddexp(0.,-m))]


def advance(s,c,observations,generation):
    """Prepare pending state; caller publishes only with a committed checkpoint."""
    require(generation==s['generation']+1,'DOMiNO must consume each generation exactly once')
    out=copy.deepcopy(validate(s,c,generation-1));before=coefficients(s,c)
    if c['mode']=='apply' and readiness(s,c)=='ready':
        residual=np.asarray(s['quality'][1:])-c['quality_ratio']*s['quality'][0]
        g=before[1:]*(1-before[1:])*residual/6
        a=out['adam'];t=a['step']+1;m=.9*np.asarray(a['m'])+.1*g;v=.999*np.asarray(a['v'])+.001*g*g
        logits=np.asarray(out['logits'])-c['dual_learning_rate']*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
        require(np.isfinite(logits).all(),'Nonfinite dual update; committed state unchanged')
        out.update(logits=logits.tolist(),adam=dict(step=t,m=m.tolist(),v=v.tolist()))
    grouped=defaultdict(list)
    for o in observations:grouped[o['strategy']].append(o)
    for z,rows in grouped.items():
        center=np.mean([r['psi'] for r in rows],axis=0,dtype=np.float64);quality=float(np.mean([r['quality'] for r in rows]))
        require(np.isfinite(center).all() and np.isfinite(quality),'Nonfinite DOMiNO observation; committed state unchanged')
        decay=c['statistics_ema_decay']
        out['centroids'][z]=(center if s['centroids'][z] is None else decay*np.asarray(s['centroids'][z])+(1-decay)*center).tolist()
        out['quality'][z]=quality if s['quality'][z] is None else decay*s['quality'][z]+(1-decay)*quality
        out['seat_games'][z]+=len(rows);out['distinct_games'][z]+=len({r['game'] for r in rows})
    out['generation']=generation
    return validate(out,c,generation)


def regime(config,share=None):
    from .utility import reward_spec
    c=config['diversity']
    return digest(dict(reward=reward_spec(config),descriptor=c['descriptor_manifest_sha256'],contexts=mapping(),
        sampling=c['context_contract'],curriculum=config['curriculum'],v4_share=bool(share and share.get('activated'))))


def change_regime(s,c,identity):
    if identity is None or identity==s['regime']:return s
    out=copy.deepcopy(s)
    out['archive'].append(dict(regime=s['regime'],end_generation=s['generation'],
        centroids=s['centroids'],quality=s['quality'],seat_games=s['seat_games'],distinct_games=s['distinct_games']))
    out.update(regime=identity,centroids=[None]*7,quality=[None]*7,seat_games=[0]*7,distinct_games=[0]*7)
    # A new evidence regime bootstraps task-only. Keep the learned dual state;
    # historical pool turnover alone is deliberately absent from the regime key.
    return validate(out,c,s['generation'])


def prepare(ds,train_ids,s,c):
    """Whole-game ordinary learner evidence; branch actor advantages stay task-only."""
    groups=defaultdict(list);train=set(train_ids);observations=[];counters=Counter();raw_rewards=[];raw_returns=[]
    ready=readiness(s,c);coeff=coefficients(s,c)
    d,near,dist=(directions(s['centroids']) if ready=='ready' else (np.zeros((7,4*width())),None,None))
    for i,row in enumerate(ds.rows):
        a=row.get('_domino')
        if a is None or row.get('_branch') or row.get('_critic_role','learner')!='learner':continue
        groups[ds.data[i][2]].append(i)
    for (game,seat),ids in groups.items():
        z=ds.rows[ids[0]]['_domino']['strategy_id']
        require(all(ds.rows[i]['_domino']['strategy_id']==z for i in ids),'Strategy changed within trajectory')
        events=[ds.rows[i].get('_domino_event') for i in ids];psi,counts=occupancy(events)
        rewards=[float(np.dot(np.asarray(e['x'])/(4*counts[e['phase']]),d[z].reshape(4,-1)[e['phase']]))
                 if e is not None and z>0 and ready=='ready' else 0. for e in events]
        returns=np.cumsum(np.asarray(rewards)[::-1])[::-1]
        for i,target in zip(ids,returns):ds.rows[i]['_domino_target']=float(target)
        counters['complete_seat_games']+=1;counters['missing_phases']+=sum(n==0 for n in counts)
        if set(ids).isdisjoint(train):continue
        require(set(ids)<=train,'DOMiNO evidence must hold out entire learner seat-games')
        raw_rewards.extend(rewards);raw_returns.extend(returns.tolist())
        first=ds.rows[ids[0]]
        quality=first['_reward_trajectory']['utility']
        observations.append(dict(strategy=z,game=game,seat=seat,psi=psi.tolist(),quality=quality,
            table=first.get('_table_kind','selfplay'),phases=counts))
    def summary(values):
        x=np.asarray(values,dtype=np.float64)
        return dict(count=len(x),mean=float(x.mean()) if len(x) else None,rms=float(np.sqrt(np.mean(x*x))) if len(x) else None,
            minimum=float(x.min()) if len(x) else None,maximum=float(x.max()) if len(x) else None,
            nonfinite=int((~np.isfinite(x)).sum()))
    fresh={str(z):dict(task_return=summary([o['quality'] for o in observations if o['strategy']==z]),
        phase_actions=[sum(o['phases'][p] for o in observations if o['strategy']==z) for p in range(4)]) for z in range(7)}
    return observations,dict(readiness=ready,coefficients=coeff.tolist(),nearest=near.tolist() if near is not None else None,
        distances=np.where(np.isfinite(dist),dist,0.).tolist() if dist is not None else None,
        zero_nearest=sum(float(dist[i,near[i]])==0 for i in range(7)) if near is not None else None,
        tied_nearest=sum(int(np.count_nonzero(dist[i]==dist[i,near[i]])>1) for i in range(7)) if near is not None else None,
        fresh_quality=fresh,intrinsic_rewards=summary(raw_rewards),intrinsic_returns=summary(raw_returns),clipped_intrinsic=0,
        constraint_residual=[j-c['quality_ratio']*s['quality'][0] if j is not None and s['quality'][0] is not None else None for j in s['quality']],
        support=dict(counters),train_seat_games=len(observations),train_distinct_games=len({o['game'] for o in observations}),
        by_strategy=dict(Counter(str(o['strategy']) for o in observations)),
        by_table_strategy=dict(Counter(o['table']+'/'+str(o['strategy']) for o in observations)),
        scope='Complete ordinary learner training games only; missing terminal phases zero without renormalization; no heldouts/snapshots/branches')
