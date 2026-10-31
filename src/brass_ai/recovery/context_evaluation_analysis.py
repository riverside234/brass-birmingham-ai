"""Paired setup statistics on compact arrays. Never fits PL or reads weights."""
from collections import Counter
import numpy as np
from brass_ai.learning.contracts import digest, require
from .context_evaluation_config import CONTRACT
from .domino_config import NAMES
from .domino_features import manifest, width, PHASES

PROJECTION='occupancy-without-investment-and-realization/1.0.0'
METRICS=('win_share','vp','placement','outrank_share','task_return','insolvent','zero_vp')


def projection_indices():
    keep=[];offset=0
    for g in manifest()['groups']:
        if g['name'] not in ('investment','realization'):keep.extend(range(offset,offset+len(g['fields'])))
        offset+=len(g['fields'])
    return [p*width()+i for p in range(4) for i in keep]


def bootstrap_indices(strata,samples,seed):
    rng=np.random.default_rng(seed);parts=[]
    for k in sorted(set(strata)):
        ids=np.flatnonzero(np.asarray(strata)==k)
        parts.append(rng.choice(ids,size=(samples,len(ids)),replace=True))
    return np.concatenate(parts,axis=1)


def distribution(x):
    centroids=x.mean(axis=(1,2))
    distances=np.linalg.norm(centroids[:,None]-centroids[None,:],axis=-1)
    within=np.sqrt(np.square(x-centroids[:,None,None]).sum(axis=-1).mean(axis=(1,2)))
    return dict(centroids=centroids.tolist(),distance_matrix=distances.tolist(),
        separation=float(np.square(distances[np.triu_indices(7,1)]).mean()),
        within_context_rms=within.tolist(),mean_within_context_rms=float(within.mean()),
        statistic='Mean squared distance over the 21 pairs of context centroids',
        variation='RMS game distance from its context centroid; four rotations remain clustered for inference')


def permutation_null(x,indices,samples,seed):
    rng=np.random.default_rng(seed);setup=x.mean(axis=2) # 7 x setups x features; average rotations FIRST
    full=[];ablated=[];ss=np.arange(setup.shape[1])
    for _ in range(samples):
        permutation=np.asarray([rng.permutation(7) for __ in ss]).T
        center=setup[permutation,ss[None,:]].mean(axis=1)
        delta=center[:,None]-center[None,:]
        full.append(float(np.square(delta).sum(-1)[np.triu_indices(7,1)].mean()))
        ablated.append(float(np.square(delta[...,indices]).sum(-1)[np.triu_indices(7,1)].mean()))
    return np.asarray(full),np.asarray(ablated)


def analyze(spec,rows):
    setups=spec['selection']['setups'];n=len(setups);by={(r['context_id'],r['setup_seed'],r['seat']):r for r in rows}
    expected={(z,s['seed'],seat) for z in range(7) for s in setups for seat in range(4)}
    require(len(rows)==len(by)==7*n*4 and set(by)==expected,'Incomplete/duplicate context support; no full-panel comparison')
    ordered=[[[by[z,s['seed'],seat] for seat in range(4)] for s in setups] for z in range(7)]
    require(all(r['panel_digest']==digest(spec) for r in rows),'Mixed panel evidence')
    a=np.asarray([[[[r[k] for k in METRICS] for r in rotations] for rotations in group] for group in ordered],dtype=float)
    x=np.asarray([[[r['descriptor'] for r in rotations] for rotations in group] for group in ordered],dtype=float)
    require(x.shape==(7,n,4,4*width()) and np.isfinite(x).all() and np.isfinite(a).all(),'Invalid panel arrays')
    seed=int(digest(dict(contract=CONTRACT,generation=spec['generation'],seed=spec['parent']['config']['seed']))[:16],16)
    indices=bootstrap_indices([s['stratum'] for s in setups],spec['config']['bootstrap_samples'],seed)
    # Mixture is derived INSIDE each paired block and resample, retaining covariance.
    values=a.mean(axis=2);values=np.concatenate([values,values.mean(axis=0,keepdims=True)],axis=0)
    draws=values[:,indices,:].mean(axis=2);point=values.mean(axis=1)
    contexts=[];warnings=[]
    def interval(data):return np.quantile(data,[.025,.975]).tolist()
    for z,name in enumerate((*NAMES,'uniform_mixture')):
        means={k:float(point[z,j]) for j,k in enumerate(METRICS)}
        cis={k:interval(draws[z,:,j]) for j,k in enumerate(METRICS)}
        paired={k:dict(mean=float(point[z,METRICS.index(k)]-point[0,METRICS.index(k)]),
            ci95=interval(draws[z,:,METRICS.index(k)]-draws[0,:,METRICS.index(k)])) for k in ('win_share','vp','task_return')}
        if z==0:paired=None # neutral is a reference, not a zero-width estimate
        task=METRICS.index('task_return');margin=point[z,task]-spec['quality_ratio']*point[0,task]
        actual=[r for rotations in ordered[z] for r in rotations] if z<7 else rows
        counts={k:sum(r[k] for r in actual) for k in ('insolvent','insolvency_events','zero_vp','passes','main_actions')}
        action=Counter();industries={'build':Counter(),'develop':Counter()}
        for r in actual:
            action.update(r['actions'])
            for k in industries:industries[k].update(r['industry'][k])
        denominators={k:sum(v.values()) for k,v in industries.items()}
        data=dict(context=name,context_id=z if z<7 else None,means=means,ci95=cis,paired=paired,
            task_quality_margin=float(margin),task_quality_margin_ci95=interval(draws[z,:,task]-spec['quality_ratio']*draws[0,:,task]),
            task_return_ratio=float(point[z,task]/point[0,task]) if point[0,task]>0 and n>=16 and np.quantile(draws[0,:,task],.025)>0 else None,
            setups=n,games=n*4 if z<7 else None,counts=counts if z<7 else None,
            event_rate=float(counts['insolvency_events']/len(actual)),
            pass_rate=float(np.mean([r['passes']/r['main_actions'] for r in actual])),
            pass_rate_scope='Equal-game mean, then equal-context mixture; main action denominators recorded',
            action_counts=dict(action) if z<7 else None,
            action_frequencies={k:float(np.mean([r['actions'].get(k,0)/r['main_actions'] for r in actual])) for k in features_actions()},
            industry_counts={k:dict(v) for k,v in industries.items()} if z<7 else None,
            industry_denominators=denominators if z<7 else None,
            industry_share_scope='Pooled selected tiles, including second develop selections on continuation rows; '
                'mixture pools selections from equally many games per context',
            industry_shares={k:{name:count/denominators[k] for name,count in v.items()} for k,v in industries.items()},
            derived=z==7,support_scope='Same '+str(n)+' setup clusters; mixture adds no games' if z==7 else 'Single candidate seat per game')
        contexts.append(data)
        if z<7 and counts['insolvent']==0:warnings.append(name+': zero insolvencies observed; not evidence of zero risk')
        if any(lo==hi for lo,hi in cis.values()):warnings.append(name+': degenerate empirical interval(s)')
    keep=projection_indices();full=distribution(x);ablated=distribution(x[...,keep])
    nulls=permutation_null(x,keep,spec['config']['permutation_samples'],seed+1)
    for result,null in zip((full,ablated),nulls):
        result['null']=dict(mean=float(null.mean()),ci95=interval(null),
            p_exploratory=float((1+sum(null>=result['separation']-1e-15))/(len(null)+1)),
            samples=len(null),seed=seed+1,method='Independent label permutation within setup, shared across all four rotations')
    ablated.update(projection=PROJECTION,removed_families=['investment','realization'],retained_coordinates=keep,
        scaling='Original coordinate scales retained; spatial features can still reflect industry')
    phase=x.reshape(7,n,4,4,width()).mean(axis=(1,2))
    family={};offset=0
    for g in manifest()['groups']:
        size=len(g['fields'])
        family[g['name']]=dict(fields=g['fields'],phase_means=phase[:,:,offset:offset+size].tolist(),
            scope='Manifest-normalized occupancy; includes fixed phase weight 1/4, missing phases zero',
            scales=g.get('scales',g.get('scale')))
        offset+=size
    representatives=[]
    for z in range(7):
        center=x[z].mean(axis=(0,1));actual=[r for rr in ordered[z] for r in rr]
        chosen=min(actual,key=lambda r:(float(np.square(np.asarray(r['descriptor'])-center).sum()),r['setup_seed'],r['seat'],r['game_id']))
        representatives.append(dict(context=NAMES[z],game_id=chosen['game_id'],setup_seed=chosen['setup_seed'],seat=chosen['seat'],
            replay=chosen['replay'],opponents=chosen['opponents'],trace=chosen['trace'],
            scope='Illustrative trajectory nearest this held-out context centroid; not selected for winning'))
    return dict(contract=CONTRACT,status='complete',generation=spec['generation'],panel_digest=digest(spec),
        protocol_id=digest(dict(contract=CONTRACT,descriptor=spec['descriptor_sha256'],reward=spec['reward'],contexts=spec['contexts'],config=spec['config'])),
        test_only=spec['test_only'],support=dict(setups=n,rotations=4,games_per_context=n*4,new_games=6*n*4,reused_games=n*4,total_games=7*n*4,
            per_context={name:n*4 for name in NAMES},per_setup={str(s['seed']):28 for s in setups}),
        contexts=contexts,quality_ratio=spec['quality_ratio'],reward=spec['reward'],
        mixture_label='Uniform candidate context mixture versus neutral-context frozen opponents',
        descriptors=dict(contract=spec['descriptor_sha256'],phases=list(PHASES),full=full,without_industry=ablated,
            heatmap_limits=[0.,max(np.max(full['distance_matrix']),np.max(ablated['distance_matrix']))],families=family,
            missing_phase_games={name:{p:sum(r['phase_counts'][i]==0 for rr in ordered[z] for r in rr) for i,p in enumerate(PHASES)} for z,name in enumerate(NAMES)}),
        main_action_counts_by_phase={name:{p:sum(r['phase_counts'][i] for rr in ordered[z] for r in rr) for i,p in enumerate(PHASES)} for z,name in enumerate(NAMES)},
        statistics=dict(method='percentile',confidence=.95,bootstrap_samples=len(indices),seed=seed,
            bootstrap_indices_sha256=digest(indices.tolist()),strata=spec['selection']['strata'],independent_clusters=n,
            interval_scope='Marginal exploratory intervals, not simultaneous certification; no PL refit'),
        representatives=representatives,warnings=warnings+(['Reduced test-only support; statistically insufficient'] if n<16 else [])+
            ['Small IL/v4 strata do not support separate context-specific qualification.',
             'No causal diversity or playing-strength claim; changing checkpoints/opponents are not pooled.'],
        selection=spec['selection'],policy=spec['policy'])


def features_actions():
    from .domino_features import ACTIONS
    return ACTIONS


def markdown(report):
    s=report['support'];lines=[f'# Context evaluation — g{report["generation"]}', '',
        'Status: '+report['status']+(' (test-only, insufficient support)' if report['test_only'] else ''), '',
        f'{s["setups"]} paired setup clusters × four rotations; {s["new_games"]} new + {s["reused_games"]} reused games.', '',
        '| Context | Win share | VP | Paired VP difference [95%] | Task-quality margin | Insolvent seats |',
        '|---|---:|---:|---|---:|---|']
    for r in report['contexts']:
        d=r['paired'];difference='Reference' if d is None else f'{d["vp"]["mean"]:+.3f} [{d["vp"]["ci95"][0]:+.3f}, {d["vp"]["ci95"][1]:+.3f}]'
        affected=f'{r["counts"]["insolvent"]}/{r["games"]}' if r['counts'] is not None else f'{r["means"]["insolvent"]:.2%} (derived rate)'
        lines.append(f'| {r["context"]} | {r["means"]["win_share"]:.2%} | {r["means"]["vp"]:.3f} | {difference} | {r["task_quality_margin"]:+.4f} | {affected} |')
    lines+=['',report['mixture_label']+'.', '',
        'Task return uses the recorded reward contract and subtracts capped insolvency once. The 0.95 task-quality ratio is not a win-rate target.', '',
        'Four rotations are averaged within setups. The same stratified setup-bootstrap indices are used for every context and the derived mixture. Percentile 95% intervals are marginal. Zero events and zero-width empirical intervals do not establish zero risk.', '',
        'Separation is the mean squared Euclidean distance of the 21 centroid pairs. The ablation removes both investment and realization, retaining original coordinate scales. Its 1,000 label permutations preserve all four rotations of a setup. This is an exploratory exchangeability diagnostic, not a D5 qualification.', '',
        'The representative traces are selected by distance to the held-out context centroid, with deterministic ties. They are illustrative, not selected winners.', '',
        'Timing, execution receipts, missing-phase counts, exact interval endpoints and family denominators are recorded in report.json.', '',
        *['- '+w for w in report['warnings']], '']
    return '\n'.join(lines)
