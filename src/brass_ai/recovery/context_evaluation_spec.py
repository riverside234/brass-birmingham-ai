"""Outcome-blind setup selection and exact, rotated context job identities."""
import copy
from collections import Counter, defaultdict
from brass_ai.bank import sha
from brass_ai.learning.contracts import digest, project_path, require
from brass_ai.learning.reward import specification
from brass_ai.step5.league import implementation_id
from .context_evaluation_config import CONTRACT
from .domino_config import mapping, NAMES
from .domino_features import manifest
from .plackett_luce import IL, V4

SELECTION='matched-balanced-setups/1.0.0'


def runtime_receipt(root, host):
    from .host_runtime import training_host
    path=project_path(root,host)
    if path==root/'artifacts/r1-host/guarded':path=training_host(root)
    return dict(path=path.relative_to(root).as_posix(),
        runtime_sha256=sha(path/'runtime.json'),host_sha256=sha(path/'BrassHost.exe'))


def select(parent, count=16):
    groups=defaultdict(list)
    for i,r in enumerate(parent['lineups']):
        if r['kind']=='rating':groups[r['seed']].append((i,r))
    require(len(groups)>=count, 'Insufficient main setups for context panel')
    candidates=[]
    for seed,rows in groups.items():
        base=rows[0][1]['agents']
        require(len(rows)==4 and {r['seat'] for _,r in rows}==set(range(4)) and
            all(r['agents']==base for _,r in rows) and base[0]==parent['candidate'] and
            base.count(parent['candidate'])==1,'Context panel requires one candidate and four exact rotations')
        # The second seat is the declared reference under both existing layouts.
        reference=base[1]
        if parent.get('coverage',{}).get('active'):
            ordered_seeds=list(groups)
            reference=parent['coverage']['reference_setup_indices'].get(str(ordered_seeds.index(seed)))
        stratum='il' if reference==IL else 'v4' if reference==V4 else 'competitive'
        candidates.append(dict(seed=seed,stratum=stratum,agents=base,
            indices=[i for i,r in sorted(rows,key=lambda p:p[1]['seat'])]))
    competitive=bool(parent.get('coverage',{}).get('active'))
    available=Counter(r['stratum'] for r in candidates)
    if competitive and count==16:
        quotas=dict(il=2,v4=2,competitive=12)
        require(all(available[k]>=n for k,n in quotas.items()), 'Competitive context strata unavailable')
    else:
        # Proportional largest-deficit allocation, including explicit startup.
        quotas=Counter()
        for _ in range(count):
            k=max((k for k in sorted(available) if quotas[k]<available[k]),
                  key=lambda k: (available[k]*count/len(candidates)-quotas[k],k))
            quotas[k]+=1
    coverage=parent.get('coverage',{})
    preferred=list(dict.fromkeys(a for a in (coverage.get('previous_rated_checkpoint'),
        coverage.get('active_kl_reference')) if a and a!=parent['candidate']))
    counts=Counter();selected=[]
    for k,n in sorted(quotas.items()):
        pool=[r for r in candidates if r['stratum']==k]
        for _ in range(n):
            def key(r):
                mids=[parent['agents'][a]['checkpoint']['model_id'] if parent['agents'][a]['kind']=='model'
                      else a for a in r['agents'][1:]]
                uncovered=sum(a in r['agents'][1:] and not any(a in s['agents'][1:] for s in selected) for a in preferred)
                return (-uncovered,sum((counts[a]+1)**2-counts[a]**2 for a in mids),
                    digest(dict(contract=SELECTION,seed=parent['config']['seed'],generation=parent['generation'],setup=r['seed'])))
            row=min(pool,key=key);pool.remove(row);selected.append(row)
            for a in row['agents'][1:]:
                ref=parent['agents'][a];counts[ref['checkpoint']['model_id'] if ref['kind']=='model' else a]+=1
    selected.sort(key=lambda r:digest(dict(contract=SELECTION,seed=parent['config']['seed'],setup=r['seed'],generation=parent['generation'])))
    appearances=Counter(a for r in selected for a in r['agents'][1:])
    return dict(contract=SELECTION,setups=selected,strata=dict(quotas),
        inherited_startup_fallback=bool(parent.get('startup_reference_fallback')),
        opponent_setup_appearances=dict(appearances),model_setup_appearances=dict(counts),
        preferred_appearances={a:appearances[a] for a in preferred},
        preferred_shortage=[a for a in preferred if not appearances[a]])


def specification_for(root,parent,config,*,test_setups=None):
    c=copy.deepcopy(config['context_evaluation']);selection=select(parent,test_setups or c['setups'])
    execution=copy.deepcopy(parent['execution']);execution['feature_cache']=False
    return dict(contract=CONTRACT,generation=parent['generation'],config=c,parent=copy.deepcopy(parent),
        parent_digest=digest(parent),selection=selection,contexts=mapping(),
        reward=specification(config['reward'],config['insolvency']),quality_ratio=config['diversity']['quality_ratio'],
        descriptor=manifest(),descriptor_sha256=digest(manifest()),
        runtime=runtime_receipt(root,parent['host']),implementation=parent['implementation'],
        execution=execution,new_games=6*len(selection['setups'])*4,reused_games=len(selection['setups'])*4,
        test_only=test_setups is not None,policy=dict(temperature=1.,exploration=0.,sampling='probabilistic',
            context='candidate-only-fixed-entire-game',opponents='unchanged-neutral-frozen',rng_index=0),
        bootstrap=dict(method='percentile',pairing='setup-cluster-all-four-rotations',confidence=.95),
        scope='Report only; not PL, training, deployment qualification or a causal DOMiNO trial')


def jobs_for(spec,directory):
    from .ratings import jobs_for as main_jobs
    from .execution import neural_spec
    base=main_jobs(spec['parent'],directory)
    jobs=[];panel_digest=digest(spec)
    for setup in spec['selection']['setups']:
        for i in setup['indices']:
            # Interleave contexts; all policies route by the same model identity.
            for z,name in enumerate(NAMES[1:],1):
                job=copy.deepcopy(base[i]);job['index']=len(jobs)
                old=job['policies'][0]
                require(old['kind']=='neural' and old['focus']=='neutral' and old['exploration']==0. and
                    old.get('temperature',1.)==1. and not old.get('opening'),'Main candidate policy incompatible')
                job['policies'][0]=neural_spec(old['model_id'],0.,name,temperature=1.)
                job['path']=str(directory/f'game-{job["index"]:06d}.jsonl.gz')
                job['rng_index']=0
                job['stable_game_id']=digest(dict(panel=panel_digest,parent_index=i,context=z))
                job['provenance']=dict(job['provenance'],producer=CONTRACT,kind='context-panel',
                    spec=panel_digest,parent_spec=spec['parent_digest'],parent_index=i,context_id=z)
                jobs.append(job)
    require(len(jobs)==spec['new_games'],'Context job count differs')
    return jobs


def validate_runtime(root,spec):
    require(spec['implementation']==implementation_id(), 'Pending context panel source changed; cannot mix sources')
    require(spec['parent_digest']==digest(spec['parent']) and spec['contexts']==mapping() and
        spec['descriptor_sha256']==digest(manifest()),'Pending context panel contracts changed')
    require(runtime_receipt(root,spec['parent']['host'])==spec['runtime'], 'Context panel runtime changed')
    for ref in spec['parent']['agents'].values():
        if ref['kind']=='model':
            e=ref['checkpoint'];require(sha(project_path(root,e['path']))==e['sha256'],'Context checkpoint changed')
