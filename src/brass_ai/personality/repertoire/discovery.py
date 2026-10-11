"""Bounded controller proposals and matched, seat-rotated discovery tables."""
from pathlib import Path
import random

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest,project_path,read_json,require
from brass_ai.learning.batching import model_id
from brass_ai.personality.contracts import PROFILES
from brass_ai.personality.search.contracts import Profile,controller,search_config
from brass_ai.personality.search.readiness import load as load_ready
from brass_ai.personality.search.collection import collect
from brass_ai.recovery.retention import owned
from brass_ai.step5.league import write_json,seeds_for
from .contracts import identity

VERSION='personality-discovery/1.0.0'


def proposals(model,parent,pool,config,*,count=64,seed=17701,decision_mode='direct'):
    require(2<=count<=4096,'Use 2..4096 candidate controllers')
    rng=random.Random(seed)
    profiles=[Profile(style='neutral',alpha=0.)]
    # Supported online profile/context inputs only. Encoder coordinates are NOT
    # fed back as untrained continuous neural style inputs.
    profiles.extend(Profile(style=s,alpha=.15 if decision_mode=='searched' else 0.) for s in PROFILES if s!='neutral')
    known={digest(p.to_dict()) for p in profiles}
    while len(profiles)<count:
        p=Profile(style=rng.choice(PROFILES),alpha=rng.choice((0.,.15,.35)) if decision_mode=='searched' else 0.,patience=rng.randint(1,3),
            hysteresis=round(rng.uniform(.025,.2),6))
        key=digest(p.to_dict())
        if key not in known:profiles.append(p);known.add(key)
    return [controller(model,parent,p,config,pool,decision_mode=decision_mode) for p in profiles[:count]]


def create(root,bundle,output,*,count=64,seed=17701,seconds=12.,workers=4,decision_mode='direct',from_archive=None):
    root,out=owned(root,output);require(not out.exists(),'Use a new immutable proposal directory')
    _,parents,ready,_,pool=load_ready(root,bundle)
    config=search_config(ready['reward'],seconds=seconds,workers=workers)
    model=ready['model']['model_id'];parent=model_id(parents[0]);ancestors=[];ancestry={};source_archive=None
    candidates=proposals(model,parent,pool,config,count=count,seed=seed,decision_mode=decision_mode)
    if from_archive is not None:
        from .archive import load as load_archive
        source_archive=project_path(root,from_archive);a=load_archive(source_archive);old=load(project_path(root,a['proposal_path']))
        require(old['ready_id']==ready['id'] and old['pool_id']==pool,'New model/pool requires fresh discovery, not elite mutation')
        same=old['candidates'][0]['decision_mode']==decision_mode and old['candidates'][0]['search']==config.to_dict()
        if same:
            require(count>=len(old['candidates']),'An archive extension cannot discard retained candidates')
            candidates=list(old['candidates']);ancestors=list(old.get('ancestors',[]))+[dict(path=a['proposal_path'],id=old['id'],sha256=sha(project_path(root,a['proposal_path'])/'manifest.json'))]
            ancestry=dict(old.get('ancestry',{}));rng=random.Random(seed)
            parents_to_mutate=[e['controller'] for e in a['entries'] if e['candidate_id'] in a['elites']]
            require(parents_to_mutate,'No qualified elite to mutate; collect more evidence first')
            known={c['candidate_id'] for c in candidates}
            while len(candidates)<count:
                base=rng.choice(parents_to_mutate);profile=Profile.parse(base['profile'])
                new=Profile(style=profile.style if rng.random()<.8 else rng.choice(PROFILES),
                    alpha=rng.choice((0.,.15,.35)) if decision_mode=='searched' else 0.,
                    patience=max(1,min(3,profile.patience+rng.choice((-1,0,1)))),
                    hysteresis=max(0.,min(.2,round(profile.hysteresis+rng.uniform(-.03,.03),6))))
                c=controller(model,parent,new,config,pool,decision_mode=decision_mode)
                if c['candidate_id'] not in known:
                    candidates.append(c);known.add(c['candidate_id']);ancestry[c['candidate_id']]=base['candidate_id']
        else:
            # Search confirmation is a NEW experiment. Transfer tested profile
            # recipes; never transfer their direct-policy coordinates/quality.
            selected=[e['controller'] for e in a['entries'] if e['candidate_id'] in a['shortlist']['members']]
            require(selected,'No tested shortlist to confirm')
            neutral=controller(model,parent,Profile(alpha=0.),config,pool,decision_mode=decision_mode);candidates=[neutral]
            for base in selected:
                profile=dict(base['profile']);profile['alpha']=.15 if decision_mode=='searched' and profile['style']!='neutral' else 0.
                c=controller(model,parent,Profile.parse(profile),config,pool,decision_mode=decision_mode)
                if c['candidate_id'] not in {x['candidate_id'] for x in candidates}:candidates.append(c)
                ancestry[c['candidate_id']]=base['candidate_id']
    result=dict(format=VERSION,contract=identity(),ready=project_path(root,bundle).relative_to(root).as_posix(),
        ready_sha256=sha(project_path(root,bundle)/'ready.json'),ready_id=ready['id'],pool_id=pool,
        seed=seed,control_id=candidates[0]['candidate_id'],candidates=candidates,ancestors=ancestors,ancestry=ancestry,
        source_archive=source_archive.relative_to(root).as_posix() if source_archive else None)
    result['id']=digest(result);out.mkdir(parents=True);write_json(out/'manifest.json',result)
    write_json(out/'pins.json',dict(paths=[result['ready']+'/ready.json']+[x['path']+'/manifest.json' for x in ancestors]+
        ([result['source_archive']+'/manifest.json'] if source_archive else [])))
    return result


def load(directory):
    value=read_json(Path(directory)/'manifest.json')
    require(value['format']==VERSION and value['contract']==identity()
        and value['id']==digest({k:v for k,v in value.items() if k!='id'}),'Discovery proposal source/manifest differs')
    require(len({c['candidate_id'] for c in value['candidates']})==len(value['candidates']),'Duplicate controllers')
    for c in value['candidates']:
        require(c['candidate_id']==digest({k:v for k,v in c.items() if k!='candidate_id'}),'Controller identity differs')
        Profile.parse(c['profile'])
    require(value['control_id'] in {c['candidate_id'] for c in value['candidates']},'Neutral control missing')
    return value


def jobs_for(proposal,seeds,partition,pool_size):
    require(partition in ('train','validation'),'Final-test setups cannot tune discovery')
    jobs=[]
    for seed in seeds:
        for c in proposal['candidates']:
            for seat in range(4):
                profiles=[Profile(style='neutral',alpha=0.).to_dict() for _ in range(4)];profiles[seat]=c['profile']
                # Rotate the same three frozen references around every tested
                # seat. Candidate ID never changes setup, sampling or opponents.
                refs=[0]*4
                for offset in range(1,4):refs[(seat+offset)%4]=(seed+offset-1)%pool_size
                jobs.append(dict(index=len(jobs),seed=seed,sampling_seed=seed*17+seat,partition=partition,
                    neural_seats=[seat],profiles=profiles,controllers=[c['candidate_id']]*4,parent_indices=refs))
    require(len(jobs)<=4096,'Split this pilot into collections of at most 4096 games')
    return jobs


def execute(root,proposal,output,*,setups=2,partition='train',seed=83000000,tables=24,device='cpu',mono=None):
    root=Path(root).resolve();proposal=project_path(root,proposal);p=load(proposal)
    ready_path=project_path(root,p['ready']);require(sha(ready_path/'ready.json')==p['ready_sha256'],'Frozen ready model changed')
    _,parents,ready,_,pool=load_ready(root,ready_path)
    require(ready['id']==p['ready_id'] and pool==p['pool_id'],'Discovery pool/readiness changed')
    require(1<=setups<=128,'Use a bounded repeated-setup pilot')
    seeds,_=seeds_for(seed,setups,partition)
    from brass_ai.search.common import SearchConfig
    cfg=SearchConfig(**p['candidates'][0]['search']).validate()
    result=collect(root,ready_path,output,jobs_for(p,seeds,partition,len(parents)),cfg,tables=tables,device=device,mono=mono,
        decision_mode=p['candidates'][0]['decision_mode'])
    receipt=dict(format=VERSION,contract=identity(),proposal=proposal.relative_to(root).as_posix(),
        proposal_sha256=sha(proposal/'manifest.json'),proposal_id=p['id'],collection_id=result['id'],partition=partition,seeds=seeds)
    receipt['id']=digest(receipt);out=project_path(root,output);write_json(out/'discovery.json',receipt)
    pins=read_json(out/'pins.json');pins['paths'].append(proposal.relative_to(root).as_posix()+'/manifest.json');write_json(out/'pins.json',pins)
    return dict(games=len(result['records']),seconds=result['seconds'],collection_id=result['id'],receipt=receipt)
