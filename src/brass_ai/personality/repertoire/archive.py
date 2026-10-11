"""Immutable, uncertainty-aware unstructured archive; no synthetic controllers."""
from dataclasses import asdict,dataclass
from pathlib import Path
import numpy as np

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest,project_path,read_json,require
from brass_ai.recovery.retention import owned
from brass_ai.step5.league import write_json
from . import descriptor,discovery
from .contracts import identity
from .traces import from_collection

VERSION='personality-behavior-archive/1.0.0'


@dataclass(frozen=True)
class Gates:
    min_setups:int=8
    min_discovery_setups:int=2
    win_floor:float=.20
    control_tolerance:float=.05
    max_dispersion:float=3.
    novelty_distance:float=.5
    capacity:int=128

    def validate(self):
        require(self.min_setups>=2 and self.min_discovery_setups>=2 and 0<=self.win_floor<=1
            and 0<=self.control_tolerance<=.1 and self.max_dispersion>0
            and self.novelty_distance>0 and 1<=self.capacity<=4096,'Invalid discovery gates')
        require(all(np.isfinite(v) for v in asdict(self).values()),'Nonfinite archive gates')
        return self


def grouped(rows):
    groups={}
    for r in rows:
        m=r['metadata'];groups.setdefault(m['setup_group'],{})
        require(m['seat'] not in groups[m['setup_group']],'Duplicate candidate/setup/seat evidence')
        groups[m['setup_group']][m['seat']]=m['quality']
    # One lucky seat/game cannot qualify an entry.
    return {s:np.mean([v[i]['win_share'] for i in range(4)]) for s,v in groups.items() if set(v)==set(range(4))}


def evidence(rows,control,gates):
    a,b=grouped(rows),grouped(control);shared=sorted(a.keys()&b.keys())
    wins=np.array([a[k] for k in shared]);delta=np.array([a[k]-b[k] for k in shared])
    n=len(shared)
    # Setup-clustered normal interval is a preliminary screen, never a release
    # certificate. Report n and raw differences for later stronger evaluation.
    error=float(1.96*delta.std(ddof=1)/np.sqrt(n)) if n>1 else 1.
    mean=float(wins.mean()) if n else 0.;difference=float(delta.mean()) if n else 0.
    return dict(setups=n,games=4*n,mean_win=mean,control_delta=difference,control_lower=difference-error,
        qualified=n>=gates.min_setups and mean>=gates.win_floor and difference-error>=-gates.control_tolerance,
        mean_vp=float(np.mean([r['metadata']['quality']['vp'] for r in rows])) if rows else 0.,
        mean_rank=float(np.mean([r['metadata']['quality']['placement'] for r in rows])) if rows else 0.,
        setup_deltas=delta.tolist(),scope='Preliminary P3 screen; final roster qualification remains P4/P5')


def select_elites(entries,gates):
    eligible=sorted((e for e in entries if e['qualified']),key=lambda e:(-e['quality']['control_lower'],-e['quality']['mean_win'],e['candidate_id']))
    elites=[]
    for entry in eligible:
        point=np.asarray(entry['centroid'])
        if all(np.linalg.norm(point-np.asarray(e['centroid']))>=gates.novelty_distance for e in elites):elites.append(entry)
        if len(elites)==gates.capacity:break
    return [e['candidate_id'] for e in elites]


def shortlist(entries,elite_ids,*,count=16,method='fps',seed=18193):
    require(1<=count<=128 and method in ('fps','kmeans'),'Invalid shortlist request')
    rows=sorted([e for e in entries if e['candidate_id'] in elite_ids and e['qualified']],key=lambda e:e['candidate_id'])
    if not rows:return []
    k=min(count,len(rows));x=np.asarray([e['centroid'] for e in rows]);rng=np.random.default_rng(seed)
    def fps():
        chosen=[max(range(len(rows)),key=lambda i:rows[i]['quality']['mean_win'])]
        while len(chosen)<k:
            distance=np.min(np.square(x[:,None]-x[chosen]).sum(-1),axis=1);distance[chosen]=-1
            chosen.append(int(np.argmax(distance)))
        return chosen
    if method=='fps':selected=fps()
    else:
        best=None
        for restart in range(8):
            centers=x[fps()] if restart==0 else x[rng.choice(len(rows),k,replace=False)]
            for _ in range(30):
                label=np.argmin(np.square(x[:,None]-centers).sum(-1),axis=1)
                new=np.array([x[label==j].mean(0) if (label==j).any() else centers[j] for j in range(k)])
                if np.allclose(centers,new):break
                centers=new
            cost=float(np.min(np.square(x[:,None]-centers).sum(-1),axis=1).sum())
            if best is None or cost<best[0]:best=(cost,centers)
        selected=[]
        for center in best[1]:
            distances=np.square(x-center).sum(-1);distances[selected]=np.inf
            selected.append(int(np.argmin(distances)))
    # Use actual tested members nearest the centroids, never the centroids as z.
    return [rows[i]['candidate_id'] for i in selected]


def load(directory):
    a=read_json(Path(directory)/'manifest.json')
    require(a['format']==VERSION and a['contract']==identity()
        and a['id']==digest({k:v for k,v in a.items() if k!='id'}),'Archive source/manifest differs')
    return a


def build(root,proposal,encoder,discovery_collections,qualification_collections,output,*,previous=None,gates=None,method='fps',count=16,device='cpu'):
    root,out=owned(root,output);require(not out.exists(),'Archive snapshots are immutable; choose a new directory')
    gates=(gates or Gates()).validate();proposal=project_path(root,proposal);p=discovery.load(proposal)
    encoder=project_path(root,encoder);enc,payload=descriptor.load(encoder)
    require(not enc['report']['collapsed'],'Collapsed learned descriptor cannot create an archive')
    old=load(project_path(root,previous)) if previous else None
    accepted_proposals={p['id']:sha(proposal/'manifest.json')}
    for ancestor in p.get('ancestors',[]):
        path=project_path(root,ancestor['path']);require(sha(path/'manifest.json')==ancestor['sha256'] and discovery.load(path)['id']==ancestor['id'],'Proposal ancestry changed')
        accepted_proposals[ancestor['id']]=ancestor['sha256']
    if old:
        require(old['proposal_id'] in accepted_proposals and old['pool_id']==p['pool_id'],'New model/pool/search settings require a separately requalified archive')
        for source in old['sources']:
            directory=project_path(root,source['path'])
            require(sha(directory/'manifest.json')==source['manifest_sha256']
                and sha(directory/'discovery.json')==source['receipt_sha256'],
                'Retained archive evidence changed; do not silently replace frozen sources')
        # A refit never keeps stale coordinates. Reload every retained source,
        # merge new evidence, then re-encode every retained controller below.
        discovery_collections=list(discovery_collections)+[s['path'] for s in old['sources'] if s['role']=='discovery']
        qualification_collections=list(qualification_collections)+[s['path'] for s in old['sources'] if s['role']=='qualification']
    sources=[];groups={};by_role={};seen=set()
    for role,directories in (('discovery',discovery_collections),('qualification',qualification_collections)):
        records=[]
        for directory in dict.fromkeys(str(project_path(root,d)) for d in directories):
            directory=Path(directory);receipt=read_json(directory/'discovery.json')
            require(receipt['id']==digest({k:v for k,v in receipt.items() if k!='id'}) and receipt['proposal_id'] in accepted_proposals
                and receipt['proposal_sha256']==accepted_proposals[receipt['proposal_id']],'Collection proposals differ')
            manifest,part=from_collection(directory)
            require(receipt['collection_id']==manifest['id'] and manifest['spec']['pool_id']==p['pool_id'],'Collection pool differs')
            require(all(r['metadata']['split']==('train' if role=='discovery' else 'validation') for r in part),'Archive split leakage')
            require(not any(r['metadata']['trace_id'] in seen for r in part),'Duplicate archive traces')
            seen.update(r['metadata']['trace_id'] for r in part);records.extend(part)
            sources.append(dict(path=directory.relative_to(root).as_posix(),role=role,manifest_sha256=sha(directory/'manifest.json'),receipt_sha256=sha(directory/'discovery.json')))
        by_role[role]=records;groups[role]={r['metadata']['setup_group'] for r in records}
    require(by_role['discovery'] and by_role['qualification'],'Need discovery and separate qualification evidence')
    require(not (groups['qualification'] & (groups['discovery']|set(enc['training_groups']))),'Qualification setup leaked into encoder fitting')
    candidate_ids={c['candidate_id'] for c in p['candidates']}
    require(all(r['metadata']['candidate_id'] in candidate_ids for r in by_role['discovery']+by_role['qualification']),'Unknown tested controller')
    z=descriptor.transform(payload,by_role['discovery'],device=device);entries=[]
    validation=by_role['qualification'];control=[r for r in validation if r['metadata']['candidate_id']==p['control_id']]
    for c in p['candidates']:
        ix=[i for i,r in enumerate(by_role['discovery']) if r['metadata']['candidate_id']==c['candidate_id']]
        if not ix:continue
        points=z[ix];centroid=points.mean(0);dispersion=float(np.sqrt(np.square(points-centroid).sum(-1).mean()))
        discovery_groups={by_role['discovery'][i]['metadata']['setup_group'] for i in ix}
        quality=evidence([r for r in validation if r['metadata']['candidate_id']==c['candidate_id']],control,gates)
        qualified=bool(quality['qualified'] and len(discovery_groups)>=gates.min_discovery_setups and dispersion<=gates.max_dispersion)
        entries.append(dict(candidate_id=c['candidate_id'],parent_candidate_id=p.get('ancestry',{}).get(c['candidate_id']),controller=c,centroid=centroid.tolist(),dispersion=dispersion,
            discovery_setups=len(discovery_groups),traces=len(ix),trace_ids=[by_role['discovery'][i]['metadata']['trace_id'] for i in ix],
            quality=quality,qualified=qualified,status='preliminary-qualified' if qualified else 'provisional'))
    if old:require({e['candidate_id'] for e in old['entries']}<={e['candidate_id'] for e in entries},'Rebuild lost a retained candidate')
    elites=select_elites(entries,gates);members=shortlist(entries,elites,count=count,method=method)
    diagnostics={m:descriptor.geometry(descriptor.transform(payload,by_role['discovery'],method=m,device=device),by_role['discovery']) for m in ('ae','pca','authored')}
    result=dict(format=VERSION,contract=identity(),proposal_id=p['id'],proposal_path=proposal.relative_to(root).as_posix(),pool_id=p['pool_id'],
        encoder=dict(path=encoder.relative_to(root).as_posix(),id=enc['id'],sha256=sha(encoder/'manifest.json')),
        previous_id=old['id'] if old else None,reencoded_candidates=len(entries),gates=asdict(gates),sources=sources,
        entries=entries,elites=elites,shortlist=dict(method=method,requested=count,members=members,complete=len(members)==count,release_qualified=False),
        diagnostics=diagnostics,shared_model=dict(ready=p['ready'],model_id=p['candidates'][0]['model_id']),
        scope='Offline P3 discovery snapshot; no 16/128 release roster is implied')
    result['id']=digest(result);out.mkdir(parents=True);write_json(out/'manifest.json',result)
    paths=[p['ready']+'/ready.json',proposal.relative_to(root).as_posix()+'/manifest.json',encoder.relative_to(root).as_posix()+'/manifest.json']
    paths.extend(s['path']+'/'+name for s in sources for name in ('manifest.json','discovery.json'))
    if previous:paths.append(project_path(root,previous).relative_to(root).as_posix()+'/manifest.json')
    write_json(out/'pins.json',dict(paths=sorted(set(paths))))
    return result
