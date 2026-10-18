"""Immutable, seeded held-out IL probe; never supplies optimizer samples."""
from collections import defaultdict,deque
import hashlib
import heapq
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import compatibility,digest,project_path,require
from brass_ai.learning.features import Encoded,check_encoded
from brass_ai.learning.local_entities import check_action_selectors
from brass_ai.learning.batching import collate
from brass_ai.step5.league import write_json
from .il_kl_limit import GROUPS

FORMAT='fixed-il-kl-probe/1.0.0'


def semantic_hash(metadata,arrays):
    tensors=hashlib.sha256()
    for key,value in sorted(arrays.items()):tensors.update(key.encode());tensors.update(value.tobytes())
    return digest(dict({k:v for k,v in metadata.items() if k not in ('probe_hash','cache')},
                       tensors_sha256=tensors.hexdigest()))


def group_for(context):
    era={'CanalEra':'canal','RailEra':'rail'}.get(context.get('era'))
    require(era is not None and isinstance(context.get('phase'),str),'Unknown IL probe era/phase')
    return era+('_action' if context['phase']=='action' else '_other')


def select_indices(data,count,seed):
    """Balanced era x phase strata, hash priority within each, no global RNG."""
    strata=defaultdict(list)
    for i in range(len(data)):
        row=data[i][0]
        if len(row['choices'])<2:continue
        context=row['view']['context'];group_for(context)
        key=(context['era'],context['phase'])
        rank=int(digest(dict(seed=seed,index=i,identity=data[i][2])),16)
        heap=strata[key];item=(-rank,-i)
        if len(heap)<count:heapq.heappush(heap,item)
        elif item>heap[0]:heapq.heapreplace(heap,item)
    queues={key:deque(-i for _,i in sorted(items,reverse=True)) for key,items in sorted(strata.items())}
    selected=[]
    while len(selected)<count and any(queues.values()):
        for queue in queues.values():
            if queue and len(selected)<count:selected.append(queue.popleft())
    require(selected,'No nonforced IL validation positions for the anchor probe')
    return selected


def raw_probabilities(logits,mask):
    p=torch.softmax(logits.float().masked_fill(~mask.bool(),-torch.inf),dim=-1)
    require(bool(torch.isfinite(p).all()),'Non-finite fixed IL probe policy')
    return p


def stable_kl(reference,current,mask,floor):
    # Apply the same numerical floor to both raw categorical distributions.
    # This defines a proper KL and preserves KL(IL || IL)=0 even in tiny tails.
    p=reference.double().clamp_min(floor)*mask
    q=current.double().clamp_min(floor)*mask
    p=p/p.sum(-1,keepdim=True);q=q/q.sum(-1,keepdim=True)
    return (p*(p.clamp_min(1e-300).log()-q.clamp_min(1e-300).log())).sum(-1).clamp_min(0)


class Probe:
    def __init__(self,metadata,arrays,config):
        self.metadata,self.arrays,self.config=metadata,arrays,config
        self.rows=metadata['rows'];self.encoded=[]
        offsets=arrays['offsets']
        require(len(offsets)==len(self.rows)+1 and offsets[0]==0 and
                offsets[-1]==len(arrays['actions']) and len(arrays['reference'])==offsets[-1],
                'Invalid fixed IL probe offsets')
        require(all(np.isfinite(v).all() for v in arrays.values()),'Non-finite fixed IL probe cache')
        for i,row in enumerate(self.rows):
            begin,end=map(int,offsets[i:i+2]);require(end-begin==len(row['kinds'])>=2,'Invalid IL probe actions')
            encoded=Encoded(arrays['states'][i],arrays['actions'][begin:end],arrays['seats'][i],tuple(row['choice_ids']))
            check_encoded(encoded);check_action_selectors(encoded.actions)
            require(row['group']==group_for(row),'Changed IL probe group')
            p=arrays['reference'][begin:end]
            require(np.all(p>=0) and abs(float(p.sum())-1)<1e-5,'Invalid cached IL probabilities')
            self.encoded.append(encoded)

    def parts(self,update):
        indices=sorted(range(len(self.rows)),key=lambda i:len(self.rows[i]['kinds']))
        part=[];maximum=0
        for i in indices:
            count=len(self.rows[i]['kinds']);maximum=max(maximum,count)
            require(count<=update['max_candidate_cells'],'IL probe position exceeds candidate budget')
            if part and (len(part)>=update['microbatch_size'] or (len(part)+1)*maximum>update['max_candidate_cells']):
                yield part;part=[];maximum=count
            part.append(i)
        if part:yield part

    def measure(self,model,update):
        began=time.perf_counter();distances={};sales=[];passes=[];was_training=model.training
        device=next(model.parameters()).device
        model.eval()
        try:
            with torch.no_grad():
                for part in self.parts(update):
                    inputs=collate([self.encoded[i] for i in part],device)
                    current=raw_probabilities(model(*inputs)[0],inputs[2])
                    reference=torch.zeros_like(current)
                    for j,i in enumerate(part):
                        a,b=map(int,self.arrays['offsets'][i:i+2])
                        reference[j,:b-a]=torch.tensor(self.arrays['reference'][a:b],device=device)
                    values=stable_kl(reference,current,inputs[2],self.config['probability_floor']).cpu().tolist()
                    for j,i in enumerate(part):
                        distances[i]=values[j];kinds=self.rows[i]['kinds']
                        selling=[k for k,kind in enumerate(kinds) if kind=='sell']
                        passing=[k for k,kind in enumerate(kinds) if kind=='pass']
                        il_sale=float(reference[j,selling].sum());now_sale=float(current[j,selling].sum())
                        if il_sale>=.2:sales.append(dict(row=i,il=il_sale,current=now_sale,ok=now_sale>=.5*il_sale))
                        il_pass=float(reference[j,passing].sum());now_pass=float(current[j,passing].sum())
                        if passing and len(passing)<len(kinds) and il_pass<.2:
                            passes.append(dict(row=i,il=il_pass,current=now_pass,ok=now_pass<=2*il_pass+.05))
        finally:model.train(was_training)
        def checks(rows):
            failed=[r for r in rows if not r['ok']]
            return dict(positions=len(rows),failures=len(failed),examples=failed[:8],ok=not failed)
        groups={}
        for group in GROUPS:
            values=[distances[i] for i,row in enumerate(self.rows) if row['group']==group]
            groups[group]=dict(positions=len(values),kl_mean=sum(values)/len(values) if values else None)
        sale_check,pass_check=checks(sales),checks(passes)
        return dict(positions=len(distances),requested_positions=self.config['probe_positions'],
            kl_mean=sum(distances.values())/len(distances),groups=groups,unassigned_positions=0,
            behavior=dict(ok=sale_check['ok'] and pass_check['ok'],sales=sale_check,passing=pass_check),
            elapsed_seconds=time.perf_counter()-began,probability_floor=self.config['probability_floor'],
            scope='Fixed IL validation positions; raw temperature-1 policies, no exploration; symmetric floor/renormalization for KL, raw behavior checks')

    def probabilities(self,model,update):
        """Raw immutable snapshot for quality-pool novelty checks, no RNG."""
        result=np.empty_like(self.arrays['reference']);device=next(model.parameters()).device
        was_training=model.training;model.eval()
        try:
            with torch.no_grad():
                for part in self.parts(update):
                    inputs=collate([self.encoded[i] for i in part],device)
                    values=raw_probabilities(model(*inputs)[0],inputs[2]).cpu().numpy()
                    for j,i in enumerate(part):
                        a,b=map(int,self.arrays['offsets'][i:i+2]);result[a:b]=values[j,:b-a]
        finally:model.train(was_training)
        return result


def source_binding(root,origin):
    path=project_path(root,origin['manifest'])
    require(sha(path)==origin['manifest_sha256'],'IL probe dataset manifest changed')
    manifest=read(path)
    sources={entry['path']:entry['sha256'] for entry in manifest['entries'] if entry['split']=='validation'}
    require(sources,'No IL validation source for the fixed probe')
    for path,expected in sources.items():
        require(sha(project_path(root,path))==expected,'IL probe validation data changed: '+path)
    return dict(manifest=origin['manifest'],manifest_sha256=origin['manifest_sha256'],sources=sources,
                il_model_id=origin['model_id'],il_checkpoint_sha256=origin['sha256'],compatibility=compatibility())


def build(root,directory,origin,model,config,update,workers):
    from .demonstrations import load_dataset
    directory.mkdir(parents=True,exist_ok=True)
    source=source_binding(root,origin)
    data,_=load_dataset(root,origin['manifest'],'validation',workers=workers,progress=True,compact=True,
                        storage=directory/'.prepared')
    try:
        indices=select_indices(data,config['probe_positions'],config['probe_seed'])
        selected=[data.record(i,features=True) for i in indices]
    finally:data.close()
    rows=[];offsets=[0]
    for i,(row,_,identity,encoded) in zip(indices,selected):
        context=row['view']['context']
        rows.append(dict(index=i,identity=list(identity),era=context['era'],phase=context['phase'],
            group=group_for(context),kinds=[c['kind'] for c in row['choices']],
            industries=[c.get('industry') for c in row['choices']],chosen=row['chosen'],
            choice_ids=list(encoded.choice_ids)))
        offsets.append(offsets[-1]+len(encoded.actions))
    arrays=dict(states=np.stack([d[3].state for d in selected]),actions=np.concatenate([d[3].actions for d in selected]),
        seats=np.stack([d[3].stable_to_relative for d in selected]),offsets=np.asarray(offsets,dtype=np.int64),
        reference=np.concatenate([np.full(len(d[3].actions),1/len(d[3].actions),dtype=np.float32) for d in selected]))
    metadata=dict(format=FORMAT,source=source,config=config,rows=rows)
    probe=Probe(metadata,arrays,config);device=next(model.parameters()).device;was_training=model.training
    model.eval()
    try:
        with torch.no_grad():
            for part in probe.parts(update):
                inputs=collate([probe.encoded[i] for i in part],device)
                p=raw_probabilities(model(*inputs)[0],inputs[2]).cpu().numpy()
                for j,i in enumerate(part):
                    a,b=offsets[i:i+2];arrays['reference'][a:b]=p[j,:b-a]
    finally:model.train(was_training)
    metadata['probe_hash']=semantic_hash(metadata,arrays)
    cache=directory/'probe.npz';temporary=directory/'probe.npz.tmp'
    with temporary.open('wb') as handle:np.savez(handle,**arrays)
    temporary.replace(cache)
    metadata['cache']=dict(path=cache.relative_to(root).as_posix(),sha256=sha(cache))
    write_json(directory/'manifest.json',metadata)
    return probe,metadata


def load(root,run,origin,model,config,update,workers,receipt=None):
    directory=run/'il-kl';path=directory/'manifest.json'
    source=source_binding(root,origin)
    if path.exists():
        metadata=read(path)
        require(metadata['format']==FORMAT and metadata['config']==config and metadata['source']==source,
                'Fixed IL probe source/config differs')
        cache=project_path(root,metadata['cache']['path'])
        require(cache.parent==directory and sha(cache)==metadata['cache']['sha256'],'Fixed IL probe cache changed')
        with np.load(cache,allow_pickle=False) as file:arrays={k:file[k] for k in file.files}
        require(metadata['probe_hash']==semantic_hash(metadata,arrays),'Fixed IL probe semantic hash differs')
        probe=Probe(metadata,arrays,config)
    else:
        require(receipt is None,'Committed fixed IL probe is missing')
        probe,metadata=build(root,directory,origin,model,config,update,workers)
    current=dict(path=path.relative_to(root).as_posix(),sha256=sha(path),probe_hash=metadata['probe_hash'])
    require(receipt is None or receipt==current,'Committed fixed IL probe manifest changed')
    return probe,current


def import_probe(root,run,from_run,origin,model,config,update):
    """Copy only a committed, compatible held-out probe into a fresh run.

    The donor's large .prepared validation cache and all PPO state stay there.
    A fresh run still checks KL(IL || IL)==0 before committing its own state.
    """
    from .retention import owned,safe,active
    root,run=owned(root,run);_,donor=owned(root,from_run)
    require(donor!=run and not (run/'state.json').exists(),'Probe import requires a different, new run')
    require(not active(donor),'Stop the donor self-play run before reusing its IL probe')
    state=read(safe(donor,donor/'state.json'))
    receipt=state.get('il_kl_limit',{}).get('probe')
    require(receipt is not None,'Donor has no committed fixed IL probe; full validation cache alone is insufficient')
    entry=state['history'][-1];checkpoint=safe(donor,project_path(root,entry['path']))
    require(sha(checkpoint)==entry['sha256'],'Donor checkpoint changed')
    raw=torch.load(checkpoint,map_location='cpu',weights_only=True)
    require(raw['training'].get('il_kl_limit',{}).get('probe')==receipt,
            'Donor fixed probe differs from its committed checkpoint')
    del raw
    probe,_=load(root,donor,origin,model,config,update,1,receipt=receipt)
    directory=safe(run,run/'il-kl');directory.mkdir(parents=True,exist_ok=True)
    if (directory/'manifest.json').exists():
        existing,current=load(root,run,origin,model,config,update,1)
        require(current['probe_hash']==receipt['probe_hash'],'New run already has a different fixed IL probe')
        return existing,current
    metadata=dict(probe.metadata);source=project_path(root,metadata['cache']['path'])
    target=safe(run,directory/'probe.npz');temporary=safe(run,directory/'probe.npz.tmp')
    try:
        shutil.copyfile(source,temporary)
        require(sha(temporary)==metadata['cache']['sha256'],'IL probe changed during copy')
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)
    metadata['cache']=dict(path=target.relative_to(root).as_posix(),sha256=sha(target))
    require(semantic_hash(metadata,probe.arrays)==receipt['probe_hash'],'Imported IL probe identity changed')
    path=directory/'manifest.json';write_json(path,metadata)
    current=dict(path=path.relative_to(root).as_posix(),sha256=sha(path),probe_hash=metadata['probe_hash'])
    write_json(directory/'import-report.json',dict(format='fixed-il-probe-import/1.0.0',
        source_run=donor.relative_to(root).as_posix(),source_receipt=receipt,destination_receipt=current,
        copied_bytes=target.stat().st_size,training_state_imported=False))
    print(f'Reused fixed IL validation probe: {len(probe.rows)} positions from {from_run}; no validation re-encoding',flush=True)
    return Probe(metadata,probe.arrays,config),current
