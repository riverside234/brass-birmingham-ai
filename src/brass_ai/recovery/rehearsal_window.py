"""Full-domain, rotating train windows with disposable, bounded feature storage.

Shard draws are proportional to audited decision counts, then rows are uniform
within each draw. Each presentation has a uniform marginal over train; rows in
a window are correlated. This is deliberately NOT IID full-corpus minibatching.
Only one source shard is prepared at a time; only selected rows are written.
"""
from bisect import bisect_right
from collections import Counter,defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import re
import time

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import digest,project_path,require
from brass_ai.step5.league import write_json
from . import demonstrations,prepared_store
from .retention import owned,safe

DEFAULT=dict(contract='rotating-train-window/1.0.0',rows_per_iteration=16384,
             shard_draws=32,max_cache_bytes=4*1024**3,delete_after_update=True)
OWNER='disposable-rehearsal-cache/1.0.0'


def validate(value):
    require(type(value) is dict and set(value)==set(DEFAULT) and value['contract']==DEFAULT['contract'],
            'Invalid rehearsal sampling contract')
    for key,limit in (('rows_per_iteration',262144),('shard_draws',256),('max_cache_bytes',64*1024**3)):
        require(type(value[key]) is int and 1<=value[key]<=limit,'Invalid rehearsal sampling '+key)
    require(value['delete_after_update'] is True,'Rehearsal cache must be deleted after each update')
    return value


def cleanup(root,run):
    """Only our flat scratch directory, under the locked run; never dataset/IL."""
    _,run=owned(root,run);directory=safe(run,run/'.rehearsal-cache')
    if not directory.exists():return
    files=list(directory.iterdir())
    if files:
        require((directory/'owner.json').is_file() and read(directory/'owner.json')==dict(format=OWNER),
                'Unrecognized rehearsal scratch directory; refusing cleanup')
    for path in files:
        safe(run,path)
        require(path.is_file() and (path.name=='owner.json' or
                re.fullmatch(r'[0-9a-f]{64}\.(bin|npz|json)(\.partial|\.tmp)?',path.name)),
                'Unexpected rehearsal scratch entry; refusing cleanup')
    for path in files:path.unlink()
    directory.rmdir()


class TrainIndex:
    def __init__(self,root,manifest_path):
        self.root=Path(root).resolve();self.path=project_path(root,manifest_path)
        self.manifest_sha256=sha(self.path)
        self.tasks,manifest=demonstrations.dataset_tasks(self.root,self.path,'train',compact=True)
        require(sha(self.path)==self.manifest_sha256,'Rehearsal manifest changed while indexing')
        self.counts=[];self.ends=[];self.games=0
        for task in self.tasks:
            count=0
            for entry in task[1]:
                teachers=entry.get('teachers')
                require(type(teachers) is dict and teachers and set(teachers)<=set(manifest['teachers']) and
                        all(type(n) is int and n>0 for n in teachers.values()),
                        'Rehearsal index requires audited per-game teacher decision counts; re-audit the manifest')
                count+=sum(teachers.values());self.games+=1
            self.counts.append(count);self.ends.append((self.ends[-1] if self.ends else 0)+count)
        self.rows=self.ends[-1]

    def select(self,config,seed,iteration):
        validate(config)
        require(type(iteration) is int and iteration>=0,'Invalid rehearsal iteration')
        generator=torch.Generator(device='cpu')
        generator.manual_seed(int(digest(dict(contract=config['contract'],manifest=self.manifest_sha256,
            seed=seed,iteration=iteration))[:16],16)%(2**63))
        count=min(config['rows_per_iteration'],self.rows);draws=min(config['shard_draws'],count)
        chosen=defaultdict(list)
        for draw,global_row in enumerate(torch.randint(self.rows,(draws,),generator=generator).tolist()):
            shard=bisect_right(self.ends,global_row);n=count//draws+int(draw<count%draws)
            chosen[shard].extend(torch.randint(self.counts[shard],(n,),generator=generator).tolist())
        return dict(chosen)


def byte_reservation(data):
    """Conservative bound before writing: dense payload, NPZ index, JSON receipt."""
    groups=set();games=set();payload=0
    for row,target,identity,encoded in data:
        payload+=4+len(json.dumps((row,target,identity),separators=(',',':'),allow_nan=False).encode('utf-8'))
        payload+=4*(encoded.state.size+encoded.actions.size)+32
        groups.add((row['behavior']['policy_id'],row['view']['context']['era'],row['choices'][row['chosen']]['kind']))
        games.add(identity[0])
    # np.savez stores fixed-width Unicode identity arrays plus numeric columns.
    identities=12*len(groups)*max(len(s) for group in groups for s in group)+4*len(games)*max(map(len,games))
    return payload+128*len(data)+2*identities+65536


class WindowDataset(prepared_store.PreparedDataset):
    def __init__(self,rows,device,identities,report):
        super().__init__(rows,device)
        self.identities=identities;self.report=report;self.used=set();self.presentations=0

    def _used(self,indices):
        self.presentations+=len(indices);self.used.update(self.identities[i] for i in indices)
        self.report.update(used_presentations=self.presentations,distinct_rows_used=len(self.used))

    def supervised_batches(self,parts,executor):
        for part,ix,inputs in super().supervised_batches(parts,executor):
            if executor is not None:self._used(part)
            yield part,ix,inputs

    def batch(self,indices,*args,**kwargs):
        # The legacy PPO path uses batch(); modern nonforced PPO uses the above.
        self._used(indices)
        return super().batch(indices,*args,**kwargs)


@contextmanager
def window(root,run,index,config,*,seed,iteration,device):
    validate(config);root,run=owned(root,run);cleanup(root,run)
    require(sha(index.path)==index.manifest_sha256,'Rehearsal manifest changed')
    directory=safe(run,run/'.rehearsal-cache');directory.mkdir(parents=True)
    write_json(directory/'owner.json',dict(format=OWNER))
    started=time.perf_counter();data=None;store=None
    selection=index.select(config,seed,iteration);receipts=[];identities=[];used_bytes=0
    report=dict(contract=config['contract'],iteration=iteration,source_rows=index.rows,source_games=index.games,
        source_shards=len(index.tasks),selected_shards=len(selection),selected_rows=sum(map(len,selection.values())),
        distinct_rows_selected=sum(len(set(v)) for v in selection.values()),used_presentations=0,distinct_rows_used=0,
        selection_sha256=digest(selection),manifest_sha256=index.manifest_sha256,
        max_cache_bytes=config['max_cache_bytes'],peak_cache_bytes=0,cache_deleted=False,
        scope='Train only; count-weighted shard draws, uniform rows with replacement; renewed each committed iteration')
    try:
        for number,(shard,indices) in enumerate(selection.items(),1):
            task=index.tasks[shard]
            # Ingress verifies raw hashes, labels and independent feature-cache
            # receipts. Missing feature caches re-encode only this shard, in RAM.
            data=demonstrations._prepared_shard(task)
            expected={e['game_id']:sum(e['teachers'].values()) for e in task[1]}
            require(Counter(row[2][0] for row in data)==expected and len(data)==index.counts[shard],
                    'Audited rehearsal decision counts differ from replay')
            selected=[data[i] for i in indices]
            reserve=byte_reservation(selected)
            require(used_bytes+reserve+4096<=config['max_cache_bytes'],
                    'Selected rehearsal window exceeds max_cache_bytes; reduce rows_per_iteration (no rows silently dropped)')
            key=digest(dict(selection=report['selection_sha256'],shard=shard))
            receipt=prepared_store.write_prepared(selected,directory,key)
            write_json(directory/(key+'.json'),receipt)
            files=prepared_store.paths(directory,key)
            size=sum(p.stat().st_size for p in files.values())
            require(size<=reserve,'Rehearsal cache reservation underestimated actual bytes')
            used_bytes+=size;report['peak_cache_bytes']=used_bytes+(directory/'owner.json').stat().st_size
            receipts.append(receipt);identities.extend((shard,i) for i in indices)
            del selected;data=None  # release the full source shard before the next
            print(f'Rehearsal sample {number}/{len(selection)} shards; {len(identities)} selected rows; '
                  f'{used_bytes/1024**2:.1f} MiB temporary cache',flush=True)
        store=prepared_store.PreparedRows(directory,receipts)
        report['preparation_seconds']=time.perf_counter()-started
        dataset=WindowDataset(store,device,identities,report)
        report['nonforced_rows_selected']=int((~dataset.forced_cpu).sum())
        yield dataset
    finally:
        data=None
        if store is not None:store.close()
        cleanup(root,run);report['cache_deleted']=True
