"""Full-domain, rotating train windows with disposable, bounded feature storage.

Shard draws are proportional to audited decision counts, then rows are uniform
within each draw. Each presentation has a uniform marginal over train; rows in
a window are correlated. This is deliberately NOT IID full-corpus minibatching.
One source shard per CPU reader; only selected rows are prepared for BC/written.
"""
from bisect import bisect_right
from collections import defaultdict
from concurrent.futures import as_completed,wait
from contextlib import contextmanager,ExitStack
import json
import os
from pathlib import Path
import re
import struct
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
        require(path.is_file() and (path.name in ('owner.json','budget.bin') or
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


def _reserve(directory,change,limit):
    """A nine-byte scratch ledger, locked by the OS and released on worker death."""
    path=safe(directory,directory/'budget.bin')
    with path.open('r+b',buffering=0) as handle:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_LOCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX)
        try:
            handle.seek(1);raw=handle.read(8)
            require(len(raw)==8,'Invalid rehearsal byte reservation')
            total=struct.unpack('<Q',raw)[0]+change
            require(4096<=total<=limit,
                    'Selected rehearsal window exceeds max_cache_bytes; reduce rows_per_iteration (no rows silently dropped)')
            handle.seek(1);handle.write(struct.pack('<Q',total))
        finally:
            if os.name=='nt':
                handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def _prepare_selection(job):
    """Only tiny receipts cross IPC; dense rows go straight to bounded disk."""
    task,indices,directory,key,limit=job;directory=Path(directory)
    started=time.perf_counter();cpu=time.process_time()
    selected=demonstrations._prepared_shard(task,selection=indices)
    prepared_at=time.perf_counter();reserve=byte_reservation(selected)
    # Reserve BEFORE writing. All concurrent files/partials count toward the
    # same cap, not a separate cap per worker. Failure never drops sampled rows.
    _reserve(directory,reserve,limit)
    receipt=prepared_store.write_prepared(selected,directory,key)
    write_json(directory/(key+'.json'),receipt)
    size=sum(p.stat().st_size for p in prepared_store.paths(directory,key).values())
    require(size<=reserve,'Rehearsal cache reservation underestimated actual bytes')
    _reserve(directory,size-reserve,limit)
    return receipt,dict(bytes=size,read_prepare_seconds=prepared_at-started,
        write_seconds=time.perf_counter()-prepared_at,worker_cpu_seconds=time.process_time()-cpu)


def _collect_selections(jobs,executor,workers):
    """Bound pending work; consume as completed, expose original sample order."""
    if executor is None:
        for i,job in enumerate(jobs):yield i,_prepare_selection(job)
        return
    remaining=iter(enumerate(jobs));pending={}
    def submit():
        item=next(remaining,None)
        if item is not None:
            i,job=item;pending[executor.submit(_prepare_selection,job)]=i
    try:
        for _ in range(min(workers,len(jobs))):submit()
        while pending:
            future=next(as_completed(pending));i=pending.pop(future)
            yield i,future.result()
            submit()
    finally:
        # Join writers before the parent deletes scratch, including Ctrl+C,
        # validation failures, byte-cap rejection and broken process pools.
        for future in pending:future.cancel()
        if pending:wait(pending)


@contextmanager
def window(root,run,index,config,*,seed,iteration,device,workers=1,executor=None):
    validate(config);root,run=owned(root,run);cleanup(root,run)
    require(type(workers) is int and 1<=workers<=64,'Invalid rehearsal reader CPU allowance')
    require(sha(index.path)==index.manifest_sha256,'Rehearsal manifest changed')
    directory=safe(run,run/'.rehearsal-cache');directory.mkdir(parents=True)
    write_json(directory/'owner.json',dict(format=OWNER))
    (directory/'budget.bin').write_bytes(b'\0'+struct.pack('<Q',4096))
    started=time.perf_counter();store=None
    selection=index.select(config,seed,iteration);receipts=[None]*len(selection);identities=[];used_bytes=0
    width=min(workers,len(selection));reused=executor is not None
    report=dict(contract=config['contract'],iteration=iteration,source_rows=index.rows,source_games=index.games,
        source_shards=len(index.tasks),selected_shards=len(selection),selected_rows=sum(map(len,selection.values())),
        distinct_rows_selected=sum(len(set(v)) for v in selection.values()),used_presentations=0,distinct_rows_used=0,
        selection_sha256=digest(selection),manifest_sha256=index.manifest_sha256,
        max_cache_bytes=config['max_cache_bytes'],peak_cache_bytes=0,cache_deleted=False,
        workers_requested=workers,workers=width,reused_reader_pool=reused,
        worker_read_prepare_seconds=0.,worker_write_seconds=0.,worker_cpu_seconds=0.,
        dense_ipc_bytes=0,
        scope='Train only; count-weighted shard draws, uniform rows with replacement; renewed each committed iteration')
    try:
        with ExitStack() as scope:
            if executor is None and width>1:
                from .ingress import reader_pool
                executor=scope.enter_context(reader_pool(width))
            jobs=[]
            for shard,indices in selection.items():
                key=digest(dict(selection=report['selection_sha256'],shard=shard))
                jobs.append((index.tasks[shard],indices,str(directory),key,config['max_cache_bytes']))
                identities.extend((shard,i) for i in indices)
            completed_rows=0
            # Explicitly join writers before a local pool/scratch exits on errors.
            from contextlib import closing
            with closing(_collect_selections(jobs,executor,width)) as pending:
                for number,(i,(receipt,metrics)) in enumerate(pending,1):
                    receipts[i]=receipt;used_bytes+=metrics['bytes'];completed_rows+=receipt['rows']
                    report['peak_cache_bytes']=used_bytes+(directory/'owner.json').stat().st_size+9
                    for key in ('read_prepare_seconds','write_seconds','cpu_seconds'):
                        report['worker_'+key]+=metrics[key if key!='cpu_seconds' else 'worker_cpu_seconds']
                    print(f'Rehearsal sample {number}/{len(selection)} shards (CPU, {width} workers); '
                          f'{completed_rows} selected rows; {used_bytes/1024**2:.1f} MiB temporary cache',flush=True)
        store=prepared_store.PreparedRows(directory,receipts)
        report['preparation_seconds']=time.perf_counter()-started
        dataset=WindowDataset(store,device,identities,report)
        report['nonforced_rows_selected']=int((~dataset.forced_cpu).sum())
        yield dataset
    finally:
        if store is not None:store.close()
        cleanup(root,run);report['cache_deleted']=True
