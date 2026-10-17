"""Restartable, disk-backed BC preparation; dense features never cross pool IPC.

Only audited heuristic ingress may create these derived files. Receipts bind the
source, preparation code and every output byte. Training retains small numeric
columns, while random-access records and file handles are bounded by minibatches.
"""
from bisect import bisect_right
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import json
import multiprocessing
from pathlib import Path
import struct
from threading import Lock
import time
from zipfile import BadZipFile

import numpy as np
import torch

from brass_ai.bank import read, sha
from brass_ai.learning.contracts import compatibility, digest, require
from brass_ai.learning.features import Encoded, check_encoded, feature_schema
from brass_ai.step5.league import write_json
from .progress import Progress
from .retention import safe
from .updates import Dataset


FORMAT = 'prepared-bc-shard/1.0.0'
TARGETS = ('win_share', 'placement', 'vp')


def preparation_identity():
    from .replay import cache_version
    return digest(dict(format=FORMAT, compatibility=compatibility(), replay=cache_version(),
        sources={name:sha(Path(__file__).with_name(name))
                 for name in ('demonstrations.py','prepared_store.py','updates.py')}))


def paths(directory, key):
    return {kind:safe(directory,directory/(key+suffix))
            for kind,suffix in (('data','.bin'),('index','.npz'),('receipt','.json'))}


def task_key(task, identity):
    path, entries, partition, allowed, compact = task
    # Paths are explicit: moving raw data requires an explicit manifest change.
    return digest(dict(preparation=identity,path=path,entries=entries,partition=partition,
                       teachers=sorted(allowed),compact=compact))


def _load_index(path, receipt):
    with np.load(path,allow_pickle=False) as file:
        columns={k:file[k] for k in ('offsets','counts','chosen','seats','old','targets','groups')}
        # These identities affect leakage and balancing, so bind them to the
        # checksummed index too, not only editable receipt bookkeeping.
        require(file['group_keys'].tolist()==[list(k) for k in receipt['group_keys']] and
                file['game_ids'].tolist()==receipt['game_ids'],'Changed prepared identity metadata')
    n=receipt['rows']; offsets=columns['offsets']; counts=columns['counts']
    require(type(n) is int and n>0,'Invalid prepared row count')
    require(isinstance(receipt['group_keys'],list) and all(isinstance(k,(list,tuple)) and len(k)==3 and
            all(isinstance(v,str) for v in k) for k in receipt['group_keys']) and
            isinstance(receipt['game_ids'],list) and receipt['game_ids'] and
            all(isinstance(g,str) for g in receipt['game_ids']),'Invalid prepared identities')
    for name,dtype,shape in (('offsets','<i8',(n+1,)),('counts','<i4',(n,)),
            ('chosen','<i8',(n,)),('seats','<i8',(n,)),('old','<f4',(n,)),
            ('targets','<f4',(n,3,4)),('groups','<i4',(n,))):
        require(columns[name].dtype==np.dtype(dtype) and columns[name].shape==shape,
                'Invalid prepared column '+name)
    require(offsets[0]==0 and offsets[-1]==receipt['data_bytes'] and np.all(np.diff(offsets)>0),
            'Invalid prepared offsets')
    require(np.all(counts>0) and np.all((columns['chosen']>=0)&(columns['chosen']<counts)) and
            np.all((columns['seats']>=0)&(columns['seats']<4)),'Invalid prepared labels')
    require(np.all((columns['groups']>=0)&(columns['groups']<len(receipt['group_keys']))) and
            len(np.unique(columns['groups']))==len(receipt['group_keys']) and
            np.isfinite(columns['targets']).all() and np.isfinite(columns['old']).all(),
            'Invalid prepared targets/groups')
    require(int(counts.sum())==receipt['candidates'],'Invalid prepared candidate count')
    return columns


def _cached(files, key):
    try:
        receipt=read(files['receipt'])
        require(receipt['format']==FORMAT and receipt['key']==key,'Stale prepared receipt')
        for kind in ('data','index'):
            require(files[kind].stat().st_size==receipt[kind+'_bytes'] and
                    sha(files[kind])==receipt[kind+'_sha256'],'Changed prepared '+kind)
        _load_index(files['index'],receipt)
        return receipt
    except (OSError,ValueError,KeyError,TypeError,EOFError,BadZipFile):
        # Derived files can be rebuilt. Raw source/admission failures cannot.
        return None


def write_prepared(data, directory, key):
    """Write one validated shard; the caller commits its receipt separately."""
    files=paths(directory,key); offsets=[0]; groups={}; group_ids=[]
    temporary=safe(directory,directory/(key+'.bin.partial'))
    with temporary.open('wb') as handle:
        for row,target,identity,encoded in data:
            check_encoded(encoded)
            metadata=json.dumps((row,target,identity),separators=(',',':'),allow_nan=False).encode('utf-8')
            handle.write(struct.pack('<I',len(metadata)));handle.write(metadata)
            handle.write(np.asarray(encoded.state,dtype='<f4').tobytes())
            handle.write(np.asarray(encoded.actions,dtype='<f4').tobytes())
            handle.write(np.asarray(encoded.stable_to_relative,dtype='<i8').tobytes())
            offsets.append(handle.tell())
            group=(row['behavior']['policy_id'],row['view']['context']['era'],row['choices'][row['chosen']]['kind'])
            group_ids.append(groups.setdefault(group,len(groups)))
    temporary.replace(files['data'])
    columns=dict(offsets=np.asarray(offsets,dtype='<i8'),
        counts=np.asarray([len(d[3].choice_ids) for d in data],dtype='<i4'),
        chosen=np.asarray([d[0]['chosen'] for d in data],dtype='<i8'),
        seats=np.asarray([d[0]['seat'] for d in data],dtype='<i8'),
        old=np.asarray([d[0]['behavior']['probability'] for d in data],dtype='<f4'),
        targets=np.asarray([[d[1][k] for k in TARGETS] for d in data],dtype='<f4'),
        groups=np.asarray(group_ids,dtype='<i4'))
    temporary=safe(directory,directory/(key+'.npz.partial'))
    game_ids=sorted({d[2][0] for d in data})
    with temporary.open('wb') as handle:
        np.savez(handle,**columns,group_keys=np.asarray(list(groups)),game_ids=np.asarray(game_ids))
    temporary.replace(files['index'])
    return dict(format=FORMAT,key=key,rows=len(data),candidates=int(columns['counts'].sum()),
        group_keys=list(groups),game_ids=game_ids,
        **{kind+suffix:value for kind in ('data','index') for suffix,value in
           (('_sha256',sha(files[kind])),('_bytes',files[kind].stat().st_size))})


def _prepare_job(job):
    task,directory,key=job;directory=Path(directory)
    # Recheck the authoritative raw input even when a prepared receipt is valid.
    entries=task[1]
    require(all(e['sha256']==entries[0]['sha256'] for e in entries),'Conflicting demonstration hashes')
    require(sha(Path(task[0]))==entries[0]['sha256'],'Demonstration source changed: '+task[0])
    files=paths(directory,key);receipt=_cached(files,key)
    if receipt is not None:return receipt,True
    from .demonstrations import _prepared_shard
    return write_prepared(_prepared_shard(task),directory,key),False


def _worker_init():
    torch.set_num_threads(1)


def prepare(tasks, directory, workers, partition, progress=False):
    """Commit each success; bounded retries after abrupt death never drop a shard."""
    started=time.perf_counter()
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    identity=preparation_identity();keys=[task_key(t,identity) for t in tasks]
    # Training-only epoch extension does not change prepared features/labels.
    # Reuse only explicitly reviewed keys; workers still verify raw/output hashes.
    from .imitation_extension import equivalent_preparations
    from .replay import cache_version
    legacy=equivalent_preparations(FORMAT,cache_version())
    existing={p.stem for p in directory.glob('*.json')}
    for i,task in enumerate(tasks):
        if keys[i] not in existing:
            keys[i]=next((key for old in legacy if (key:=task_key(task,old)) in existing),keys[i])
    receipts=[None]*len(tasks);width=min(workers,len(tasks));cached=0;retries=[]
    log=Progress(f'imitation load {partition} (disk-backed, CPU, {width} workers)',len(tasks),'shards') if progress else None
    while any(r is None for r in receipts):
        todo=iter(i for i,r in enumerate(receipts) if r is None);pending={};failed=None
        with ProcessPoolExecutor(max_workers=width,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_worker_init) as executor:
            def submit():
                i=next(todo,None)
                if i is not None:
                    pending[executor.submit(_prepare_job,(tasks[i],str(directory),keys[i]))]=i
            try:
                for _ in range(width):submit()
                while pending:
                    future=next(as_completed(pending));i=pending.pop(future)
                    receipt,reused=future.result()
                    require(receipt['key']==keys[i] and receipt['format']==FORMAT,'Wrong prepared shard receipt')
                    # A worker killed during writing leaves no committed receipt.
                    files=paths(directory,keys[i]);safe(directory,files['receipt'].with_suffix('.json.tmp'))
                    write_json(files['receipt'],receipt)
                    receipts[i]=receipt;cached+=int(reused)
                    if log:log(sum(r is not None for r in receipts))
                    submit()
            except BrokenProcessPool as error:
                failed=error
            except Exception as error:
                for future in pending:future.cancel()
                source=tasks[i][0] if 'i' in locals() else 'worker startup'
                raise ValueError(f'Imitation preparation failed for {source}: {error}') from error
        if failed:
            completed=sum(r is not None for r in receipts)
            unfinished=[tasks[i][0] for i,r in enumerate(receipts) if r is None]
            if width==1:
                raise RuntimeError(f'Imitation loader worker died with one worker; {completed}/{len(tasks)} shards '
                    f'are cached. Unfinished source: {unfinished[0]}. Check OS/job logs for OOM or external '
                    'termination and available RAM/disk; no shards were skipped. Rerun the same command '
                    'to reuse verified preparation.') from failed
            previous=width;width=max(1,width//2)
            retries.append(dict(previous_workers=previous,workers=width,completed_shards=completed))
            print(f'Imitation loader worker died; keeping {completed}/{len(tasks)} prepared shards and '
                  f'retrying unfinished shards with {width} workers. Check OS/job logs for the cause.',flush=True)
    rows=PreparedRows(directory,receipts)
    rows.report=dict(storage='disk-backed',shards=len(tasks),rows=len(rows),cache_hits=cached,
        initial_workers=min(workers,len(tasks)),final_workers=width,worker_retries=retries,
        prepared_bytes=sum(r['data_bytes']+r['index_bytes'] for r in receipts),elapsed_seconds=time.perf_counter()-started,
        resident_index_bytes=rows.index_bytes,max_open_files=rows.max_open_files)
    if progress:print('Imitation prepared '+partition+': '+json.dumps(rows.report),flush=True)
    return rows


class _Row:
    def __init__(self, owner, index):self.owner,self.index=owner,index
    def __len__(self):return 4
    def __getitem__(self, field):
        if isinstance(field,slice):return tuple(self[i] for i in range(4)[field])
        if field<0:field+=4
        if field not in range(4):raise IndexError(field)
        return self.owner.record(self.index,features=field==3)[field]


class _Column:
    def __init__(self, data, field):self.data,self.field=data,field
    def __len__(self):return len(self.data)
    def __getitem__(self,index):return self.data[index][self.field]


class PreparedRows:
    max_open_files=8

    def __init__(self,directory,receipts):
        self.directory,self.receipts=Path(directory),receipts
        self.ends=np.cumsum([r['rows'] for r in receipts]).tolist()
        self.game_ids={g for r in receipts for g in r['game_ids']}
        self.files=[paths(self.directory,r['key']) for r in receipts]
        self.offsets=[];parts=[];vocabulary={};self.group_keys=[]
        for file,receipt in zip(self.files,receipts):
            columns=_load_index(file['index'],receipt);self.offsets.append(columns.pop('offsets'))
            remap=[]
            for key in receipt['group_keys']:
                key=tuple(key)
                if key not in vocabulary:vocabulary[key]=len(vocabulary);self.group_keys.append(key)
                remap.append(vocabulary[key])
            columns['groups']=np.asarray(remap,dtype=np.int32)[columns['groups']];parts.append(columns)
        self.columns={name:np.concatenate([p[name] for p in parts]) for name in parts[0]}
        self.index_bytes=sum(c.nbytes for c in self.columns.values())+sum(o.nbytes for o in self.offsets)
        self.handles=OrderedDict();self.lock=Lock();self.closed=False

    def __len__(self):return self.ends[-1]
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        if index<0:index+=len(self)
        if not 0<=index<len(self):raise IndexError(index)
        return _Row(self,index)

    def close(self):
        with self.lock:
            for handle in self.handles.values():handle.close()
            self.handles.clear();self.closed=True

    def record(self,index,features=False):
        shard=bisect_right(self.ends,index);local=index-(self.ends[shard-1] if shard else 0)
        start,end=map(int,self.offsets[shard][local:local+2])
        with self.lock:
            require(not self.closed,'Prepared dataset is closed')
            handle=self.handles.pop(shard,None)
            if handle is None:handle=self.files[shard]['data'].open('rb')
            self.handles[shard]=handle
            while len(self.handles)>self.max_open_files:self.handles.popitem(last=False)[1].close()
            handle.seek(start);prefix=handle.read(4)
            require(len(prefix)==4,'Truncated prepared record')
            size=struct.unpack('<I',prefix)[0];require(size<=end-start-4,'Invalid prepared metadata length')
            metadata=handle.read(size)
            payload=handle.read(end-start-4-size) if features else None
        row,target,identity=json.loads(metadata);encoded=None
        if features:
            schema=feature_schema();s=schema['state_dim'];n=int(self.columns['counts'][index]);a=schema['action_dim']
            require(len(payload)==4*(s+n*a)+32,'Truncated prepared features')
            state=np.frombuffer(payload,dtype='<f4',count=s)
            actions=np.frombuffer(payload,dtype='<f4',count=n*a,offset=4*s).reshape(n,a)
            seats=np.frombuffer(payload,dtype='<i8',count=4,offset=4*(s+n*a))
            encoded=Encoded(state,actions,seats,tuple(c['id'] for c in row['choices']))
            check_encoded(encoded)
        return row,target,tuple(identity),encoded


class PreparedDataset(Dataset):
    """Supervised Dataset with identical batching/loss semantics and lazy features."""
    def __init__(self,data,device,weights=None):
        self.data,self.device=data,device;columns=data.columns
        self.rows=_Column(data,0);self.encoded=_Column(data,3)
        self.chosen=torch.tensor(columns['chosen'],device=device)
        self.seats=torch.tensor(columns['seats'],device=device)
        self.old=torch.tensor(columns['old'],device=device)
        self.targets=tuple(torch.tensor(columns['targets'][:,i],device=device) for i in range(3))
        self.utility=self.remaining=self.q=None
        self.forced_cpu=columns['counts']==1
        self.forced=torch.tensor(self.forced_cpu,device=device)
        self.bc_weights=torch.tensor(np.ones(len(data),dtype=np.float32) if weights is None else weights,
                                     dtype=torch.float32,device=device)

    def microbatches(self,indices,size,cells,bucket=True):
        require(type(size) is int and size>0,'Invalid minibatch size')
        require(cells is None or (type(cells) is int and cells>0),'Invalid candidate-cell limit')
        indices=list(indices);counts=self.data.columns['counts']
        if bucket:indices.sort(key=lambda i:counts[i])
        part=[];maximum=0
        for i in indices:
            count=int(counts[i]);require(cells is None or count<=cells,
                'Single position exceeds training candidate capacity; increase max_candidate_cells')
            candidate_max=max(maximum,count)
            if part and (len(part)==size or (cells is not None and (len(part)+1)*candidate_max>cells)):
                yield part;part=[];maximum=0
            part.append(i);maximum=max(maximum,count)
        if part:yield part
