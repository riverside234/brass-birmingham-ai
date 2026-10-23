"""Ordered CPU-only process collation with a bounded, iteration-local transport.

Input arrays are copied once to a private mapped file, never to each worker.
Only small row descriptors cross the task queue. Output files are bounded by
the prefetch width and removed after consumption. No model, CUDA tensor or RNG
enters a worker. Sampling and gradient accumulation stay in the main process.
"""
from collections import deque
from contextlib import contextmanager,ExitStack
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from brass_ai.learning.batching import collate
from brass_ai.learning.contracts import require,SOURCE_ROOT
from brass_ai.learning.features import Encoded
from .batch_cache import array_digest


def initialize():
    torch.set_num_threads(1)


def ready():return os.getpid()


def write_array(stream,array):
    array=np.ascontiguousarray(array)
    spec=(stream.tell(),array.shape,array.dtype.str)
    stream.write(memoryview(array).cast('B'))
    return spec


def view(buffer,spec):
    offset,shape,dtype=spec
    return np.ndarray(shape,dtype=dtype,buffer=buffer,offset=offset)


def prepare(path,size,rows,indices,output):
    """Pure deterministic worker; NumPy mmap avoids multi-GB queue pickles."""
    began=time.perf_counter();cpu=time.process_time()
    mapped=np.memmap(path,dtype=np.uint8,mode='r',shape=(size,))
    encoded=[];distributions=[]
    try:
        for state,actions,seats,q,version in rows:
            a=view(mapped,actions)
            encoded.append(Encoded(view(mapped,state),a,view(mapped,seats),tuple(range(len(a))),version))
            distributions.append(view(mapped,q) if q is not None else None)
        tensors=collate(encoded)
        mixture=np.zeros(tensors[2].shape,dtype=np.float32)
        for i,q in enumerate(distributions):
            if q is not None:mixture[i,:len(q)]=q
        arrays=(np.asarray(indices,dtype=np.int64),*(t.numpy() for t in tensors),mixture)
        with open(output,'wb') as stream:layout=[write_array(stream,a) for a in arrays]
        return dict(path=output,layout=layout,bytes=Path(output).stat().st_size,
            seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu,pid=os.getpid())
    finally:
        encoded.clear();distributions.clear()
        mapped._mmap.close()


CRITIC_COLUMNS=('targets','utility','remaining','descriptors','seat')


def prepare_critic(sources,indices,output):
    """Read already-sealed dense PPG mmaps directly; no staging or unpacking."""
    from .value_buffer import row_dtype
    from brass_ai.learning.local_entities import check_action_selectors
    began=time.perf_counter();cpu=time.process_time()
    ids=np.asarray(indices,dtype=np.int64);ends=np.cumsum([n for _,n in sources])
    require(ids.ndim==1 and ((ids>=0)&(ids<ends[-1])).all(),'Invalid prepared critic indices')
    owners=np.searchsorted(ends,ids,side='right');rows=np.empty(len(ids),dtype=row_dtype())
    for p in np.unique(owners):
        mapped=np.load(sources[p][0],mmap_mode='r',allow_pickle=False)
        try:
            require(mapped.dtype==rows.dtype and mapped.shape==(sources[p][1],),'Prepared critic layout changed')
            selected=np.flatnonzero(owners==p);local=ids[selected]-(ends[p-1] if p else 0)
            rows[selected]=mapped[local]
        finally:mapped._mmap.close()
    check_action_selectors(rows['action'])
    arrays=(ids,rows['state'],rows['action'][:,None,:],np.ones((len(ids),1),dtype=np.bool_),
        rows['mapping'],np.zeros((len(ids),1),dtype=np.float32),*(rows[k] for k in CRITIC_COLUMNS))
    shared=SharedMemory(name=output);layout=[];offset=0
    try:
        for array in arrays:
            layout.append((offset,array.shape,array.dtype.str))
            require(offset+array.nbytes<=shared.size,'PPG shared batch exceeds transport capacity')
            np.ndarray(array.shape,dtype=array.dtype,buffer=shared.buf,offset=offset)[...]=array
            offset+=array.nbytes
    finally:shared.close()
    return dict(shared=output,layout=layout,bytes=offset,
        seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu,pid=os.getpid(),critic=True)


@contextmanager
def phase_preparation(pool,workers,*,directory,enabled):
    """Share the PPO pool in training; own a bounded pool in standalone calls."""
    with ExitStack() as stack:
        if not enabled or not workers:
            yield None;return
        if pool is None:pool=stack.enter_context(PreparationPool(workers))
        require(pool.workers==workers,'PPG/PPO preparation worker count differs')
        yield stack.enter_context(Preparation(pool,directory=directory))


class PreparationPool:
    """Persistent across iterations; at most 4 or 8 single-thread CPU workers."""
    def __init__(self,workers):
        require(type(workers) is int and workers in (0,4,8),'PPO preparation_workers must be 0, 4 or 8')
        self.workers=workers;self.pool=None;self.ready=[]

    def __enter__(self):
        if self.workers:
            self.pool=ProcessPoolExecutor(max_workers=self.workers,
                mp_context=multiprocessing.get_context('spawn'),initializer=initialize)
            self.ready=[self.pool.submit(ready) for _ in range(self.workers)]
        return self

    def __exit__(self,*args):
        if self.pool is not None:self.pool.shutdown(wait=True,cancel_futures=True)


class Preparation:
    def __init__(self,pool,*,directory=None):
        require(pool.pool is not None,'PPO CPU preparation pool is not running')
        self.pool=pool;self.directory=directory or SOURCE_ROOT/'artifacts'
        self.cache={};self.serial=0;self.futures=set();self.pids=set();self.shared={};self.free_shared=[]
        self.metrics=dict(workers=pool.workers,batches=0,discarded_batches=0,input_bytes=0,output_bytes=0,
            staging_seconds=0.,worker_seconds=0.,worker_cpu_seconds=0.,wait_seconds=0.,shared_memory_batches=0,
            transfer_seconds=0.,peak_inflight=0,peak_output_bytes=0)

    def __enter__(self):
        # TemporaryDirectory owns this new path. Never place replay or checkpoints inside it.
        self.temp=tempfile.TemporaryDirectory(prefix='brass-ppo-',dir=self.directory)
        self.path=Path(self.temp.name);self.input_path=self.path/'inputs.bin'
        self.stream=self.input_path.open('w+b',buffering=1024*1024)
        return self

    def __exit__(self,*args):
        try:
            for future in self.futures:future.cancel()
            # Workers must release mapped files before Windows can retire them.
            for future in self.futures:
                if not future.cancelled():
                    try:future.result()
                    except BaseException:pass
        finally:
            for shared in self.shared.values():
                shared.close();shared.unlink()
            self.shared.clear();self.free_shared.clear()
            self.stream.close();self.cache.clear();self.temp.cleanup()

    def _shared_slot(self,count):
        from .value_buffer import row_dtype
        # The complete row contains more fields than the transport, plus an
        # index/mask/mixture. The worker checks every write against this bound.
        size=count*(row_dtype().itemsize+64)
        if self.free_shared:
            name=self.free_shared.pop();shared=self.shared[name]
            if shared.size>=size:return name
            shared.close();shared.unlink();del self.shared[name]
        require(len(self.shared)<self.pool.workers,'PPG shared prefetch exceeds worker bound')
        shared=SharedMemory(create=True,size=size);self.shared[shared.name]=shared
        return shared.name

    def _rows(self,dataset,part,exploration,value_only):
        began=time.perf_counter();result=[]
        for index,row in zip(part,dataset._batch_rows(part,value_only)):
            q=dataset.q[index] if exploration else None
            arrays=(row.state,row.actions,row.stable_to_relative)
            # Recheck source bytes on reuse: stale snapshots cannot hide mutation.
            seal=tuple(array_digest(a) for a in arrays)+(array_digest(q) if q is not None else None,)
            key=(id(dataset),index,bool(exploration),value_only)
            cached=self.cache.get(key)
            if cached is None:
                specs=[write_array(self.stream,a) for a in arrays]
                spec=(*specs,write_array(self.stream,q) if q is not None else None,row.feature_version)
                self.cache[key]=(dataset,seal,spec) # keep owner alive; object IDs cannot be reused
            else:
                require(cached[0] is dataset and cached[1]==seal,'PPO preparation inputs changed after admission')
                spec=cached[2]
            result.append(spec)
        self.stream.flush();self.metrics['input_bytes']=self.stream.tell()
        self.metrics['staging_seconds']+=time.perf_counter()-began
        return result

    def _record(self,record,*,discarded=False):
        self.metrics['worker_seconds']+=record['seconds'];self.metrics['worker_cpu_seconds']+=record['cpu_seconds']
        self.metrics['discarded_batches' if discarded else 'batches']+=1
        self.metrics['output_bytes']+=record['bytes'];self.pids.add(record['pid'])
        if record.get('shared'):self.metrics['shared_memory_batches']+=1
        # Largest completed output times the prefetch width bounds disk results.
        self.metrics['peak_output_bytes']=max(self.metrics['peak_output_bytes'],record['bytes']*self.pool.workers)

    def batches(self,dataset,parts,exploration=0,*,value_only=False,cache=False):
        parts=iter(parts);pending=deque()
        sources=dataset.preparation_source() if value_only and hasattr(dataset,'preparation_source') else None
        require(sources is None or not exploration,'Critic preparation cannot alter behavior probabilities')
        def submit():
            part=next(parts,None)
            if part is None:return
            if sources is not None:
                output=self._shared_slot(len(part))
                future=self.pool.pool.submit(prepare_critic,sources,part,output)
            else:
                output=str(self.path/f'batch-{self.serial}.bin');self.serial+=1
                rows=self._rows(dataset,part,exploration,value_only)
                future=self.pool.pool.submit(prepare,str(self.input_path),self.stream.tell(),rows,part,output)
            self.futures.add(future);pending.append((part,future,output))
            self.metrics['peak_inflight']=max(self.metrics['peak_inflight'],len(pending))
        for _ in range(self.pool.workers):submit()
        try:
            while pending:
                part,future,output=pending.popleft();began=time.perf_counter()
                record=future.result();self.futures.remove(future)
                self.metrics['wait_seconds']+=time.perf_counter()-began
                self._record(record)
                began=time.perf_counter()
                mapped=self.shared[output].buf if sources is not None else np.memmap(record['path'],dtype=np.uint8,mode='r')
                try:
                    # Own buffers before releasing the file; CUDA pins are independent snapshots.
                    host=tuple(torch.from_numpy(view(mapped,s).copy()) for s in record['layout'])
                    if torch.device(dataset.device).type=='cuda':host=tuple(t.pin_memory() for t in host)
                    tensors=tuple(t.to(dataset.device,non_blocking=True) for t in host)
                finally:
                    if sources is not None:self.free_shared.append(output)
                    else:mapped._mmap.close();Path(record['path']).unlink()
                self.metrics['transfer_seconds']+=time.perf_counter()-began
                submit()
                if record.get('critic'):
                    dataset.data.prepared=(tensors[0],list(part),dict(zip(CRITIC_COLUMNS,tensors[6:])))
                    try:yield part,tensors[0],tensors[1:5],tensors[5]
                    finally:dataset.data.prepared=None
                else:yield part,tensors[0],tensors[1:-1],tensors[-1]
        finally:
            # No result from a cancelled/early-stopped scan can enter a later batch.
            for _,future,_ in pending:future.cancel()
            for _,future,output in pending:
                try:
                    if not future.cancelled():self._record(future.result(),discarded=True)
                except BaseException:
                    # An unused prefetched batch cannot replace the original
                    # failure/early stop. Iteration cleanup still owns the file.
                    pass
                finally:
                    self.futures.discard(future)
                    if sources is not None:self.free_shared.append(output)
                    else:Path(output).unlink(missing_ok=True)

    def report(self):
        return dict(self.metrics,worker_pids=sorted(self.pids),
            scope='Ordered CPU-only process collation; RNG and sampling in main process. '
                  'PPO uses private mapped staging; dense PPG reads sealed mmaps directly and uses at most one shared result slot per worker. Owned files/shared buffers are released before return. '
                  'peak_output_bytes is a conservative bound, not process RSS.')
