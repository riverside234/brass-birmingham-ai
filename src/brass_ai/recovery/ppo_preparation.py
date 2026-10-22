"""Ordered CPU-only process collation with a bounded, iteration-local transport.

Input arrays are copied once to a private mapped file, never to each worker.
Only small row descriptors cross the task queue. Output files are bounded by
the prefetch width and removed after consumption. No model, CUDA tensor or RNG
enters a worker. Sampling and gradient accumulation stay in the main process.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
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
        self.cache={};self.serial=0;self.futures=set();self.pids=set()
        self.metrics=dict(workers=pool.workers,batches=0,discarded_batches=0,input_bytes=0,output_bytes=0,
            staging_seconds=0.,worker_seconds=0.,worker_cpu_seconds=0.,wait_seconds=0.,
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
            self.stream.close();self.cache.clear();self.temp.cleanup()

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
        # Largest completed output times the prefetch width bounds disk results.
        self.metrics['peak_output_bytes']=max(self.metrics['peak_output_bytes'],record['bytes']*self.pool.workers)

    def batches(self,dataset,parts,exploration=0,*,value_only=False,cache=False):
        parts=iter(parts);pending=deque()
        def submit():
            part=next(parts,None)
            if part is None:return
            rows=self._rows(dataset,part,exploration,value_only)
            output=str(self.path/f'batch-{self.serial}.bin');self.serial+=1
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
                began=time.perf_counter();mapped=np.memmap(record['path'],dtype=np.uint8,mode='r')
                try:
                    # Own buffers before releasing the file; CUDA pins are independent snapshots.
                    host=tuple(torch.from_numpy(view(mapped,s).copy()) for s in record['layout'])
                    if torch.device(dataset.device).type=='cuda':host=tuple(t.pin_memory() for t in host)
                    tensors=tuple(t.to(dataset.device,non_blocking=True) for t in host)
                finally:mapped._mmap.close();Path(record['path']).unlink()
                self.metrics['transfer_seconds']+=time.perf_counter()-began
                submit()
                yield part,tensors[0],tensors[1:-1],tensors[-1]
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
                    Path(output).unlink(missing_ok=True)

    def report(self):
        return dict(self.metrics,worker_pids=sorted(self.pids),
            scope='Ordered CPU-only process collation; RNG and sampling in main process. '
                  'Private iteration-local mapped inputs, bounded output files; deleted before return. '
                  'peak_output_bytes is a conservative bound, not process RSS.')
