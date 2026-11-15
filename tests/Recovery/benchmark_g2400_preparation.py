"""Paired in-memory PPO packing benchmark; no lab disk or GPU speed claims."""
import io
import json
from pathlib import Path
import statistics
import tempfile
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch

from brass_ai.learning.batching import collate
from brass_ai.learning.features import Encoded
from brass_ai.learning.contracts import FEATURE_VERSION,SOURCE_ROOT
from brass_ai.recovery import ppo_preparation as P


def main():
    torch.set_num_threads(1)
    widths=[64]*24+[128]*24+[512]*16
    rows=[]
    for n in widths:
        a=np.zeros((n,327),dtype=np.float32);a[:,0]=1
        rows.append(Encoded(np.ones(3128,dtype=np.float32),a,np.arange(4,dtype=np.int64),tuple(map(str,range(n))),FEATURE_VERSION))
    with tempfile.TemporaryDirectory(dir=SOURCE_ROOT/'artifacts') as folder:
        source=Path(folder)/'input';specs=[]
        with source.open('wb') as stream:
            for row in rows:
                s=[P.write_array(stream,a) for a in (row.state,row.actions,row.stable_to_relative)]
                specs.append((*s,None,row.feature_version))
        layout,size=P.batch_layout(specs);shared=SharedMemory(create=True,size=size)
        destination=dict(shared=shared.name,layout=layout,bytes=size)
        indices=list(range(len(rows)));times={'previous':[],'shared':[]}
        def previous():
            # Previous worker packed/validated arrays, then copied all of them
            # to RAM-backed files. BytesIO models that copy without disk latency.
            tensors=collate(rows)
            arrays=(np.asarray(indices,dtype=np.int64),*(t.numpy() for t in tensors),np.zeros(tensors[2].shape,np.float32))
            stream=io.BytesIO();old_layout=[P.write_array(stream,a) for a in arrays]
            return stream.getvalue(),old_layout
        def current():return P.prepare(source,source.stat().st_size,specs,indices,destination)
        try:
            expected,old_layout=previous();current()
            for old,new in zip(old_layout,layout):np.testing.assert_array_equal(P.view(expected,old),P.view(shared.buf,new))
            for i in range(12):
                for name,fn in ([('previous',previous),('shared',current)] if i%2 else [('shared',current),('previous',previous)]):
                    started=time.perf_counter();result=fn();times[name].append(time.perf_counter()-started);del result
            report=dict(rows=len(rows),legal_candidates=sum(widths),padded_cells=len(rows)*max(widths),output_bytes=size,
                        seconds={name:statistics.median(v) for name,v in times.items()},exact_tensor_parity=True,
                        scope='Windows CPU, paired warm in-memory packing; previous output copied to BytesIO to exclude disk latency. Not a full lab PPO benchmark.')
            report['speedup']=report['seconds']['previous']/report['seconds']['shared']
            output=SOURCE_ROOT/'artifacts/rust-cotton-g2400/packing-benchmark.json'
            output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2))
        finally:shared.close();shared.unlink()


if __name__=='__main__':main()
