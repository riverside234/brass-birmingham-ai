"""Owned ragged policy inputs and frozen probabilities, local to one PPG phase."""
import time
import numpy as np
import torch

from brass_ai.learning.contracts import require
from brass_ai.learning.features import check_encoded
from brass_ai.learning.local_entities import check_action_selectors


class PolicyInputs:
    def __init__(self,dataset,ids):
        began=time.perf_counter();self.device=torch.device(dataset.device)
        self.ids=list(ids);self.lookup={i:j for j,i in enumerate(self.ids)}
        require(len(self.lookup)==len(ids) and ids,'Invalid PPG policy cache rows')
        rows=[dataset.encoded[i] for i in ids]
        self.lengths=np.array([len(r.choice_ids) for r in rows],dtype=np.int64)
        offsets=np.r_[0,np.cumsum(self.lengths)[:-1]];total=int(self.lengths.sum())
        width=rows[0].actions.shape[1];state_width=len(rows[0].state)
        self.bytes=len(ids)*(state_width*4+4*8+2*8)+total*(width+1)*4
        self.storage=self.device
        if self.device.type=='cuda' and self.bytes>min(8*1024**3,torch.cuda.mem_get_info(self.device)[0]//3):
            self.storage=torch.device('cpu')
        self.states=torch.empty((len(ids),state_width),dtype=torch.float32,device=self.storage)
        self.mapping=torch.empty((len(ids),4),dtype=torch.long,device=self.storage)
        self.actions=torch.empty((total,width),dtype=torch.float32,device=self.storage)
        self.old=torch.empty(total,dtype=torch.float32,device=self.storage)
        self.offsets=torch.tensor(offsets,device=self.storage)
        self.sizes=torch.tensor(self.lengths,device=self.storage)
        # Full selector admission before taking our own immutable tensor copies.
        # Upload bounded contiguous chunks, never one transfer per row or an
        # eight-age globally padded action array.
        for start in range(0,len(rows),64):
            group=rows[start:start+64];end=start+len(group)
            for row in group:check_encoded(row);check_action_selectors(row.actions)
            self.states[start:end]=torch.from_numpy(np.stack([r.state for r in group])).to(self.storage)
            self.mapping[start:end]=torch.from_numpy(np.stack([r.stable_to_relative for r in group])).to(self.storage)
            a=int(offsets[start]);b=int(offsets[end]) if end<len(rows) else total
            self.actions[a:b]=torch.from_numpy(np.concatenate([r.actions for r in group])).to(self.storage)
        self.build_seconds=time.perf_counter()-began;self.batches=0

    def microbatches(self,indices,size,cells,bucket=True):
        indices=list(indices)
        if bucket:indices.sort(key=lambda i:int(self.lengths[self.lookup[i]]))
        part=[];maximum=0
        for i in indices:
            n=int(self.lengths[self.lookup[i]])
            if part and (len(part)>=size or (len(part)+1)*max(maximum,n)>cells):
                yield part;part=[];maximum=0
            part.append(i);maximum=max(maximum,n)
        if part:yield part

    def _indices(self,part,width=None):
        local=[self.lookup[i] for i in part]
        width=int(self.lengths[local].max()) if width is None else width
        ix=torch.tensor(local,device=self.storage)
        column=torch.arange(width,device=self.storage)[None,:]
        mask=column<self.sizes[ix,None]
        positions=(self.offsets[ix,None]+column).clamp_max(len(self.old)-1)
        return ix,positions,mask

    def inputs(self,part):
        ix,positions,mask=self._indices(part);self.batches+=1
        actions=self.actions[positions].masked_fill(~mask[:,:,None],0.)
        return tuple(t.to(self.device) for t in (self.states[ix],actions,mask,self.mapping[ix]))

    def store_old(self,part,probabilities):
        # Only executed once before the first auxiliary update.
        _,positions,mask=self._indices(part,probabilities.shape[1])
        self.old[positions[mask]]=probabilities.detach().to(self.storage)[mask]

    def probabilities(self,part,width):
        _,positions,mask=self._indices(part,width)
        return self.old[positions].masked_fill(~mask,0.).to(self.device)

    def report(self):
        return dict(rows=len(self.ids),candidate_rows=len(self.old),bytes=self.bytes,
            device=str(self.storage),build_seconds=self.build_seconds,batches=self.batches,
            scope='Phase-local owned complete legal inputs and post-PPO raw probabilities; current policy recomputed after every update. No cross-phase reuse.')
