"""Phase-local frozen inputs for the stock head-only utility calibration."""
import time
import torch

from brass_ai.learning.utility_model import UtilityNet
from brass_ai.learning.seat_descriptors import critic_inputs


class FrozenUtilityInputs:
    def __init__(self,model,dataset,indices,config,executor=None):
        self.model=model;self.dataset=dataset;self.lookup={i:j for j,i in enumerate(indices)}
        self.columns=None;start=time.perf_counter();self.rows=len(indices)
        # Cache only frozen outputs. Keep the original predict_utility path so
        # trainable head arithmetic, seat mapping and reward priors stay exact.
        parts=dataset.microbatches(indices,config['microbatch_size'],config['max_candidate_cells'],config['bucket'])
        with torch.no_grad():
            for part,ix,inputs in dataset.value_batches(parts,model,executor):
                encoded=model.encode_state(inputs[0])
                embeddings=model.descriptor_embeddings(encoded,critic_inputs(dataset,part))
                heads=model.official_heads(encoded,inputs[3],embeddings)
                remaining=None if dataset.remaining is None else dataset.remaining[ix]
                values=(encoded.context,inputs[3],*heads,remaining,embeddings)
                if self.columns is None:
                    size=sum(v[0].numel()*v.element_size()*len(indices) for v in values if v is not None)
                    storage=inputs[0].device
                    if storage.type=='cuda' and size>min(1024**3,torch.cuda.mem_get_info(storage)[0]//8):
                        storage=torch.device('cpu')
                    self.columns=tuple(None if v is None else torch.empty((len(indices),*v.shape[1:]),dtype=v.dtype,device=storage) for v in values)
                    self.storage=storage;self.bytes=size
                local=torch.tensor([self.lookup[i] for i in part],device=self.storage)
                for dest,value in zip(self.columns,values):
                    if dest is not None:dest[local]=value.detach().to(self.storage)
        self.seconds=time.perf_counter()-start

    def predict(self,part):
        local=torch.tensor([self.lookup[i] for i in part],device=self.storage)
        values=[None if v is None else v[local].to(self.dataset.device) for v in self.columns]
        context,mapping,win,rank,vp,remaining,embeddings=values
        return self.model.predict_utility(context,mapping,(win,rank,vp),remaining,embeddings)

    def report(self):
        return dict(rows=self.rows,bytes=self.bytes,device=str(self.storage),build_seconds=self.seconds,
            scope='One calibration call only; frozen trunk/official heads/descriptors. Utility head recomputed every step; no cache survives PPO updates.')


def build(model,dataset,indices,config,executor=None):
    # Custom subclasses may have action-dependent or stochastic utility paths.
    return FrozenUtilityInputs(model,dataset,indices,config,executor) if type(model) is UtilityNet else None
