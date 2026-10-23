"""Frozen true-descriptor critic outputs; no terminal targets in PPO fitting."""
from collections import Counter
import time

import torch
from torch.nn import functional as F

from brass_ai.learning.contracts import require

CONTRACT='dcpg-value-preservation/1.0.0'


def outputs(model,inputs,ds,ix,companion=None):
    from brass_ai.learning.seat_descriptors import critic_inputs
    ids=ix.detach().cpu().tolist()
    encoded=model.encode_state(inputs[0])
    embeddings=model.descriptor_embeddings(encoded,critic_inputs(ds,ids))
    heads=model.official_heads(encoded,inputs[3],embeddings)
    utility=model.predict_utility(encoded.context,inputs[3],heads,ds.remaining[ix],embeddings)
    result=dict(win=heads[0].float(),rank=heads[1].float(),vp=heads[2].float(),utility=utility.float())
    if companion is not None and companion.active:
        for kind,(owner,vp) in companion.heads(encoded,embeddings,companion.scale).items():
            result[kind+'_owner']=owner.float().softmax(-1)
            result[kind+'_vp']=vp.float()
    return result


def divergences(current,old,companion=None):
    require(current.keys()==old.keys(),'Preservation output identity changed')
    def kl(a,b):
        return (a*(a.clamp_min(1e-30).log()-b.clamp_min(1e-30).log())).sum(-1)
    win=kl(old['win'],current['win'])
    # v11.2 rank is a bounded scalar per seat, not a rank categorical head.
    # Keep the existing normalized Huber forms instead of inventing logits.
    rank=F.smooth_l1_loss((current['rank']-1)/3,(old['rank']-1)/3,reduction='none').mean(-1)
    vp=F.smooth_l1_loss(current['vp']/100,old['vp']/100,reduction='none').mean(-1)
    utility=(current['utility']-old['utility']).square().mean(-1)
    local=torch.zeros_like(win)
    if companion is not None and companion.active:
        w=companion.config['loss_weights']
        for kind in ('slots','links'):
            local+=w[kind+'_owner']*kl(old[kind+'_owner'],current[kind+'_owner']).mean(-1)
            local+=w[kind+'_vp']*F.smooth_l1_loss(current[kind+'_vp'],old[kind+'_vp'],reduction='none').mean((1,2))
    return dict(core=win+.25*rank+.1*vp+utility,win=win,rank=rank,vp=vp,utility=utility,local=local)


class FrozenValues:
    def __init__(self,model,ds,update,companion=None,executor=None):
        from .updates import autocast
        from .value_phase import value_parts
        self.ds=ds;self.update=update;self.companion=companion;self.values={};self.gradients=[]
        started=time.perf_counter();was_training=model.training;model.eval()
        self.rows=len(ds.data)
        try:
            with torch.no_grad():
                for part,ix,inputs in ds.value_batches(value_parts(ds,range(self.rows),update,model),model,executor):
                    with autocast(update,ds.device):out=outputs(model,inputs,ds,ix,companion)
                    for k,v in out.items():
                        require(bool(torch.isfinite(v).all()),'Nonfinite frozen critic output')
                        if k not in self.values:self.values[k]=torch.empty((self.rows,*v.shape[1:]),dtype=torch.float32,device='cpu')
                        self.values[k][part]=v.detach().cpu()
        finally:model.train(was_training)
        self.seconds=time.perf_counter()-started

    def selected(self,indices,device):
        ids=indices.detach().cpu() if isinstance(indices,torch.Tensor) else indices
        return {k:v[ids].to(device) for k,v in self.values.items()}

    def losses(self,model,inputs,ix):
        out=outputs(model,inputs,self.ds,ix,self.companion)
        return divergences(out,self.selected(ix,inputs[0].device),self.companion)

    def report(self,model,train_ids,probe_ids,executor=None):
        from .updates import autocast
        from .value_phase import value_parts
        totals=torch.zeros(7,device=self.ds.device);weighted=torch.zeros_like(totals)
        keys=('core','win','rank','vp','utility','local');model.eval()
        with torch.no_grad():
            for part,ix,inputs in self.ds.value_batches(value_parts(self.ds,train_ids,self.update,model),model,executor):
                with autocast(self.update,self.ds.device):v=self.losses(model,inputs,ix)
                combined=self.update['value_weight']*v['core']+v['local']
                tensors=[v[k] for k in keys]+[combined]
                totals+=torch.stack([x.sum() for x in tensors])
                weighted+=torch.stack([(x*self.ds.critic_weights[ix]).sum() for x in tensors])
            drift=torch.zeros((),device=self.ds.device)
            for part,ix,inputs in self.ds.value_batches(value_parts(self.ds,probe_ids,self.update,model),model,executor):
                with autocast(self.update,self.ds.device):out=outputs(model,inputs,self.ds,ix)
                drift+=(out['utility']-self.values['utility'][part].to(inputs[0].device)).square().sum()
        means=dict(zip((*keys,'weighted_objective'),(totals/max(1,len(train_ids))).cpu().tolist()))
        return dict(contract=CONTRACT,beta=1.,value_weight=self.update['value_weight'],fresh_target_steps=0,
            frozen_rows=self.rows,train_rows=len(train_ids),frozen_by_role=dict(Counter(self.ds.roles)),
            frozen_bytes=sum(v.numel()*v.element_size() for v in self.values.values()),
            floats_per_row=sum(v[0].numel() for v in self.values.values()) if self.rows else 0,
            storage='CPU float32, iteration-local; never reused by PPO or replay',snapshot_seconds=self.seconds,
            end_loss=means,end_weighted_objective=float(weighted[-1])/max(1,len(train_ids)),
            probe_rows=len(probe_ids),utility_probe_rms=float((drift/max(1,4*len(probe_ids))).sqrt()),
            gradients=self.gradients,
            scope='After PPO, before PPG. All critic-train rows for loss; nonforced learner PPO probe for four-seat utility RMS. True descriptors, no dropout; heldouts excluded from fitting. Core uses beta*value_weight; existing local-head weights are preserved separately.')


class TrunkGradients:
    """Exact pre-clipping actor and preservation gradients without changing them."""
    def __init__(self,model):
        self.parameters=tuple(p for name,p in model.named_parameters()
            if name.startswith(('state_encoder.','entity_encoder.')) and p.requires_grad)
        self.actor=[torch.zeros_like(p) for p in self.parameters]

    def add_actor(self,loss):
        values=torch.autograd.grad(loss,self.parameters,retain_graph=True,allow_unused=True)
        for total,g in zip(self.actor,values):
            if g is not None:total.add_(g.detach())

    def before_value(self):
        # Keep actor+entropy gradients aside while accumulating preservation.
        # Subtracting two nearly equal accumulated float32 gradients destroys
        # the direction of a small preservation term. Recombine before clipping.
        self.before=[p.grad for p in self.parameters]
        for p in self.parameters:p.grad=None

    def finish(self,step):
        norm_a=norm_v=dot=self.actor[0].new_zeros(())
        for a,p,b in zip(self.actor,self.parameters,self.before):
            v=p.grad.detach() if p.grad is not None else torch.zeros_like(p)
            norm_a=norm_a+a.square().sum();norm_v=norm_v+v.square().sum();dot=dot+(a*v).sum()
            if b is not None:
                if p.grad is None:p.grad=b
                else:p.grad.add_(b)
        a,v,d=torch.stack((norm_a.sqrt(),norm_v.sqrt(),dot)).double().cpu().tolist()
        return dict(step=step,actor_norm=a,preservation_norm=v,cosine=d/(a*v) if a*v>0 else None,
            scope='Shared state/entity encoders; actor excludes entropy/rehearsal, preservation includes configured core/local weights; before clipping')
