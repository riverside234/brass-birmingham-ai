"""Critic-only seat metadata; independent of Features-4 and the actor inputs."""
import hashlib
import math
import numpy as np
import torch

from .contracts import require

CONTRACT='seat-descriptors/1.0.0'
ROLES=('learner','snapshot','il','rule_bot','unknown')
FAMILIES=('build','network','develop','sell','loan','scout','pass','other')
INDUSTRIES=('CottonMill','Manufacturer','Pottery','Brewery','IronWorks','CoalMine')
DIM=34
EMBEDDING=32
DEFAULT=dict(contract=CONTRACT,roles=list(ROLES),lag_scale=256,
    fingerprint=dict(phase='action',families=list(FAMILIES),build_industries=True),
    embedding=EMBEDDING,dropout_games=.25)


def validate_config(c):
    require(type(c) is dict and set(c)==set(DEFAULT) and
            all(c[k]==DEFAULT[k] for k in DEFAULT if k!='dropout_games'),
            'Invalid seat descriptor contract/layout')
    require(type(c['dropout_games']) in (int,float) and math.isfinite(c['dropout_games'])
            and 0<=c['dropout_games']<=1,'Invalid descriptor game dropout')
    return c


def unknown_numpy():
    result=np.zeros((4,DIM),dtype=np.float32);result[:,4]=1
    return result


def unknown_tensor(reference):
    result=reference.new_zeros((len(reference),4,DIM));result[:,:,4]=1
    return result


def validate(values):
    a=np.asarray(values,dtype=np.float32)
    require(a.shape==(4,DIM),'Invalid seat descriptor values')
    return _validate_values(a)


def validate_batch(values):
    a=np.asarray(values,dtype=np.float32)
    require(a.shape==(len(values),4,DIM),'Invalid seat descriptor values')
    return _validate_values(a)


def _validate_values(a):
    require(np.isfinite(a).all() and (a>=0).all() and (a<=1).all(),'Invalid seat descriptor values')
    roles=a[...,:5]
    require(((roles==0)|(roles==1)).all() and (roles.sum(-1)==1).all(),
            'Seat descriptor role must be one-hot')
    require(np.all(a[...,5:][a[...,4]==1]==0),'Unknown descriptors must have zero lag/style')
    return a


def descriptor(role,lag,fingerprint):
    require(role in ROLES and type(lag) is int and lag>=0,'Invalid seat role/generation lag')
    fp=np.asarray(fingerprint,dtype=np.float32)
    require(fp.shape==(28,) and np.isfinite(fp).all() and (fp>=0).all() and (fp<=1).all(),
            'Invalid policy fingerprint')
    a=np.zeros(DIM,dtype=np.float32);a[ROLES.index(role)]=1
    if role!='unknown':a[5]=min(1.,math.log1p(lag)/math.log1p(256));a[6:]=fp
    return a.tolist()


def relative(stable,mapping):
    """mapping[stable seat] -> relative seat, identical to outcome-head mapping."""
    a=validate(stable);m=np.asarray(mapping)
    require(m.shape==(4,) and sorted(m.tolist())==list(range(4)),'Invalid descriptor seat mapping')
    return a[np.argsort(m)].copy()


def drop_game(game_id,generation,epoch,fraction,phase):
    """The dropout rule: one decision per game for a (generation, epoch, phase). Trainers apply it through
    `dropout_mask`, which evaluates it once per game, never per row."""
    key=f'{CONTRACT}:dropout:{phase}:{generation}:{epoch}:{game_id}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')<int(fraction*2**64)


def dropout_mask(dataset,indices,dropout):
    """`drop_game` of each row's game, as a boolean array. Rows of one game share the decision, so it is made once
    per game and dropout setting and kept on the dataset. A dataset that numbers its games (`game_codes`, `game_names`:
    the sealed PPG critic replay) maps rows to games with one vectorized gather; any other reads each row's game id."""
    generation,epoch,fraction,phase=dropout
    if isinstance(indices,torch.Tensor):indices=indices.detach().cpu().tolist()
    decided=getattr(dataset,'_descriptor_dropout',None)
    if decided is None:decided=dataset._descriptor_dropout={}
    key=(generation,epoch,fraction,phase)
    codes=getattr(dataset,'game_codes',None)
    if codes is not None:
        table=decided.get(key)
        if table is None:
            names=dataset.game_names()
            table=decided[key]=np.fromiter((drop_game(g,generation,epoch,fraction,phase) for g in names),dtype=bool,count=len(names))
        return table[codes(indices)]
    games=decided.setdefault(key,{});result=np.empty(len(indices),dtype=bool)
    for j,i in enumerate(indices):
        g=dataset.data[i][2][0];value=games.get(g)
        if value is None:value=games[g]=drop_game(g,generation,epoch,fraction,phase)
        result[j]=value
    return result


def critic_inputs(dataset,indices,*,dropout=None,unknown=False,index=None):
    """Return true relative descriptors unless explicitly fitting critic loss.

    A column that gathers by host row lists (`host_indices`: the PPG critic replay, whose batches carry their prepared
    columns) is read with the list itself, so no index tensor makes a device round trip. `index`: the same rows as a
    tensor on the dataset's device (a batch's own index), used by device-resident descriptors instead of a new upload."""
    if not hasattr(dataset,'descriptors'):
        return unknown_tensor(torch.empty((len(indices),),device=dataset.device))
    if getattr(dataset.descriptors,'host_indices',False) and not isinstance(indices,torch.Tensor):
        descriptors=dataset.descriptors[list(indices)]
    else:
        descriptors=dataset.descriptors[index if index is not None else torch.as_tensor(indices,device=dataset.device)]
    if unknown:return unknown_tensor(descriptors)
    if dropout is not None:
        mask=dropout_mask(dataset,indices,dropout)
        if mask.any():
            rows=torch.from_numpy(mask)
            # A pinned non-blocking upload: no stream wait behind the batch's queued kernels.
            rows=rows.pin_memory().to(dataset.device,non_blocking=True) if torch.device(dataset.device).type=='cuda' else rows.to(dataset.device)
            # The same values as assigning the unknown descriptors to the dropped rows, without a boolean-mask index
            # (which reads the selected count back from the device).
            descriptors=torch.where(rows[:,None,None],unknown_tensor(descriptors),descriptors)
    return descriptors
