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
    require(a.shape==(4,DIM) and np.isfinite(a).all() and (a>=0).all() and (a<=1).all(),
            'Invalid seat descriptor values')
    require(np.isin(a[:,:5],[0,1]).all() and (a[:,:5].sum(1)==1).all(),
            'Seat descriptor role must be one-hot')
    require(np.all(a[a[:,4]==1,5:]==0),'Unknown descriptors must have zero lag/style')
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
    key=f'{CONTRACT}:dropout:{phase}:{generation}:{epoch}:{game_id}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')<int(fraction*2**64)


def critic_inputs(dataset,indices,*,dropout=None,unknown=False):
    """Return true relative descriptors unless explicitly fitting critic loss."""
    ix=torch.as_tensor(indices,device=dataset.device)
    if not hasattr(dataset,'descriptors'):
        return unknown_tensor(torch.empty((len(indices),),device=dataset.device))
    descriptors=dataset.descriptors[ix]
    if unknown:return unknown_tensor(descriptors)
    if dropout is not None:
        generation,epoch,fraction,phase=dropout
        mask=[drop_game(dataset.data[i][2][0],generation,epoch,fraction,phase) for i in indices]
        if any(mask):
            descriptors=descriptors.clone();descriptors[torch.tensor(mask,device=dataset.device)]=unknown_tensor(descriptors)[0]
    return descriptors
