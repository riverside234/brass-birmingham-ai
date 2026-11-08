"""Once-only canal VP payout, then the capped terminal VP difference.

The NN and undiscounted search keep a full-game value coordinate. Future values
subtract the known paid credit analytically; features and old Adam tensors do
not change. This also avoids asking a memoryless NN to infer an earlier score.
"""
from functools import lru_cache

import numpy as np

from .contracts import require
from .reward import ERA_VERSION


def vp_payment(scores,spec):
    a=np.asarray(scores,dtype=float)
    require(a.shape==(4,) and np.isfinite(a).all(),'Invalid four-seat VP payout')
    return spec['vp']*np.minimum(a,spec['vp_cap'])


def player_scores(view):
    players=view['players']
    require(len(players)==4 and [p['seat'] for p in players]==list(range(4)),
            'VP payout requires stable-seat public scores')
    return [p['vp'] for p in players]


def replay_payments(records,result,spec,*,initial_vp_paid=None):
    if spec['version']!=ERA_VERSION:return None,None
    paid=np.zeros((len(records),4),dtype=np.float64)
    require(records,'VP payout requires a complete replay')
    eras=[r['view']['context']['era'] for r in records]
    require(all(e in ('CanalEra','RailEra') for e in eras),'Invalid payout era')
    first=next((i for i,e in enumerate(eras) if e=='RailEra'),None)
    require(first is not None and all(e=='RailEra' for e in eras[first:]),'Missing/reversed canal scoring boundary')
    if first==0:
        require(initial_vp_paid is not None,'Rail-only replay needs its parent canal payout')
        canal=np.asarray(initial_vp_paid,dtype=float)
        require(canal.shape==(4,) and np.isfinite(canal).all() and (canal>=0).all() and
                (canal<=spec['vp']*spec['vp_cap']).all(),'Invalid inherited canal payout')
    else:
        require(initial_vp_paid is None or not np.asarray(initial_vp_paid).any(),'Canal-start replay cannot inherit a paid credit')
        context=records[first]['view']['context']
        require(context['round']==1,'First rail observation is past the canal boundary')
        canal=vp_payment(player_scores(records[first]['view']),spec)
    paid[first:]=canal
    final=vp_payment([s['vp'] for s in result['seats']],spec)
    return paid,dict(contract='canal-and-final-delta/1.0.0',canal_before_record=first,
        canal=canal.tolist(),final_delta=(final-canal).tolist(),total=final.tolist(),
        inherited=first==0,coordinate='NN/search full-game utility; future utility subtracts paid canal credit')


@lru_cache(maxsize=1)
def vp_indices():
    from .state_layout import state_layout
    segments={s.name:s for s in state_layout().segments}
    return tuple(segments[f'player.{s}.vp'].start for s in range(4))


def teacher_payments(rows,result,spec):
    """Read the existing independently bound first rail root, no new IPC."""
    if spec['version']!=ERA_VERSION:return None
    rail=[r for r in rows if r['turn_key'][0]=='RailEra']
    require(len(rail)==32 and rail[0]['index']==33 and rail[0]['turn_key'][1]==1,
            'Teacher payout requires the first rail root and both complete eras')
    root=rail[0];encoded=np.asarray(root['state'],dtype=np.float32)
    # The frozen feature schema stores VP / 100 without clipping. Reconstruct
    # the public integer exactly, including observer-relative seat rotation.
    raw=np.asarray([encoded[vp_indices()[root['mapping'][s]]]*100 for s in range(4)])
    scores=np.rint(raw)
    require(np.isfinite(raw).all() and (scores>=0).all() and (scores<=10000).all() and
            np.allclose(raw,scores,rtol=0,atol=.001),'Nonintegral encoded canal scores')
    canal=vp_payment(scores,spec);final=vp_payment([s['vp'] for s in result['seats']],spec)
    return dict(contract='canal-and-final-delta/1.0.0',canal=canal.tolist(),
        final_delta=(final-canal).tolist(),total=final.tolist(),canal_before_root=33,
        coordinate='NN/search full-game utility; future utility subtracts paid canal credit')


def policy_returns(dataset):
    value=dataset.utility if dataset.utility is not None else dataset.targets[0]
    paid=getattr(dataset,'vp_paid',None)
    return value if paid is None else value-paid
