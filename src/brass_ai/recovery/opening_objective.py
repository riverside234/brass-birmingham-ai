"""Opening-only T1 PPO surrogate; collection probabilities remain authoritative."""
import copy

import torch

from brass_ai.learning.contracts import require

CONTRACT='opening-ppo-objective/1.0.0'
MIGRATION='decoupled-opening-ppo/1.0.0'
ZERO_EXPLORATION_MIGRATION='ordinary-zero-exploration/1.0.0'
DEFAULT=dict(contract=CONTRACT,scope='temperature-2-action-openings',target_temperature=1.,
    importance_cap=2.,normalization='none',old_policy='frozen-before-ppo',
    kl_guard='behavior-and-raw-temperature-1')
ROW_KEY='_ppo_opening_objective'
T15=dict(DEFAULT,contract='opening-ppo-objective/2.0.0',scope='temperature-1.5-action-openings')
T13=dict(DEFAULT,contract='opening-ppo-objective/3.0.0',scope='temperature-1.3-three-action-openings')
T2_THREE=dict(DEFAULT,contract='opening-ppo-objective/4.0.0',scope='temperature-2-three-action-openings')


def opening_spec(config):
    from .exploration import OPENING,REVISED_OPENING,THREE_OPENING,THREE_T2_OPENING
    validate(config)
    return THREE_T2_OPENING if config==T2_THREE else THREE_OPENING if config==T13 else REVISED_OPENING if config==T15 else OPENING


def opening_temperature(config):return opening_spec(config)['temperature']


def validate(c):
    require(type(c) is dict and c in (DEFAULT,T15,T13,T2_THREE) and
        all(type(c[k]) in (int,float) for k in ('target_temperature','importance_cap')),
        'Invalid decoupled opening PPO objective')
    return c


def configure(config):
    result=copy.deepcopy(config)
    from .exploration import OPENING
    require(result.get('opening')==OPENING and result['update'].get('actor_nonforced'),
        'Decoupled PPO requires temperature-2 openings and the nonforced actor')
    result['update']['opening_objective']=copy.deepcopy(DEFAULT)
    return result


def zero_exploration(config):
    result=copy.deepcopy(config)
    require(type(result.get('exploration')) in (int,float) and result['exploration'] in (0.,.02),
        'Zero exploration migration requires the current .02 recipe')
    result['exploration']=0.
    return result


def losses(new_behavior,old_behavior,new_t1,old_t1,opening,advantages,clip,cap=2.):
    """Per-row loss; caller divides by the full batch size, never sum(weights)."""
    old_behavior=old_behavior.detach();old_t1=old_t1.detach()
    ratio=torch.where(opening,new_t1/old_t1,new_behavior/old_behavior)
    weight=torch.where(opening,(old_t1/old_behavior).clamp(max=cap),torch.ones_like(old_behavior)).detach()
    return -weight*torch.minimum(ratio*advantages,ratio.clamp(1-clip,1+clip)*advantages),ratio,weight


class Reference:
    """Selected raw T1 probabilities captured during strict behavior verification."""
    def __init__(self,dataset,behavior,config):
        self.dataset=dataset;self.config=validate(config);self.frozen=False
        spec=opening_spec(config);temperature=spec['temperature'];decisions=spec['action_decisions_per_seat']
        require(getattr(behavior,'exploration_version',2)==3 and behavior.temperature==1. and
            (behavior.opening is None or (behavior.opening['temperature']==temperature and
             behavior.opening['action_decisions_per_seat']==decisions)),'Decoupled PPO behavior contract differs')
        selected=[]
        for row in dataset.rows:
            require(row.get(ROW_KEY)==config,'Missing/stale opening objective collection provenance')
            require(not row.get('_copied_prefix',False) and row['behavior'].get('component')!='branch_move',
                'Fork moves/copied prefixes cannot enter PPO')
            b=row['behavior'];opening=b['temperature']==temperature
            require(b['temperature'] in (1.,temperature),'Stale opening temperature in PPO rows')
            require(not opening or (row['view']['context']['phase']=='action' and
                type(b['action_decision']) is int and 0<=b['action_decision']<decisions),
                'Tempered row is not an opening action')
            selected.append(opening and len(row['choices'])>1)
        self.mask=torch.tensor(selected,device=dataset.device,dtype=torch.bool)
        self.old=torch.ones_like(dataset.old)
        self.seen=torch.zeros_like(self.mask)

    def capture(self,indices,logits):
        require(not self.frozen,'Old T1 probabilities are already frozen')
        with torch.no_grad():
            p=logits.float().softmax(-1).gather(1,self.dataset.chosen[indices,None]).squeeze(1)
            self.old[indices]=p.clamp_min(1e-30);self.seen[indices]=True

    def freeze(self):
        require(bool(self.seen.all()) and bool(torch.isfinite(self.old).all()) and bool((self.old>0).all()),
            'Incomplete/nonfinite pre-update T1 reference')
        require(not self.old.requires_grad,'Old T1 reference must be detached')
        self.frozen=True

    def loss(self,indices,logits,new_behavior,advantages,clip):
        require(self.frozen,'T1 reference must be frozen before PPO')
        t1=logits.float().softmax(-1).gather(1,self.dataset.chosen[indices,None]).squeeze(1).clamp_min(1e-30)
        actor,ratio,_=losses(new_behavior,self.dataset.old[indices],t1,self.old[indices],
            self.mask[indices],advantages,clip,self.config['importance_cap'])
        return actor,ratio

    def report(self):
        weights=(self.old[self.mask]/self.dataset.old[self.mask]).clamp(max=self.config['importance_cap'])
        ids=self.mask.nonzero().flatten().cpu().tolist()
        return dict(config=copy.deepcopy(self.config),opening_rows=len(ids),
            branch_opening_rows=sum(bool(self.dataset.rows[i].get('_branch')) for i in ids),
            weight_mean=float(weights.mean()) if ids else None,
            weight_min=float(weights.min()) if ids else None,weight_max=float(weights.max()) if ids else None,
            weight_capped_rows=int((weights==self.config['importance_cap']).sum()),
            behavior_verified=True,old_t1_frozen=self.frozen,
            scope='Eligible nonforced opening actions only, including post-fork openings. Detached capped old-T1/actual-behavior weights, no minibatch renormalization. Returns/GAE remain from the collected behavior trajectories.')
