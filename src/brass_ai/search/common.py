"""Neutral observation, deadline and progressive expansion contracts."""
from dataclasses import asdict, dataclass
import math

import numpy as np

from brass_ai.learning.contracts import require


class SearchIncomplete(RuntimeError):
    pass


def turn_key(context):
    return tuple(context[k] for k in ('era', 'round', 'turn', 'turn_owner'))


@dataclass(frozen=True)
class SearchConfig:
    backend: str = 'masked_ismcts'
    seconds: float = 1.0
    simulations: int = 100000
    contract: str = 'masked-ismcts/1.1.0'
    evaluator_kind: str = 'neural'
    q_transform: str = 'none'
    leaf_policy: str = 'new_root_information_frontier'
    candidate_policy: str = 'supported_progressive'
    fpu_reduction: float = .02
    minimum_plan_visits: int = 2
    reopen_support: int = 4
    tail_admission: float = .05
    max_choices: int = 512
    exploration: float = 1.4
    root_top_k: int = 16
    interior_top_k: int = 8
    pending_paths: int = 8
    batch_size: int = 8
    flush_ms: float = .5
    prior_mix: float = .1
    turn_interval: int = 4
    max_searches_per_seat: int = 4
    workers: int = 1
    cpu_role: str = 'reference'
    virtual_loss: float = .02
    reward: dict | None = None
    utility_scale: float = 1.0

    def validate(self):
        require(self.backend == 'masked_ismcts', 'Only Masked ISMCTS is supported')
        require(math.isfinite(self.virtual_loss) and 0<=self.virtual_loss<=1,'Invalid selection-only virtual loss')
        require(type(self.workers) is int and 1<=self.workers<=16 and self.cpu_role in ('lab','laptop','reference'),'Invalid search CPU allowance')
        require(self.contract in ('masked-ismcts/1.1.0','masked-ismcts/1.2.0') and self.evaluator_kind=='neural' and self.q_transform=='none' and
                self.leaf_policy=='new_root_information_frontier' and self.candidate_policy=='supported_progressive',
                'NN search cannot inherit heuristic Q transforms or reply/depth caps')
        require(math.isfinite(self.utility_scale) and self.utility_scale>0,'Invalid fixed search utility scale')
        require((self.reward is not None)==(self.contract=='masked-ismcts/1.2.0'),'Search reward/contract differs')
        if self.reward is not None:
            from brass_ai.learning.reward import validate
            validate(self.reward)
        require(math.isfinite(self.fpu_reduction) and 0<=self.fpu_reduction<=1 and
                math.isfinite(self.tail_admission) and 0<=self.tail_admission<=.5,'Invalid FPU/admission')
        require(type(self.minimum_plan_visits) is int and self.minimum_plan_visits>=2,'Insufficient complete-plan support')
        for key in ('simulations','reopen_support','max_choices','root_top_k','interior_top_k',
                    'pending_paths','batch_size','turn_interval'):
            require(type(getattr(self,key)) is int and getattr(self,key)>0, 'Invalid '+key)
        require(self.pending_paths<=32 and self.batch_size<=256, 'Excessive search capacity')
        require(type(self.max_searches_per_seat) is int and self.max_searches_per_seat>=0, 'Invalid search limit')
        require(all(math.isfinite(v) and v>0 for v in (self.seconds,self.exploration,self.flush_ms)), 'Invalid search clock/selection')
        require(self.flush_ms<=10 and 0<=self.prior_mix<=.5, 'Invalid batching/teacher mixture')
        return self

    def to_dict(self):
        result=asdict(self)
        if self.reward is None:
            result.pop('reward');result.pop('utility_scale')
        return result


def ranked_candidates(choices, priors):
    order=sorted(range(len(choices)),key=lambda i:(-priors[i],choices[i]['id']))
    families={}
    for i in order:
        c=choices[i];key=(c['kind'],c.get('industry') if c['kind'] in ('build','develop') else None,
                         c.get('target') if c['kind']=='beer' else None)
        families.setdefault(key,i)
    return order,list(families.values())


def expansion_indices(choices, priors, visits, expanded, rng, *, root=True, config=None, ranked=None, admitted=None):
    """A temporary candidate set; full legality and policy denominators are intact.

    New choices compete with visited choices through FPU/PUCT. Only completed
    support reopens ordinary candidates; a small tail admits variety after that.
    Card/location variants remain exact, deferred choices with eventual access.
    """
    config=config or SearchConfig()
    require(len(choices)==len(priors)>0 and len({c['id'] for c in choices})==len(choices), 'Invalid legal choice order')
    p=np.asarray(priors,dtype=float)
    require(np.isfinite(p).all() and (p>=0).all() and p.sum()>0, 'Invalid legal prior')
    order,protected=ranked or ranked_candidates(choices,p)
    leading=max((getattr(e,'visits',e.get('visits',0) if isinstance(e,dict) else 0) for e in expanded.values()),default=visits)
    width=min(len(choices),(config.root_top_k if root else config.interior_top_k)+leading//config.reopen_support)
    active=set(order[:width]) | set(protected) | set(admitted or ())
    active.update(i for i,c in enumerate(choices) if c['id'] in expanded)
    tail=[i for i in order if i not in active]
    if tail and leading>=config.reopen_support and rng.random()<config.tail_admission:
        i=rng.choices(tail,weights=[.9*p[j]+.1/len(choices) for j in tail])[0];active.add(i)
        if admitted is not None:admitted.add(i)
    return sorted(active)
