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
    future_turns: int = 1
    max_choices: int = 512
    exploration: float = 1.4
    root_top_k: int = 16
    interior_top_k: int = 8
    widening: float = 2.0
    pending_paths: int = 8
    batch_size: int = 8
    flush_ms: float = .5
    prior_mix: float = .1
    turn_interval: int = 4
    max_searches_per_seat: int = 4
    workers: int = 1
    cpu_role: str = 'reference'

    def validate(self):
        require(self.backend == 'masked_ismcts', 'Only Masked ISMCTS is supported')
        require(type(self.workers) is int and 1<=self.workers<=16 and self.cpu_role in ('lab','laptop','reference'),'Invalid search CPU allowance')
        for key in ('simulations','future_turns','max_choices','root_top_k','interior_top_k',
                    'pending_paths','batch_size','turn_interval'):
            require(type(getattr(self,key)) is int and getattr(self,key)>0, 'Invalid '+key)
        require(self.future_turns<=4 and self.pending_paths<=32 and self.batch_size<=256, 'Excessive search capacity')
        require(type(self.max_searches_per_seat) is int and self.max_searches_per_seat>=0, 'Invalid search limit')
        require(all(math.isfinite(v) and v>0 for v in (self.seconds,self.exploration,self.widening,self.flush_ms)), 'Invalid search clock/selection')
        require(self.flush_ms<=10 and 0<=self.prior_mix<=.5, 'Invalid batching/teacher mixture')
        return self

    def to_dict(self): return asdict(self)


def expansion_indices(choices, priors, visits, expanded, rng, *, root=True, config=None):
    """Temporary top-K plus protected families and independently sampled tail.

    Legal/padding masks never depend on this shortlist. Completed existing edges
    are retained. Every fourth expansion and the first permit a tail admission.
    """
    config=config or SearchConfig()
    require(len(choices)==len(priors)>0 and len({c['id'] for c in choices})==len(choices), 'Invalid legal choice order')
    p=np.asarray(priors,dtype=float)
    require(np.isfinite(p).all() and (p>=0).all() and p.sum()>0, 'Invalid legal prior')
    width=min(len(choices),(config.root_top_k if root else config.interior_top_k)+int(config.widening*math.sqrt(visits)))
    order=sorted(range(len(choices)),key=lambda i:(-p[i],choices[i]['id']))
    active=set(order[:width]) | {i for i,c in enumerate(choices) if c['id'] in expanded}
    families={}
    for i in order:
        c=choices[i]; key=(c['kind'],c.get('industry') if c['kind'] in ('build','develop') else None,
                          c.get('merchant') if c['kind']=='beer' else None)
        families.setdefault(key,i)
    active.update(families.values())
    tail=[i for i in order if i not in active]
    if tail and (len(expanded)==0 or len(expanded)%4==0):
        weights=[.9*p[i]+.1/len(choices) for i in tail]
        active.add(rng.choices(tail,weights=weights)[0])
    return sorted(active)
