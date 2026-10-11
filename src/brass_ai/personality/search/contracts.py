from dataclasses import asdict,dataclass
from pathlib import Path
from functools import lru_cache
import math

from brass_ai.learning.contracts import digest,keys,require
from brass_ai.personality.contracts import contract,PROFILES
from brass_ai.search.common import SearchConfig

VERSION='personality-search/1.0.0'
TEACHER='personality-root-visits/1.0.0'
TRACES='personality-search-traces/1.0.0'
READY='personality-search-ready/1.0.0'


def source_id():
    return digest({p.name:p.read_text(encoding='utf-8') for p in sorted(Path(__file__).parent.glob('*.py'))})


@lru_cache(maxsize=1)
def identity():return dict(version=VERSION,source=source_id(),personality=contract())


@dataclass(frozen=True)
class Profile:
    style: str='neutral'
    alpha: float=.15
    epsilon: float=.10
    unbiased_fraction: float=.25
    patience: int=2
    hysteresis: float=.10

    def validate(self):
        require(self.style in PROFILES,'Unsupported neural style input')
        require(all(math.isfinite(x) for x in (self.alpha,self.epsilon,self.unbiased_fraction,self.hysteresis)), 'Nonfinite profile')
        require(0<=self.alpha<=.5 and self.epsilon==.10 and self.unbiased_fraction==.25,'Unreviewed prior recipe')
        require(type(self.patience) is int and 1<=self.patience<=3 and 0<=self.hysteresis<=.2,'Invalid intent hysteresis')
        return self

    def to_dict(self):return asdict(self)

    @classmethod
    def parse(cls,value):
        keys(value,cls.__dataclass_fields__);return cls(**value).validate()


def search_config(reward,*,seconds=12.,workers=4,simulations=100000,unpruned=False):
    return SearchConfig(seconds=seconds,workers=workers,cpu_role='lab',pending_paths=2,batch_size=2,
        reward=reward,contract='masked-ismcts/1.2.0',utility_scale=1.2,simulations=simulations,
        root_top_k=1000000 if unpruned else 16,interior_top_k=1000000 if unpruned else 8,
        turn_interval=1,max_searches_per_seat=0).validate()


def controller(model,parent,profile,config,pool,*,decision_mode='searched'):
    profile.validate();config.validate()
    require(decision_mode in ('searched','direct'),'Unsupported personality decision mode')
    value=dict(contract=identity(),model_id=model,parent_incumbent_id=parent,profile=profile.to_dict(),
        search=config.to_dict(),pool_id=pool,reward_id=digest(config.reward),decision_mode=decision_mode)
    return dict(candidate_id=digest(value),**value)
