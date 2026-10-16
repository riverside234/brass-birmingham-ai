"""Versioned extension of the byte-pinned IL backbone; no old-head changes."""
from dataclasses import dataclass
import torch
from torch import nn

from .contracts import require
from .model import FourSeatNet,ModelConfig,ARCHITECTURE
from .reward import validate


@dataclass(frozen=True)
class UtilityConfig(ModelConfig):
    utility: dict | None = None

    @classmethod
    def from_dict(cls,raw):
        require(raw.get('architecture')==ARCHITECTURE,
                'Use fresh version-11.1 IL weights; old architectures including v11 cannot migrate')
        require(set(raw)==set(cls.__dataclass_fields__),'Invalid utility model config')
        result=cls(**raw);result.validate();return result

    def validate(self):
        super().validate()
        require(self.architecture==ARCHITECTURE,'Utility head requires the current versioned residual backbone')
        validate(self.utility)


def utility_config(config,spec):return UtilityConfig(**config.to_dict(),utility=spec)


class UtilityNet(FourSeatNet):
    def __init__(self,config):
        super().__init__(config)
        self.add_utility_head();self.utility_calibration=None;self.validate_parameter_count()

    def add_utility_head(self):
        self.utility_head=nn.Sequential(nn.Linear(self.config.hidden+4,32),nn.GELU(),nn.Linear(32,4))
        nn.init.zeros_(self.utility_head[-1].weight);nn.init.zeros_(self.utility_head[-1].bias)

    def predict_utility(self,context,mapping,heads,remaining=None):
        c=self.config.utility
        if remaining is None:remaining=context.new_full((len(context),4),c['insolvency']['max_cost'])
        relative=remaining.gather(1,mapping.argsort(1))
        residual=self.utility_head(torch.cat((context,relative.to(context.dtype)),-1)).gather(1,mapping)
        win,rank,vp=heads
        # Deterministic initialization; the learned correction handles ties and
        # future costs. Utility fitting does not relabel the official heads.
        prior=(c['win']*win+c['placement']*(4-rank)/3+c['vp']*vp).detach()
        return prior+residual

    def heads_from_encoded(self,encoded,mapping,remaining=None):
        heads=self.official_heads(encoded,mapping)
        return heads+(self.predict_utility(encoded.context,mapping,heads,remaining),)
