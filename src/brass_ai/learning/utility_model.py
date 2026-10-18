"""Versioned extension of the byte-pinned IL backbone; no old-head changes."""
from dataclasses import dataclass
import torch
from torch import nn

from .contracts import require
from .model import FourSeatNet,ModelConfig,ARCHITECTURE,LEGACY_ARCHITECTURE
from .reward import validate


@dataclass(frozen=True)
class UtilityConfig(ModelConfig):
    utility: dict | None = None

    @classmethod
    def from_dict(cls,raw,*,allow_legacy=False):
        require(raw.get('architecture')==ARCHITECTURE or (allow_legacy and raw.get('architecture')==LEGACY_ARCHITECTURE),
                'Architecture 11.2 required; explicitly upgrade completed v11.1 IL')
        require(set(raw)==set(cls.__dataclass_fields__),'Invalid utility model config')
        result=cls(**raw);result.validate();return result

    def validate(self):
        super().validate()
        require(self.architecture in (ARCHITECTURE,LEGACY_ARCHITECTURE),'Utility head requires the current versioned residual backbone')
        validate(self.utility)


def utility_config(config,spec):return UtilityConfig(**config.to_dict(),utility=spec)


class UtilityNet(FourSeatNet):
    def __init__(self,config):
        super().__init__(config)
        self.add_utility_head();self.utility_calibration=None;self.validate_parameter_count()

    def add_utility_head(self):
        extra=32 if self.config.architecture==ARCHITECTURE else 0
        self.utility_head=nn.Sequential(nn.Linear(self.config.hidden+4+extra,32),nn.GELU(),nn.Linear(32,4))
        if extra:
            with torch.no_grad():self.utility_head[0].weight[:,-extra:].zero_()
        nn.init.zeros_(self.utility_head[-1].weight);nn.init.zeros_(self.utility_head[-1].bias)

    def predict_utility(self,context,mapping,heads,remaining=None,embeddings=None):
        c=self.config.utility
        if remaining is None:remaining=context.new_full((len(context),4),c['insolvency']['max_cost'])
        relative=remaining.gather(1,mapping.argsort(1))
        inputs=torch.cat((context,relative.to(context.dtype)),-1);first=self.utility_head[0]
        pre=torch.nn.functional.linear(inputs,first.weight[:,:self.config.hidden+4],first.bias)
        if embeddings is not None:pre=pre+torch.nn.functional.linear(embeddings.mean(1),first.weight[:,self.config.hidden+4:])
        residual=self.utility_head[2](self.utility_head[1](pre)).gather(1,mapping)
        win,rank,vp=heads
        # Deterministic initialization; the learned correction handles ties and
        # future costs. Utility fitting does not relabel the official heads.
        prior=(c['win']*win+c['placement']*(4-rank)/3+c['vp']*vp).detach()
        return prior+residual

    def heads_from_encoded(self,encoded,mapping,remaining=None,descriptors=None):
        embeddings=self.descriptor_embeddings(encoded,descriptors)
        heads=self.official_heads(encoded,mapping,embeddings)
        return heads+(self.predict_utility(encoded.context,mapping,heads,remaining,embeddings),)
