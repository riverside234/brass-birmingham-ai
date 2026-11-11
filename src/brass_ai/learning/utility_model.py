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
    utility_hidden: int = 32

    @classmethod
    def from_dict(cls,raw,*,allow_legacy=False):
        require(raw.get('architecture')==ARCHITECTURE or (allow_legacy and raw.get('architecture')==LEGACY_ARCHITECTURE),
                'Architecture 11.2 required; explicitly upgrade completed v11.1 IL')
        require(set(raw) in (set(cls.__dataclass_fields__),set(cls.__dataclass_fields__)-{'utility_hidden'}),
                'Invalid utility model config')
        result=cls(**raw);result.validate();return result

    def to_dict(self):
        raw=super().to_dict()
        # Keep every existing width-32 checkpoint's config and model ID intact.
        if self.utility_hidden==32:raw.pop('utility_hidden')
        return raw

    def validate(self):
        super().validate()
        require(self.architecture in (ARCHITECTURE,LEGACY_ARCHITECTURE),'Utility head requires the current versioned residual backbone')
        validate(self.utility)
        require(type(self.utility_hidden) is int and self.utility_hidden in (32,256),'Invalid utility hidden width')


def utility_config(config,spec,*,hidden=32):return UtilityConfig(**config.to_dict(),utility=spec,utility_hidden=hidden)


class UtilityNet(FourSeatNet):
    def __init__(self,config):
        super().__init__(config)
        self.add_utility_head();self.utility_calibration=None;self.validate_parameter_count()

    def add_utility_head(self):
        extra=32 if self.config.architecture==ARCHITECTURE else 0
        width=self.config.utility_hidden
        self.utility_head=nn.Sequential(nn.Linear(self.config.hidden+4+extra,width),nn.GELU(),nn.Linear(width,4))
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
        reward_vp=vp.clamp_max(c['vp_cap']) if 'vp_cap' in c else vp
        # Deterministic initialization; the learned correction handles ties and
        # future costs. Utility fitting does not relabel the official heads.
        prior=(c['win']*win+c['placement']*(4-rank)/3+c['vp']*reward_vp).detach()
        return prior+residual

    def heads_from_encoded(self,encoded,mapping,remaining=None,descriptors=None):
        embeddings=self.descriptor_embeddings(encoded,descriptors)
        heads=self.official_heads(encoded,mapping,embeddings)
        return heads+(self.predict_utility(encoded.context,mapping,heads,remaining,embeddings),)
