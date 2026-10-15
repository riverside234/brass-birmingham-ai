"""One copied decision network plus zero-initialized context and a goal branch."""
import copy
from dataclasses import asdict, dataclass
import torch
from torch import nn

from brass_ai.learning.contracts import keys,require
from brass_ai.learning.utility_model import UtilityConfig,UtilityNet
from .contracts import MODEL,CONTEXT_DIM,FAMILIES,FACTORS,HORIZONS,contract


@dataclass(frozen=True)
class PersonalityConfig:
    base: dict
    model: str = MODEL
    goal_hidden: int = 128
    parameter_limit: int = 3100000
    extension_limit: int = 100000

    def validate(self):
        UtilityConfig.from_dict(self.base)
        require(self.model==MODEL and self.goal_hidden==128 and self.parameter_limit==3100000
            and self.extension_limit==100000,'Unknown personality layout/budget')
        return self

    def to_dict(self):return asdict(self)

    @classmethod
    def from_dict(cls,raw):
        keys(raw,cls.__dataclass_fields__)
        return cls(**raw).validate()


class PersonalityNet(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config.validate()
        self.core=UtilityNet(UtilityConfig.from_dict(config.base))
        width=self.core.config.hidden
        self.context_projection=nn.Linear(CONTEXT_DIM,width,bias=False)
        nn.init.zeros_(self.context_projection.weight)
        self.goal_trunk=nn.Sequential(nn.Linear(width+CONTEXT_DIM,config.goal_hidden),nn.GELU())
        self.goal_heads=nn.ModuleDict({name:nn.Linear(config.goal_hidden,len(vocab)) for name,vocab in FACTORS.items()})
        self.confidence_head=nn.Linear(config.goal_hidden,len(FAMILIES)*len(HORIZONS))
        self.validate_parameter_count()

    @property
    def parameter_count(self):return sum(p.numel() for p in self.parameters())

    def validate_parameter_count(self):
        self.core.validate_parameter_count()
        require(self.parameter_count<=self.config.parameter_limit and
            self.parameter_count-self.core.parameter_count<=self.config.extension_limit,'Personality parameter budget exceeded')
        return dict(base=self.core.parameter_count,extension=self.parameter_count-self.core.parameter_count,total=self.parameter_count)

    def forward(self,states,actions,legal_mask,stable_to_relative,*,personality,remaining):
        require(personality.shape==(len(states),CONTEXT_DIM),'Personality tensor shape mismatch')
        require(remaining.shape==(len(states),4),'Remaining utility budgets required')
        h=self.core.state_encoder(states)+self.context_projection(personality)
        a=self.core.action_encoder(actions)
        logits=self.core.policy_scorer(torch.cat((h[:,None].expand(-1,a.shape[1],-1),a),-1)).squeeze(-1)
        logits=logits.masked_fill(~legal_mask,-torch.inf)
        outcomes=self.core.outcomes(h)
        w=torch.softmax(outcomes[:,:4],-1).gather(1,stable_to_relative)
        r=(1+3*torch.sigmoid(outcomes[:,4:8])).gather(1,stable_to_relative)
        v=(100*outcomes[:,8:12]).gather(1,stable_to_relative)
        utility=self.core.predict_utility(h,stable_to_relative,(w,r,v),remaining)
        g=self.goal_trunk(torch.cat((h,personality),-1))
        return dict(policy_logits=logits,win_share_4=w,placement_4=r,vp_4=v,
            remaining_utility_4=utility,goals={k:head(g) for k,head in self.goal_heads.items()},
            confidence_logits=self.confidence_head(g).reshape(-1,len(FAMILIES),len(HORIZONS)))

    def freeze_core(self):
        self.requires_grad_(True);self.core.requires_grad_(False);self.context_projection.requires_grad_(False)


def import_parent(parent):
    require(type(parent) is UtilityNet,'Personality import requires the trained utility extension')
    model=PersonalityNet(PersonalityConfig(copy.deepcopy(parent.config.to_dict())))
    model.core.load_state_dict(parent.state_dict(),strict=True)
    model.core.utility_calibration=copy.deepcopy(parent.utility_calibration)
    copied=['core.'+k for k in parent.state_dict()]
    report=dict(contract=contract(),copied=copied,initialized=[k for k in model.state_dict() if k not in copied],
        rejected=[],optimizer='Fresh; no Step 5 Adam/RNG imported',parameters=model.validate_parameter_count())
    return model,report


def base_outputs(output):
    return tuple(output[k] for k in ('policy_logits','win_share_4','placement_4','vp_4','remaining_utility_4'))
