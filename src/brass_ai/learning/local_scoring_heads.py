"""Trainer-owned heads. Not registered on FourSeatNet or exported for play."""
import torch
from torch import nn

CONTRACT='local-scoring-heads/1.0.0'


def scale_gradient(value,scale):
    return value.detach()+(value-value.detach())*scale


class EntityScoringHead(nn.Module):
    def __init__(self,context_width):
        super().__init__()
        self.global_projection=nn.Linear(context_width+128,64)
        self.entity_projection=nn.Linear(64,64,bias=False)
        self.owner=nn.Linear(64,5);self.vp=nn.Linear(64,4)
        for layer in (self.owner,self.vp):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,global_input,entities):
        hidden=torch.nn.functional.gelu(self.global_projection(global_input)[:,None,:]+self.entity_projection(entities))
        return self.owner(hidden),self.vp(hidden)


class LocalScoringHeads(nn.Module):
    def __init__(self,context_width=512):
        super().__init__()
        self.slots=EntityScoringHead(context_width);self.links=EntityScoringHead(context_width)

    def forward(self,encoded,descriptors,scale=1.):
        shared=scale_gradient(torch.cat((encoded.context,descriptors.flatten(1)),-1),scale)
        entities=scale_gradient(encoded.targets,scale)
        return dict(slots=self.slots(shared,entities[:,:49]),links=self.links(shared,entities[:,49:118]))
