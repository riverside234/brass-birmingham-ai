"""Function-preserving utility widening; allocate only the enlarged head/moments."""
from dataclasses import replace

import torch
from torch import nn

from .contracts import require


def widen(model):
    """Keep the learned 32 units; new random features initially contribute zero."""
    require(model.config.utility_hidden==32,'Utility widening requires a width-32 model')
    first,last=model.utility_head[0],model.utility_head[2]
    shapes={name:tuple(p.shape) for name,p in model.named_parameters() if name.startswith('utility_head.')}
    # Construct on CPU with an isolated seed, then transfer only the small head.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2561529)
        head=nn.Sequential(nn.Linear(first.in_features,256),nn.GELU(),nn.Linear(256,4))
    head.to(device=first.weight.device,dtype=first.weight.dtype)
    with torch.no_grad():
        head[0].weight[:32].copy_(first.weight);head[0].bias[:32].copy_(first.bias)
        head[2].weight.zero_();head[2].weight[:,:32].copy_(last.weight);head[2].bias.copy_(last.bias)
    head.train(model.utility_head.training)
    for old,new in zip(model.utility_head.parameters(),head.parameters(),strict=True):
        new.requires_grad_(old.requires_grad)
    model.utility_head=head
    model.config=replace(model.config,utility_hidden=256)
    model.validate_parameter_count()
    return shapes


def optimizer_state(saved,model,old_shapes):
    """Extend the PPO or PPG Adam moments in its original ordinary-parameter order.

    Existing tensor slices and scalar steps survive; added slices start at zero.
    All unrelated optimizer storage is shared read-only, never copied wholesale.
    """
    parameters=[(name,p) for name,p in model.named_parameters() if not name.startswith('search_value_head.')]
    require(saved is not None and len(saved['param_groups'])==1 and
        len(saved['param_groups'][0]['params'])==len(parameters),'Utility widening requires the ordinary optimizer order')
    states=dict(saved['state'])
    for index,(name,p) in zip(saved['param_groups'][0]['params'],parameters,strict=True):
        old=old_shapes.get(name)
        if old is None or old==tuple(p.shape) or index not in states:continue
        values=dict(states[index])
        for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if key not in values:continue
            value=values[key]
            require(tuple(value.shape)==old,'Utility Adam moment shape differs')
            expanded=value.new_zeros(p.shape)
            expanded[tuple(slice(0,n) for n in old)].copy_(value)
            values[key]=expanded
        states[index]=values
    return dict(saved,state=states)
