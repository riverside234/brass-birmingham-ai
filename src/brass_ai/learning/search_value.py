"""Opt-in search-only residual. Ordinary forward and legacy tensor IDs stay intact."""
from dataclasses import fields, replace

import torch
from torch import nn

from .contracts import require

CONTRACT = 'search-value-correction/1.0.0'
PROTOCOL = 'search-utility/1.0.0'


def install(model, *, seed=4504):
    require(getattr(model.config, 'utility', None) is not None, 'Search value requires utility')
    if hasattr(model, 'search_value_head'):
        return model.search_value_head
    # Adding this optional head must not consume ordinary training RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = nn.Sequential(nn.Linear(model.config.hidden + 4 + 32, 32), nn.GELU(), nn.Linear(32, 4))
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
    model.add_module('search_value_head', head.to(next(model.parameters()).device))
    model.validate_parameter_count()
    return head


def binding(weight):
    require(type(weight) in (int, float) and 0 <= weight <= 1 and
            abs(weight * 20 - round(weight * 20)) < 1e-8, 'Invalid search blend')
    return dict(protocol=PROTOCOL, value_source='main' if weight == 0 else 'corrected', w=float(weight))


def validate_binding(value):
    require(type(value) is dict and value == binding(value.get('w')), 'Search value binding differs')
    return value


def correction(model, encoded, mapping, remaining, *, scale=0.):
    from brass_ai.recovery.ppg_phasic import stop_value_gradient
    from .local_scoring_heads import scale_gradient
    require(hasattr(model, 'search_value_head') and 0 <= scale <= 1, 'Missing head/invalid trunk scale')
    if scale == 0:
        encoded = stop_value_gradient(encoded)
    else:
        encoded = replace(encoded, **{f.name: scale_gradient(getattr(encoded, f.name), scale) for f in fields(encoded)})
    embeddings = model.descriptor_embeddings(encoded, None)
    # Unknown descriptors: no gradient to descriptor parameters from this loss.
    if scale == 0:
        embeddings = embeddings.detach()
    else:
        # The descriptor encoder is shared too; scaling only the state tensors
        # would leave this branch at full gradient during the trunk ramp.
        embeddings = scale_gradient(embeddings, scale)
    if remaining is None:
        remaining = encoded.context.new_full((len(mapping), 4), model.config.utility['insolvency']['max_cost'])
    relative = remaining.gather(1, mapping.argsort(1))
    inputs = torch.cat((encoded.context, relative.to(encoded.context.dtype), embeddings.mean(1)), -1)
    return model.search_value_head(inputs).gather(1, mapping)


def head_parameters(model):
    return list(model.search_value_head.parameters())
