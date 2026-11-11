"""Opt-in search-only residual. Ordinary forward and legacy tensor IDs stay intact."""
from dataclasses import fields, replace

import torch
from torch import nn

from .contracts import require

CONTRACT = 'search-value-correction/1.0.0'
# A head of another reviewed hidden width records it in the checkpoint marker.
WIDE_CONTRACT = 'search-value-correction/1.1.0'
WIDTHS = (32, 256)
PROTOCOL = 'search-utility/1.0.0'


def install(model, *, seed=4504, hidden=None):
    """The optional search head; ``hidden`` is its width (32 unless a recipe says 256)."""
    require(getattr(model.config, 'utility', None) is not None, 'Search value requires utility')
    if hasattr(model, 'search_value_head'):
        require(hidden is None or width(model) == hidden, 'Search value head width differs')
        return model.search_value_head
    hidden = 32 if hidden is None else hidden
    require(hidden in WIDTHS, 'Unreviewed search value head width')
    # Adding this optional head must not consume ordinary training RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        head = nn.Sequential(nn.Linear(model.config.hidden + 4 + 32, hidden), nn.GELU(), nn.Linear(hidden, 4))
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
    model.add_module('search_value_head', head.to(next(model.parameters()).device))
    model.validate_parameter_count()
    return head


def width(model):
    return model.search_value_head[0].out_features


def marker(model):
    """The checkpoint marker: the original contract for width 32, else the width itself."""
    hidden = width(model)
    return CONTRACT if hidden == 32 else dict(contract=WIDE_CONTRACT, hidden=hidden)


def restore(model, value):
    """Re-create a saved head at its own width before its tensors load."""
    if value == CONTRACT:
        return install(model, hidden=32)
    require(type(value) is dict and set(value) == {'contract', 'hidden'} and value['contract'] == WIDE_CONTRACT
            and value['hidden'] in WIDTHS and value['hidden'] != 32, 'Unknown search value head')
    return install(model, hidden=value['hidden'])


def binding(weight):
    require(type(weight) in (int, float) and 0 <= weight <= 1 and
            abs(weight * 100 - round(weight * 100)) < 1e-8, 'Invalid search blend')  # .02 steps over fifty B collections
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


def ordinary_parameters(model):
    """Stable legacy order; the search head has exactly one Adam owner."""
    return (p for name, p in model.named_parameters() if not name.startswith('search_value_head.'))
