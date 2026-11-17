"""Trainer-owned local-scoring heads by era (g3140). Not registered on FourSeatNet or exported.

Each slot and link reads its own slice of the trunk, so two empty slots in
different places can have different predictions. Canal rows, Rail rows and the
Canal rows' game-end (Rail-end) map each have their own owner/VP outputs on
one shared hidden layer per entity kind.
"""
import torch
from torch import nn

from .local_scoring_heads import scale_gradient

CONTRACT = 'local-scoring-heads/2.0.0'
SLOTS, LINKS = 49, 69
KINDS = (('slots', 0, SLOTS), ('links', SLOTS, SLOTS + LINKS))
HORIZONS = ('canal', 'rail', 'rail_end')
SHARED_WIDTH, SLICE_WIDTH, HIDDEN_WIDTH, ENTITY_WIDTH, DESCRIPTOR_WIDTH = 256, 16, 96, 64, 128


def output_names():
    return tuple(f'{horizon}_{kind}' for horizon in HORIZONS for kind, _, _ in KINDS)


class EraScoringHeads(nn.Module):
    def __init__(self, context_width=512):
        super().__init__()
        self.shared = nn.Linear(context_width + DESCRIPTOR_WIDTH, SHARED_WIDTH)
        self.slices = nn.Linear(SHARED_WIDTH, (SLOTS + LINKS) * SLICE_WIDTH)
        self.hidden = nn.ModuleDict({kind: nn.Linear(ENTITY_WIDTH + SLICE_WIDTH, HIDDEN_WIDTH) for kind, _, _ in KINDS})
        self.owner = nn.ModuleDict({name: nn.Linear(HIDDEN_WIDTH, 5) for name in output_names()})
        self.vp = nn.ModuleDict({name: nn.Linear(HIDDEN_WIDTH, 4) for name in output_names()})
        for layer in (*self.owner.values(), *self.vp.values()):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, encoded, descriptors, scale=1.):
        """{horizon_kind: (owner logits [B,n,5], VP [B,n,4])}; n is 49 slots or 69 links."""
        context = scale_gradient(torch.cat((encoded.context, descriptors.flatten(1)), -1), scale)
        entities = scale_gradient(encoded.targets[:, :SLOTS + LINKS], scale)
        shared = torch.nn.functional.gelu(self.shared(context))
        slices = self.slices(shared).view(len(context), SLOTS + LINKS, SLICE_WIDTH)
        result = {}
        for kind, lo, hi in KINDS:
            hidden = torch.nn.functional.gelu(self.hidden[kind](torch.cat((entities[:, lo:hi], slices[:, lo:hi]), -1)))
            for horizon in HORIZONS:
                name = f'{horizon}_{kind}'
                result[name] = (self.owner[name](hidden), self.vp[name](hidden))
        return result
