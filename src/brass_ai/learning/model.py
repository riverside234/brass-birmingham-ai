# SPDX-License-Identifier: AGPL-3.0-only
# Daiyu architectural adaptation; source, revision and changes: NOTICE.md.
"""Separate state/candidate encoders, shared scorer and three four-seat heads."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .contracts import keys, require
from .features import Encoder


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int
    action_dim: int
    hidden: int
    action_hidden: int
    players: int = 4
    architecture: str = 'daiyu-four-seat-dense/1.0.0'

    @classmethod
    def preset(cls, size='medium'):
        encoder = Encoder()
        require(size in ('small', 'medium', 'large'), 'Unknown model preset')
        width = {'small': 128, 'medium': 416, 'large': 1152}[size]
        return cls(encoder.state_dim, encoder.action_dim, width, width // 2)

    @classmethod
    def from_dict(cls, raw):
        keys(raw, cls.__dataclass_fields__)
        config = cls(**raw)
        config.validate()
        return config

    def validate(self):
        encoder = Encoder()
        require(type(self.players) is int and self.players == 4, 'Four-seat model required')
        require(self.architecture == 'daiyu-four-seat-dense/1.0.0', 'Unknown model architecture')
        require((self.state_dim, self.action_dim) == (encoder.state_dim, encoder.action_dim), 'Feature dimensions differ')
        require(type(self.hidden) is int and 32 <= self.hidden <= 2048 and
                type(self.action_hidden) is int and 16 <= self.action_hidden <= 1024, 'Invalid hidden size')

    def to_dict(self):
        return asdict(self)


class FourSeatNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or ModelConfig.preset()
        c.validate()
        self.state_encoder = nn.Sequential(nn.Linear(c.state_dim, c.hidden), nn.GELU(),
            nn.LayerNorm(c.hidden), nn.Linear(c.hidden, c.hidden), nn.GELU())
        # Dense semantic fields replace upstream's sparse EmbeddingBag for a
        # simple variable-candidate ONNX graph with no custom runtime operators.
        self.action_encoder = nn.Sequential(nn.Linear(c.action_dim, c.action_hidden),
            nn.GELU(), nn.LayerNorm(c.action_hidden))
        self.policy_scorer = nn.Sequential(nn.Linear(c.hidden + c.action_hidden, c.hidden),
            nn.GELU(), nn.Linear(c.hidden, 1))
        self.outcomes = nn.Sequential(nn.Linear(c.hidden, c.hidden // 2), nn.GELU(),
                                     nn.Linear(c.hidden // 2, 12))

    def forward(self, states, actions, legal_mask, stable_to_relative):
        context = self.state_encoder(states)
        candidates = self.action_encoder(actions)
        expanded = context.unsqueeze(1).expand(-1, actions.shape[1], -1)
        logits = self.policy_scorer(torch.cat((expanded, candidates), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~legal_mask, -1e9)
        outcomes = self.outcomes(context)
        # Each prediction uses ONE observer's features. Relative heads are
        # mapped independently to stable identities, never current turn order.
        win = torch.softmax(outcomes[:, :4], dim=-1).gather(1, stable_to_relative)
        placement = (1 + 3 * torch.sigmoid(outcomes[:, 4:8])).gather(1, stable_to_relative)
        vp = (100 * outcomes[:, 8:12]).gather(1, stable_to_relative)
        return logits, win, placement, vp

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
