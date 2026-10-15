# SPDX-License-Identifier: AGPL-3.0-only
# Daiyu architectural adaptation; source, revision and changes: NOTICE.md.
"""Separate state/candidate encoders, shared scorer and three four-seat heads."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .contracts import keys, require
from .features import Encoder

ARCHITECTURE='brass-four-seat-mlp-resnet/4.0.0'
PARAMETER_LIMIT=3000000


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int
    action_dim: int
    hidden: int
    action_hidden: int
    players: int = 4
    architecture: str = ARCHITECTURE
    residual_blocks: int = 5
    head_hidden: int = 128
    parameter_limit: int = PARAMETER_LIMIT

    @classmethod
    def preset(cls, size='medium'):
        encoder = Encoder()
        require(size in ('small', 'medium'), 'Unknown version-4 residual model preset')
        width = {'small': 128, 'medium': 304}[size]
        return cls(encoder.state_dim, encoder.action_dim, width, 128,
                   architecture=ARCHITECTURE, residual_blocks=5)

    @classmethod
    def from_dict(cls, raw):
        allowed=set(cls.__dataclass_fields__)
        legacy=allowed-{'residual_blocks','head_hidden','parameter_limit'}
        require(set(raw) in (allowed,legacy), 'Invalid model config fields')
        config = cls(**raw)
        config.validate()
        return config

    def validate(self):
        encoder = Encoder()
        require(type(self.players) is int and self.players == 4, 'Four-seat model required')
        require(self.architecture==ARCHITECTURE, 'Use fresh version-4 five-block model weights; old architectures cannot migrate')
        require((self.state_dim, self.action_dim) == (encoder.state_dim, encoder.action_dim), 'Feature dimensions differ')
        require(type(self.hidden) is int and 32 <= self.hidden <= 2048 and
                type(self.action_hidden) is int and 16 <= self.action_hidden <= 1024, 'Invalid hidden size')
        if self.architecture==ARCHITECTURE:
            require(type(self.residual_blocks) is int and 1<=self.residual_blocks<=8 and
                    type(self.head_hidden) is int and 16<=self.head_hidden<=256 and
                    self.parameter_limit==PARAMETER_LIMIT, 'Version-4 residual model/budget contract')

    def to_dict(self):
        raw=asdict(self)
        if self.architecture=='daiyu-four-seat-dense/1.0.0':
            for k in ('residual_blocks','head_hidden','parameter_limit'):raw.pop(k)
        return raw


class ResidualBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.branch=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU(),nn.Linear(width,width))
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self,x):return x+self.branch(x)


class FourSeatNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or ModelConfig.preset()
        c.validate()
        residual=c.architecture==ARCHITECTURE
        self.state_encoder = (nn.Sequential(nn.Linear(c.state_dim,c.hidden),nn.GELU(),
            *[ResidualBlock(c.hidden) for _ in range(c.residual_blocks)],nn.LayerNorm(c.hidden)) if residual else
            nn.Sequential(nn.Linear(c.state_dim, c.hidden), nn.GELU(),
            nn.LayerNorm(c.hidden), nn.Linear(c.hidden, c.hidden), nn.GELU()))
        # Dense semantic fields replace upstream's sparse EmbeddingBag for a
        # simple variable-candidate ONNX graph with no custom runtime operators.
        self.action_encoder = nn.Sequential(nn.Linear(c.action_dim, c.action_hidden),
            nn.GELU(), nn.LayerNorm(c.action_hidden))
        ph=c.head_hidden if residual else c.hidden
        vh=c.head_hidden if residual else c.hidden//2
        self.policy_scorer = nn.Sequential(nn.Linear(c.hidden + c.action_hidden, ph),
            nn.GELU(), nn.Linear(ph, 1))
        self.outcomes = nn.Sequential(nn.Linear(c.hidden, vh), nn.GELU(),nn.Linear(vh,12))
        self.validate_parameter_count()

    def validate_parameter_count(self):
        if self.config.architecture==ARCHITECTURE:
            require(self.parameter_count<=self.config.parameter_limit,'Residual model exceeds versioned parameter limit')
        return self.parameter_count

    def forward(self, states, actions, legal_mask, stable_to_relative):
        context = self.state_encoder(states)
        candidates = self.action_encoder(actions)
        expanded = context.unsqueeze(1).expand(-1, actions.shape[1], -1)
        logits = self.policy_scorer(torch.cat((expanded, candidates), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(~legal_mask, -torch.inf)
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
