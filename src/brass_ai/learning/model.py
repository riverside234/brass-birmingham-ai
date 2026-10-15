# SPDX-License-Identifier: AGPL-3.0-only
# Daiyu architectural adaptation; source, revision and changes: NOTICE.md.
"""Separate state/candidate encoders, shared scorer and three four-seat heads."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from .contracts import keys, require
from .features import Encoder
from .state_layout import state_layout,PROJECTION_WIDTHS

ARCHITECTURE='brass-four-seat-mlp-resnet/9.0.0'
PARAMETER_LIMIT=3000000


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int
    action_dim: int
    hidden: int
    action_hidden: int
    players: int = 4
    architecture: str = ARCHITECTURE
    residual_blocks: int = 6
    head_hidden: int = 128
    parameter_limit: int = PARAMETER_LIMIT
    group_layout: str = field(default_factory=lambda:state_layout().identity)
    projection_widths: tuple = PROJECTION_WIDTHS

    def __post_init__(self):
        if type(self.projection_widths) is list:
            object.__setattr__(self,'projection_widths',tuple(self.projection_widths))

    @classmethod
    def preset(cls, size='medium'):
        encoder = Encoder()
        require(size in ('small', 'medium'), 'Unknown version-9 grouped model preset')
        width = {'small': 128, 'medium': 304}[size]
        return cls(encoder.state_dim, encoder.action_dim, width, 128,
                   architecture=ARCHITECTURE, residual_blocks=6)

    @classmethod
    def from_dict(cls, raw):
        allowed=set(cls.__dataclass_fields__)
        require(raw.get('architecture')==ARCHITECTURE,
                'Use fresh version-9 grouped model weights; old architectures cannot migrate')
        require(set(raw)==allowed, 'Invalid grouped model config fields')
        config = cls(**raw)
        config.validate()
        return config

    def validate(self):
        encoder = Encoder()
        require(type(self.players) is int and self.players == 4, 'Four-seat model required')
        require(self.architecture==ARCHITECTURE, 'Use fresh version-9 grouped model weights; old architectures cannot migrate')
        require((self.state_dim, self.action_dim) == (encoder.state_dim, encoder.action_dim), 'Feature dimensions differ')
        require(type(self.hidden) is int and 32 <= self.hidden <= 2048 and
                type(self.action_hidden) is int and 16 <= self.action_hidden <= 1024, 'Invalid hidden size')
        if self.architecture==ARCHITECTURE:
            require(type(self.residual_blocks) is int and 1<=self.residual_blocks<=8 and
                    type(self.head_hidden) is int and 16<=self.head_hidden<=256 and
                    self.parameter_limit==PARAMETER_LIMIT, 'Version-9 grouped model/budget contract')
        require(self.group_layout==state_layout().identity,'Grouped state layout identity differs')
        require(type(self.projection_widths) is tuple and all(type(n) is int for n in self.projection_widths)
                and self.projection_widths==PROJECTION_WIDTHS,'Grouped projection widths differ')

    def to_dict(self):
        raw=asdict(self)
        raw['projection_widths']=list(self.projection_widths)
        return raw


class ResidualBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.branch=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU(),nn.Linear(width,width))
        nn.init.zeros_(self.branch[-1].weight)
        nn.init.zeros_(self.branch[-1].bias)

    def forward(self,x):return x+self.branch(x)


class GroupedStateEncoder(nn.Module):
    """One static gather, six group projections, mandatory joint fusion."""
    def __init__(self,config):
        super().__init__();layout=state_layout();self.input_dim=config.state_dim
        self.split_sizes=tuple(len(g.indices) for g in layout.groups)
        self.register_buffer('permutation',torch.tensor(layout.permutation,dtype=torch.int64))
        self.projections=nn.ModuleList(nn.Sequential(nn.Linear(size,width),nn.GELU())
            for size,width in zip(self.split_sizes[:-1],config.projection_widths))
        self.fusion=nn.Sequential(nn.Linear(sum(config.projection_widths)+12,config.hidden),nn.GELU())
        self.backbone=nn.Sequential(*[ResidualBlock(config.hidden) for _ in range(config.residual_blocks)],nn.LayerNorm(config.hidden))

    def validate_layout(self):
        expected=state_layout()
        require(self.split_sizes==tuple(len(g.indices) for g in expected.groups)
                and self.permutation.dtype==torch.int64
                and torch.equal(self.permutation.detach().cpu(),torch.tensor(expected.permutation,dtype=torch.int64)),
                'Grouped state index buffer/layout differs')

    def forward(self,states):
        groups=torch.split(states.index_select(1,self.permutation),self.split_sizes,dim=1)
        fused=torch.cat([projection(values) for projection,values in zip(self.projections,groups[:-1])]+[groups[-1]],dim=1)
        return self.backbone(self.fusion(fused))


class FourSeatNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = c = config or ModelConfig.preset()
        c.validate()
        residual=c.architecture==ARCHITECTURE
        self.state_encoder = GroupedStateEncoder(c)
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
            self.state_encoder.validate_layout()
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


def dense_multiply_adds(model):
    """Linear MACs; padding is real work. Nonlinear/gather/normalization excluded."""
    prefix='core.' if hasattr(model,'core') else ''
    candidate=fixed=0
    for name,module in model.named_modules():
        if not isinstance(module,nn.Linear):continue
        mac=module.in_features*module.out_features
        if name.startswith((prefix+'action_encoder.',prefix+'policy_scorer.')):candidate+=mac
        else:fixed+=mac
    return dict(fixed_per_position=fixed,per_padded_candidate=candidate,
        formula='batch * (fixed_per_position + padded_candidates * per_padded_candidate)',
        scope='Dense Linear MACs only; one MAC is two FLOPs. Excludes gather, GELU, LayerNorm, softmax, masks and scalar utility algebra.')
