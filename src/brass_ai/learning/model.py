# SPDX-License-Identifier: AGPL-3.0-only
# Daiyu architectural adaptation; source, revision and changes: NOTICE.md.
"""Separate state/candidate encoders, shared scorer and three four-seat heads."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import keys, require
from .features import Encoder
from .state_layout import model_layout as state_layout,MODEL_PROJECTION_WIDTHS as PROJECTION_WIDTHS
from .local_entities import (ENTITY_HIDDEN, COUNTS, LocalEntityEncoder, LocalCandidates,
                            entity_layout)

ARCHITECTURE='brass-four-seat-mlp-resnet/11.0.0'
PARAMETER_LIMIT=6500000


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int
    action_dim: int
    hidden: int
    action_hidden: int
    players: int = 4
    architecture: str = ARCHITECTURE
    residual_blocks: int = 8
    head_hidden: int = 192
    outcome_hidden: int = 128
    parameter_limit: int = PARAMETER_LIMIT
    group_layout: str = field(default_factory=lambda:state_layout().identity)
    projection_widths: tuple = PROJECTION_WIDTHS
    entity_layout: str = field(default_factory=lambda:entity_layout()['identity'])
    entity_hidden: int = ENTITY_HIDDEN

    def __post_init__(self):
        if type(self.projection_widths) is list:
            object.__setattr__(self,'projection_widths',tuple(self.projection_widths))

    @classmethod
    def preset(cls, size='medium'):
        encoder = Encoder()
        require(size in ('small', 'medium'), 'Unknown version-11 grouped model preset')
        width = {'small': 128, 'medium': 512}[size]
        return cls(encoder.state_dim, encoder.action_dim, width, 160,
                   architecture=ARCHITECTURE, residual_blocks=8)

    @classmethod
    def from_dict(cls, raw):
        allowed=set(cls.__dataclass_fields__)
        require(raw.get('architecture')==ARCHITECTURE,
                'Use fresh version-11 grouped model weights; old architectures cannot migrate')
        require(set(raw)==allowed, 'Invalid grouped model config fields')
        config = cls(**raw)
        config.validate()
        return config

    def validate(self):
        encoder = Encoder()
        require(type(self.players) is int and self.players == 4, 'Four-seat model required')
        require(self.architecture==ARCHITECTURE, 'Use fresh version-11 grouped model weights; old architectures cannot migrate')
        require((self.state_dim, self.action_dim) == (encoder.state_dim, encoder.action_dim), 'Feature dimensions differ')
        require(type(self.hidden) is int and 32 <= self.hidden <= 2048 and
                type(self.action_hidden) is int and 16 <= self.action_hidden <= 1024, 'Invalid hidden size')
        if self.architecture==ARCHITECTURE:
            require(type(self.residual_blocks) is int and self.residual_blocks==8 and
                    type(self.head_hidden) is int and self.head_hidden==192 and
                    type(self.outcome_hidden) is int and self.outcome_hidden==128 and
                    self.action_hidden==160 and self.parameter_limit==PARAMETER_LIMIT,
                    'Version-11 grouped model/budget contract')
        require(self.group_layout==state_layout().identity,'Grouped state layout identity differs')
        require(type(self.projection_widths) is tuple and all(type(n) is int for n in self.projection_widths)
                and self.projection_widths==PROJECTION_WIDTHS,'Grouped projection widths differ')
        require(self.entity_layout==entity_layout()['identity'] and type(self.entity_hidden) is int
                and self.entity_hidden==ENTITY_HIDDEN, 'V11 entity layout/width differs')

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
        self.entity_encoder = LocalEntityEncoder()
        self.entity_projections = nn.ModuleDict({name: nn.Linear(c.entity_hidden, c.action_hidden, bias=False)
                                                for name in ('target', 'merchant', 'industry')})
        for layer in self.entity_projections.values():nn.init.zeros_(layer.weight)
        self.reference_slices = tuple((f['start'], f['stop']) for f in entity_layout()['action_fields'].values())
        # Dense semantic fields replace upstream's sparse EmbeddingBag for a
        # simple variable-candidate ONNX graph with no custom runtime operators.
        self.action_encoder = nn.Sequential(nn.Linear(c.action_dim, c.action_hidden),
            nn.GELU(), nn.LayerNorm(c.action_hidden))
        ph=c.head_hidden if residual else c.hidden
        vh=c.outcome_hidden
        self.policy_scorer = nn.Sequential(nn.Linear(c.hidden + c.action_hidden, ph),
            nn.GELU(), nn.Linear(ph, 1))
        self.outcomes = nn.Sequential(nn.Linear(c.hidden + c.entity_hidden, vh), nn.GELU(),nn.Linear(vh,3))
        self.validate_parameter_count()

    def validate_parameter_count(self):
        if self.config.architecture==ARCHITECTURE:
            self.state_encoder.validate_layout()
            self.entity_encoder.validate_layout()
            require(self.parameter_count<=self.config.parameter_limit,'Residual model exceeds versioned parameter limit')
        return self.parameter_count

    def encode_state(self, states, *, context=None):
        context = self.state_encoder(states) if context is None else context
        return self.entity_encoder(states, context)

    def prepare_candidates(self, encoded):
        # Project each bank once per position, including across candidate chunks.
        # Row zero is the exact zero embedding for an absent reference.
        def project(name, bank):
            projected = self.entity_projections[name](bank)
            return torch.cat((torch.zeros_like(projected[:, :1]), projected), dim=1)
        return LocalCandidates(self.project_policy_context(encoded.context),
            project('target', encoded.targets), project('merchant', encoded.merchants),
            project('industry', encoded.industries))

    def reference_context(self, actions, prepared):
        result = None
        absent = torch.zeros_like(actions[..., :1])
        for (start, stop), bank in zip(self.reference_slices,
                                      (prepared.targets, prepared.merchants, prepared.industries)):
            # Admitted fields are one-hot or all-zero. Keep argmax in the graph
            # so Features-4 data and the public ONNX input contract stay intact.
            # Argmax chooses the first maximum: all-zero fields select row zero,
            # while each present one-hot selects its bank row shifted by one.
            index = torch.cat((absent, actions[..., start:stop]), dim=-1).argmax(dim=-1)
            offsets = torch.arange(bank.shape[0], device=bank.device) * bank.shape[1]
            # Flatten only the bank; Gather uses [B,A] indices instead of
            # expanding them to [B,A,160] for GatherElements.
            value = F.embedding(index + offsets[:, None], bank.flatten(0, 1))
            result = value if result is None else result + value
        return result

    def score_prepared(self, actions, prepared):
        candidates = self.action_encoder(actions) + self.reference_context(actions, prepared)
        return self.score_candidates(candidates, prepared.context)

    def official_heads(self, encoded, mapping):
        first = self.outcomes[0]; width = self.config.hidden
        # Shared player head; retain the factored state contribution here too.
        joint = F.linear(encoded.players, first.weight[:, width:]) + \
                F.linear(encoded.context, first.weight[:, :width], first.bias).unsqueeze(1)
        raw = self.outcomes[2](self.outcomes[1](joint))  # [B, relative player, outcome]
        return (torch.softmax(raw[:, :, 0], dim=-1).gather(1, mapping),
                (1 + 3 * torch.sigmoid(raw[:, :, 1])).gather(1, mapping),
                (100 * raw[:, :, 2]).gather(1, mapping))

    def heads_from_encoded(self, encoded, mapping, remaining=None):
        return self.official_heads(encoded, mapping)

    def forward(self, states, actions, legal_mask, stable_to_relative, remaining=None):
        encoded = self.encode_state(states)
        logits = self.score_prepared(actions, self.prepare_candidates(encoded))
        logits = logits.masked_fill(~legal_mask, -torch.inf)
        # Each prediction uses ONE observer's features. Relative heads are
        # mapped independently to stable identities, never current turn order.
        return (logits,) + self.heads_from_encoded(encoded, stable_to_relative, remaining)

    def project_policy_context(self, context):
        """Compute the first scorer layer's state contribution once per position.

        Slice the existing Parameter rather than creating new layers: checkpoint
        keys, optimizer ownership and the bias remain exactly the same.
        """
        first = self.policy_scorer[0]
        return F.linear(context, first.weight[:, :self.config.hidden], first.bias)

    def score_candidates(self, candidates, projected_context):
        """Add both linear contributions before the original shared nonlinearity."""
        first = self.policy_scorer[0]
        combined = F.linear(candidates, first.weight[:, self.config.hidden:]) + projected_context.unsqueeze(1)
        return self.policy_scorer[2](self.policy_scorer[1](combined)).squeeze(-1)

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


def dense_multiply_adds(model):
    """Linear MACs; padding is real work. Nonlinear/gather/normalization excluded."""
    prefix='core.' if hasattr(model,'core') else ''
    core=model.core if prefix else model
    # Role projections remain once per position; selectors now use index gathers.
    reference_sizes=(128,9,6)
    candidate=fixed=0
    for name,module in model.named_modules():
        if not isinstance(module,nn.Linear):continue
        mac=module.in_features*module.out_features
        if name==prefix+'policy_scorer.0':
            state_mac=core.config.hidden*module.out_features
            fixed+=state_mac;candidate+=mac-state_mac
        elif name.startswith((prefix+'action_encoder.',prefix+'policy_scorer.')):candidate+=mac
        elif name.startswith(prefix+'entity_encoder.encoders.'):
            family=name[len(prefix+'entity_encoder.encoders.'):].split('.')[0]
            fixed+=mac*COUNTS[family]
        elif name.startswith(prefix+'entity_projections.'):
            role=name.rsplit('.',1)[1]
            fixed+=mac*dict(zip(('target','merchant','industry'),reference_sizes))[role]
        elif name==prefix+'outcomes.0':
            fixed+=core.config.hidden*module.out_features+4*core.config.entity_hidden*module.out_features
        elif name==prefix+'outcomes.2':fixed+=4*mac
        else:fixed+=mac
    return dict(fixed_per_position=fixed,per_padded_candidate=candidate,
        formula='batch * (fixed_per_position + padded_candidates * per_padded_candidate)',
        scope='Dense Linear MACs; entity/player multiplicities included. One MAC is two FLOPs. Excludes selector argmax/gather, GELU, LayerNorm, softmax, masks and scalar utility algebra.')
