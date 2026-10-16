"""V11 local entity views of unchanged Features-4 tensors.

No rules calls or new observations: indices are derived from the named layout.
Candidate references retain the Encoder's ordering and empty-reference zeros.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from .contracts import CONFIG_DIR, FEATURE_SCHEMA, digest, read_json, require
from .state_layout import state_layout
from .board_topology import topology

ENTITY_VERSION = 'grouped-mlp-local-entities/2.0.0'
ENTITY_HIDDEN = 64
COUNTS = dict(slot=49, link=69, merchant=9, industry=6, market=1,
              own_player=1, other_player=3)
WIDTHS = dict(slot=72, link=148, merchant=7, industry=219, market=2,
              own_player=284, other_player=287)


@lru_cache(maxsize=1)
def entity_layout():
    schema = read_json(CONFIG_DIR / FEATURE_SCHEMA)
    layout = state_layout()
    def gather(*prefixes):
        return [i for s in layout.segments if s.name.startswith(prefixes)
                for i in range(s.start, s.stop)]
    players = [gather(f'player.{p}.', f'finance.{p}.', f'order.{p}.',
                      f'history.{p}.', f'mat.{p}.') for p in range(4)]
    board=topology()
    def slot_rows(names,capacity):
        require(len(names)<=capacity,'V11 local slot capacity exceeded')
        return [i for name in names for i in gather('slot.'+name+'.')]+[-1]*(18*(capacity-len(names)))
    slots={name:slot_rows([name]+[s for s in board['location_slots'][name.split(':')[0]] if s!=name],4)
           for name in schema['slots']}
    links={name:gather('link.'+name)+slot_rows([s for region in board['link_endpoints'][name]
            for s in board['location_slots'].get(region,[])],8) for name in schema['links']}
    groups = dict(
        slot=[slots[name] for name in schema['slots']],
        link=[links[name] for name in schema['links']],
        merchant=[gather('merchant.' + name + '.') for name in schema['merchants']],
        industry=[sum((gather(f'mat.{p}.{name}.') for p in range(4)),[])+gather(f'development.{name}.')
                  for name in schema['industries']],
        market=[gather('global.coal_market', 'global.iron_market')],
        own_player=players[:1], other_player=players[1:])
    for name, rows in groups.items():
        require(len(rows) == COUNTS[name] and all(len(r) == WIDTHS[name] for r in rows),
                'V11 entity feature layout differs: ' + name)
        require(all(len([i for i in r if i>=0])==len({i for i in r if i>=0}) for r in rows), 'Duplicate local entity component')
    targets = schema['slots'] + schema['links'] + ['merchant:' + m for m in schema['merchants']] + ['market']
    fields = {}; offset = len(schema['kinds'])
    for name, vocabulary in (('target', targets), ('merchant', schema['merchants']), ('industry', schema['industries'])):
        fields[name] = dict(start=offset, stop=offset + len(vocabulary), identities=vocabulary)
        offset += len(vocabulary)
    require(offset + len(schema['card_types']) == 184 and schema['action_dim'] == 327,
            'V11 action-reference layout differs')
    result = dict(format=ENTITY_VERSION, feature_schema_sha256=digest(schema),
                  state_layout=layout.identity, topology=board, topology_sha256=digest(board), groups=groups, action_fields=fields,
                  target_order=['slot', 'link', 'merchant', 'market'],
                  player_order='observer-relative-0-1-2-3', entity_hidden=ENTITY_HIDDEN)
    return dict(result, identity=digest(result))


def check_action_selectors(actions):
    """Admission check for the unchanged categorical Features-4 reference fields.

    Run on CPU arrays before batching/upload, not inside a GPU forward. Empty
    references are all-zero; soft, multi-hot and nonfinite selectors are invalid.
    """
    require(actions.ndim == 2 and actions.shape[1] == 327, 'Invalid action selector tensor shape')
    for name, field in entity_layout()['action_fields'].items():
        values = actions[:, field['start']:field['stop']]
        require(np.all((values == 0) | (values == 1)) and np.all(values.sum(-1) <= 1),
                'Action selector must be one-hot or absent: ' + name)


@dataclass
class LocalState:
    context: torch.Tensor
    targets: torch.Tensor
    merchants: torch.Tensor
    industries: torch.Tensor
    players: torch.Tensor


@dataclass
class LocalCandidates:
    context: torch.Tensor
    targets: torch.Tensor
    merchants: torch.Tensor
    industries: torch.Tensor


class LocalEntityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({name: nn.Sequential(nn.Linear(width, ENTITY_HIDDEN),
            nn.GELU(), nn.LayerNorm(ENTITY_HIDDEN)) for name, width in WIDTHS.items()})
        self.player_identity = nn.Embedding(4, ENTITY_HIDDEN)
        nn.init.normal_(self.player_identity.weight, std=.02)
        for name, rows in entity_layout()['groups'].items():
            self.register_buffer(name + '_indices', torch.tensor(rows, dtype=torch.int64))

    def gather(self, states, name):
        indices = getattr(self, name + '_indices')
        values=states.index_select(1, indices.clamp_min(0).reshape(-1)).reshape(-1, COUNTS[name], WIDTHS[name])
        return values.masked_fill(indices.unsqueeze(0)<0,0)

    def forward(self, states, context):
        banks = {name: encoder(self.gather(states, name)) for name, encoder in self.encoders.items()}
        targets = torch.cat([banks[name] for name in ('slot', 'link', 'merchant', 'market')], dim=1)
        players = torch.cat((banks['own_player'], banks['other_player']), dim=1)
        return LocalState(context, targets, banks['merchant'], banks['industry'],
                          players + self.player_identity.weight.unsqueeze(0))

    def validate_layout(self):
        for name, rows in entity_layout()['groups'].items():
            actual = getattr(self, name + '_indices')
            require(actual.dtype == torch.int64 and torch.equal(actual.detach().cpu(), torch.tensor(rows)),
                    'V11 entity index buffer/layout differs: ' + name)
