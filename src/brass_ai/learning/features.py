"""Explicit permitted-view encoding; no engine object or checkpoint is accepted.

IDs identify public board locations, never hidden state. Internal player slots run
clockwise from the observer. Model outputs are converted back to stable seat IDs.
The version-2 layout contains the six permitted feature groups A–F.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from brass_ai.runtime_contracts import validate
from .contracts import CONFIG_DIR, FEATURE_SCHEMA, FEATURE_VERSION, digest, finite_vector, read_json, require
from .encoding_budget import check_budget


@dataclass
class Encoded:
    state: np.ndarray
    actions: np.ndarray
    stable_to_relative: np.ndarray
    choice_ids: tuple[str, ...]
    feature_version: str = FEATURE_VERSION


def check_encoded(row):
    require(getattr(row,'feature_version',None)==FEATURE_VERSION and row.state.shape==(5489,) and
            row.actions.shape==(len(row.choice_ids),548),'Incompatible or mixed-version encoded features')


class Encoder:
    def __init__(self):
        from brass_ai.feature_engineering.context import DerivedContextBuilder
        from brass_ai.feature_engineering.contracts import STATE_FIELDS,ACTION_FIELDS,COMPONENTS
        self.schema = s = read_json(CONFIG_DIR / FEATURE_SCHEMA)
        self.derived = DerivedContextBuilder(cache_size=0)
        require(digest(s['derived'])==self.derived.schema,'Frozen derived feature layout differs')
        self.cards, self.industries, self.kinds = s['card_types'], s['industries'], s['kinds']
        self.targets = s['slots'] + s['links'] + ['merchant:' + x for x in s['merchants']] + ['market']
        self.action_dim = len(self.kinds) + len(self.targets) + len(s['merchants']) + len(self.industries) + len(self.cards)
        tile_dim = len(s['tile_fields'])
        self.state_dim = (2 + len(s['phases']) + 3 + 4 + 16 + len(self.cards) +
            4 * (len(s['player_fields']) + len(self.cards) + 2 + 6 * (8 + 2 * (1 + tile_dim))) +
            len(s['slots']) * (6 + 1 + 4 + 6 + tile_dim) +
            len(s['links']) * 4 + len(s['merchants']) * (6 + 1 + 4) +
            5 + len(self.kinds) + len(self.targets) + len(s['merchants']) + 4 +
            4 * 2 * len(self.kinds) + 4 + s['intent_dim'])
        self.base_state_dim,self.base_action_dim=self.state_dim,self.action_dim
        self.state_dim+=len(STATE_FIELDS)*len(COMPONENTS)
        self.action_dim+=len(ACTION_FIELDS)*len(COMPONENTS)
        require((self.state_dim,self.action_dim)==(s['state_dim'],s['action_dim']),'Frozen input dimensions differ')

    def append_derived(self,x,actions,view,choices,style,intent,*,source=None,deadline=None,cancelled=None):
        context=self.derived.build(view,choices,style,intent,validated=True,input_identity=source,deadline=deadline,cancelled=cancelled)
        # Keep the existing neutral-BC/personality tail contract explicit.
        state=np.asarray(list(x[:-12])+list(context.state_tensor())+list(x[-12:]),dtype=np.float32)
        dynamic=np.asarray(context.action_tensors(),dtype=np.float32)
        actions=np.concatenate((actions,dynamic),axis=1)
        check_budget(deadline,cancelled)
        return state,actions

    @staticmethod
    def one(value, vocabulary, *, empty=False):
        if empty and value in ('', None):
            return [0.0] * len(vocabulary)
        require(value in vocabulary, f'Unknown categorical feature: {value}')
        return [float(value == item) for item in vocabulary]

    @staticmethod
    def counts(items, vocabulary, scale=1):
        counts = Counter(items)
        require(not set(counts) - set(vocabulary), 'Unknown count feature')
        return [counts[item] / scale for item in vocabulary]

    def tile(self, tile):
        if tile is None:
            return [0.0] * (1 + len(self.schema['tile_fields']))
        require(1 <= tile['level'] <= 8, 'Unknown tile level')
        return [1.0] + [float(tile[k]) / self.schema['tile_fields'][k] for k in sorted(self.schema['tile_fields'])]

    def choice_features(self, choice):
        # Scout discards three physical cards, possibly with repeated semantics.
        cards = ([choice['card']] if 'card' in choice else []) + choice.get('cards', [])
        return (self.one(choice['kind'], self.kinds) +
                self.one(choice.get('target'), self.targets, empty=True) +
                self.one(choice.get('merchant'), self.schema['merchants'], empty=True) +
                self.one(choice.get('industry'), self.industries, empty=True) +
                self.counts(cards, self.cards, 3))

    def encode(self, view, choices, style=None, intent=None):
        validate('observation', view)
        validate('choices', choices)
        require(view['context']['actor'] == view['seat'] and not view['context']['terminal'],
                'Only the actual acting observer can submit a policy request')
        ids = tuple(c['id'] for c in choices)
        require(len(ids) > 0 and len(ids) == len(set(ids)), 'Missing/duplicate choices')
        style = [0.0] * 4 if style is None else style
        intent = [0.0] * self.schema['intent_dim'] if intent is None else intent
        finite_vector(style, 4, -1, 1)
        finite_vector(intent, self.schema['intent_dim'], -1, 1)
        s, ctx, observer = self.schema, view['context'], view['seat']
        relative = lambda seat: (seat - observer) % 4
        x = self.one(ctx['era'], ['CanalEra', 'RailEra']) + self.one(ctx['phase'], s['phases'])
        x += [ctx['round'] / 8, ctx['turn'] / 4, ctx['actions_remaining'] / 2]
        x += self.one(relative(ctx['turn_owner']), list(range(4)))
        for seat in view['order']:
            x += self.one(relative(seat), list(range(4)))
        x += self.counts(view['own_hand'], self.cards, 8)
        for offset in range(4):
            player = view['players'][(observer + offset) % 4]
            x += [player[k] / s['player_fields'][k] for k in sorted(s['player_fields'])]
            x += self.counts(player['discards'], self.cards, 8)
            x += self.counts(player['known_wild'], ['wild:industry', 'wild:location'])
            mats = {mat['type']: mat['tiles'] for mat in player['mat']}
            require(len(player['mat']) == 6 and set(mats) == set(self.industries), 'Malformed player mat')
            for kind in self.industries:
                tiles = mats[kind]
                require(all(t['seat'] == player['seat'] and t['type'] == kind for t in tiles), 'Tile ownership/type mismatch')
                x += self.counts([t['level'] for t in tiles], list(range(1, 9)), 4)
                x += self.tile(tiles[0] if tiles else None)
                x += self.tile(tiles[1] if len(tiles) > 1 else None)
        for group in ('slots', 'links', 'merchants'):
            by_id = {item['id']: item for item in view[group]}
            require(len(by_id) == len(view[group]) and set(by_id) == set(s[group]), 'Board vocabulary mismatch')
            for identity in s[group]:
                item = by_id[identity]
                if group == 'slots':
                    tile = item.get('tile')
                    x += self.counts(item['types'], self.industries)
                    x += self.tile(tile)[:1]
                    x += self.one(relative(tile['seat']) if tile else None, list(range(4)), empty=True)
                    x += self.one(tile['type'] if tile else None, self.industries, empty=True)
                    x += self.tile(tile)[1:]
                elif group == 'links':
                    x += self.one(relative(item['owner']) if 'owner' in item else None, list(range(4)), empty=True)
                else:
                    x += self.counts(item.get('types', []), self.industries)
                    x += [item['beer']]
                    x += self.one(item['bonus'], ['Develop', 'VP', 'Income', 'Money'])
        x += [view['coal_market'] / 14, view['iron_market'] / 10, view['deck_count'] / 64,
              view['wild_industry_count'] / 4, view['wild_location_count'] / 4]
        pending = view['pending']
        x += self.one(pending['kind'], self.kinds, empty=True)
        x += self.one(pending['target'], self.targets, empty=True)
        x += self.one(pending['merchant'], s['merchants'], empty=True)
        x += [pending[k] / 4 for k in ('parts', 'coal', 'iron', 'beer')]
        counts = Counter((relative(e['seat']), e['era'], e['choice']['kind']) for e in view['history'])
        x += [counts[(seat, era, kind)] / 32 for seat in range(4)
              for era in ('CanalEra', 'RailEra') for kind in self.kinds]
        x += list(style) + list(intent)
        actions = np.asarray([self.choice_features(c) for c in choices], dtype=np.float32)
        state,actions = self.append_derived(x,actions,view,choices,style,intent)
        require(state.shape == (self.state_dim,) and actions.shape == (len(choices), self.action_dim), 'Feature layout mismatch')
        require(np.isfinite(state).all() and np.isfinite(actions).all(), 'Non-finite features')
        return Encoded(state, actions, np.asarray([relative(i) for i in range(4)], dtype=np.int64), ids)
