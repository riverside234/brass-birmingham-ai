"""Same R2 tensor layout, with cached categorical lookups.

The pinned R2 encoder remains unchanged. validated=True is used only after the
recovery ingress boundary has checked the original permitted view and choices.
"""
from collections import Counter
from functools import lru_cache
import numpy as np
from brass_ai.runtime_contracts import validate
from brass_ai.learning.contracts import finite_vector, require
from brass_ai.learning.features import Encoder, Encoded

@lru_cache(maxsize=128)
def vocabulary(values):
    return {v:i for i,v in enumerate(values)}

@lru_cache(maxsize=8192)
def categorical(value, values, empty):
    out=[0.0]*len(values)
    if empty and value in ('',None): return tuple(out)
    require(value in vocabulary(values), 'Unknown categorical feature: '+str(value))
    out[vocabulary(values)[value]]=1.0
    return tuple(out)

class PreparedEncoder(Encoder):
    @staticmethod
    def one(value, values, *, empty=False):
        return list(categorical(value,tuple(values),empty))

    @staticmethod
    def counts(items, values, scale=1):
        values=tuple(values); indices=vocabulary(values); out=[0.0]*len(values)
        for value, count in Counter(items).items():
            require(value in indices,'Unknown count feature')
            out[indices[value]]=count/scale
        return out

    def encode(self, view, choices, style=None, intent=None, *, validated=False):
        if not validated:
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
        state = np.asarray(x, dtype=np.float32)
        require(state.shape == (self.state_dim,) and actions.shape == (len(choices), self.action_dim), 'Feature layout mismatch')
        require(np.isfinite(state).all() and np.isfinite(actions).all(), 'Non-finite features')
        return Encoded(state, actions, np.asarray([relative(i) for i in range(4)], dtype=np.int64), ids)
