"""Six-group compact version-4 layout, with separate categorical/context caches.

The original raw fields retain their meaning. validated=True is used only after the
recovery ingress boundary has checked the original permitted view and choices.
"""
from collections import Counter
from collections import OrderedDict
from functools import lru_cache
import numpy as np
from brass_ai.runtime_contracts import validate
from brass_ai.learning.contracts import finite_vector, require
from brass_ai.learning.features import Encoder, Encoded
from brass_ai.learning.encoding_budget import check_budget
from brass_ai.learning.contracts import digest
from .inputs import BOUNDARY

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
    def __init__(self, cache_size=256):
        super().__init__()
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.action_cache=OrderedDict()
        # The complete encoded context cache owns reuse; derived numeric effects
        # are local to each miss and never share the categorical action cache.

    def choice_features(self, choice):
        # Semantic fields, including physical Scout multiplicity; IDs alone are
        # not trusted as a cache key. The caller has validated the wire choice.
        key=(choice['kind'],choice.get('target'),choice.get('merchant'),choice.get('industry'),
             choice.get('card'),tuple(choice.get('cards',[])))
        if key not in self.action_cache:
            self.action_cache[key]=tuple(super().choice_features(choice))
        self.action_cache.move_to_end(key)
        while len(self.action_cache)>8192:self.action_cache.popitem(last=False)
        return self.action_cache[key]

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

    def encode(self, view, choices, style=None, intent=None, *, validated=False,deadline=None,cancelled=None):
        # The boundary already hashed the complete permitted inputs. Reuse that
        # immutable content identity instead of serializing the whole growing
        # history again merely to append twelve policy-context values.
        check_budget(deadline,cancelled)
        source = digest(dict(view=view, choices=choices)) if validated else BOUNDARY.check(view, choices)
        key = digest(dict(schema=self.derived.schema,input=source, style=style, intent=intent))
        check_budget(deadline,cancelled)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
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
                    x += self.one(relative(tile['seat']) if tile else None, list(range(4)), empty=True)
                    x += self.one(tile['type'] if tile else None, self.industries, empty=True)
                    x += self.board_tile(tile)
                elif group == 'links':
                    x += self.one(relative(item['owner']) if 'owner' in item else None, list(range(4)), empty=True)
                else:
                    x += self.counts(item.get('types', []), self.industries)
                    x += [item['beer']]
        x += [view['coal_market'] / 14, view['iron_market'] / 10, view['deck_count'] / 64,
              view['wild_industry_count'] / 4, view['wild_location_count'] / 4]
        pending = view['pending']
        x += self.one(pending['kind'], self.kinds, empty=True)
        x += self.one(pending['target'], self.targets, empty=True)
        x += self.one(pending['merchant'], s['merchants'], empty=True)
        x += [pending[k] / 4 for k in ('parts', 'coal', 'iron', 'beer')]
        counts = Counter((relative(e['seat']), e['era'], e['choice']['kind']) for e in view['history'])
        x += [counts[(seat, era, kind)] / 32 for seat in range(4)
              for era in ('CanalEra', 'RailEra') for kind in s['history_kinds']]
        x += list(style) + list(intent)
        actions = np.asarray([self.choice_features(c) for c in choices], dtype=np.float32)
        state,actions = self.append_derived(x,actions,view,choices,style,intent,source=source,deadline=deadline,cancelled=cancelled)
        require(state.shape == (self.state_dim,) and actions.shape == (len(choices), self.action_dim), 'Feature layout mismatch')
        require(np.isfinite(state).all() and np.isfinite(actions).all(), 'Non-finite features')
        encoded = Encoded(state, actions, np.asarray([relative(i) for i in range(4)], dtype=np.int64), ids)
        for array in (encoded.state, encoded.actions, encoded.stable_to_relative): array.setflags(write=False)
        check_budget(deadline,cancelled)
        self.cache[key] = encoded
        while len(self.cache) > self.cache_size: self.cache.popitem(last=False)
        return encoded
