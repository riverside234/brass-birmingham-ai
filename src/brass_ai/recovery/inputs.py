"""Exact wire validation, caching independent subtrees by immutable content.

The pinned schema checks cache misses. Canonical JSON hashes intentionally do
not distinguish equivalent lists and tuples on cache hits; that known gap is
accepted to keep the fast paths free of additional recursive container scans.
Array bounds, unknown fields and seat/hand checks remain in the validators.
"""
from collections import OrderedDict
import copy
from functools import lru_cache
import threading

from brass_ai.learning.contracts import digest, require
from brass_ai.runtime_contracts import validate,validator


GROUPS=('players','slots','links','merchants','history')


@lru_cache(maxsize=1)
def split_validators():
    # Start with the fingerprint-checked schema. Keep all outer constraints,
    # and validate each independent item against its ORIGINAL subschema.
    original=validator('observation');schema=copy.deepcopy(original.schema)
    for name in GROUPS:
        schema['$defs']['observation']['properties'][name].pop('items')
    outer=type(original)(schema)
    items={name:original.evolve(schema=dict(original.schema,
        **{'$ref':f'#/$defs/observation/properties/{name}/items'})) for name in GROUPS}
    # Cash, spending and hand counts change much more often than industry
    # stacks. Rechecking every tile whenever a player changes dominates the
    # actor boundary. Keep the complete player/array constraints and validate
    # each mat against its original schema, cached by current content below.
    players=copy.deepcopy(items['players'].schema)
    players['$defs']['observation']['properties']['players']['items']['properties']['mat'].pop('items')
    items['players']=type(original)(players)
    items['player_mat']=original.evolve(schema=dict(original.schema,
        **{'$ref':'#/$defs/observation/properties/players/items/properties/mat/items'}))
    choice_schema=copy.deepcopy(validator('choices').schema)
    choice_schema['$defs']['choices'].pop('items')
    choices_outer=type(original)(choice_schema)
    # Choice branches have disjoint required kind enums. Dispatch directly to
    # the exact branch instead of trying every oneOf branch on a cache miss.
    branches={}
    for i,branch in enumerate(original.schema['$defs']['choice']['oneOf']):
        require('kind' in branch['required'],'Choice discriminator must be required')
        for kind in branch['properties']['kind']['enum']:
            require(kind not in branches,'Choice discriminator must be unique')
            branches[kind]=original.evolve(schema=dict(original.schema,**{'$ref':f'#/$defs/choice/oneOf/{i}'}))
    return outer,choices_outer,items,branches


class InputBoundary:
    def __init__(self, frames=4096, events=65536, parts=16384):
        self.frames, self.events = OrderedDict(), OrderedDict()
        self.parts=OrderedDict();self.part_limit=parts
        self.frame_limit, self.event_limit = frames, events
        self.lock = threading.RLock()
        self.checked_frames = self.checked_events = self.frame_hits = 0
        self.checked_parts=self.part_hits=0

    @staticmethod
    def _remember(cache, key, limit):
        cache[key] = True
        cache.move_to_end(key)
        while len(cache) > limit: cache.popitem(last=False)

    def _part(self,name,value,check):
        key=(name,digest(value))
        with self.lock:
            if key in self.parts:
                self.parts.move_to_end(key);self.part_hits+=1
                return
        check(value)
        with self.lock:
            self._remember(self.parts,key,self.part_limit);self.checked_parts+=1

    def _history(self, history):
        for event in history:
            token = digest(event)
            with self.lock: known = token in self.events
            if not known:
                validate('history_entry', event)
                with self.lock:
                    self._remember(self.events, token, self.event_limit)
                    self.checked_events += 1

    def check(self, view, choices):
        key = digest(dict(view=view, choices=choices))
        with self.lock:
            if key in self.frames:
                self.frames.move_to_end(key); self.frame_hits += 1
                return key
        outer,choices_outer,items,branches=split_validators()
        outer.validate(view);choices_outer.validate(choices)
        def player(value):
            items['players'].validate(value)
            for mat in value['mat']:self._part('player_mat',mat,items['player_mat'].validate)
        for name in GROUPS[:-1]:
            check=player if name=='players' else items[name].validate
            for item in view[name]:self._part(name,item,check)
        # These are the same cross-field checks as runtime_contracts.validate.
        require([p['seat'] for p in view['players']]==list(range(4)) and sorted(view['order'])==list(range(4)),
                'Four stable seats required')
        require(len(view['own_hand'])==view['players'][view['seat']]['hand_count'],'Private hand count inconsistent')
        for choice in choices:
            kind=choice.get('kind') if isinstance(choice,dict) else None
            branch=branches.get(kind) if isinstance(kind,str) else None
            self._part('choice',choice,(branch or validator('choice')).validate)
        history = view['history']
        require(type(history) is list, 'Observation history must be an array')
        # Public history usually grows by appending one event. Hash bounded
        # chunks in the JSON encoder instead of rehashing every prior event in
        # Python at every subchoice. Content keys still detect edits anywhere;
        # misses use the same exact per-event validator and bounded caches.
        for start in range(0, len(history), 32):
            self._part('history_chunk', history[start:start+32], self._history)
        with self.lock:
            self._remember(self.frames, key, self.frame_limit)
            self.checked_frames += 1
        return key

    def report(self):
        return dict(validated_frames=self.checked_frames, validated_history_events=self.checked_events,
                    frame_cache_hits=self.frame_hits, retained_frames=len(self.frames),
                    retained_history_events=len(self.events),validated_parts=self.checked_parts,
                    part_cache_hits=self.part_hits,retained_parts=len(self.parts))


BOUNDARY = InputBoundary()
