"""Exact wire validation with bounded caches of immutable content digests.

History entries have no cross-entry schema constraints. Validate every distinct
entry against the pinned schema, rather than walking the same prefix again.
Unknown fields, private data, and changed content still cross the full boundary.
"""
from collections import OrderedDict
import threading

from brass_ai.learning.contracts import digest, require
from brass_ai.runtime_contracts import validate


class InputBoundary:
    def __init__(self, frames=4096, events=65536):
        self.frames, self.events = OrderedDict(), OrderedDict()
        self.frame_limit, self.event_limit = frames, events
        self.lock = threading.RLock()
        self.checked_frames = self.checked_events = self.frame_hits = 0

    @staticmethod
    def _remember(cache, key, limit):
        cache[key] = True
        cache.move_to_end(key)
        while len(cache) > limit: cache.popitem(last=False)

    def check(self, view, choices):
        key = digest(dict(view=view, choices=choices))
        with self.lock:
            if key in self.frames:
                self.frames.move_to_end(key); self.frame_hits += 1
                return key
        history = view.get('history')
        require(type(history) is list, 'Observation history must be an array')
        validate('observation', dict(view, history=[]))
        validate('choices', choices)
        for event in history:
            token = digest(event)
            with self.lock: known = token in self.events
            if not known:
                validate('history_entry', event)
                with self.lock:
                    self._remember(self.events, token, self.event_limit)
                    self.checked_events += 1
        with self.lock:
            self._remember(self.frames, key, self.frame_limit)
            self.checked_frames += 1
        return key

    def report(self):
        return dict(validated_frames=self.checked_frames, validated_history_events=self.checked_events,
                    frame_cache_hits=self.frame_hits, retained_frames=len(self.frames),
                    retained_history_events=len(self.events))


BOUNDARY = InputBoundary()
