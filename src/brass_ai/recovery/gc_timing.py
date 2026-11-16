"""Diagnostic-only record of Python cyclic-GC pauses; collection itself is unchanged."""
from contextlib import contextmanager
import gc
import time

FULL_LIMIT = 32


class Watch:
    def __init__(self):
        self.began = time.perf_counter(); self.start = None
        self.collections = {}; self.durations = {}; self.full = []

    def __call__(self, phase, info):
        if phase == 'start':
            self.start = time.perf_counter(); return
        if self.start is None: return
        seconds = time.perf_counter() - self.start; self.start = None
        generation = str(info.get('generation', 0))
        self.collections[generation] = self.collections.get(generation, 0) + 1
        self.durations[generation] = self.durations.get(generation, 0.) + seconds
        if generation == '2' and len(self.full) < FULL_LIMIT:
            self.full.append(dict(at_seconds=round(self.start_offset(), 3), seconds=seconds, collected=info.get('collected')))

    def start_offset(self): return time.perf_counter() - self.began

    def seconds(self): return sum(self.durations.values())

    def report(self):
        return dict(collections=dict(self.collections), seconds=dict(self.durations), full_collections=self.full,
                    scope='Python cyclic GC pauses inside this window (all threads), from gc.callbacks; at_seconds is the '
                          'end of each full collection after the window opened. Diagnostic only.')


@contextmanager
def watch():
    collector = Watch(); gc.callbacks.append(collector)
    try: yield collector
    finally:
        try: gc.callbacks.remove(collector)
        except ValueError: pass
