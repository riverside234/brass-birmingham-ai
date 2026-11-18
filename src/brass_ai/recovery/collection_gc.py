"""The garbage collector during collection (g3270, build step 2: search workers never wait).

A full CPython collection pauses every thread of the trainer process. In g3268/g3269 all
teacher tables stopped writing at the same moments, 3-4 times per collection for 3-16 s, and
resumed within half a second of each other: the trainer had stopped draining their pipes.
The process keeps large long-lived trees (label views, value rows, replay metadata); a full
pass walks all of them even though none can be garbage during collection.

At the start of collection the existing objects move to the permanent generation
(gc.freeze), so a collection during the window walks only the objects made since. At the end
they return to the oldest generation (gc.unfreeze) and the normal thresholds apply again.
Reference counting frees garbage as usual meanwhile, frozen or not; only cyclic garbage older
than the window waits until it ends. Every pause inside the window is timed for the report.
BRASS_COLLECTION_GC_FREEZE=0 keeps the timing and skips the freeze.
"""
from collections import Counter
import gc
import os
import time

from brass_ai.learning.contracts import require

SWITCH = 'BRASS_COLLECTION_GC_FREEZE'
CONTRACT = 'collection-gc-window/1.0.0'


def enabled():
    value = os.environ.get(SWITCH, '1')
    require(value in ('0', '1'), SWITCH + ' must be 0 or 1')
    return value == '1'


class Window:
    def __init__(self):
        self.active = False; self.report = None
        self._started = None

    def _callback(self, phase, info):
        if phase == 'start':
            self._started = time.perf_counter()
        elif self._started is not None:
            elapsed = time.perf_counter() - self._started; self._started = None
            generation = str(info.get('generation'))
            self._counts[generation] += 1; self._seconds[generation] += elapsed
            self._longest = max(self._longest, elapsed)

    def start(self):
        if self.active: self.stop()
        self._counts, self._seconds, self._longest = Counter(), Counter(), 0.
        self.freeze = enabled(); self.frozen = None; self._began = time.perf_counter()
        gc.callbacks.append(self._callback)
        if self.freeze:
            gc.freeze(); self.frozen = gc.get_freeze_count()
        self.active = True

    def stop(self):
        if not self.active: return self.report
        if self.freeze: gc.unfreeze()
        try:gc.callbacks.remove(self._callback)
        except ValueError:pass
        self.active = False
        self.report = dict(contract=CONTRACT, freeze=self.freeze, frozen_objects=self.frozen,
            window_seconds=time.perf_counter() - self._began, collections=dict(self._counts),
            pause_seconds={k: round(v, 6) for k, v in self._seconds.items()},
            total_pause_seconds=sum(self._seconds.values()), longest_pause_seconds=self._longest,
            scope='Collector passes inside ordinary and teacher collection; every trainer thread waits during each')
        return self.report
