"""The garbage collector over a training iteration (g3270 collection window; g3658 iteration policy).

A full CPython collection pauses every thread of the trainer process. In g3268/g3269 all
teacher tables stopped writing at the same moments, 3-4 times per collection for 3-16 s, and
resumed within half a second of each other: the trainer had stopped draining their pipes.
The process keeps large long-lived trees (label views, value rows, replay metadata); a full
pass walks all of them even though none can be garbage during collection.

g3270: at the start of collection the existing objects move to the permanent generation
(gc.freeze), so a collection during the window walks only the objects made since. At the end
they return to the oldest generation (gc.unfreeze) and the normal thresholds apply again.

g3658: freezing alone made full passes more frequent, not rarer. CPython 3.12 starts a full
pass when the objects promoted since the last one exceed a quarter of the old generation, and
after a freeze the old generation is only the window's own survivors. The ingested replay grew
past that quarter again and again: the g3654-g3657 windows paused 39-57 s, 26-39 s of it in
7-18 full passes of 2-5 s each. After the window (unfrozen) the next full passes walked the
whole heap, 11M frozen objects plus the ingested games, and landed wherever they fell:
sampling (+20 s), PPO critic preparation (+21 s), the value phase's replay open (+16 s).

The iteration policy, on by default:
- start (collection begins): freeze the heap and turn automatic collection off. Reference
  counting frees garbage as usual; only cyclic garbage made during the window waits.
- stop (collection ends): freeze again. That is O(1): the window's survivors join the permanent
  generation without any pass. Automatic collection resumes with full passes off, so the rest
  of the iteration runs young passes over objects made after collection only.
- finish (iteration end, after the commit): unfreeze, restore the thresholds and run one full
  pass over everything. It is timed and reports the cyclic garbage it found, so no cycle lives
  longer than one iteration.

g5466: the full pass walked the whole heap, about 14M long-lived objects, for 15-16 s every iteration to free
the ~3.7M objects of cyclic garbage the iteration had made. Two cycles held most of what the fixtures make: every
SearchRows (its rows, era maps and device tensors) through a closure over itself, and every InputAudit with its
per-row proofs through a weakref callback over the audit; both are broken where they are made, so refcounting
frees them. The full pass now runs every FULL_EVERY iterations (BRASS_GC_FULL_EVERY, 1 restores every iteration),
and every iteration again while the last pass found more than GARBAGE_PER_ITERATION cyclic objects per iteration
since the one before. In between the heap stays frozen with young passes only, as in the tail after collection,
and the next collection start refreezes it; cyclic garbage from those iterations waits for the next full pass.

BRASS_ITERATION_GC=0 keeps the g3270 window (freeze, unfreeze at the end of collection).
BRASS_COLLECTION_GC_FREEZE=0 keeps the timing and skips any freeze or policy.
"""
from collections import Counter
import gc
import os
import time

from brass_ai.learning.contracts import require

SWITCH = 'BRASS_COLLECTION_GC_FREEZE'
POLICY = 'BRASS_ITERATION_GC'
CONTRACT = 'collection-gc-window/1.0.0'
ITERATION_CONTRACT = 'iteration-gc/2.0.0'
FULL_EVERY_SWITCH = 'BRASS_GC_FULL_EVERY'
FULL_EVERY = 5  # iterations between full passes while cyclic garbage stays low
GARBAGE_PER_ITERATION = 500_000  # cyclic objects per iteration above which the full pass runs every iteration
# The oldest generation never reaches this many young collections inside one iteration.
NO_FULL = 2 ** 31 - 1

# (thresholds, window) to restore if an iteration ended without finish() (an exception or a break).
_outstanding = None


def _flag(name):
    value = os.environ.get(name, '1')
    require(value in ('0', '1'), name + ' must be 0 or 1')
    return value == '1'


def enabled():
    return _flag(SWITCH)


def iteration_policy():
    return enabled() and _flag(POLICY)


def full_every():
    value = os.environ.get(FULL_EVERY_SWITCH, str(FULL_EVERY))
    require(value.isdigit() and 1 <= int(value) <= 20, FULL_EVERY_SWITCH + ' must be an integer from 1 to 20')
    return int(value)


def settle():
    """Undo a policy an earlier iteration left in force; the next window starts clean."""
    global _outstanding
    if _outstanding is None: return False
    thresholds, window = _outstanding; _outstanding = None
    window._detach(); window.active = window.tail = False
    gc.set_threshold(*thresholds); gc.unfreeze(); gc.enable()
    return True


class Window:
    def __init__(self):
        self.active = False; self.report = None; self.tail = False
        self._started = None
        self.since_full = 0; self.every_iteration = False  # g5466: full-pass cadence, per trainer process

    def _callback(self, phase, info):
        if phase == 'start':
            self._started = time.perf_counter()
        elif self._started is not None:
            elapsed = time.perf_counter() - self._started; self._started = None
            generation = str(info.get('generation'))
            self._counts[generation] += 1; self._seconds[generation] += elapsed
            self._longest = max(self._longest, elapsed)

    def _reset(self):
        self._counts, self._seconds, self._longest = Counter(), Counter(), 0.

    def _passes(self):
        return dict(collections=dict(self._counts), pause_seconds={k: round(v, 6) for k, v in self._seconds.items()},
                    total_pause_seconds=sum(self._seconds.values()), longest_pause_seconds=self._longest)

    def _detach(self):
        try: gc.callbacks.remove(self._callback)
        except ValueError: pass

    def start(self):
        global _outstanding
        if self.active: self.stop()
        if self.tail: self.finish()
        settle()
        self._reset()
        self.freeze = enabled(); self.policy = iteration_policy(); self.frozen = None; self._began = time.perf_counter()
        gc.callbacks.append(self._callback)
        if self.freeze:
            gc.freeze(); self.frozen = gc.get_freeze_count()
        if self.policy:
            _outstanding = (gc.get_threshold(), self); gc.disable()
        self.active = True

    def stop(self):
        if not self.active: return self.report
        if self.policy:
            # The window's survivors join the frozen heap without a pass; young passes resume, full passes wait for finish().
            young = _outstanding[0]
            gc.freeze(); gc.set_threshold(young[0], young[1], NO_FULL); gc.enable()
        else:
            if self.freeze: gc.unfreeze()
            self._detach()
        self.active = False
        self.report = dict(contract=CONTRACT, freeze=self.freeze, frozen_objects=self.frozen,
            window_seconds=time.perf_counter() - self._began, **self._passes(),
            policy='iteration' if self.policy else 'window',
            scope='Collector passes inside ordinary and teacher collection; every trainer thread waits during each')
        if self.policy:
            self.tail = True; self._tail_began = time.perf_counter(); self._reset()
        return self.report

    def finish(self):
        """Iteration end: the passes since collection, then a timed full pass over the whole heap when one is due
        (g5466: every full_every() iterations, or every iteration while cyclic garbage is high). Otherwise the
        policy stays in force (frozen heap, young passes only) until the next collection start settles it."""
        if not self.tail: return None
        after = dict(seconds=time.perf_counter() - self._tail_began, **self._passes())
        every = full_every(); self.since_full += 1
        scope = ('After collection: young passes only, over objects made since; the heap stays frozen. A full pass '
                 f'every {every} iterations, or every iteration while the last one found more than '
                 f'{GARBAGE_PER_ITERATION} cyclic objects per iteration; unreachable counts the cyclic garbage it freed.')
        if not (self.every_iteration or self.since_full >= every):
            self._detach(); self.tail = False
            return dict(contract=ITERATION_CONTRACT, after_collection=after, scope=scope,
                        full_collection=dict(seconds=0., ran=False, unreachable=None, iterations_since_full=self.since_full,
                                             next_full_in=every - self.since_full))
        settle()
        began = time.perf_counter(); unreachable = gc.collect()
        per_iteration = unreachable / self.since_full
        full = dict(seconds=time.perf_counter() - began, ran=True, unreachable=unreachable, iterations=self.since_full,
                    unreachable_per_iteration=per_iteration)
        self.every_iteration = per_iteration > GARBAGE_PER_ITERATION; self.since_full = 0
        full['next_full_in'] = 1 if self.every_iteration else every
        return dict(contract=ITERATION_CONTRACT, after_collection=after, full_collection=full, scope=scope)
