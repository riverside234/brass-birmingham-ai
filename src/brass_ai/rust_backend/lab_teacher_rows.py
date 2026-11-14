"""Teacher label-row preparation in helper processes.

A streamed teacher turn is verified in the trainer process. Its schema
validation and feature encoding are pure functions of the turn's rows and the
frozen job identity, but in the trainer every one of them competes for the same
interpreter lock as the other tables' turns and ordinary collection. With 36
tables the native searches then wait for that lock instead of searching.

Here that work runs in a few spawned processes. Each table always goes to the
same process, so its validation cache stays warm as in a single process. The
trainer keeps its own frozen rows and receives exactly the arrays prepare_rows
computes in process; identity, legality, turn, logit and publication checks
are unchanged and stay in the trainer.
"""
import atexit
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeout
from concurrent.futures.process import BrokenProcessPool
import hashlib
import json
import multiprocessing
import threading
import time
import uuid

from brass_ai.learning.contracts import require

WORKERS = 4
SESSIONS = 8

_encoder = None
_sessions = OrderedDict()


def _initialize():
    import signal
    # The trainer owns shutdown; an interrupt reaches it and retires the pool.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def _instance():
    global _encoder
    if _encoder is None:
        from brass_ai.recovery.encoding import PreparedEncoder
        _encoder = PreparedEncoder()
    return _encoder


def _warm():
    import os
    from brass_ai.recovery.inputs import split_validators
    from brass_ai.recovery import teacher_labels  # noqa: F401
    split_validators()
    _instance()
    return os.getpid()


def _configure(key, history, sources):
    _sessions[key] = (history, frozenset(sources))
    _sessions.move_to_end(key)
    while len(_sessions) > SESSIONS:
        _sessions.popitem(last=False)
    return key


def _prepare(key, expected, rows):
    from brass_ai.recovery.teacher_labels import prepare_indexed
    require(key in _sessions, 'Teacher row session is not configured')
    history, sources = _sessions[key]
    began = time.perf_counter()
    accepted, rejected = prepare_indexed(json.loads(rows), expected=expected, history=history,
                                         admitted_sources=sources, encoder=_instance(), private=True)
    arrays = [(index, row.encoded.state, row.encoded.actions, row.encoded.stable_to_relative,
               row.encoded.choice_ids, row.encoded.feature_version) for index, row in accepted]
    return arrays, rejected, time.perf_counter() - began


class RowPool:
    """Persistent single-process executors; a table is always prepared by the same one."""
    def __init__(self, workers):
        require(type(workers) is int and 1 <= workers <= 16, 'Invalid teacher row worker count')
        context = multiprocessing.get_context('spawn')
        self.executors = [ProcessPoolExecutor(max_workers=1, mp_context=context, initializer=_initialize)
                          for _ in range(workers)]
        self.ready = [executor.submit(_warm) for executor in self.executors]
        self.broken = False

    def shutdown(self):
        for executor in self.executors:
            executor.shutdown(wait=False, cancel_futures=True)


_pool = None
_lock = threading.Lock()
_registered = False


def shared(workers=WORKERS):
    """The process-wide pool, created and warmed on first use; None if helpers cannot start."""
    global _pool, _registered
    with _lock:
        if _pool is not None and (_pool.broken or len(_pool.executors) != workers):
            _pool.shutdown()
            _pool = None
        if _pool is None:
            try:
                _pool = RowPool(workers)
            except (OSError, RuntimeError, BrokenProcessPool) as error:
                # Rows are then prepared in the trainer, exactly as before.
                print(f'Teacher row helpers unavailable; preparing rows in process: {error}', flush=True)
                return None
            if not _registered:
                atexit.register(close)
                _registered = True
        return _pool


def close():
    global _pool
    with _lock:
        if _pool is not None:
            _pool.shutdown()
            _pool = None


def route(identity, count):
    return int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % count


class RowSession:
    """One frozen collection's history and admitted sources in every helper process.

    If a helper dies, ``unavailable`` is set and the caller prepares the rest of
    the collection in process (identical results); the next collection starts
    a new pool.
    """
    def __init__(self, pool, history, sources, cancel):
        self.pool, self.cancel = pool, cancel
        self.key = uuid.uuid4().hex
        self.unavailable = False
        self.lock = threading.Lock()
        self.metrics = dict(turns=0, rows=0, worker_seconds=0., wait_seconds=0., fallback=False)
        lineage = [dict(generation=e['generation'], model_id=e['model_id']) for e in history]
        try:
            self.configured = [executor.submit(_configure, self.key, lineage, sorted(sources))
                               for executor in pool.executors]
        except (BrokenProcessPool, RuntimeError):
            self._broken()

    def _broken(self):
        self.pool.broken = True
        self.unavailable = True
        with self.lock:
            self.metrics['fallback'] = True

    def _wait(self, future):
        while True:
            require(not self.cancel.is_set(), 'Turn verification cancelled')
            try:
                return future.result(timeout=.05)
            except FutureTimeout:
                continue

    def prepare(self, identity, expected, rows):
        """Accepted LabelRows over the caller's own frozen rows and rejection counts; None if helpers died."""
        from brass_ai.learning.features import Encoded
        from brass_ai.recovery.teacher_labels import LabelRow
        if self.unavailable:
            return None
        index = route(identity, len(self.pool.executors))
        began = time.perf_counter()
        try:
            self._wait(self.configured[index])
            payload = json.dumps(list(rows), separators=(',', ':'))
            future = self.pool.executors[index].submit(_prepare, self.key, dict(expected), payload)
            arrays, rejected, spent = self._wait(future)
        except (BrokenProcessPool, RuntimeError) as error:
            if isinstance(error, RuntimeError) and not isinstance(error, BrokenProcessPool) and \
                    'after shutdown' not in str(error):
                raise
            self._broken()
            return None
        accepted = []
        for position, state, actions, mapping, ids, version in arrays:
            for array in (state, actions, mapping):
                array.setflags(write=False)
            accepted.append(LabelRow(rows[position], Encoded(state, actions, mapping, tuple(ids), version)))
        with self.lock:
            self.metrics['turns'] += 1
            self.metrics['rows'] += len(rows)
            self.metrics['worker_seconds'] += spent
            self.metrics['wait_seconds'] += time.perf_counter() - began
        return accepted, rejected

    def report(self):
        with self.lock:
            return dict(self.metrics, workers=len(self.pool.executors),
                        scope='Schema validation and feature encoding of labelled teacher turns in helper processes; '
                              'wait_seconds is the trainer-side wall time including the handoff')
