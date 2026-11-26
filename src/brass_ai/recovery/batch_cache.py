"""Bounded, iteration-local tensor reuse; content changes never hit a cache.

The public collator still validates untrusted input. An audit can reuse selector
admission only after checking the current array's complete content, shape and
layout. Hashing runs in C and avoids repeating the Python/NumPy selector walks.
No cache or counter is checkpointed and no model output is cached.
"""
from collections import OrderedDict
import functools
import hashlib
import threading
import weakref

import torch

from brass_ai.learning.contracts import require
from brass_ai.learning.features import check_encoded
from brass_ai.learning.local_entities import check_action_selectors


SCAN_CACHE_BYTES = 128 * 1024 * 1024
PROBE_CACHE_BYTES = 256 * 1024 * 1024


def array_digest(array):
    value = hashlib.sha256()
    value.update(str((array.shape, array.dtype.str)).encode())
    value.update(memoryview(array) if array.flags.c_contiguous else array.tobytes())
    return value.digest()


def _forget(audit, key, dead):
    """A proof's weakref callback. It holds the audit weakly (g5466): a closure over the audit made the audit and
    every proof a reference cycle, freed only by a full GC pass."""
    audit = audit()
    if audit is not None: audit._forget(key, dead)


class InputAudit:
    def __init__(self):
        self.proofs = {}
        self.checks = self.hits = 0
        self.lock = threading.RLock()
        self._self = weakref.ref(self)

    def check(self, row):
        from .replay_binding import checked_encoded
        check_encoded(row)
        if checked_encoded(row):
            # g6120: a sealed row was admitted when it was sealed and is immutable since; its proof is the object
            # itself (alive while any batch of it is cached), with no SHA-256 of its actions.
            with self.lock:self.hits+=1
            return b'sealed:%d' % id(row)
        seal = array_digest(row.actions)
        # A weak reference prevents object-ID reuse from becoming an admission
        # proof, without retaining temporary state-only candidate slices.
        with self.lock:
            previous = self.proofs.get(id(row))
            if previous is not None and previous[0]() is row and previous[1] == seal:
                self.hits += 1
            else:
                check_action_selectors(row.actions)
                key = id(row)
                try:
                    reference = weakref.ref(row, functools.partial(_forget, self._self, key))
                except TypeError:
                    reference = None  # Duck-typed inputs retain full admission.
                if reference is not None:self.proofs[key] = (reference, seal)
                self.checks += 1
        return seal

    def _forget(self, key, dead):
        with self.lock:
            if self.proofs.get(key, (None,))[0] is dead:
                self.proofs.pop(key)

    def seal(self, rows, distributions=()):
        value = hashlib.sha256()
        for row in rows:
            token=self.check(row);value.update(token)
            if token.startswith(b'sealed:'):continue  # g6120: the immutable sealed object covers its state and mapping
            for field in ('state','stable_to_relative'):
                value.update(array_digest(getattr(row,field)))
        for q in distributions:
            value.update(array_digest(q))
        return value.digest()

    def report(self):
        return dict(selector_checks=self.checks, selector_reuses=self.hits,
                    scope='Content-checked selector admission within one Dataset')


def tensor_bytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


class BatchCache:
    """Reserve a bounded prefix, rather than thrash during full dataset scans.

Entries own their collated buffers. They are private to the read-only model
input path; source arrays are content-checked on every hit. Once full, later
parts use the existing packing/prefetch path without displacing reusable parts.
"""
    def __init__(self, limit=None, max_entries=256):
        self.limit, self.max_entries = SCAN_CACHE_BYTES if limit is None else limit, max_entries
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.refused = 0
        self.lock = threading.RLock()

    def get(self, key, seal):
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            require(entry[0] == seal, 'Cached training inputs changed after admission')
            self.hits += 1
            return entry[1]

    def put(self, key, seal, tensors):
        size = tensor_bytes(tensors)
        with self.lock:
            if key in self.entries:
                require(self.entries[key][0] == seal, 'Cached training inputs changed after admission')
                return self.entries[key][1]
            if size > self.limit - self.bytes or len(self.entries) >= self.max_entries:
                self.refused += 1
                return tensors
            self.entries[key] = (seal, tensors)
            self.bytes += size
            return tensors

    def report(self):
        with self.lock:
            return dict(bytes=self.bytes, limit_bytes=self.limit, entries=len(self.entries),
                        hits=self.hits, misses=self.misses, uncached_parts=self.refused)


def probe_device_limit(device):
    if torch.device(device).type != 'cuda':
        return PROBE_CACHE_BYTES
    # Leave space for activations, Adam and other models; do not reserve a
    # fraction of total VRAM that may already belong to another process.
    free, _ = torch.cuda.mem_get_info(device)
    return min(PROBE_CACHE_BYTES, free // 8)
