"""Bounded, iteration-local tensor reuse; content changes never hit a cache.

The public collator still validates untrusted input. An audit can reuse selector
admission only after checking the current array's complete content, shape and
layout. Hashing runs in C and avoids repeating the Python/NumPy selector walks.
No cache or counter is checkpointed and no model output is cached.
"""
from collections import OrderedDict
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


class InputAudit:
    def __init__(self):
        self.proofs = {}
        self.checks = self.hits = 0
        self.lock = threading.RLock()

    def check(self, row):
        check_encoded(row)
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
                    reference = weakref.ref(row, lambda dead: self._forget(key, dead))
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
            value.update(self.check(row))
            value.update(array_digest(row.state))
            value.update(array_digest(row.stable_to_relative))
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
