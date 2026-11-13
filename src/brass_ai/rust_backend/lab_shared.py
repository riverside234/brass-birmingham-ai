"""Bounded parent-owned transient buffers. No replay payload touches disk.

The parent allocates *before* an actor writes: even an actor killed during export
cannot strand an unknown shared segment. Publishers detach each sealed chunk
once into immutable bytes. This keeps NumPy views alive after lease retirement.
"""
import hashlib
import json
from multiprocessing import shared_memory
import threading
import time
import uuid

from brass_ai.learning.contracts import require

FORMAT = 'brass-ordinary-shared-part/1'
CHUNK = 16 * 1024**2
LIMIT = 512 * 1024**2
BUDGET = 6 * 1024**3


class Registry:
    def __init__(self, cancel, *, budget=BUDGET):
        self.cancel, self.budget = cancel, budget
        self.condition = threading.Condition(); self.leases = {}; self.used = self.peak = 0
        self.waiting = 0

    def allocate(self, size):
        require(type(size) is int and 0 < size <= min(self.budget, LIMIT), 'Shared part exceeds bound')
        with self.condition:
            if self.used + size > self.budget:
                self.waiting += 1
                try:
                    while self.used + size > self.budget:
                        require(not self.cancel.is_set(), 'Shared allocation canceled')
                        self.condition.wait(.05)
                finally:self.waiting -= 1
            require(not self.cancel.is_set(), 'Shared allocation canceled')
            self.used += size; self.peak = max(self.peak, self.used)
        owners = []
        try:
            for start in range(0, size, CHUNK):
                owners.append(shared_memory.SharedMemory(create=True, size=min(CHUNK, size-start)))
            handle = dict(format=FORMAT, lease=uuid.uuid4().hex, size=size,
                          chunks=[dict(name=s.name, size=s.size) for s in owners])
            with self.condition:self.leases[handle['lease']] = (handle, owners)
            return handle
        except BaseException:
            for s in owners:
                s.close(); s.unlink()
            with self.condition:
                self.used -= size; self.condition.notify_all()
            raise

    def verify(self, handle):
        with self.condition:
            require(handle.get('lease') in self.leases, 'Unknown shared lease')
            expected, _ = self.leases[handle['lease']]
            require({k:v for k,v in handle.items() if k not in ('sha256', 'chunk_hashes', 'branch_expected', 'transfer_timings')}
                    == expected, 'Shared lease identity changed')

    def release(self, handle):
        with self.condition:
            item = self.leases.pop(handle['lease'], None)
        if item is None:return
        expected, owners = item
        for s in owners:
            try:s.close()
            finally:
                try:s.unlink()
                except FileNotFoundError:pass
        with self.condition:
            self.used -= expected['size']; self.condition.notify_all()

    def dispatch_limit(self, lanes):
        """Pause new parents while completed games wait for handoff space."""
        with self.condition:
            return 0 if self.waiting else lanes

    def close(self):
        for handle, _ in list(self.leases.values()):self.release(handle)

    def report(self):
        with self.condition:return dict(active_bytes=self.used, peak_bytes=self.peak,
                                         budget_bytes=self.budget, active_leases=len(self.leases))


class Export:
    """Raw native record pages plus exact feature bytes; no decoded game tree."""
    def __init__(self, metadata, records, features):
        self.metadata, self.records, self.features = metadata, records, features


def write(part, allocate, *, admit=None, cancel=None):
    """Actor writes only a parent-owned lease; control IPC carries no history."""
    import struct
    began = time.perf_counter()
    require(isinstance(part, Export), 'Shared export requires native wire pages')
    pages = [dict(meta=meta, size=len(raw)) for meta, raw in part.records]
    header = json.dumps(dict(metadata=part.metadata, records=pages,
                             features=[len(raw) for raw in part.features]),
                        allow_nan=False, separators=(',', ':')).encode()
    require(len(header) <= 4*1024**2, 'Shared export header exceeds bound')
    values = [struct.pack('<I', len(header)), header,
              *(raw for _, raw in part.records), *part.features]
    size = sum(map(len, values))
    queued=time.perf_counter(); handle = allocate(size); allocated=time.perf_counter()
    from contextlib import nullcontext
    sources = iter(values); source = memoryview(next(sources)); hashes = []; total = hashlib.sha256()
    # Capacity waits happen outside CPU admission, so a full buffer cannot
    # consume every actor slot and starve the publishers that would free it.
    with (admit() if admit else nullcontext()):
        for chunk in handle['chunks']:
            require(cancel is None or not cancel.is_set(),'Shared export canceled')
            shm = shared_memory.SharedMemory(name=chunk['name'])
            try:
                offset = 0; hashed = hashlib.sha256()
                while offset < chunk['size']:
                    if not len(source):source = memoryview(next(sources))
                    count = min(len(source), chunk['size']-offset)
                    shm.buf[offset:offset+count] = source[:count]
                    hashed.update(source[:count]); total.update(source[:count])
                    offset += count; source = source[count:]
                hashes.append(hashed.hexdigest())
            finally:shm.close()
    require(not len(source) and next(sources, None) is None, 'Shared export length differs')
    return dict(handle, sha256=total.hexdigest(), chunk_hashes=hashes,
                branch_expected=part.metadata['branch_expected'],
                transfer_timings=dict(shared_write_seconds=time.perf_counter()-began,
                                      shared_allocate_wait_seconds=allocated-queued,
                                      shared_bytes=size, temporary_payload_write_bytes=0))


def read(handle):
    """Verify exact bytes once, detach, then release all OS mappings promptly."""
    import struct
    require(handle.get('format') == FORMAT and 0 < handle['size'] <= LIMIT and
            len(handle['chunks']) == len(handle['chunk_hashes']) and
            sum(x['size'] for x in handle['chunks']) == handle['size'], 'Invalid shared descriptor')
    buffers = []; total = hashlib.sha256()
    for chunk, fingerprint in zip(handle['chunks'], handle['chunk_hashes'], strict=True):
        require(0 < chunk['size'] <= CHUNK, 'Invalid shared chunk size')
        shm = shared_memory.SharedMemory(name=chunk['name'])
        try:
            require(shm.size >= chunk['size'], 'Shared chunk truncated')
            raw = bytes(shm.buf[:chunk['size']])
        finally:shm.close()
        require(hashlib.sha256(raw).hexdigest() == fingerprint, 'Shared chunk changed')
        total.update(raw); buffers.append(memoryview(raw))
    require(total.hexdigest() == handle['sha256'], 'Shared part changed')
    index = offset = consumed = 0
    def take(size):
        nonlocal index, offset, consumed
        require(type(size) is int and size >= 0 and consumed+size <= handle['size'], 'Shared slice exceeds bound')
        parts = []; remaining = size
        while remaining:
            if offset == len(buffers[index]):index += 1; offset = 0
            count = min(remaining, len(buffers[index])-offset)
            parts.append(buffers[index][offset:offset+count]); offset += count; remaining -= count
        consumed += size
        return parts[0] if len(parts) == 1 else memoryview(b''.join(parts))
    header_size = struct.unpack('<I', take(4))[0]
    require(0 < header_size <= 4*1024**2, 'Invalid shared header size')
    header = json.loads(bytes(take(header_size)))
    pages = [(p['meta'], take(p['size'])) for p in header['records']]
    features = [take(size) for size in header['features']]
    require(consumed == handle['size'] and header['metadata']['branch_expected'] == handle['branch_expected'],
            'Shared part coverage changed')
    return Export(header['metadata'], pages, features)


def decode(part):
    """Publisher-only materialization, with the original Rust byte seals."""
    import numpy as np
    from .lab_delta import Decoder
    from .lab_transport import decode_arrays
    from brass_ai.recovery.replay import CheckedGame
    value = dict(part.metadata); receipt = value['native_receipt']
    decoder = Decoder(); records = []; evidence = []; start = 0
    hashes = [hashlib.sha256(b'['),hashlib.sha256(b'[')]
    for meta, raw in part.records:
        require(meta['token']==receipt['token'] and meta['rows']==receipt['rows'] and meta['start']==start
                and start < meta['end'] <= min(start+64,receipt['rows']) and
                len(meta['lengths'])==meta['end']-start, 'Shared record page binding differs')
        for row, proof in decoder.records(meta,raw):
            for destination, hashed, raw_json in zip((records,evidence),hashes,(row,proof),strict=True):
                if start:hashed.update(b',')
                hashed.update(raw_json);destination.append(json.loads(bytes(raw_json)))
            start += 1
        require(start==meta['end'],'Shared page row count differs')
    for hashed in hashes:hashed.update(b']')
    require(start==receipt['rows']==len(part.features) and
            [h.hexdigest() for h in hashes]==[receipt['records_sha256'],receipt['evidence_sha256']],
            'Native completion data seal differs')
    arrays = []
    for row, raw in zip(records,part.features,strict=True):
        e = decode_arrays(raw,row['choices'])
        require(np.array_equal(e.state[-12:],np.asarray(row['style']+row['intent'],dtype=np.float32)) and
                np.array_equal(e.stable_to_relative,[(s-row['seat'])%4 for s in range(4)]),
                'Shared feature context/seat differs')
        arrays.append(e)
    game = dict(value['game'],records=records)
    value['game'] = CheckedGame(game).bind(arrays)
    value['game'].actor_validated = value.pop('actor_validated')
    value['game']._native_arrays = arrays
    value['evidence'] = evidence
    return value
