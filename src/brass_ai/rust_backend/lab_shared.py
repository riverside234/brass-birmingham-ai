"""Bounded parent-owned transient buffers. No replay payload touches disk.

The parent allocates *before* an actor writes: even an actor killed during export
cannot strand an unknown shared segment. Publishers detach each sealed chunk
once into immutable bytes. This keeps NumPy views alive after lease retirement.

The parent issues each ordinary game's lease with the job (reserve), sized from the
largest export so far. The lane then writes its export inside the CPU ticket it played
the game under: no allocation round trip to the parent and no second ticket wait. The
parent shrinks the lease to the written chunks when the result arrives (settle). An
export that does not fit takes the original exact allocation path. Each chunk is hashed
once; the part's seal binds the ordered chunk sizes and hashes.
"""
import hashlib
import json
from multiprocessing import shared_memory
import threading
import time
import uuid

from brass_ai.learning.contracts import require

FORMAT = 'brass-ordinary-shared-part/2'
SHARED_ROWS = 'BRASS_SHARED_ROWS'          # '0': publishers parse every row in full (json.loads)
PREISSUED = 'BRASS_PREISSUED_LEASES'       # '0': every export asks the parent for an exact lease


def switch(name):
    import os
    value = os.environ.get(name, '1')
    require(value in ('0', '1'), name + ' must be 0 or 1')
    return value == '1'
CHUNK = 16 * 1024**2
LIMIT = 512 * 1024**2
BUDGET = 6 * 1024**3
OFFER = 64 * 1024**2      # first pre-issued lease; then 1.25 x the largest export so far
_TIMING_KEYS = ('capacity_wait_seconds', 'mapping_setup_seconds')


def seal(chunks, hashes):
    """One digest over the ordered chunk sizes and SHA-256 hashes, each chunk hashed once."""
    return hashlib.sha256(json.dumps([[c['size'], h] for c, h in zip(chunks, hashes, strict=True)],
                                     separators=(',', ':')).encode()).hexdigest()


class Registry:
    def __init__(self, cancel, *, budget=BUDGET):
        self.cancel, self.budget = cancel, budget
        self.condition = threading.Condition(); self.leases = {}; self.used = self.peak = 0
        self.waiting = 0; self.largest = 0
        self.offers = dict(issued=0, used=0, released=0, too_small=0)

    def allocate(self, size):
        require(type(size) is int and 0 < size <= min(self.budget, LIMIT), 'Shared part exceeds bound')
        began=time.perf_counter()
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
        admitted=time.perf_counter()
        owners = []
        try:
            for start in range(0, size, CHUNK):
                owners.append(shared_memory.SharedMemory(create=True, size=min(CHUNK, size-start)))
            handle = dict(format=FORMAT, lease=uuid.uuid4().hex, size=size,
                          chunks=[dict(name=s.name, size=s.size) for s in owners])
            with self.condition:self.leases[handle['lease']] = (handle, owners)
            return dict(handle, capacity_wait_seconds=admitted-began,
                        mapping_setup_seconds=time.perf_counter()-admitted)
        except BaseException:
            for s in owners:
                s.close(); s.unlink()
            with self.condition:
                self.used -= size; self.condition.notify_all()
            raise

    def offer_size(self):
        with self.condition:largest = self.largest
        size = OFFER if not largest else -(-(largest * 5 // 4) // 2**20) * 2**20
        return min(size, LIMIT, self.budget)

    def reserve(self):
        """A lease issued with a job, before the game is played. It counts fully against the budget."""
        handle = self.allocate(self.offer_size())
        with self.condition:self.offers['issued'] += 1
        return handle

    def _expected(self, handle):
        with self.condition:
            require(handle.get('lease') in self.leases, 'Unknown shared lease')
            return self.leases[handle['lease']]

    def verify(self, handle):
        """The handle names this lease's leading chunks: all full but the last, which is not empty."""
        expected, _ = self._expected(handle)
        chunks = handle.get('chunks')
        require(handle.get('format') == FORMAT and type(handle.get('size')) is int and type(chunks) is list
                and 0 < len(chunks) <= len(expected['chunks']) and 0 < handle['size'] <= expected['size']
                and sum(c['size'] for c in chunks) == handle['size']
                and all(set(c) == {'name', 'size'} and c['name'] == e['name'] and type(c['size']) is int
                        and 0 < c['size'] <= e['size'] and (c['size'] == e['size'] or i == len(chunks)-1)
                        for i, (c, e) in enumerate(zip(chunks, expected['chunks']))),
                'Shared lease identity changed')

    def settle(self, offered, result):
        """After a lane's reply: shrink a used pre-issued lease to its written chunks, else release it."""
        if offered is None:return
        shared = isinstance(result, dict) and result.get('format') == FORMAT
        if not (shared and result.get('lease') == offered['lease']):
            # Unused: a larger export took the exact allocation path, or the part is not shared.
            self.release(offered)
            with self.condition:self.offers['too_small' if shared else 'released'] += 1
            return
        self.verify(result)
        with self.condition:
            expected, owners = self.leases[result['lease']]
            names = {c['name'] for c in result['chunks']}
            dropped = [s for s in owners if s.name not in names]
            kept = [s for s in owners if s.name in names]
            self.leases[result['lease']] = (dict(expected, size=result['size'], chunks=result['chunks']), kept)
        for s in dropped:
            try:s.close()
            finally:
                try:s.unlink()
                except FileNotFoundError:pass
        with self.condition:
            self.used -= expected['size'] - result['size']; self.largest = max(self.largest, result['size'])
            self.offers['used'] += 1; self.condition.notify_all()

    def observe(self, handle):
        """Record an exactly allocated export's size, so later offers fit it."""
        if isinstance(handle, dict) and handle.get('format') == FORMAT and type(handle.get('size')) is int:
            with self.condition:self.largest = max(self.largest, handle['size'])

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
                                         budget_bytes=self.budget, active_leases=len(self.leases),
                                         preissued_leases=dict(self.offers), largest_part_bytes=self.largest)


class Export:
    """Raw native record pages plus exact feature bytes; no decoded game tree."""
    def __init__(self, metadata, records, features):
        self.metadata, self.records, self.features = metadata, records, features


def write(part, allocate=None, *, admit=None, cancel=None, lease=None):
    """Actor writes only a parent-owned lease; control IPC carries no history.

    With `lease` (issued with the job, written inside the caller's CPU ticket) it returns
    None when the export does not fit; the caller then takes the allocate path.
    """
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
    from contextlib import nullcontext
    if lease is not None:
        require(lease.get('format') == FORMAT and type(lease.get('size')) is int, 'Invalid pre-issued lease')
        if size > lease['size']:return None
        handle = dict(lease); queued = allocated = time.perf_counter(); context = nullcontext()
    else:
        queued=time.perf_counter(); handle = dict(allocate(size)); allocated=time.perf_counter()
        # Capacity waits happen outside CPU admission, so a full buffer cannot
        # consume every actor slot and starve the publishers that would free it.
        context = admit() if admit else nullcontext()
    capacity_wait=handle.pop('capacity_wait_seconds',None)
    mapping_setup=handle.pop('mapping_setup_seconds',None)
    sources = iter(values); source = memoryview(next(sources)); hashes = []; chunks = []; remaining = size
    with context:
        for chunk in handle['chunks']:
            if not remaining:break
            require(cancel is None or not cancel.is_set(),'Shared export canceled')
            count_chunk = min(chunk['size'], remaining)
            shm = shared_memory.SharedMemory(name=chunk['name'])
            try:
                offset = 0; hashed = hashlib.sha256()
                while offset < count_chunk:
                    if not len(source):source = memoryview(next(sources))
                    count = min(len(source), count_chunk-offset)
                    shm.buf[offset:offset+count] = source[:count]
                    hashed.update(source[:count])
                    offset += count; source = source[count:]
                hashes.append(hashed.hexdigest())
            finally:shm.close()
            chunks.append(dict(name=chunk['name'], size=count_chunk)); remaining -= count_chunk
    require(not remaining and not len(source) and next(sources, None) is None, 'Shared export length differs')
    return dict(format=FORMAT, lease=handle['lease'], size=size, chunks=chunks, sha256=seal(chunks, hashes),
                chunk_hashes=hashes, branch_expected=part.metadata['branch_expected'],
                transfer_timings=dict(shared_write_seconds=time.perf_counter()-began,
                                      shared_allocate_wait_seconds=allocated-queued,
                                      shared_capacity_wait_seconds=capacity_wait,
                                      shared_mapping_setup_seconds=mapping_setup,
                                      shared_preissued=lease is not None,
                                      shared_bytes=size, temporary_payload_write_bytes=0))


def read(handle):
    """Verify exact bytes once, detach, then release all OS mappings promptly."""
    import struct
    require(handle.get('format') == FORMAT and 0 < handle['size'] <= LIMIT and
            len(handle['chunks']) == len(handle['chunk_hashes']) and
            sum(x['size'] for x in handle['chunks']) == handle['size'] and
            seal(handle['chunks'], handle['chunk_hashes']) == handle['sha256'], 'Invalid shared descriptor')
    buffers = []
    for chunk, fingerprint in zip(handle['chunks'], handle['chunk_hashes'], strict=True):
        require(0 < chunk['size'] <= CHUNK, 'Invalid shared chunk size')
        shm = shared_memory.SharedMemory(name=chunk['name'])
        try:
            require(shm.size >= chunk['size'], 'Shared chunk truncated')
            raw = bytes(shm.buf[:chunk['size']])
        finally:shm.close()
        require(hashlib.sha256(raw).hexdigest() == fingerprint, 'Shared chunk changed')
        buffers.append(memoryview(raw))
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
    from .lab_delta import Decoder, SharedRows
    from .lab_transport import decode_arrays
    from brass_ai.recovery.replay import CheckedGame
    value = dict(part.metadata); receipt = value['native_receipt']
    decoder = Decoder(); shared = SharedRows() if switch(SHARED_ROWS) else None
    records = []; evidence = []; start = 0
    hashes = [hashlib.sha256(b'['),hashlib.sha256(b'[')]
    for meta, raw in part.records:
        require(meta['token']==receipt['token'] and meta['rows']==receipt['rows'] and meta['start']==start
                and start < meta['end'] <= min(start+64,receipt['rows']) and
                len(meta['lengths'])==meta['end']-start, 'Shared record page binding differs')
        for prefix, history, suffix, base, proof in decoder.split_records(meta,raw):
            if start:
                for hashed in hashes:hashed.update(b',')
            # The seal covers the row's exact bytes; hash them in place, without joining.
            for piece in ((prefix,) if history is None else (prefix,history,suffix)):hashes[0].update(piece)
            hashes[1].update(proof)
            # Unchanged history events, board fields, players and choices are parsed once and shared.
            records.append(json.loads(bytes(prefix)) if history is None else
                           json.loads(b''.join((prefix,history,suffix))) if shared is None else
                           shared.row(prefix,history,suffix,base))
            evidence.append(json.loads(bytes(proof)))
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
    # Not part of the replay: an attribute, reported in the publication timings.
    value['game'].decode_sharing = (dict(parsed=shared.parsed, reused=shared.reused, fallbacks=shared.fallbacks)
                                    if shared is not None else dict(disabled=True))
    value['evidence'] = evidence
    return value
