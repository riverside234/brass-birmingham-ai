"""Private, bounded, deadline-controlled ordinary lane transport."""
from __future__ import annotations
import json
import queue
import struct
import subprocess
import threading
import time
import uuid

import numpy as np
from brass_ai.learning.contracts import require, digest
from brass_ai.learning.features import Encoded, check_encoded
from brass_ai.learning.encoding_budget import EncodingStopped
from .lab_contract import SOURCE
from brass_ai.native.features import FORMAT as FEATURE_FORMAT,VERSION,SCHEMA,GROUP,ENTITY

CONTRACT = 'brass-lab-collection/1.0.0'
MAX_FRAME = 4 * 1024**2


class Pipe:
    def __init__(self, command, *, binary=True, timeout=15):
        self.binary, self.timeout, self.seq, self.closed = binary, timeout, 0, False
        self.lock = threading.RLock()
        self.jobs, self.results = queue.Queue(1), queue.Queue(1)
        self.process = subprocess.Popen([str(x) for x in command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.thread = threading.Thread(target=self._loop, daemon=True); self.thread.start()

    def _read(self, n):
        out = bytearray()
        while len(out) < n:
            part = self.process.stdout.read(n-len(out))
            require(part, 'Collector closed output'); out.extend(part)
        return out

    def _loop(self):
        try:
            while (payload := self.jobs.get()) is not None:
                view = memoryview(payload)
                while view:
                    count = self.process.stdin.write(view)
                    require(count, 'Collector closed input'); view = view[count:]
                if self.binary:
                    n = struct.unpack('<I', self._read(4))[0]
                    require(0 < n <= MAX_FRAME, 'Collector frame limit'); result = self._read(n)
                else:
                    result = self.process.stdout.readline(MAX_FRAME+1)
                    require(result.endswith(b'\n') and len(result) <= MAX_FRAME, 'Bootstrap frame limit')
                self.results.put(json.loads(result, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))))
        except BaseException as error:
            self.results.put(error)

    def call(self, op, **values):
        with self.lock:
            require(not self.closed, 'Closed collector')
            self.seq += 1
            request = dict(op=op, **values)
            if self.binary: request.update(protocol=CONTRACT, id=self.seq)
            payload = json.dumps(request, separators=(',', ':'), allow_nan=False).encode()
            require(len(payload) <= MAX_FRAME, 'Collector request limit')
            try:
                self.jobs.put(struct.pack('<I', len(payload))+payload if self.binary else payload+b'\n', timeout=self.timeout)
                reply = self.results.get(timeout=self.timeout)
                if isinstance(reply, BaseException): raise reply
                require(not self.binary or (reply['id'] == self.seq and reply['protocol'] == CONTRACT), 'Collector envelope differs')
                require(reply.get('ok') is True, str(reply.get('error', 'Collector error')))
                return reply['data']
            except BaseException:
                self.close(); raise

    def close(self):
        if self.closed: return
        self.closed = True
        if self.process.poll() is None: self.process.kill()
        self.process.wait(timeout=10)
        try: self.jobs.put_nowait(None)
        except queue.Full: pass
        self.thread.join(timeout=2)
        self.process.stdin.close(); self.process.stdout.close()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


class Collector(Pipe):
    def __init__(self, executable):
        super().__init__([executable])
        try:
            self.hello = self.call('hello')
            require(self.hello['contract'] == CONTRACT and self.hello['core']['source_sha256'] == SOURCE
                    and self.hello['search_enabled'] is False, 'Collector uses an unqualified shared core')
        except BaseException:
            self.close(); raise
        self.token = None; self.current = None

    def begin(self, job_id, origin, cap, *, checkpoint=None, indices=None):
        self.token = uuid.uuid4().hex
        fields = dict(job=job_id, token=self.token, cap=cap)
        self.current = self.call('begin', **fields, state=origin['state'], rail_keys=origin['rail_keys']) if checkpoint is None else \
            self.call('fork', **fields, checkpoint=checkpoint, indices=indices)

    def append(self, row, evidence):
        self.current = self.call('append', token=self.token, revision=row['view']['context']['revision'], row=row, evidence=evidence)

    def checkpoint(self, revision): return self.call('checkpoint', token=self.token, revision=revision)
    def release(self): self.call('release'); self.current = self.token = None


class DecisionEncoder:
    """Existing RemoteEvaluator receives Rust tensors and keeps its GPU queues."""
    def __init__(self, collector, model_id):
        self.collector, self.model_id, self.cached = collector, model_id, None

    def encode(self, view, choices, style=(), intent=(), *, deadline=None, cancelled=None, validated=False):
        if cancelled is not None and cancelled.is_set(): raise EncodingStopped('cancelled')
        if deadline is not None and time.monotonic() >= deadline: raise EncodingStopped('expired')
        c = self.collector
        require(c.current['decision']['view'] == view and c.current['decision']['choices'] == choices, 'Foreign native input')
        context = dict(model_id=self.model_id, reward_id='ordinary-policy-only', profile_id=CONTRACT,
            style=list(style), intent=list(intent), remaining=[0.]*4, max_cost=0., stable_descriptors=None)
        key = (c.token, view['context']['revision'], digest(context))
        if self.cached and self.cached[0] == key: return self.cached[1]
        parts = []; first = None
        for start in range(0, len(choices), 64):
            if deadline is not None and time.monotonic() >= deadline: raise EncodingStopped('expired')
            if cancelled is not None and cancelled.is_set(): raise EncodingStopped('cancelled')
            page = c.call('features', token=c.token, revision=view['context']['revision'], context=context, start=start)
            end = min(start+64, len(choices))
            require(page['choice_ids'] == [x['id'] for x in choices[start:end]] and page['context'] == context
                    and page['candidate_offsets'] == [start, end] and page['total_candidates'] == len(choices), 'Native feature page differs')
            require((page['format'],page['feature_version'],page['schema_sha256'],page['group_layout'],page['entity_layout'])==
                    (FEATURE_FORMAT,VERSION,SCHEMA,GROUP,ENTITY) and page['legal_mask']==[True]*(end-start),
                    'Native feature contract/mask differs')
            if first is None: first = page
            require(all(page[k] == first[k] for k in ('state', 'stable_to_relative', 'descriptors')), 'Native state page changed')
            parts.append(np.asarray(page['actions'], np.float32).reshape(end-start, 327))
        encoded = Encoded(np.asarray(first['state'], np.float32), np.concatenate(parts),
            np.asarray(first['stable_to_relative'], np.int64), tuple(x['id'] for x in choices))
        check_encoded(encoded)
        self.cached = key, encoded
        return encoded
