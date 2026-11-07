"""Private, bounded, deadline-controlled ordinary lane transport."""
from __future__ import annotations
import io
import json
import queue
import struct
import subprocess
import threading
import time
import uuid
from collections import Counter, deque

import numpy as np
from brass_ai.learning.contracts import require, digest
from brass_ai.learning.features import Encoded, check_encoded
from brass_ai.learning.encoding_budget import EncodingStopped
from .lab_contract import SOURCE
from brass_ai.native.features import FORMAT as FEATURE_FORMAT,VERSION,SCHEMA,GROUP,ENTITY

CONTRACT = 'brass-lab-collection/1.0.0'
MAX_FRAME = 4 * 1024**2
MAX_ARRAY_FRAME = 16 * 1024**2


class Pipe:
    def __init__(self, command, *, binary=True, timeout=15):
        self.binary, self.timeout, self.seq, self.closed = binary, timeout, 0, False
        self.lock = threading.RLock()
        self.metrics = Counter()
        self.jobs, self.results = queue.Queue(1), queue.Queue(1)
        self.process = subprocess.Popen([str(x) for x in command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        # FileIO.readline on an unbuffered pipe makes one OS read per byte.
        # Oracle decisions are hundreds of KB; concurrent lanes amplify this
        # cost. Buffer only stdout: requests must still be written immediately.
        if not binary:
            self.process.stdout = io.BufferedReader(self.process.stdout, buffer_size=64*1024)
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
                    arrays = bool(n & (1 << 31)); n &= (1 << 31)-1
                    require(0 < n <= (MAX_ARRAY_FRAME if arrays else MAX_FRAME), 'Collector frame limit')
                    result = self._read(n)
                else:
                    arrays = False
                    result = self.process.stdout.readline(MAX_FRAME+1)
                    require(result.endswith(b'\n') and len(result) <= MAX_FRAME, 'Bootstrap frame limit')
                self.metrics['received_bytes'] += len(result)
                began = time.perf_counter()
                if arrays:
                    require(len(result)>=4, 'Truncated array header')
                    h = struct.unpack_from('<I',result)[0]
                    require(0<h<=MAX_FRAME and 4+h<=len(result), 'Invalid array header')
                    reply = json.loads(result[4:4+h])
                    require(reply.get('ok') is True and reply['data']['bytes']==len(result)-4-h, 'Array size differs')
                    reply['data']['_arrays'] = memoryview(result)[4+h:]
                else:
                    reply = json.loads(result, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                self.metrics['reply_decode_seconds'] += time.perf_counter()-began
                self.results.put(reply)
        except BaseException as error:
            self.results.put(error)

    def call(self, op, **values):
        with self.lock:
            require(not self.closed, 'Closed collector')
            self.seq += 1
            request = dict(op=op, **values)
            if self.binary: request.update(protocol=CONTRACT, id=self.seq)
            began = time.perf_counter()
            payload = json.dumps(request, separators=(',', ':'), allow_nan=False).encode()
            self.metrics['request_encode_seconds'] += time.perf_counter()-began
            self.metrics['sent_bytes'] += len(payload)
            self.metrics['calls'] += 1
            self.metrics[op+'_calls'] += 1
            require(len(payload) <= MAX_FRAME, 'Collector request limit')
            try:
                began = time.perf_counter()
                self.jobs.put(struct.pack('<I', len(payload))+payload if self.binary else payload+b'\n', timeout=self.timeout)
                cancel = getattr(self,'cancel_event',None)
                if cancel is None: reply = self.results.get(timeout=self.timeout)
                else:
                    until=time.monotonic()+self.timeout
                    while True:
                        require(not cancel.is_set(), 'Collection canceled during IPC')
                        try: reply=self.results.get(timeout=min(.05,max(0.,until-time.monotonic())));break
                        except queue.Empty:
                            if time.monotonic()>=until:raise
                self.metrics['round_trip_seconds'] += time.perf_counter()-began
                self.metrics[op+'_seconds'] += time.perf_counter()-began
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
    def __init__(self, executable, *, batch_forced=True):
        super().__init__([executable])
        try:
            self.hello = self.call('hello')
            require(self.hello['contract'] == CONTRACT and self.hello['core']['source_sha256'] == SOURCE
                    and self.hello['search_enabled'] is False
                    and self.hello.get('collection_wire')=='compact-arrays-forced/1', 'Collector uses an unqualified shared core/wire')
        except BaseException:
            self.close(); raise
        self.token = None; self.current = None
        self.batch_forced = batch_forced
        self.forced = deque(); self.acknowledgements = []

    def begin(self, job_id, origin, cap, *, checkpoint=None, indices=None):
        self.token = uuid.uuid4().hex
        fields = dict(job=job_id, token=self.token, cap=cap)
        self.current = self.call('begin', **fields, state=origin['state'], rail_keys=origin['rail_keys']) if checkpoint is None else \
            self.call('fork', **fields, checkpoint=checkpoint, indices=indices)

    def append(self, row, evidence):
        require(row['teacher'] is None and row['view']==self.current['decision']['view']
                and row['choices']==self.current['decision']['choices'], 'Compact ordinary input differs')
        context = feature_context(evidence.get('model_id') or self.context['model_id'],row['style'],row['intent'])
        record = dict(revision=row['view']['context']['revision'], action_id=row['choices'][row['chosen']]['id'],
                      context=context, seconds=row['seconds'], behavior=row['behavior'], evidence=evidence)
        if self.forced:
            require(self.current == self.forced[0], 'Forced cursor differs')
            self.acknowledgements.append(record); self.forced.popleft()
            if self.forced: self.current = self.forced[0]
            else:
                self.current = self.call('append_forced',token=self.token,records=self.acknowledgements)
                self.acknowledgements = []
        else:
            reply = self.call('append_compact',token=self.token,record=record,forced_limit=16 if self.batch_forced else 0)
            self.forced.extend(reply['forced'])
            if self.forced:
                require(reply['current'] is None and len(self.forced)<=16 and all(
                    frame['token']==self.token and frame['step']==row['step']+i+1
                    and frame['decision']['view']['context']['revision']==record['revision']+i+1
                    and len(frame['decision']['choices'])==1 for i,frame in enumerate(self.forced)),
                    'Forced preview binding differs')
            self.current = self.forced[0] if self.forced else reply['current']

    def checkpoint(self, revision): return self.call('checkpoint', token=self.token, revision=revision)
    def release(self):
        self.call('release'); self.current = self.token = None
        self.forced.clear(); self.acknowledgements.clear()

    def records(self, receipt):
        start = 0
        while start < receipt['rows']:
            page = self.call('record_arrays',token=self.token,start=start)
            require(page['wire']=='ordinary-records/1' and page['token']==self.token and page['start']==start
                    and page['rows']==receipt['rows'] and start<page['end']<=min(start+64,receipt['rows'])
                    and len(page['lengths'])==page['end']-start, 'Record batch binding differs')
            raw=page['_arrays']; offset=0
            for a,b in page['lengths']:
                require(isinstance(a,int) and isinstance(b,int) and a>0 and b>0 and offset+a+b<=len(raw), 'Record batch lengths differ')
                yield raw[offset:offset+a],raw[offset+a:offset+a+b]
                offset += a+b
            require(offset==len(raw), 'Trailing record batch bytes')
            start=page['end']


def feature_context(model_id,style,intent):
    return dict(model_id=model_id,reward_id='ordinary-policy-only',profile_id=CONTRACT,
        style=list(style),intent=list(intent),remaining=[0.]*4,max_cost=0.,stable_descriptors=None)


class DecisionEncoder:
    """Existing RemoteEvaluator receives Rust tensors and keeps its GPU queues."""
    def __init__(self, collector, model_id):
        self.collector, self.model_id, self.cached = collector, model_id, None

    def encode(self, view, choices, style=(), intent=(), *, deadline=None, cancelled=None, validated=False):
        if cancelled is not None and cancelled.is_set(): raise EncodingStopped('cancelled')
        if deadline is not None and time.monotonic() >= deadline: raise EncodingStopped('expired')
        c = self.collector
        require(c.current['decision']['view'] == view and c.current['decision']['choices'] == choices, 'Foreign native input')
        context = feature_context(self.model_id,style,intent)
        c.context = context
        key = (c.token, view['context']['revision'], digest(context))
        if self.cached and self.cached[0] == key: return self.cached[1]
        page = c.call('feature_arrays', token=c.token,revision=view['context']['revision'],context=context)
        if cancelled is not None and cancelled.is_set(): raise EncodingStopped('cancelled')
        if deadline is not None and time.monotonic()>=deadline: raise EncodingStopped('expired')
        count=len(choices); raw=page['_arrays']
        require(page['wire']=='ordinary-arrays/1' and page['token']==c.token and page['revision']==view['context']['revision']
                and page['choice_ids']==[x['id'] for x in choices] and page['context']==context
                and page['total_candidates']==count and page['state_dim']==3128 and page['action_dim']==327
                and len(raw)==(3128+count*327)*4+32, 'Native feature array binding/shape differs')
        require((page['format'],page['feature_version'],page['schema_sha256'],page['group_layout'],page['entity_layout'])==
                (FEATURE_FORMAT,VERSION,SCHEMA,GROUP,ENTITY), 'Native feature contract differs')
        encoded = Encoded(np.frombuffer(raw,dtype='<f4',count=3128),
            np.frombuffer(raw,dtype='<f4',count=count*327,offset=3128*4).reshape(count,327),
            np.frombuffer(raw,dtype='<i8',count=4,offset=(3128+count*327)*4),tuple(x['id'] for x in choices))
        for value in (encoded.state,encoded.actions,encoded.stable_to_relative): value.flags.writeable=False
        check_encoded(encoded)
        self.cached = key, encoded
        return encoded
