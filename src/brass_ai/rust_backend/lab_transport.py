"""Private, bounded, deadline-controlled ordinary lane transport."""
from __future__ import annotations
import io
import hashlib
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
MAX_EXPORT_FRAME = 2 * MAX_ARRAY_FRAME + MAX_FRAME


class Pipe:
    def __init__(self, command, *, binary=True, timeout=15):
        from pathlib import Path, PureWindowsPath
        from .lab_paths import runtime_path
        command = [runtime_path(x) if Path(x).is_absolute() or PureWindowsPath(str(x)).is_absolute() else x for x in command]
        self.binary, self.timeout, self.seq, self.closed = binary, timeout, 0, False
        self.lock = threading.RLock()
        self.metrics = Counter()
        self.sample_every = 1
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
        # One bounded allocation, filled directly by the pipe. read()+extend()
        # allocated/copies every partial OS read of a multi-MB array frame.
        out = bytearray(n); view = memoryview(out); offset = 0
        while offset < n:
            count = self.process.stdout.readinto(view[offset:])
            require(count, 'Collector closed output'); offset += count
        self.metrics['receive_buffer_bytes'] += n
        return out

    def _loop(self):
        try:
            while (item := self.jobs.get()) is not None:
                payload, op, sampled = item
                view = memoryview(payload)
                while view:
                    count = self.process.stdin.write(view)
                    require(count, 'Collector closed input'); view = view[count:]
                if self.binary:
                    n = struct.unpack('<I', self._read(4))[0]
                    arrays = bool(n & (1 << 31)); n &= (1 << 31)-1
                    limit=MAX_EXPORT_FRAME if arrays and op=='export_page' else (MAX_ARRAY_FRAME if arrays else MAX_FRAME)
                    require(0 < n <= limit, 'Collector frame limit')
                    result = self._read(n)
                else:
                    arrays = False
                    result = self.process.stdout.readline(MAX_FRAME+1)
                    require(result.endswith(b'\n') and len(result) <= MAX_FRAME, 'Bootstrap frame limit')
                self.metrics['received_bytes'] += len(result)
                self.metrics[op+'_received_bytes'] += len(result)
                began = time.perf_counter() if sampled else None
                if arrays:
                    require(len(result)>=4, 'Truncated array header')
                    h = struct.unpack_from('<I',result)[0]
                    require(0<h<=MAX_FRAME and 4+h<=len(result), 'Invalid array header')
                    reply = json.loads(result[4:4+h])
                    require(reply.get('ok') is True and reply['data']['bytes']==len(result)-4-h, 'Array size differs')
                    reply['data']['_arrays'] = memoryview(result)[4+h:]
                else:
                    reply = json.loads(result, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                if sampled:
                    key='reply_decode_seconds' if self.sample_every==1 else 'reply_decode_sampled_seconds'
                    self.metrics[key] += time.perf_counter()-began
                self.results.put(reply)
        except BaseException as error:
            self.results.put(error)

    def call(self, op, **values):
        with self.lock:
            require(not self.closed, 'Closed collector')
            self.seq += 1
            sampled=self.sample_every==1 or self.seq % self.sample_every==0
            self.metrics['timing_sample_every']=self.sample_every
            self.metrics['timing_eligible_calls']+=1
            self.metrics['timing_sampled_calls']+=int(sampled)
            request = dict(op=op, **values)
            if self.binary: request.update(protocol=CONTRACT, id=self.seq)
            began = time.perf_counter() if sampled else None
            payload = json.dumps(request, separators=(',', ':'), allow_nan=False).encode()
            if sampled:
                key='request_encode_seconds' if self.sample_every==1 else 'request_encode_sampled_seconds'
                self.metrics[key] += time.perf_counter()-began
            self.metrics['sent_bytes'] += len(payload)
            self.metrics[op+'_sent_bytes'] += len(payload)
            self.metrics['calls'] += 1
            self.metrics[op+'_calls'] += 1
            require(len(payload) <= MAX_FRAME, 'Collector request limit')
            try:
                began = time.perf_counter() if sampled else None
                self.jobs.put((struct.pack('<I', len(payload))+payload if self.binary else payload+b'\n', op, sampled), timeout=self.timeout)
                cancel = getattr(self,'cancel_event',None)
                if cancel is None: reply = self.results.get(timeout=self.timeout)
                else:
                    until=time.monotonic()+self.timeout
                    while True:
                        require(not cancel.is_set(), 'Collection canceled during IPC')
                        try: reply=self.results.get(timeout=min(.05,max(0.,until-time.monotonic())));break
                        except queue.Empty:
                            if time.monotonic()>=until:raise
                if sampled:
                    elapsed=time.perf_counter()-began
                    key='round_trip_seconds' if self.sample_every==1 else 'round_trip_sampled_seconds'
                    self.metrics[key] += elapsed
                    self.metrics[op+('_seconds' if self.sample_every==1 else '_sampled_seconds')] += elapsed
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
    def __init__(self, executable, *, batch_forced=True, optimize=True):
        from .lab_perf import mode
        from .lab_delta import Decoder
        selected_mode=mode()
        super().__init__([executable])
        self.sample_every=64 if selected_mode=='production_fast' else 1
        self.delta = Decoder(); self.optimized = False; self.retained_features = {};self.retained_bytes=0
        try:
            self.hello = self.call('hello')
            require(self.hello['contract'] == CONTRACT and self.hello['core']['source_sha256'] == SOURCE
                    and self.hello['search_enabled'] is False
                    and self.hello.get('collection_wire')=='compact-arrays-forced/1'
                    and self.hello.get('training_arrays')=='ordinary-training-arrays/1', 'Collector uses an unqualified shared core/wire')
            if optimize and self.hello.get('performance_wire')=='ordinary-collector-fast/1':
                require(self.call('configure_wire',enabled=True).get('enabled') is True, 'Performance wire unavailable')
                self.optimized = True
        except BaseException:
            self.close(); raise
        self.token = None; self.current = None; self.retired = False
        self.batch_forced = batch_forced
        self.forced = deque(); self.acknowledgements = []

    def call(self, op, **fields):
        if op in ('begin','fork'):
            from .lab_delta import Decoder
            self.delta = Decoder(); self.retained_features.clear()
            self.retained_bytes=0
        result = super().call(op, **fields)
        if self.optimized:result=self.delta.reply(result)
        if hasattr(self,'token') and self.token is not None and isinstance(result,dict):
            frames=[*result.get('forced',()),result.get('current',result)]
            for frame in frames:
                if isinstance(frame,dict) and ('decision' in frame or frame.get('terminal')):
                    require(frame.get('token')==self.token,'Foreign native decision token')
        return result

    def begin(self, job_id, origin, cap, *, checkpoint=None, indices=None):
        self.retired = False
        self.token = uuid.uuid4().hex
        fields = dict(job=job_id, token=self.token, cap=cap)
        self.current = self.call('begin', **fields, state=origin['state'], rail_keys=origin['rail_keys']) if checkpoint is None else \
            self.call('fork', **fields, checkpoint=checkpoint, indices=indices)

    def append(self, row, evidence, *, allow_forced=True):
        require(row['teacher'] is None and
                (row['view'] is self.current['decision']['view'] or row['view']==self.current['decision']['view']) and
                (row['choices'] is self.current['decision']['choices'] or row['choices']==self.current['decision']['choices']),
                'Compact ordinary input differs')
        model = 'ordinary-external' if row['behavior']['kind']=='external' else evidence.get('model_id') or self.context['model_id']
        context = feature_context(model,row['style'],row['intent'])
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
            reply = self.call('append_compact',token=self.token,record=record,forced_limit=16 if self.batch_forced and allow_forced else 0)
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
        if not self.retired:self.call('release')
        self.current = self.token = None; self.retired = False
        self.forced.clear(); self.acknowledgements.clear()
        self.context = None
        self.retained_features.clear()
        self.retained_bytes=0

    def completed_export(self):
        """Finish, drain records/features together and retire in the last reply.

        Independent page cursors preserve the core's original byte/64-row
        bounds. No look-ahead policy context or RNG consumption is involved.
        """
        if not (self.optimized and self.hello.get('export_pages')=='ordinary-export-pages/1'):
            receipt=self.call('finish')
            return receipt,list(self.record_pages(receipt)),list(self.raw_training_features(receipt))
        record_start=feature_start=0;receipt=None;records=[];features=[]
        while True:
            kept=[self.retained_features.get(i) for i in range(feature_start,feature_start+64)]
            page=self.call('export_page',token=self.token,record_start=record_start,feature_start=feature_start,
                           reuse=[v[1] if v else None for v in kept])
            self.metrics['combined_export_pages']+=1
            require(page['wire']=='ordinary-export-pages/1' and page['token']==self.token,
                    'Combined export identity differs')
            if receipt is None:
                receipt=page['receipt']
                require(receipt['contract']==CONTRACT and receipt['token']==self.token and
                        type(receipt['rows']) is int and receipt['rows']>0,'Combined completion differs')
            else:require(page['receipt'] is None,'Duplicate combined completion')
            raw=page['_arrays'];offset=0
            meta=page['record']
            require((meta is None)==(record_start==receipt['rows']),'Combined record coverage differs')
            if meta is not None:
                require(meta['wire']=='ordinary-records-delta/1' and meta['token']==self.token and
                        meta['start']==record_start and meta['rows']==receipt['rows'] and
                        record_start<meta['end']<=min(record_start+64,receipt['rows']) and
                        len(meta['lengths'])==meta['end']-record_start and
                        type(meta['bytes']) is int and 0<meta['bytes']<=min(MAX_ARRAY_FRAME,len(raw)),'Combined record page differs')
                offset=meta['bytes'];records.append((meta,raw[:offset]));record_start=meta['end']
            meta=page['training']
            require((meta is None)==(feature_start==receipt['rows']),'Combined feature coverage differs')
            if meta is not None:
                require(type(meta['bytes']) is int and 0<=meta['bytes']<=MAX_ARRAY_FRAME,'Combined feature section exceeds bound')
                meta=dict(meta,_arrays=raw[offset:])
                features.extend(self._raw_features(meta,receipt,feature_start,kept))
                feature_start=meta['end'];offset+=meta['bytes']
            require(offset==len(raw) and type(page['done']) is bool and
                    page['done']==(record_start==feature_start==receipt['rows']),'Combined export bounds differ')
            if page['done']:
                self.retired=True;break
        return receipt,records,features

    def _raw_features(self,page,receipt,start,reuse):
        require(page['wire'] == ('ordinary-training-reuse/1' if self.optimized else 'ordinary-training-arrays/1') and
                page['token'] == self.token and page['start'] == start and page['rows'] == receipt['rows'] and
                start < page['end'] <= min(start+64,receipt['rows']) and len(page['lengths']) == page['end']-start and
                page['feature_version'] == VERSION and page['schema_sha256'] == SCHEMA,
                'Raw training page binding differs')
        raw=page['_arrays'];offset=0
        require(page['bytes']==len(raw),'Raw training payload size differs')
        if self.optimized:
            require(len(page['sizes'])==len(page['reuse_sha256'])==len(page['lengths']),
                    'Retained feature coverage differs')
        for j,size in enumerate(page['lengths']):
            require(type(size) is int and size>=0,'Invalid feature length')
            if self.optimized and size==0:
                kept=reuse[j]
                require(kept is not None and len(kept[0])==page['sizes'][j] and kept[1]==page['reuse_sha256'][j],
                        'Retained feature binding differs')
                value=kept[0];self.metrics['training_reused_rows']+=1;self.metrics['training_reused_bytes']+=len(value)
            else:
                require(size>0 and offset+size<=len(raw),'Training payload exceeds bounds')
                value=raw[offset:offset+size];offset+=size;self.metrics['training_transferred_bytes']+=size
            yield value
        require(offset==len(raw),'Trailing raw training bytes')

    def record_pages(self, receipt):
        """Raw pages, decoded only by the publisher. Native seals survive drain."""
        start = 0
        while start < receipt['rows']:
            page = self.call('drain_records_compact' if self.optimized else 'drain_records',token=self.token,start=start)
            require(page['wire'] == ('ordinary-records-delta/1' if self.optimized else 'ordinary-records/1') and
                    page['token'] == self.token and page['start'] == start and page['rows'] == receipt['rows'] and
                    start < page['end'] <= min(start+64,receipt['rows']) and
                    len(page['lengths']) == page['end']-start, 'Raw record page binding differs')
            raw = page.pop('_arrays')
            require(page['bytes'] == len(raw), 'Raw record payload size differs')
            yield page,raw
            start = page['end']

    def raw_training_features(self, receipt):
        """Keep inference bytes once; Rust verifies omitted feature hashes."""
        start = 0
        while start < receipt['rows']:
            reuse = [self.retained_features.get(i) for i in range(start,min(start+64,receipt['rows']))]
            page = self.call('drain_training_reuse' if self.optimized else 'drain_training',token=self.token,start=start,
                             **(dict(reuse=[v[1] if v else None for v in reuse]) if self.optimized else {}))
            yield from self._raw_features(page,receipt,start,reuse)
            start = page['end']

    def training_features(self, receipt, rows, *, drain=False):
        require(len(rows)==receipt['rows'], 'Incomplete training records')
        start=0
        while start<len(rows):
            page=self.call('drain_training' if drain else 'training_arrays',token=self.token,start=start)
            require(page['wire']=='ordinary-training-arrays/1' and page['token']==self.token
                    and page['start']==start and page['rows']==len(rows)
                    and start<page['end']<=min(start+64,len(rows))
                    and len(page['lengths'])==page['end']-start
                    and page['feature_version']==VERSION and page['schema_sha256']==SCHEMA,
                    'Training array binding differs')
            raw=page['_arrays'];offset=0
            for index,size in enumerate(page['lengths'],start):
                row=rows[index];count=len(row['choices'])
                require(type(size) is int and size==(3128+count*327)*4+32 and offset+size<=len(raw),
                        'Training array size differs')
                e=decode_arrays(raw[offset:offset+size],row['choices'])
                require(np.array_equal(e.state[-12:],np.asarray(row['style']+row['intent'],dtype=np.float32))
                        and np.array_equal(e.stable_to_relative,[(s-row['seat'])%4 for s in range(4)]),
                        'Training array context/seat differs')
                yield e;offset+=size
            require(offset==len(raw),'Trailing training arrays')
            start=page['end']

    def records(self, receipt, *, drain=False):
        start = 0
        while start < receipt['rows']:
            page = self.call('drain_records' if drain else 'record_arrays',token=self.token,start=start)
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


def decode_arrays(raw,choices):
    count=len(choices)
    require(len(raw)==(3128+count*327)*4+32, 'Native array dimensions differ')
    e=Encoded(np.frombuffer(raw,dtype='<f4',count=3128),
        np.frombuffer(raw,dtype='<f4',count=count*327,offset=3128*4).reshape(count,327),
        np.frombuffer(raw,dtype='<i8',count=4,offset=(3128+count*327)*4),tuple(x['id'] for x in choices))
    for a in (e.state,e.actions,e.stable_to_relative):a.flags.writeable=False
    check_encoded(e)
    return e


class DecisionEncoder:
    """Existing RemoteEvaluator receives Rust tensors and keeps its GPU queues."""
    def __init__(self, collector, model_id):
        self.collector, self.model_id, self.cached = collector, model_id, None

    def encode(self, view, choices, style=(), intent=(), *, deadline=None, cancelled=None, validated=False):
        if cancelled is not None and cancelled.is_set(): raise EncodingStopped('cancelled')
        if deadline is not None and time.monotonic() >= deadline: raise EncodingStopped('expired')
        c = self.collector
        require((c.current['decision']['view'] is view or c.current['decision']['view'] == view) and
                (c.current['decision']['choices'] is choices or c.current['decision']['choices'] == choices), 'Foreign native input')
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
        raw = bytes(raw)
        encoded = decode_arrays(raw,choices)
        if getattr(c,'optimized',False):
            previous=c.retained_features.get(c.current['step'])
            size=c.retained_bytes-(len(previous[0]) if previous else 0)+len(raw)
            require(size<=256*1024**2,'Retained features exceed native lane bound')
            c.retained_bytes=size
            c.retained_features[c.current['step']] = (raw,hashlib.sha256(raw).hexdigest())
        self.cached = key, encoded
        return encoded
