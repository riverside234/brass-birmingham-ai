"""Opt-in B04 native teacher collection; returns no ordinary replay records."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import queue
import struct
import subprocess
import threading
import time

from brass_ai.learning.contracts import digest, require
from brass_ai.recovery.teacher_phase_config import defaults, SCHEDULE, PROFILE
from .lab_contract import admit_search_request, pin
from .lab_teacher_overlay import CONTRACT, source_fingerprint
from .lab_teacher_inference import Client
from .lab_transport import Pipe

LIMIT = 64*1024**2


def validate_job(job):
    require(job['kind'] == 'teacher_phase' and job['ordinary_training'] is False,
            'Teacher jobs cannot be ordinary training games')
    expected = defaults(job['search']['reward'])
    require(digest(job['search']) == digest(expected['search']) and job['unsearched'] == expected['unsearched']
            and job['schedule'] == SCHEDULE, 'Unqualified teacher profile')
    admit_search_request(PROFILE,{k:v for k,v in job['search'].items() if k != 'v205_enabled'})
    e = job['expected']
    require(job.get('readiness',{}).get('ready') is True and job['readiness']['reward_id'] == e['reward_id'],
            'Teacher readiness deferred; no local deployment exception')
    require(e['source_fingerprint'] == source_fingerprint() and e['reward_id'] == digest(job['search']['reward'])
            and job['seats'] == [e['model_id']]*4, 'Teacher model/source/reward differs')
    require(job['split'] in ('train','heldout') and type(e['generation']) is int and e['generation'] >= 0
            and type(e['epoch']) is int and e['epoch'] >= 0, 'Invalid teacher attribution')
    from brass_ai.recovery.teacher_phase import reserved_rounds
    require(job['reservations'] == {str(s):reserved_rounds(job['seed'],job['sampling_seed'],s) for s in range(4)},
            'Teacher reservation stream differs')
    return e


class Worker:
    """Length-bounded stdout pump permits cancellation even in native traversal."""
    def __init__(self, executable, request, client, cancel):
        self.client, self.cancel = client, cancel
        self.process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.events = queue.Queue(2); self.stop = threading.Event(); self.stderr = bytearray()
        def read(n):
            data = bytearray()
            while len(data) < n:
                part = self.process.stdout.read(n-len(data)); require(part,'Native teacher closed output')
                data.extend(part)
            return bytes(data)
        def pump():
            try:
                while not self.stop.is_set():
                    n = struct.unpack('<I',read(4))[0]; require(4 < n <= LIMIT,'Teacher frame bound')
                    data = read(n); size = struct.unpack('<I',data[:4])[0]
                    require(0 < size <= n-4,'Teacher metadata bound')
                    event = (json.loads(data[4:4+size]),data[4+size:])
                    while not self.stop.is_set():
                        try:self.events.put(event,timeout=.1);break
                        except queue.Full:pass
            except BaseException as e:
                while not self.stop.is_set():
                    try:self.events.put(e,timeout=.1);break
                    except queue.Full:pass
        def errors():
            while part := self.process.stderr.read(4096):
                if len(self.stderr) < 32768:self.stderr.extend(part[:32768-len(self.stderr)])
        self.reader = threading.Thread(target=pump,daemon=True); self.reader.start()
        self.err_reader = threading.Thread(target=errors,daemon=True); self.err_reader.start()
        try:
            self.send(request)
        except BaseException:
            self.close()
            raise

    def send(self, value):
        data = json.dumps(value,separators=(',',':'),allow_nan=False).encode()
        require(len(data) <= LIMIT,'Teacher input bound')
        view = memoryview(struct.pack('<I',len(data))+data)
        while view:
            n = self.process.stdin.write(view);require(n,'Teacher input closed');view = view[n:]

    def collect(self, expected, core_source, *, calibration=False):
        rows = []; hello = False; started = last = time.monotonic()
        try:
            while True:
                require(not self.cancel.is_set(),'Teacher table cancelled; discard unpublished labels')
                require(time.monotonic()-started < 180,'Teacher table watchdog; incomplete table discarded')
                require(time.monotonic()-last < 30,'Stuck native search watchdog; no labels admitted')
                try:event = self.events.get(timeout=.05)
                except queue.Empty:continue
                if isinstance(event,BaseException):
                    self.err_reader.join(timeout=.2)
                    raise ValueError(str(event)+': '+self.stderr.decode(errors='replace'))
                meta,payload = event; last = time.monotonic()
                kind = meta['event']
                if kind == 'hello':
                    require(not hello and not payload and meta['contract'] == CONTRACT and meta['job'] == expected
                        and meta['core']['source_sha256'] == core_source and meta['method'] == 'search_v1'
                        and meta['v205_enabled'] is False,'Native teacher handshake differs')
                    hello = True;continue
                require(hello,'Missing teacher handshake')
                if kind == 'evaluate':self.send(self.client.evaluate(meta,payload));continue
                require(not payload,'Unexpected teacher binary payload')
                if kind == 'label':
                    require(not calibration and len(rows) < 32,'Unexpected/excess teacher label')
                    rows.append(meta['row'])
                elif kind == 'calibration':
                    require(calibration and meta['completed'] == 8 and meta['elapsed_ns'] > 0,'Incomplete calibration')
                    require(not self.cancel.is_set(),'Teacher calibration cancelled before publication')
                    return meta
                elif kind == 'finished':
                    require(not calibration and meta['job'] == expected and meta['ordinary_training'] is False
                            and meta['metrics']['labels'] == len(rows) and meta['metrics']['reserved_turns'] == 16,
                            'Incomplete/misrouted teacher table')
                    require(not self.cancel.is_set(),'Teacher table cancelled before publication')
                    return dict(job_id=expected['job_id'],rows=rows,metrics=meta['metrics'],result=meta['result'])
                else:raise ValueError('Unknown teacher event')
        finally:self.close()

    def close(self):
        self.stop.set()
        if self.process.poll() is None:self.process.kill()
        self.process.wait(timeout=5)
        self.reader.join(timeout=2);self.err_reader.join(timeout=2)
        self.process.stdin.close();self.process.stdout.close();self.process.stderr.close()


class TeacherCollector:
    """Explicit frozen-model collector used by the phase collection barrier.

    No legacy allowance/work budget is consulted. Production adoption remains
    gated by B05/B07; this class is also the exact two-table qualification path.
    """
    def __init__(self, executable, core_source, bootstrap, service, calibration, *, lanes=2, cancel=None):
        self.executable, self.core_source, self.bootstrap = executable,core_source,bootstrap
        self.service, self.calibration = service, calibration
        require(type(lanes) is int and 1 <= lanes <= 64,'Invalid teacher lanes')
        self.lanes = lanes;self.cancel = cancel or threading.Event()

    def collect(self, jobs):
        if not jobs:return []
        jobs = copy.deepcopy(jobs)
        require(len({j['expected']['job_id'] for j in jobs}) == len(jobs),'Duplicate teacher job')
        for job in jobs:validate_job(job)
        c = self.calibration
        require(c['contract'] == CONTRACT and c['source'] == source_fingerprint() and c['core_source'] == self.core_source
                and c['model_id'] == jobs[0]['expected']['model_id'] and c['export_hash'] == jobs[0]['expected']['export_hash']
                and c['reward_id'] == jobs[0]['expected']['reward_id'] and c['device'] == str(self.service.device)
                and c['profile'] == digest(jobs[0]['search']) and c['samples'] and c['estimate_ns'] > 0,
                'GPU trajectory calibration identity differs')
        require(all(j['expected']['model_id'] == c['model_id'] and j['expected']['export_hash'] == c['export_hash']
                    and digest(j['search']) == c['profile'] for j in jobs),'Teacher round is not frozen')
        lease = self.service.channels(min(self.lanes,len(jobs)))
        endpoints = queue.Queue();owned = [lease.acquire() for _ in range(min(self.lanes,len(jobs)))]
        for e in owned:endpoints.put(e)
        def one(job):
            endpoint = endpoints.get(timeout=30)
            try:
                from brass_ai.recovery.telemetry import Timings
                timings=Timings()
                with self.service.cpu.acquire(1,timings):
                    require(not self.cancel.is_set(),'Teacher collection cancelled')
                    with Pipe(self.bootstrap,binary=False) as bootstrap:origin = bootstrap.call('init',seed=job['seed'])
                    request = dict(contract=CONTRACT,op='run',job=job,origin=origin,core_source=self.core_source,
                                   estimate_ns=c['estimate_ns'],calibration_id=digest(c))
                    client = Client(endpoint,job['expected'],self.cancel)
                    result = Worker(self.executable,request,client,self.cancel).collect(job['expected'],self.core_source)
                result['metrics']['python_gpu_wait_seconds'] = client.wait_seconds
                result['metrics']['cpu_admission'] = timings.report()
                return result
            except BaseException:self.cancel.set();raise
            finally:endpoints.put(endpoint)
        try:
            with ThreadPoolExecutor(max_workers=len(owned),thread_name_prefix='rust-teacher') as pool:
                results = list(pool.map(one,jobs))
            require(not self.cancel.is_set(),'Teacher collection cancelled before publication')
            return results
        finally:
            for e in owned:e.close()


def calibrate(executable, core_source, origins, job, service):
    validate_job(job);lease = service.channels(1);e = lease.acquire();cancel = threading.Event();samples = []
    try:
        for origin in origins:
            request = dict(contract=CONTRACT,op='calibrate',job=job,origin=origin,core_source=core_source)
            client = Client(e,job['expected'],cancel)
            samples.append(Worker(executable,request,client,cancel).collect(job['expected'],core_source,calibration=True))
    finally:e.close()
    require(samples,'No GPU calibration samples')
    # Includes root setup/encoding, GPU queue and finalisation: conservative and
    # hardware-measured, never the local CPU estimate. Frozen before tables run.
    estimate = max((s['elapsed_ns']+7)//8 for s in samples)
    return dict(contract=CONTRACT,source=source_fingerprint(),core_source=core_source,
        model_id=job['expected']['model_id'],export_hash=job['expected']['export_hash'],reward_id=job['expected']['reward_id'],
        device=str(service.device),profile=digest(job['search']),estimate_ns=estimate,samples=samples)


def collect_barrier(phase, state, model, *, export_hash, epoch, backend, ordinary_jobs=(), ordinary_collect=None):
    """Teacher jobs precede ordinary jobs; labels never reach ordinary callbacks."""
    jobs = phase.begin(state,model,export_hash=export_hash,epoch=epoch,native_source=source_fingerprint())
    results = backend.collect(jobs) if jobs else []
    phase.admit_collection(results,model)
    ordinary = ordinary_collect(ordinary_jobs) if ordinary_jobs and ordinary_collect is not None else []
    return ordinary, results
