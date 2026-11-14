"""Opt-in B04 native teacher collection; returns no ordinary replay records."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
import copy
import json
from pathlib import Path
import queue
import struct
import subprocess
import threading
import time
import traceback

from brass_ai.learning.contracts import digest, require
from brass_ai.recovery.teacher_phase_config import defaults, SCHEDULE, PROFILE
from brass_ai.recovery.search_value_config import FAST_POST_SECONDS
from .lab_contract import admit_search_request, pin
from .lab_teacher_overlay import CONTRACT, source_fingerprint
from .lab_teacher_inference import Client
from .lab_transport import Pipe

LIMIT = 64*1024**2
COUNT_TABLE_SECONDS = 1800
COUNT_TURN_SECONDS = 120
STREAM_TURNS = 'verified-stream/1'


def validate_job(job):
    return _validate_job(job,source_fingerprint())


def _validate_job(job,producer):
    require(job['kind'] == 'teacher_phase' and job['ordinary_training'] is False,
            'Teacher jobs cannot be ordinary training games')
    expected = defaults(job['search']['reward'])
    require(digest(job['search']) == digest(expected['search']) and job['unsearched'] == expected['unsearched']
            and job['schedule'] == SCHEDULE, 'Unqualified teacher profile')
    admit_search_request(PROFILE,{k:v for k,v in job['search'].items() if k != 'v205_enabled'})
    if job.get('seating') == 'all-seat-search':
        from brass_ai.recovery.search_value_config import fast_profile
        from brass_ai.learning.search_value import validate_binding
        require(job['fast_search'] == fast_profile(job['search']), 'Fast teacher profile differs')
        validate_binding(job['search_value'])
        require(not job.get('shadow_probe') or job['split']=='heldout', 'From-zero probes must stay held out')
        if job.get('h2h'):
            from brass_ai.learning.search_value import binding
            require(len(job['seat_values']) == 4 and job['seat_values'].count(binding(0)) == 2
                and job['seat_values'].count(binding(1)) == 2, 'Head-to-head needs two main and two corrected seats')
    e = job['expected']
    require(job.get('readiness',{}).get('ready') is True and job['readiness']['reward_id'] == e['reward_id'],
            'Teacher readiness deferred; no local deployment exception')
    require(e['source_fingerprint'] == producer and e['reward_id'] == digest(job['search']['reward'])
            and job['seats'] == [e['model_id']]*4, 'Teacher model/source/reward differs')
    require(job['split'] in ('train','heldout') and type(e['generation']) is int and e['generation'] >= 0
            and type(e['epoch']) is int and e['epoch'] >= 0, 'Invalid teacher attribution')
    from brass_ai.recovery.teacher_phase import reserved_rounds
    require(job['reservations'] == {str(s):reserved_rounds(job['seed'],job['sampling_seed'],s) for s in range(4)},
            'Teacher reservation stream differs')
    from brass_ai.recovery.training_revision_g1491 import validate_job as validate_revision
    validate_revision(job)
    return e


class Worker:
    """Length-bounded stdout pump permits cancellation even in native traversal."""
    def __init__(self, executable, request, client, cancel, cpu=None):
        from .lab_paths import runtime_path
        self.client, self.cancel = client, cancel
        self.job = request.get('job', {})
        # A fixed-count table runs to its quota; only a stall should stop it.
        self.counted = request.get('inference', {}).get('budget', {}).get('mode') == 'count'
        self.stream_turns = request.get('turn_handoff') == STREAM_TURNS
        self.process = subprocess.Popen([str(runtime_path(executable))], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if cpu is not None:
            from .lab_cpu_runtime import pin_child
            pin_child(self.process.pid,cpu)
        self.events = queue.Queue(2); self.stop = threading.Event(); self.stderr = bytearray()
        def read(n):
            data = bytearray(n)
            remaining = memoryview(data)
            while remaining:
                got = self.process.stdout.readinto(remaining)
                require(got, 'Native teacher closed output')
                remaining = remaining[got:]
            return data
        def pump():
            try:
                while not self.stop.is_set():
                    n = struct.unpack('<I',read(4))[0]; require(4 < n <= LIMIT,'Teacher frame bound')
                    data = read(n); size = struct.unpack('<I',data[:4])[0]
                    require(0 < size <= n-4,'Teacher metadata bound')
                    received = time.monotonic()
                    event = (json.loads(data[4:4+size]),memoryview(data)[4+size:].toreadonly(),received)
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

    def collect(self, expected, core_source, *, calibration=False, verifier=None, turn_probe=False, defer_logits=False):
        rows = []; values = []; hello = False; started = last = time.monotonic()
        meta = None; self.failure = None; verified_turns = 0; stream_turns = False
        try:
            while True:
                require(not self.cancel.is_set(),'Teacher table cancelled; discard unpublished labels')
                counted = getattr(self, 'counted', False)
                require(time.monotonic()-started < (COUNT_TABLE_SECONDS if counted else 420 if getattr(self, 'job', {}).get('h2h') else 180),'Teacher table watchdog; incomplete table discarded')
                require(time.monotonic()-last < (COUNT_TURN_SECONDS if counted else 30),'Stuck native search watchdog; no labels admitted')
                try:event = self.events.get(timeout=.05)
                except queue.Empty:continue
                if isinstance(event,BaseException):
                    self.err_reader.join(timeout=.2)
                    raise ValueError(str(event)+': '+self.stderr.decode(errors='replace'))
                meta,payload,*stamp = event; last = time.monotonic()
                received = stamp[0] if stamp else last
                kind = meta['event']
                if kind == 'hello':
                    require(not hello and not payload and meta['contract'] == CONTRACT and meta['job'] == expected
                        and meta['core']['source_sha256'] == core_source and meta['method'] == 'search_v1'
                        and meta['v205_enabled'] is False,'Native teacher handshake differs')
                    stream_turns = meta.get('turn_handoff') == STREAM_TURNS
                    require(not stream_turns or getattr(self, 'stream_turns', False), 'Unexpected streamed teacher turns')
                    hello = True;continue
                require(hello,'Missing teacher handshake')
                if kind == 'evaluate':
                    require(not stream_turns, 'Streamed CPU teacher requested remote inference')
                    self.send(self.client.evaluate(meta,payload));continue
                require(not payload,'Unexpected teacher binary payload')
                if kind == 'turn':
                    # Long labels: 16 reserved turns, two actions each. A g1491 job
                    # also labels its short searches: all 64 turns.
                    revision = getattr(self, 'job', {}).get('teacher_revision')
                    require(not calibration and verifier is not None and
                            len(rows)+len(meta['rows']) <= (128 if revision else 32),
                            'Unexpected/unverified/excess teacher turn')
                    if stream_turns:
                        verifier.verify(meta,received)
                    elif counted:
                        # Older binaries require an ACK before the next search.
                        # Verify in order before any table can be admitted.
                        self.send(dict(job=meta['job'],index=meta['index'],status='ok'))
                        verifier.verify(meta,received)
                    else:self.send(verifier.verify(meta,received))
                    # Verification is outside the active search/stuck-worker
                    # watchdog. Keep generic search/IPC watchdogs and cancel.
                    ready = time.monotonic()
                    started += ready-last; last = ready
                    verified_turns += 1
                    rows.extend(meta['rows'])
                    if meta.get('value_row') is not None: values.append(meta['value_row'])
                elif kind == 'calibration':
                    require(calibration and meta['completed'] == 8 and meta['elapsed_ns'] > 0,'Incomplete calibration')
                    require(not self.cancel.is_set(),'Teacher calibration cancelled before publication')
                    return meta
                elif kind == 'turn_probe':
                    require(turn_probe and meta['job'] == expected and meta.get('diagnostic') is True
                            and len(meta['metrics']['turns']) == 1 and meta['metrics']['labels'] == len(rows)
                            and not self.cancel.is_set(), 'Incomplete/late/misrouted turn probe')
                    return dict(meta,rows=rows,value_rows=values)
                elif kind == 'finished':
                    require(not calibration and not turn_probe and meta['job'] == expected and meta['ordinary_training'] is False
                            and meta['metrics']['labels'] == len(rows) and meta['metrics']['reserved_turns'] == (64 if getattr(self,'job',{}).get('h2h') else 16),
                            'Incomplete/misrouted teacher table')
                    require(not self.cancel.is_set(),'Teacher table cancelled before publication')
                    if stream_turns:
                        require(meta['value_rows']==[] and type(meta['value_count']) is int and meta['value_count']==len(values),
                                'Incomplete streamed root values')
                        expected_turns = 64 if self.job.get('seating') == 'all-seat-search' else 16
                        require(verified_turns==meta['metrics']['searched_turns']==expected_turns and
                                (len(values)==64 if expected_turns==64 else not values),
                                'Incomplete streamed teacher table')
                    else:
                        require(meta.get('value_rows', []) == values, 'Changed root value stream')
                    revision = getattr(self, 'job', {}).get('teacher_revision')
                    # The worker echoes the revision it ran; a stale binary would label nothing short.
                    require(revision is None or
                            (meta['metrics'].get('cpu_inference') or {}).get('revision', {}).get('job') == revision,
                            'CPU teacher did not run the g1491 revision; rerun scripts/prepare-rust-cpu-inference.py')
                    from brass_ai.recovery.teacher_cotton import validate_receipt
                    validate_receipt((meta['metrics'].get('cpu_inference') or {}).get('revision',{}).get('cotton'),
                                     getattr(self,'job',{}), rows)
                    # Batched model-logit parity of every labelled row, once, as the table ends.
                    if not defer_logits and revision is not None and getattr(verifier,'settle',None) is not None:
                        verifier.settle(expected['job_id'])
                    return dict(job_id=expected['job_id'],rows=rows,metrics=meta['metrics'],result=meta['result'],
                                value_rows=values, final_spent=meta.get('final_spent'))
                else:raise ValueError('Unknown teacher event')
        except BaseException as error:
            self.failure = dict(error=str(error),type=type(error).__name__,last_frame=meta,
                                verified_turns=verified_turns,elapsed_seconds=time.monotonic()-started)
            raise
        finally:
            try:self.close()
            except BaseException as close_error:
                if self.failure is None:raise
                self.failure['close_error']=str(close_error)
            if self.failure is not None:
                self.failure.update(stderr=getattr(self,'stderr',b'').decode(errors='replace'),
                                    exit_code=getattr(getattr(self,'process',None),'returncode',None))

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
    def __init__(self, executable, core_source, bootstrap, service, calibration, *, lanes=2, cancel=None,
                 model=None, history=(), inference=None, cores=(), core_plan=None):
        self.executable, self.core_source, self.bootstrap = executable,core_source,bootstrap
        self.service, self.calibration = service, calibration
        require(type(lanes) is int and 1 <= lanes <= 256,'Invalid teacher lanes')
        self.lanes = lanes;self.cancel = cancel or threading.Event()
        self.model,self.history = model,history
        self.inference,self.cores = inference,tuple(cores)
        self.core_plan = core_plan
        self.verifier = None
        self.failures = []
        self.finalization = {}

    def collect(self, jobs, *, on_result=None):
        if not jobs:return []
        jobs = copy.deepcopy(jobs)
        require(len({j['expected']['job_id'] for j in jobs}) == len(jobs),'Duplicate teacher job')
        producer=source_fingerprint()
        for job in jobs:_validate_job(job,producer)
        require(self.model is not None, 'Frozen independent model required for bounded turn verification')
        from .lab_teacher_finalization import TableLogitBatcher, TurnVerifier
        from .lab_cpu_runtime import count_search_limits
        from brass_ai.recovery.collector_runtime import teacher_job_budget
        self.verifier = TurnVerifier(self.model,self.history,jobs,self.cancel,search_limits=count_search_limits(self.inference))
        c = self.calibration
        require(c['contract'] == CONTRACT and c['source'] == producer and c['core_source'] == self.core_source
                and c['model_id'] == jobs[0]['expected']['model_id'] and c['export_hash'] == jobs[0]['expected']['export_hash']
                and c['reward_id'] == jobs[0]['expected']['reward_id']
                and c['device'] == ('cpu-ort' if self.inference is not None else str(self.service.device))
                and c['profile'] == digest(jobs[0]['search']) and c['samples'] and c['estimate_ns'] > 0,
                'GPU trajectory calibration identity differs')
        require(all(j['expected']['model_id'] == c['model_id'] and j['expected']['export_hash'] == c['export_hash']
                    and digest(j['search']) == c['profile'] for j in jobs),'Teacher round is not frozen')
        count = min(self.lanes,len(jobs))
        lease = None if self.inference is not None else self.service.channels(count)
        owned = [_CpuLane(self.cores[i] if i < len(self.cores) else None) for i in range(count)] if lease is None else [lease.acquire() for _ in range(count)]
        failure_lock = threading.Lock(); first_failure = []; self.failures = []
        work = iter(enumerate(jobs))
        results = [None] * len(jobs)
        defer_logits = (self.inference is not None and
                        self.inference['budget']['mode'] == 'count' and all(j.get('teacher_revision') for j in jobs))
        batcher = TableLogitBatcher(self.verifier, len(jobs), pending=2*count) if defer_logits else None
        began = time.monotonic()
        search_ended = 0.
        self.finalization = {}
        calibration_id = digest(c)

        def release_core(endpoint):
            core = getattr(endpoint, 'cpu', None)
            if self.core_plan is not None and core is not None:
                self.core_plan.release(core)

        def failed(job, worker, error):
            # Record the initiating exception before waking sibling work.
            with failure_lock:
                primary = not first_failure
                self.failures.append(dict(job_id=job['expected']['job_id'],split=job['split'],seed=job['seed'],
                    error=str(error),type=type(error).__name__,traceback=traceback.format_exc(),
                    worker=getattr(worker,'failure',None),primary=primary))
                if primary:first_failure.append((error,error.__traceback__))
                self.cancel.set()
            if primary and self.core_plan is not None:self.core_plan.fail(error)

        def complete(job, result):
            require(not self.cancel.is_set(), 'Teacher collection cancelled before publication')
            if on_result is not None:
                result['metrics']['shadow_cache_seconds'] = on_result(result,job)
            return result

        def finalize(job, worker, result, checked):
            try:
                checked.result()
                return complete(job, result)
            except BaseException as error:
                failed(job, worker, error)
                raise
            finally:
                batcher.release()

        def claim():
            with failure_lock:
                return next(work, None)

        def one(job, endpoint, reserved):
            nonlocal search_ended
            worker = None
            try:
                from brass_ai.recovery.telemetry import Timings
                timings=Timings()
                with self.service.cpu.acquire(1,timings):
                    require(not self.cancel.is_set(),'Teacher collection cancelled')
                    mono_began = time.monotonic()
                    with Pipe(self.bootstrap,binary=False) as bootstrap:origin = bootstrap.call('init',seed=job['seed'])
                    mono_seconds = time.monotonic()-mono_began
                    request = dict(contract=CONTRACT,op='run',job=job,origin=origin,core_source=self.core_source,
                                   estimate_ns=c['estimate_ns'],calibration_id=calibration_id)
                    if self.inference is not None:
                        budget = self.inference['budget']
                        table_budget = teacher_job_budget(budget, job)
                        request['inference'] = (self.inference if table_budget is budget else
                                                dict(self.inference, budget=table_budget))
                        if table_budget['mode'] == 'count':
                            request['turn_handoff'] = STREAM_TURNS
                    client = Client(endpoint,job['expected'],self.cancel,job.get('search_value'),job.get('seat_values'))
                    worker = Worker(self.executable,request,client,self.cancel,cpu=getattr(endpoint,'cpu',None))
                    result = worker.collect(job['expected'],self.core_source,verifier=self.verifier,
                                            **({'defer_logits': True} if defer_logits else {}))
                with failure_lock:
                    search_ended = max(search_ended, time.monotonic())
                following = claim()
                if following is None:
                    release_core(endpoint)
                result['metrics']['mono_bootstrap_seconds'] = mono_seconds
                result['metrics']['python_gpu_wait_seconds'] = client.wait_seconds
                result['metrics']['cpu_admission'] = timings.report()
                if batcher is None:
                    return complete(job, result), following
                checked = batcher.submit(job['expected']['job_id'])
                pending = finalizers.submit(finalize, job, worker, result, checked)
                reserved = False
                return pending, following
            except BaseException as error:
                failed(job, worker, error)
                raise
            finally:
                if reserved:
                    batcher.release()

        def lane(endpoint):
            item = claim()
            try:
                while item is not None and not self.cancel.is_set():
                    if batcher is not None:
                        batcher.reserve()
                    index, job = item
                    results[index], item = one(job, endpoint, batcher is not None)
            finally:
                release_core(endpoint)

        try:
            try:
                from .lab_families import _failure_scope
                finalization_pool = (ThreadPoolExecutor(max_workers=min(4,count),thread_name_prefix='teacher-publish')
                                     if batcher is not None else nullcontext())
                with batcher or nullcontext(), finalization_pool as finalizers, \
                        ThreadPoolExecutor(max_workers=len(owned),thread_name_prefix='rust-teacher') as pool, \
                        _failure_scope(self.cancel,None):
                    list(pool.map(lane,owned))
                    require(not self.cancel.is_set() and all(r is not None for r in results),
                            'Teacher collection cancelled before publication')
                    if batcher is not None:
                        results = [future.result() for future in results]
            except BaseException as interruption:
                if isinstance(interruption,Exception) and first_failure:
                    error,tb=first_failure[0]
                    raise error.with_traceback(tb)
                raise
            require(not self.cancel.is_set(),'Teacher collection cancelled before publication')
            if batcher is not None:
                self.finalization = dict(batcher.metrics, last_search_seconds=search_ended-began,
                                         tables_published_seconds=time.monotonic()-began)
            return results
        finally:
            close_errors = []
            for e in owned:
                try:e.close()
                except BaseException as error:close_errors.append(error)
            if lease is not None:
                try:self.service.release_channels(lease)
                except BaseException as error:close_errors.append(error)
            if close_errors:
                if first_failure:
                    self.failures[0]['endpoint_cleanup_errors'] = [str(e) for e in close_errors]
                else:raise close_errors[0]


class _CpuLane:
    """A search lane with no GPU reply channel; workers evaluate in process."""
    def __init__(self, cpu):self.cpu = cpu
    def close(self):pass


def count_calibration(core_source, job):
    """Fixed-count searches freeze N from their budget; no time estimate exists."""
    return dict(contract=CONTRACT,source=source_fingerprint(),core_source=core_source,
        model_id=job['expected']['model_id'],export_hash=job['expected']['export_hash'],reward_id=job['expected']['reward_id'],
        device='cpu-ort',profile=digest(job['search']),estimate_ns=1000,samples=['fixed-count'],root_medians_ns=[],
        rule='not-applicable-fixed-count')


def calibrate_cpu(executable, core_source, origins, job, inference, cores):
    """Time trajectories on every lane at once, matching collection load."""
    validate_job(job);cancel = threading.Event();samples = []
    def one(index):
        lane = []
        for root_index, origin in enumerate(origins):
            for repetition in range(4):
                request = dict(contract=CONTRACT,op='calibrate',job=job,origin=origin,core_source=core_source,inference=inference)
                worker = Worker(executable,request,None,cancel,cpu=cores[index] if index < len(cores) else None)
                sample = worker.collect(job['expected'],core_source,calibration=True)
                lane.append(dict(sample,root_index=root_index,warmup=repetition == 0,lane=index))
        return lane
    with ThreadPoolExecutor(max_workers=len(cores) or 1,thread_name_prefix='rust-teacher-calibrate') as pool:
        for lane in pool.map(one,range(len(cores) or 1)):samples.extend(lane)
    import statistics
    medians = [statistics.median((s['elapsed_ns']+7)//8 for s in samples
               if s['root_index'] == i and not s['warmup']) for i in range(len(origins))]
    return dict(contract=CONTRACT,source=source_fingerprint(),core_source=core_source,
        model_id=job['expected']['model_id'],export_hash=job['expected']['export_hash'],reward_id=job['expected']['reward_id'],
        device='cpu-ort',profile=digest(job['search']),estimate_ns=int(max(medians)),samples=samples,root_medians_ns=medians,
        rule='concurrent-all-lanes; exclude-one-warmup-per-root; median; max-root-median')


def calibrate(executable, core_source, origins, job, service):
    validate_job(job);lease = service.channels(1);e = lease.acquire();cancel = threading.Event();samples = []
    try:
        for root_index, origin in enumerate(origins):
            for repetition in range(4):
                request = dict(contract=CONTRACT,op='calibrate',job=job,origin=origin,core_source=core_source)
                client = Client(e,job['expected'],cancel,job.get('search_value'),job.get('seat_values'))
                sample = Worker(executable,request,client,cancel).collect(job['expected'],core_source,calibration=True)
                samples.append(dict(sample, root_index=root_index, warmup=repetition == 0))
    finally:
        e.close();service.release_channels(lease)
    require(samples,'No GPU calibration samples')
    # Per-root warmup is measured but excluded. Freeze the slower root median;
    # a single cold CUDA allocation can no longer halve K for an entire run.
    import statistics
    medians = [statistics.median((s['elapsed_ns']+7)//8 for s in samples
               if s['root_index'] == i and not s['warmup']) for i in range(len(origins))]
    estimate = int(max(medians))
    return dict(contract=CONTRACT,source=source_fingerprint(),core_source=core_source,
        model_id=job['expected']['model_id'],export_hash=job['expected']['export_hash'],reward_id=job['expected']['reward_id'],
        device=str(service.device),profile=digest(job['search']),estimate_ns=estimate,samples=samples,root_medians_ns=medians,
        rule='exclude-one-warmup-per-root; median-of-three-warm; max-root-median')


def collect_barrier(phase, state, model, *, export_hash, epoch, backend, ordinary_jobs=(), ordinary_collect=None):
    """Teacher jobs precede ordinary jobs; labels never reach ordinary callbacks."""
    jobs = phase.begin(state,model,export_hash=export_hash,epoch=epoch,native_source=source_fingerprint())
    results = backend.collect(jobs) if jobs else []
    phase.admit_collection(results,model,verifier=backend.verifier if jobs else None)
    ordinary = ordinary_collect(ordinary_jobs) if ordinary_jobs and ordinary_collect is not None else []
    return ordinary, results


def turn_probes(executable, core_source, origins, job, service, calibration, model, history, *, profile="full"):
    """Two bounded typical/wide 4s/K16 turns; diagnostic rows cannot train."""
    from .lab_teacher_finalization import TurnVerifier
    validate_job(job)
    lease=service.channels(1);endpoint=lease.acquire();cancel=threading.Event();results=[]
    try:
        require(profile in ('full','fast'), 'Unknown probe profile')
        for origin in origins:
            verifier=TurnVerifier(model,history,[job],cancel,diagnostic=True)
            request=dict(contract=CONTRACT,op='turn_probe',job=job,origin=origin,core_source=core_source,probe_profile=profile,
                estimate_ns=calibration['estimate_ns'],calibration_id=digest(calibration))
            client=Client(endpoint,job['expected'],cancel,job.get('search_value'),job.get('seat_values'))
            result=Worker(executable,request,client,cancel).collect(job['expected'],core_source,
                verifier=verifier,turn_probe=True)
            result['verification']=verifier.cache[job['expected']['job_id']]['timings']
            results.append(result)
    finally:
        endpoint.close();service.release_channels(lease)
    return dict(passed=True,diagnostic=True,turns=results,search_seconds=.8 if profile=='fast' else 4.,post_search_seconds=FAST_POST_SECONDS if profile=='fast' else 1.,k=4 if profile=='fast' else 16,
                scope='Fixed-root timing and verification probes; no full games or training labels')
