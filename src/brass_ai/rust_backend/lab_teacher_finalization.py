"""Once-only verification of a completed native teacher turn, sparsely timed.

The cache is private to one frozen collection. It cannot authorize persisted
rows or a changed model/job. The ordinary shard boundary still seals bytes.
"""
from collections import Counter, deque
from concurrent.futures import Future
from contextlib import contextmanager
import copy
import time
import threading

from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import digest, require
from brass_ai.recovery.admitted_inputs import _freeze
from brass_ai.recovery.teacher_phase_config import TIMING
from brass_ai.recovery.search_value_config import FAST_POST_SECONDS
from .lab_admission import admit_teacher_rows
from .lab_admission import teacher_sources
from brass_ai.recovery.teacher_labels import prepare_rows, check_logits, check_turn


class TableLogitBatcher:
    """One verifier for immutable, completed tables of a frozen collection."""
    MAX_ROWS = 64
    MAX_CELLS = 16_384
    FLUSH_SECONDS = .002

    def __init__(self, verifier, count, *, pending):
        self.verifier, self.count = verifier, count
        self.pending = threading.BoundedSemaphore(pending)
        self.condition = threading.Condition()
        self.queue = deque()
        self.keys = set()
        self.closed = False
        self.error = None
        self.thread = threading.Thread(target=self._run, name='teacher-logits')
        self.metrics = Counter()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join()

    def reserve(self):
        # Search workers wait before taking a new table, outside CPU admission.
        while not self.pending.acquire(timeout=.05):
            require(not self.verifier.cancel.is_set(), 'Teacher logit verification cancelled')

    def release(self):
        self.pending.release()

    def submit(self, key):
        record = self.verifier.cache[key]
        future = Future()
        with self.condition:
            if self.error is not None:
                raise self.error
            require(not self.closed and not self.verifier.cancel.is_set(), 'Teacher logit verification cancelled')
            require(key not in self.keys and 'settled' not in record, 'Repeated teacher logit verification')
            self.keys.add(key)
            self.queue.append((record, future, time.monotonic()))
            self.condition.notify()
        return future

    def _run(self):
        group = []
        try:
            while True:
                with self.condition:
                    while not self.queue and not self.closed:
                        self.condition.wait()
                    if not self.queue:
                        return
                    until = self.queue[0][2] + self.FLUSH_SECONDS
                    while not self.closed and len(self.keys) < self.count and len(self.queue) < 2:
                        delay = until - time.monotonic()
                        if delay <= 0:
                            break
                        self.condition.wait(delay)
                    group = [self.queue.popleft() for _ in range(min(len(self.queue), 4))]
                self._verify(group)
                group = []
        except BaseException as error:
            with self.condition:
                self.error = error
                group.extend(self.queue)
                self.queue.clear()
            for _, future, _ in group:
                if not future.done():
                    future.set_exception(error)

    def _verify(self, group):
        from brass_ai.recovery.teacher_term import chunks
        require(not self.verifier.cancel.is_set(), 'Teacher logit verification cancelled')
        began = time.monotonic()
        rows = [row for record, _, _ in group for row in record['deferred']]
        parts = [[rows[i] for i in part] for part in chunks(rows, self.MAX_ROWS, self.MAX_CELLS)]
        if rows:
            with self.verifier._policy_lock():
                self.verifier._check_model()
                check_logits(self.verifier.model, [row for part in parts for row in part],
                             batches=parts, timings=self.metrics)
        require(not self.verifier.cancel.is_set(), 'Teacher logit verification cancelled')
        ended = time.monotonic()
        self.metrics.update(groups=1, tables=len(group), rows=len(rows), batches=len(parts),
                            check_seconds=ended-began)
        for record, future, queued in group:
            count = len(record['deferred'])
            record['settled'] = dict(rows=count, seconds=ended-queued, queue_seconds=began-queued,
                                     check_seconds=ended-began) if count else {}
            future.set_result(None)


class TurnVerifier:
    def __init__(self, model, history, jobs, cancel, *, diagnostic=False, search_limits=None):
        self.model, self.history, self.cancel = model, copy.deepcopy(history), cancel
        self.search_limits = search_limits  # Fixed-count searches may run past the job's own allowance.
        self.model_id = model_id(model)
        self.model_config = copy.deepcopy(model.config.to_dict())
        self.model_binding = self._binding()
        self.jobs = {j['expected']['job_id']: copy.deepcopy(j) for j in jobs}
        require(all(j['expected']['model_id'] == self.model_id for j in jobs), 'Verifier model differs')
        self.cache = {}
        self.value_tables = {}
        self.lock = threading.Lock()
        self.diagnostic = diagnostic
        self.sources = teacher_sources(jobs) if not diagnostic else set()
        self.encoders = threading.local()
        self.handoff_model_checked = False

    def _binding(self):
        # No GPU synchronization or weight copy. The collection is exclusive
        # of optimizer updates; take() still hashes the full model at handoff.
        return tuple((name,id(t),t._version,t.data_ptr(),tuple(t.shape),t.dtype,t.device,t.requires_grad)
                     for name,t in self.model.state_dict(keep_vars=True).items())

    def _check_model(self):
        require(self._binding() == self.model_binding and self.model.config.to_dict() == self.model_config,
                'Frozen verification model changed')

    @contextmanager
    def _policy_lock(self):
        while not self.lock.acquire(timeout=.05):
            require(not self.cancel.is_set(),'Turn verification cancelled')
        try:
            require(not self.cancel.is_set(),'Turn verification cancelled')
            yield
        finally:self.lock.release()

    def verify(self, meta, received):
        identity = meta['job']['job_id']; job = self.jobs[identity]
        require(meta['job'] == job['expected'] and meta['timing'] == TIMING,
                'Stale turn/job/timing identity')
        require(meta.get('diagnostic',False) is self.diagnostic, 'Wrong turn verification scope')
        profile = meta.get('profile','full')
        require(profile in ('fast','full'),'Invalid teacher search profile')
        fast = profile == 'fast'
        search_ns, post_ns = (800_000_000,int(FAST_POST_SECONDS*1e9)) if fast else (4_000_000_000,1_000_000_000)
        if self.search_limits is not None:search_ns = self.search_limits[profile]
        # Only a g1491 job labels its short searches; those rows say so themselves.
        short_labels = (job.get('teacher_revision') or {}).get('short_labels') is True and not self.diagnostic
        # Every revised table (long-only g1490 restart included) checks logits once per table.
        revised = job.get('teacher_revision') is not None and not self.diagnostic
        require(not fast or (job.get('seating') == 'all-seat-search' and (short_labels or not meta['rows'])),
                'Fast searches cannot label')
        require(all((row.get('teacher', {}).get('profile') == 'fast') == fast for row in meta['rows']),
                'Teacher label profile differs from its search')
        require(type(meta['remaining_ns']) is int and 0 <= meta['remaining_ns'] <= post_ns
                and type(meta['native_post_ns']) is int and 0 <= meta['native_post_ns'] < 2**64
                and meta['remaining_ns'] <= max(0,post_ns-meta['native_post_ns'])
                and type(meta['search_ns']) is int and 0 <= meta['search_ns'] <= search_ns,
                'Invalid post-search timing metadata')
        if meta.get('value_row') is not None:
            from brass_ai.recovery.search_value_rows import freeze_root, validate_root
            validate_root(meta['value_row'], job)
            require(meta['value_row']['index'] == meta['index'] and meta['value_row']['profile'] == meta['profile'], 'Root turn differs')
            meta['value_row'] = freeze_root(meta['value_row'])
        def check():
            require(not self.cancel.is_set(), 'Turn verification cancelled')
        check()
        record = self.cache.setdefault(identity, dict(raw=[], accepted=[], rejected=Counter(), timings=[], last=0,
                                                      sampled_profiles=set(), deferred=[], values=[]))
        require(meta['index'] == record['last'] + 1 and len(meta['rows']) <= 2,
                'Duplicate/out-of-order/excess teacher turn')
        sampled = profile not in record['sampled_profiles'] or meta['index'] % 64 == 0
        began = time.monotonic() if sampled else None
        # A single detached immutable owner, shared by the admitted LabelRows.
        # deepcopy(FrozenDict) is identity; cold/mutable callers retain copies.
        frozen = [_freeze(row) for row in meta['rows']]
        if not meta['rows']:
            # Fast/value-only turns never wait for the shared policy model.
            accepted, rejected = [], {}
        elif self.diagnostic:
            # policy_context preserves the process RNG/model mode. Serialize
            # reference checks across diagnostic lanes as in cold admission.
            with self._policy_lock():
                check()
                from brass_ai.recovery.teacher_labels import admit_rows
                require(all(r.get('diagnostic') is True for r in meta['rows']), 'Unmarked diagnostic probe')
                accepted,rejected=admit_rows(frozen,model=self.model,expected=job['expected'],
                    history=self.history,admitted_sources={job['expected']['source_fingerprint']})
        else:
            require(all(not row.get('diagnostic',False) for row in frozen), 'Calibration probes cannot enter the teacher buffer')
            self._check_model()
            from brass_ai.recovery.encoding import PreparedEncoder
            if not hasattr(self.encoders,'value'):self.encoders.value = PreparedEncoder()
            accepted,rejected = prepare_rows(frozen, expected=job['expected'], history=self.history,
                                             admitted_sources=self.sources, encoder=self.encoders.value)
            prepared = time.monotonic() if sampled else None
            if fast or revised:
                # A g1491 table labels every turn, four times as many as before, and its
                # long turns reach the shared lock together at the phase boundaries
                # (0.5 s median, 8 s worst wait at g1490). Their model-logit parity runs
                # once per table instead, as the table finishes, before anything publishes.
                locked = None
            else:
                with self._policy_lock():
                    check();self._check_model()
                    locked = time.monotonic() if sampled else None
                    check_logits(self.model,accepted)
        if not self.diagnostic:
            for row in accepted: check_turn(row.label, row.row['view']['context']['round'], job)
        check()
        # Publish only after identity, legality, target and independent batched
        # model-logit checks pass. Verification has no search-time deadline.
        # Reuse the existing immutable JSON snapshot containers. Detach once
        # inside this turn's post-work budget, then share safely through take,
        # policy publication and value-label ages instead of copying growing
        # histories again at the final fast turn. No actor admission is granted.
        from brass_ai.recovery.teacher_labels import LabelRow
        meta['rows'] = frozen
        accepted = [LabelRow(_freeze(row.row),row.encoded) for row in accepted]
        check()
        record['raw'].extend(meta['rows'])
        if meta.get('value_row') is not None:record['values'].append(meta['value_row'])
        record['accepted'].extend(accepted); record['rejected'].update(rejected)
        batched = fast or revised
        if batched: record['deferred'].extend(accepted)
        record['last'] = meta['index']
        if sampled:
            ended=time.monotonic()
            timing=dict(index=meta['index'],profile=profile,rows=len(meta['rows']),
                verification_seconds=ended-began, receive_to_ready_seconds=ended-received,
                verification_limit_seconds=None,sampling='first-per-profile-and-every-64-turns')
            if meta['rows'] and not self.diagnostic and batched:
                timing.update(preparation_seconds=prepared-began,model_check='batched-per-table')
            elif meta['rows'] and not self.diagnostic:
                timing.update(preparation_seconds=prepared-began,model_queue_seconds=locked-prepared,
                              model_and_handoff_seconds=ended-locked)
            record['timings'].append(timing);record['sampled_profiles'].add(profile)
        try:check()
        except BaseException:
            # Cancellation cannot leave a complete-looking cache at publication.
            self.cache.pop(identity,None)
            raise
        return dict(job=job['expected'], index=meta['index'], status='ok')

    def settle(self, key):
        """The model-logit parity of a table's batched rows, once, in one pass.

        A worker calls this as the table finishes, so the shared lock is taken once per
        table at staggered times; take() runs it if nobody did. Nothing publishes before it.
        """
        record = self.cache.get(key)
        require(record is not None and not self.diagnostic,'Changed/unverified teacher table')
        if 'settled' not in record:
            deferred = record.get('deferred', [])
            began = time.monotonic()
            if deferred:
                with self._policy_lock():
                    ready = time.monotonic()
                    self._check_model();check_logits(self.model,deferred)
                done = time.monotonic()
                record['settled'] = dict(rows=len(deferred),seconds=done-began,
                                         queue_seconds=ready-began,check_seconds=done-ready)
            else:record['settled'] = {}
        return record['settled']

    def take(self, result, job, model):
        began = time.monotonic()
        require(not self.diagnostic, 'Diagnostic probes cannot enter the teacher buffer')
        require(not self.cancel.is_set(),'Retired/changed verifier model')
        if hasattr(self,'model_binding'):
            require(model is self.model,'Retired/changed verifier model')
            self._check_model()
            if not self.handoff_model_checked:
                require(model_id(model)==self.model_id,'Retired/changed verifier model')
                self.handoff_model_checked=True
        else:
            # Foreign/offline diagnostic construction has no private binding.
            require(model_id(model)==self.model_id,'Retired/changed verifier model')
        key = job['expected']['job_id']; record = self.cache.get(key)
        require(record is not None and job == self.jobs[key] and result['job_id'] == key,
                'Changed/unverified teacher table')
        # The live Worker holds these exact immutable per-turn rows. Plain JSON
        # reloads/copies still take the original full canonical-byte check.
        from brass_ai.recovery.admitted_inputs import FrozenDict
        same = len(result['rows']) == len(record['raw']) and all(
            type(expected) is FrozenDict and actual is expected
            for actual,expected in zip(result['rows'],record['raw'],strict=True))
        require(same or digest(result['rows']) == digest(record['raw']), 'Changed/unverified teacher table')
        require(record['last'] == result['metrics'].get('searched_turns',result['metrics']['reserved_turns'])
                and record['last'] == (64 if job.get('seating') == 'all-seat-search' else 16),
                'Incomplete turn verification')
        settled = self.settle(key)
        if settled: result['metrics']['deferred_logit_check'] = dict(settled)
        if record.get('values'):
            self.value_tables[key] = record['values']
        # A verified table can cross the phase boundary exactly once. Fresh
        # loads always use full sealed-shard validation instead of this cache.
        del self.cache[key]
        result['metrics']['verification'] = record['timings']
        result['metrics']['verified_reuse_seconds'] = time.monotonic()-began
        return record['accepted'], dict(record['rejected'])

    def take_value_rows(self, result, job):
        """Reuse completed root validation only for the live immutable rows."""
        require(not self.cancel.is_set(), 'Teacher value admission cancelled')
        key = job['expected']['job_id']
        expected = self.value_tables.pop(key, None)
        if expected is None:
            return False
        rows = result['value_rows']
        return job == self.jobs[key] and result['job_id'] == key and len(rows) == len(expected) and all(
            actual is checked for actual, checked in zip(rows, expected, strict=True))

    def compare_previous_barrier(self, results):
        """Qualification-only audit of the former one-batch table verifier.

        This deliberately repeats checks outside measured collection. Compare
        the final pause and total verification work; do not invent a two-pass
        baseline or label relocated work as an overall speed improvement.
        """
        comparisons=[]
        for result in results:
            key=result['job_id'];job=self.jobs[key];record=self.cache[key]
            began=time.monotonic()
            rows,rejected=admit_teacher_rows(result['rows'],job=job,model=self.model,history=self.history)
            elapsed=time.monotonic()-began
            require(digest([r.row for r in rows]) == digest([r.row for r in record['accepted']])
                and rejected == dict(record['rejected']), 'Streamed/barrier verification differs')
            comparisons.append(dict(job_id=key,rows=len(rows),previous_barrier_seconds=elapsed,
                sampled_verification_seconds=sum(r['verification_seconds'] for r in record['timings']),
                sampled_turns=len(record['timings'])))
        return dict(passed=True,tables=comparisons,scope='Qualification-only audit, outside timed collection; '
            'compare former end-of-table pause with verified_reuse_seconds; sparse samples are not a streamed total')
