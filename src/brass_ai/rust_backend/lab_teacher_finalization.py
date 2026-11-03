"""Bounded, once-only verification of a completed native teacher turn.

The cache is private to one frozen collection. It cannot authorize persisted
rows or a changed model/job. The ordinary shard boundary still seals bytes.
"""
from collections import Counter
import copy
import time
import threading

from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import digest, require
from brass_ai.recovery.teacher_phase_config import TIMING
from .lab_admission import admit_teacher_rows


class TurnVerifier:
    def __init__(self, model, history, jobs, cancel, *, diagnostic=False):
        self.model, self.history, self.cancel = model, copy.deepcopy(history), cancel
        self.model_id = model_id(model)
        self.jobs = {j['expected']['job_id']: copy.deepcopy(j) for j in jobs}
        require(all(j['expected']['model_id'] == self.model_id for j in jobs), 'Verifier model differs')
        self.cache = {}
        self.lock = threading.Lock()
        self.diagnostic = diagnostic

    def verify(self, meta, received):
        identity = meta['job']['job_id']; job = self.jobs[identity]
        require(meta['job'] == job['expected'] and meta['timing'] == TIMING,
                'Stale turn/job/timing identity')
        require(meta.get('diagnostic',False) is self.diagnostic, 'Wrong turn verification scope')
        require(type(meta['remaining_ns']) is int and 0 < meta['remaining_ns'] <= 1_000_000_000
                and 0 <= meta['native_post_ns'] < 1_000_000_000
                and meta['remaining_ns'] <= 1_000_000_000 - meta['native_post_ns']
                and 0 <= meta['search_ns'] <= 4_000_000_000, 'Invalid post-search allowance')
        deadline = received + meta['remaining_ns'] / 1e9
        def check():
            require(not self.cancel.is_set(), 'Turn verification cancelled')
            require(time.monotonic() < deadline, 'Post-search verification deadline exceeded')
        check()
        record = self.cache.setdefault(identity, dict(raw=[], accepted=[], rejected=Counter(), timings=[], last=0))
        require(meta['index'] == record['last'] + 1 and len(meta['rows']) <= 2,
                'Duplicate/out-of-order/excess teacher turn')
        began = time.monotonic()
        # policy_context preserves the process RNG/model mode. Serialize that
        # small reference check across lanes; queue wait consumes this deadline.
        with self.lock:
            check()
            if self.diagnostic:
                from brass_ai.recovery.teacher_labels import admit_rows
                require(all(r.get('diagnostic') is True for r in meta['rows']), 'Unmarked diagnostic probe')
                accepted,rejected=admit_rows(meta['rows'],model=self.model,expected=job['expected'],
                    history=self.history,admitted_sources={job['expected']['source_fingerprint']})
            else:
                accepted, rejected = admit_teacher_rows(meta['rows'], job=job, model=self.model, history=self.history)
        for row in accepted:
            ctx = row.row['view']['context']
            require(self.diagnostic or ctx['round'] == job['reservations'][str(row.label['seat'])][row.label['phase']],
                    'Teacher label outside reserved turn')
        check()
        # Publish only after identity, legality, target and independent batched
        # model-logit checks pass within the original allowance.
        record['raw'].extend(copy.deepcopy(meta['rows']))
        record['accepted'].extend(accepted); record['rejected'].update(rejected)
        record['last'] = meta['index']
        record['timings'].append(dict(index=meta['index'], rows=len(meta['rows']),
            verification_seconds=time.monotonic()-began, receive_to_ready_seconds=time.monotonic()-received,
            allowance_seconds=meta['remaining_ns']/1e9))
        check()
        return dict(job=job['expected'], index=meta['index'], status='ok')

    def take(self, result, job, model):
        began = time.monotonic()
        require(not self.diagnostic, 'Diagnostic probes cannot enter the teacher buffer')
        require(not self.cancel.is_set() and model_id(model) == self.model_id, 'Retired/changed verifier model')
        key = job['expected']['job_id']; record = self.cache.get(key)
        require(record is not None and job == self.jobs[key] and result['job_id'] == key
                and digest(result['rows']) == digest(record['raw']), 'Changed/unverified teacher table')
        require(record['last'] == result['metrics']['reserved_turns'] == 16,
                'Incomplete turn verification')
        # A verified table can cross the phase boundary exactly once. Fresh
        # loads always use full sealed-shard validation instead of this cache.
        del self.cache[key]
        result['metrics']['verification'] = record['timings']
        result['metrics']['verified_reuse_seconds'] = time.monotonic()-began
        return record['accepted'], dict(record['rejected'])

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
                streamed_verification_seconds=sum(r['verification_seconds'] for r in record['timings'])))
        return dict(passed=True,tables=comparisons,scope='Qualification-only audit, outside timed collection; '
            'compare former end-of-table pause with verified_reuse_seconds, and report streamed total separately')
