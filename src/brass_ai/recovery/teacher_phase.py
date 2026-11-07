"""Trainer lifecycle and the Python side of the future B04 collection barrier."""
import copy
import random
import time
from collections import Counter

from brass_ai.learning.contracts import digest, require
from brass_ai.step5.league import seeds_for
from .teacher_phase_config import CONTRACT, SCHEDULE, SOURCE, validate, buffer_identity
from .teacher_buffer import TeacherBuffer
from .teacher_labels import PHASES
from .teacher_term import TeacherTerm, heldout_measure, summary

KEY = 'teacher_phase'


def initial_cursor(config):
    return int(digest(dict(contract=CONTRACT, stream='teacher-setup', seed=config['seed']))[:8], 16) % 1_000_000_000


def readiness(previous, current, reward_id):
    """Null means no new fit; explicit not-ready always supersedes an old fit."""
    receipt = copy.deepcopy(current if current is not None else previous)
    if receipt is not None:
        require(isinstance(receipt, dict) and type(receipt.get('ready')) is bool
                and isinstance(receipt.get('reward_id'), str), 'Invalid teacher readiness receipt')
    ready = bool(receipt and receipt['reward_id'] == reward_id and receipt['ready'])
    return receipt, ready


def reserved_rounds(setup_seed, sampling_seed, seat):
    # Keep the same four ranges and independent streams; version the new labels.
    ranges = (('canal_early', 1, 4), ('canal_late', 5, 8), ('rail_early', 1, 6), ('rail_late', 7, 8))
    return {phase: random.Random(digest(dict(contract=SCHEDULE, setup_seed=setup_seed,
        sampling_seed=sampling_seed, seat=seat, phase=phase))).randint(first, last) for phase, first, last in ranges}


def plan_jobs(config, state, *, model_id, export_hash, epoch, ready, native_source=SOURCE):
    """Frozen separate teacher stream; the ordinary plan/cursor is never touched."""
    c = validate(config); cursor = state.get('teacher_seed_cursor', initial_cursor(config))
    if c['mode'] == 'off' or not ready: return [], cursor
    jobs = []; generation = state['iteration']; reward_id = digest(c['search']['reward'])
    for split, partition, count in (('train', 'train', c['tables']['train']), ('heldout', 'validation', c['tables']['held_out'])):
        seeds, cursor = seeds_for(cursor, count, partition=partition)
        for seed in seeds:
            identity = dict(contract=CONTRACT, generation=generation, split=split, seed=seed, model_id=model_id,
                            profile=buffer_identity(config))
            job_id = digest(identity); sampling = int(job_id[:8], 16)
            jobs.append(dict(kind='teacher_phase', split=split, seed=seed, sampling_seed=sampling,
                expected=dict(job_id=job_id, game_id='teacher/' + job_id, generation=generation, model_id=model_id,
                    export_hash=export_hash, reward_id=reward_id, source_fingerprint=native_source, epoch=epoch),
                reservations={str(seat): reserved_rounds(seed, sampling, seat) for seat in range(4)},
                search=copy.deepcopy(c['search']), unsearched=copy.deepcopy(c['unsearched']),
                seats=[model_id] * 4, schedule=SCHEDULE, ordinary_training=False))
    if c.get('seating') == 'all-seat-search':
        from brass_ai.learning.search_value import binding
        progress = state.get('search_value', {}).get('stage', state.get('search_value', {}))
        # A saved machine owns the dynamic blend; it is not a manual recipe
        # edit and must not reset when the ordinary iteration starts again.
        if not isinstance(progress, dict): progress = state.get('search_value', {})
        weight = progress.get('w', c['search_value']['blend']['w'])
        if c['search_value']['blend']['w'] > 0:
            require(progress.get('stage') in ('B','head_to_head','C') and
                    progress.get('w') == c['search_value']['blend']['w'], 'Search blend requires committed stage progress')
        for job in jobs:
            job.update(seating=c['seating'], fast_search=copy.deepcopy(c['fast_search']),
                       search_value=binding(0. if c['search_value']['stage']=='off' else weight))
        if c['startup'] == 'shadow-until-search-informative' and not progress.get('g0_passed', False):
            jobs = [j for j in jobs if j['split'] == 'heldout']
            for job in jobs: job['shadow_probe'] = True
    return jobs, cursor


def collection_order(teacher_jobs, ordinary_jobs):
    require(all(j['kind'] == 'teacher_phase' and j['ordinary_training'] is False for j in teacher_jobs),
            'Unidentified teacher collection job')
    return [*teacher_jobs, *ordinary_jobs]


class TeacherPhase:
    def __init__(self, root, run, config, state, model, raw=None):
        self.config = config; self.options = validate(config); self.state = state
        self.generation = state['iteration']; self.reward_id = digest(self.options['search']['reward'])
        saved = copy.deepcopy(state.get(KEY, {}))
        checkpoint = (raw or {}).get('training', {}).get(KEY)
        require((not saved and checkpoint is None) or saved == checkpoint, 'Teacher checkpoint/state differs')
        if saved:
            require(saved['contract'] == CONTRACT and saved['generation'] == self.generation
                    and saved['teacher_seed_cursor'] == state.get('teacher_seed_cursor'), 'Teacher resume generation/seed differs')
        cache = saved.get('buffer', {})
        require(all(any(e['generation'] == r['generation'] and e['model_id'] == r['model_id']
                        for e in state['history']) for r in cache.get('receipts', ())),
                'Teacher buffer model is not in committed run lineage')
        self.buffer = TeacherBuffer(root, run, buffer_identity(config), receipts=cache.get('receipts', ()), retired=cache.get('retired', ()))
        self.weight = saved.get('weight', {}); self.term = None; self.jobs = []; self.heldout = []
        if cache and cache.get('identity') != self.buffer.identity:
            self.weight = {}  # Static producer/profile changes retire beta with its labels.
        self.cursor = state.get('teacher_seed_cursor', initial_cursor(config))
        self.receipt, self.ready = readiness(saved.get('readiness'), getattr(model, 'utility_calibration', None), self.reward_id)
        self.metrics = {}; self.admission_seconds = 0.; self.pre = None; self.post = None
        self.search_value = None
        if 'search_value' in self.options:
            from .search_value_iteration import SearchValueIteration
            self.search_value = SearchValueIteration(self.buffer.run,config,state,model,raw)

    def begin(self, state, model, *, export_hash=None, epoch=None, native_source=SOURCE):
        self.state = state; self.generation = state['iteration']
        self.receipt, self.ready = readiness(self.receipt, getattr(model, 'utility_calibration', None), self.reward_id)
        self.jobs = []; self.heldout = []; self.term = None; self.metrics = {}; self.pre = self.post = None
        self.admission_seconds = 0.
        self.buffer.window(self.generation)
        if self.options['mode'] != 'off' and self.ready:
            require(isinstance(export_hash, str) and len(export_hash) == 64 and epoch is not None,
                    'B04 frozen export/epoch required before teacher collection')
            from brass_ai.learning.batching import model_id
            self.jobs, self.cursor = plan_jobs(self.config, state, model_id=model_id(model),
                                               export_hash=export_hash, epoch=epoch, ready=True, native_source=native_source)
            for job in self.jobs:job['readiness'] = copy.deepcopy(self.receipt)
        return copy.deepcopy(self.jobs)

    def admit_collection(self, results, model, *, verifier=None):
        """Separate barrier results; returns no ordinary games or PPO targets."""
        began = time.perf_counter(); by_id = {r['job_id']: r for r in results}
        require(len(by_id) == len(results) and set(by_id) == {j['expected']['job_id'] for j in self.jobs},
                'Missing/duplicate/misrouted teacher table')
        splits = dict(train=[], heldout=[]); rejected = Counter(); labels = []; table_metrics = []
        for job in self.jobs:
            result = by_id[job['expected']['job_id']]
            from brass_ai.rust_backend.lab_admission import admit_teacher_rows
            rows, dropped = (verifier.take(result,job,model) if verifier is not None else
                admit_teacher_rows(result['rows'],job=job,model=model,history=self.state['history']))
            for r in rows:
                ctx = r.row['view']['context']
                require(ctx['round'] == job['reservations'][str(r.label['seat'])][r.label['phase']],
                        'Teacher label outside reserved turn')
            splits[job['split']].extend(rows); rejected.update(dropped); labels.extend(r.label for r in rows)
            table_metrics.append(copy.deepcopy(result.get('metrics', {})))
        if self.jobs:
            for split, rows in splits.items():
                self.buffer.publish(self.generation, split, rows, model_id=self.jobs[0]['expected']['model_id'], reward_id=self.reward_id)
        self.heldout = splits['heldout']
        self.metrics = dict(tables=dict(Counter(j['split'] for j in self.jobs)), labels=len(labels),
            reserved_turns_by_phase={p: 4 * len(self.jobs) for p in PHASES},
            labels_by_kind=dict(Counter(l['kind'] for l in labels)), labels_by_phase=dict(Counter(l['phase'] for l in labels)),
            discarded=dict(rejected), completed_trajectories=summary([l['completed_trajectories'] for l in labels]),
            effective_k=dict(Counter(str(l['effective_k']) for l in labels)), completed_rounds=summary([l['completed_rounds'] for l in labels]),
            deadline_hit_fraction=sum(l['deadline_hit'] for l in labels) / len(labels) if labels else None,
            overrun_seconds=summary([l['overrun_seconds'] for l in labels]),
            max_overrun_seconds=max((l['overrun_seconds'] for l in labels), default=0.),
            fallback_played_fraction=sum(l['fallback_played'] for l in labels) / len(labels) if labels else None,
            collection_metrics=table_metrics,
            scope='Native search/lane/inference metrics supplied by B04; absent metrics are unmeasured')
        self.admission_seconds = time.perf_counter() - began
        if verifier is not None:
            # Deferred shard sealing is charged to the final searched turn;
            # it receives no extra allowance of its own.
            require(all(r['metrics']['turns'] and
                r['metrics']['turns'][-1]['post_search_ns']/1e9 + self.admission_seconds <=
                    (.2 if r['metrics']['turns'][-1].get('profile')=='fast' else 1.)
                for r in results), 'Teacher sealing exceeded the remaining post-search allowance')
            self.metrics['deferred_sealing_seconds'] = self.admission_seconds
        if self.search_value is not None:
            self.search_value.admit(results,self.jobs,post_spent=self.admission_seconds if verifier is not None else 0.)

    def prepare(self, model):
        self.pre = heldout_measure(model, self.heldout)
        rows = self.buffer.rows(self.generation) if self.options['mode'] != 'off' else []
        self.term = TeacherTerm(model, self.options, rows, seed=self.config['seed'], generation=self.generation, state=self.weight)
        return self.term if self.term.active else None

    def finish(self, model, update):
        if self.term is not None: self.weight = self.term.finish()
        self.post = heldout_measure(model, self.heldout)
        self.receipt, self.ready = readiness(self.receipt, getattr(model, 'utility_calibration', None), self.reward_id)
        before = self.pre or heldout_measure(model, [])
        changes = {p: self.post[p]['kl'] - before[p]['kl'] if self.post[p]['kl'] is not None and before[p]['kl'] is not None else None for p in before}
        return dict(contract=CONTRACT, mode=self.options['mode'], ready=self.ready, collection=self.metrics,
            term=self.term.report() if self.term else {}, heldout=dict(before=before, after=self.post, delta=changes),
            admission_encoding_seconds=self.admission_seconds, buffer=self.buffer.payload(),
            side_effects=dict(ppo_steps=update.get('optimizer_steps'), kl_guard=update.get('kl_guard'),
                              categorical_probe=update.get('categorical_probe')))

    def payload(self, generation):
        require(generation == self.generation + 1, 'Teacher commit generation differs')
        return dict(contract=CONTRACT, generation=generation, teacher_seed_cursor=self.cursor,
                    readiness=copy.deepcopy(self.receipt), weight=copy.deepcopy(self.weight), buffer=self.buffer.payload())

    def committed(self, generation): self.buffer.committed(generation)
