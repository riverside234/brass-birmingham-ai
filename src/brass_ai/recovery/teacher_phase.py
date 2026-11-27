"""Trainer lifecycle and the Python side of the future B04 collection barrier."""
import copy
import random
import time
from collections import Counter, defaultdict

from brass_ai.learning.contracts import digest, require
from brass_ai.step5.league import seeds_for
from brass_ai.step5.policy import cotton_seats
from .teacher_phase_config import CONTRACT, SCHEDULE, SOURCE, validate, buffer_identity
from .teacher_buffer import TeacherBuffer
from .teacher_cotton import NETWORK_CONTRACTS, SEATS
from .teacher_labels import PHASES
from .teacher_term import TeacherTerm, heldout_measure, movement, policy_logits, summary
from .search_value_schedule import validation_plan
from .training_revision_g1491 import count_search, cotton_only, cotton_settings, job_block, seated_cotton, teacher_search_head_weight
from .training_revision_g2090 import teacher_weight as opening_revision_weight
from .teacher_vp import summarize as summarize_vp

KEY = 'teacher_phase'


def committed_weight(config, state):
    """The committed teacher weight, or {} when this recipe's labels retire it with them.

    The same rule as TeacherPhase: a changed label identity (the g1491 revision)
    retires the fixed beta and its qualification.
    """
    saved = state.get(KEY, {})
    cache = saved.get('buffer', {})
    if cache and cache.get('identity') != buffer_identity(config): return {}
    from .training_revision_g3270 import teacher_weight as teacher_revision_weight
    return teacher_revision_weight(config, state, opening_revision_weight(config, state, saved.get('weight', {})))


def initial_cursor(config):
    return int(digest(dict(contract=CONTRACT, stream='teacher-setup', seed=config['seed']))[:8], 16) % 1_000_000_000


def readiness(previous, current, reward_id, *, skip=False):
    """Null means no new fit; explicit not-ready always supersedes an old fit."""
    if isinstance(previous, dict) and previous.get('utility_qualification') == 'skipped':
        previous = previous.get('utility_calibration')
    receipt = current if current is not None else previous
    if receipt is not None:
        require(isinstance(receipt, dict) and type(receipt.get('ready')) is bool
                and isinstance(receipt.get('reward_id'), str), 'Invalid teacher readiness receipt')
    if skip:
        # Ready to collect by explicit choice, not a claim of a successful utility fit.
        return dict(ready=True, reward_id=reward_id, utility_qualification='skipped', utility_calibration=receipt), True
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
    if c['mode'] == 'off' or not (ready or c.get('skip_utility_qualification', False)): return [], cursor
    jobs = []; generation = state['iteration']; reward_id = digest(c['search']['reward'])
    validation=validation_plan(config,state)
    profile = buffer_identity(config)
    teacher_weight = teacher_search_head_weight(config)
    progress = state.get('search_value', {}).get('stage', state.get('search_value', {}))
    # The committed stage owns the blend until Stage B finishes.
    if not isinstance(progress, dict): progress = state.get('search_value', {})
    stage = progress.get('stage', c['search_value']['stage']) if c.get('seating') == 'all-seat-search' else None
    seated = seated_cotton(config)
    single = cotton_only(config)
    cotton = cotton_settings(config)
    spacing = round(1 / cotton.get('table_fraction', .5))
    # g3270: 36 train and 4 validation tables every iteration; the qualified recipe keeps 32+4.
    from .training_revision_g3270 import tables as revised_tables
    table_counts = revised_tables(config)
    for split, partition, count in (('train', 'train', table_counts['train']), ('heldout', 'validation', table_counts['held_out'])):
        seeds, cursor = seeds_for(cursor, count, partition=partition)
        # Reserve the same independent seed stream, but create no held-out jobs
        # or search/inference work on intervening iterations.
        if split=='heldout' and not validation['due']:continue
        for index, seed in enumerate(seeds):
            identity = dict(contract=CONTRACT, generation=generation, split=split, seed=seed, model_id=model_id,
                            profile=profile)
            job_id = digest(identity); sampling = int(job_id[:8], 16)
            # Interleave cotton tables independently within training and held-out seeds.
            cotton_table = (index + generation) % spacing == 0
            if cotton.get('contract') in NETWORK_CONTRACTS:
                # Exactly 3/20 over consecutive training tables (4/5 of 32).
                # Heldouts retain one cotton and three standard games each time.
                ordinal = generation * table_counts['train'] + index
                cotton_table = ((ordinal+1)*3//20 > ordinal*3//20 if split == 'train'
                                else (index+generation)%4 == 0)
            seats = None
            if seated and cotton_table:
                count = 1 if single else 1 + (index//2 + generation) % 2
                seats = cotton_seats(sampling, count, cotton['contract'])
            revision = job_block(config, split=split, cotton_table=cotton_table, stage=stage, seats=seats)
            if teacher_weight is not None:
                job_id = digest(dict(identity, teacher_revision=revision))
            jobs.append(dict(kind='teacher_phase', split=split, seed=seed, sampling_seed=sampling,
                expected=dict(job_id=job_id, game_id='teacher/' + job_id, generation=generation, model_id=model_id,
                    export_hash=export_hash, reward_id=reward_id, source_fingerprint=native_source, epoch=epoch),
                reservations={str(seat): reserved_rounds(seed, sampling, seat) for seat in range(4)},
                search=copy.deepcopy(c['search']), unsearched=copy.deepcopy(c['unsearched']),
                seats=[model_id] * 4, schedule=SCHEDULE, ordinary_training=False))
            if revision is not None:
                jobs[-1]['teacher_revision'] = revision
    if c.get('seating') == 'all-seat-search':
        from brass_ai.learning.search_value import binding
        weight = progress.get('w', c['search_value']['blend']['w'])
        if 'schedule' in c['search_value'] and progress.get('stage')=='B':
            # The first B collection uses one step (.05, or .02 over fifty); its committed
            # counter then owns that weight. B collections reach one without a zero step.
            from .search_value_schedule import blend_step
            weight=min(1.,round(weight+blend_step(c['search_value']),2))
        if c['search_value']['blend']['w'] > 0:
            require(progress.get('stage') in ('B','head_to_head','C') and
                    progress.get('w') == c['search_value']['blend']['w'], 'Search blend requires committed stage progress')
        use_override = teacher_weight is not None and (not seated or stage == 'C')
        value = binding(teacher_weight if use_override else
                        (0. if c['search_value']['stage']=='off' else weight))
        for job in jobs:
            job.update(seating=c['seating'], fast_search=copy.deepcopy(c['fast_search']),
                       search_value=dict(value))
        if c['startup'] == 'shadow-until-search-informative' and not progress.get('g0_passed', False):
            jobs = [j for j in jobs if j['split'] == 'heldout']
            for job in jobs: job['shadow_probe'] = True
    return jobs, cursor


def collection_order(teacher_jobs, ordinary_jobs):
    require(all(j['kind'] == 'teacher_phase' and j['ordinary_training'] is False for j in teacher_jobs),
            'Unidentified teacher collection job')
    return [*teacher_jobs, *ordinary_jobs]


def action_summary(table_metrics, jobs, weight):
    """Reduce the eight fixed era/seat counters from each completed teacher table."""
    total, tables = Counter(), []
    by_split = defaultdict(lambda: dict(total=Counter(), cotton_exploration=Counter(), standard_exploration=Counter()))
    for metrics, job in zip(table_metrics, jobs, strict=True):
        revision = metrics['cpu_inference']['revision']
        require(revision['job'] == job['teacher_revision'], 'Teacher action report job differs')
        cells = revision['actions']['by_era_seat']
        require(len(cells) == 2 and all(len(era) == 4 for era in cells), 'Teacher action report shape differs')
        counts = Counter()
        for era in cells:
            for seat in era:
                counts.update(seat)
        split = job['split']
        cotton = 'cotton' in revision['job']['exploration']
        tables.append(dict(seed=job['seed'], split=split, cotton_exploration=cotton, **counts))
        if seats := revision['job'].get(SEATS):
            tables[-1][SEATS] = seats
        counts['tables'] = 1
        total.update(counts)
        split_counts = by_split[split]
        split_counts['total'].update(counts)
        split_counts['cotton_exploration' if cotton else 'standard_exploration'].update(counts)
    return dict(search_head_weight=weight, total=dict(total), per_table=tables,
                by_split={split: {name: dict(counts) for name, counts in groups.items()}
                          for split, groups in by_split.items()})


class TeacherPhase:
    def __init__(self, root, run, config, state, model, raw=None, *, rebuild=None):
        from . import training_revision_g3270 as g3270
        # The trainer runs the g3270 overlay (tables, label window, rows per step) when it is adopted.
        self.config = config; self.options = g3270.phase_options(config, validate(config)); self.state = state
        self.generation = state['iteration']; self.reward_id = digest(self.options['search']['reward'])
        saved = copy.deepcopy(state.get(KEY, {}))
        checkpoint = (raw or {}).get('training', {}).get(KEY)
        require((not saved and checkpoint is None) or saved == checkpoint, 'Teacher checkpoint/state differs')
        if saved:
            require(saved['contract'] == CONTRACT and saved['generation'] == self.generation
                    and saved['teacher_seed_cursor'] == state.get('teacher_seed_cursor'), 'Teacher resume generation/seed differs')
        cache = saved.get('buffer', {})
        if rebuild is not None:
            cache = dict(cache, receipts=[], retired=[])
            # An explicit reward migration retires labels, not the user's
            # fixed policy-teacher beta/qualification. Keep its provenance.
            cache['identity']=buffer_identity(config)
        require(all(any(e['generation'] == r['generation'] and e['model_id'] == r['model_id']
                        for e in state['history']) for r in cache.get('receipts', ())),
                'Teacher buffer model is not in committed run lineage')
        from .collector_runtime import teacher_c_scale
        profile = count_search(config) or {}
        self.buffer = TeacherBuffer(root, run, buffer_identity(config), receipts=cache.get('receipts', ()),
                                    retired=cache.get('retired', ()), c_scale=teacher_c_scale(profile or None),
                                    cotton_c_scale=profile.get('cotton_c_scale'),
                                    iterations=self.options['buffer']['iterations'],
                                    heldout_lag=g3270.heldout_lag(config))
        # Static producer/profile changes retire beta with its labels.
        self.weight = ({} if cache and cache.get('identity') != self.buffer.identity
                       else g3270.teacher_weight(config, state, opening_revision_weight(config, state, saved.get('weight', {}))))
        self.term = None; self.jobs = []; self.heldout = []; self.pre_logits = None
        # g3423 (soft policy S1): the trainer-owned soft head's committed state, beside `weight`.
        from .soft_policy import carry
        self.soft_state = carry(saved.get('soft_policy'), self.generation); self.soft = None; self.pre_soft = None
        self.cursor = state.get('teacher_seed_cursor', initial_cursor(config))
        self.receipt, self.ready = readiness(saved.get('readiness'), getattr(model, 'utility_calibration', None), self.reward_id,
                                            skip=self.options.get('skip_utility_qualification', False))
        self.metrics = {}; self.admission_seconds = 0.; self.pre = None; self.post = None
        self.search_value = None
        if 'search_value' in self.options:
            from .search_value_iteration import SearchValueIteration
            self.search_value = SearchValueIteration(self.buffer.run,config,state,model,raw,rebuild=rebuild)

    def begin(self, state, model, *, export_hash=None, epoch=None, native_source=SOURCE):
        self.state = state; self.generation = state['iteration']
        self.receipt, self.ready = readiness(self.receipt, getattr(model, 'utility_calibration', None), self.reward_id,
                                            skip=self.options.get('skip_utility_qualification', False))
        self.jobs = []; self.heldout = []; self.term = None; self.metrics = {}; self.pre = self.post = None
        self.pre_logits = None; self.admission_seconds = 0.; self.soft = None; self.pre_soft = None; self.post_ppo = None
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
        splits = dict(train=[], heldout=[]); rejected = Counter(); labels = []; table_metrics = [];post_spent={}
        for job in self.jobs:
            table_began=time.perf_counter()
            result = by_id[job['expected']['job_id']]
            from brass_ai.rust_backend.lab_admission import admit_teacher_rows
            rows, dropped = (verifier.take(result,job,model) if verifier is not None else
                admit_teacher_rows(result['rows'],job=job,model=model,history=self.state['history']))
            from .teacher_labels import check_turn
            for r in rows: check_turn(r.label, r.row['view']['context']['round'], job)
            splits[job['split']].extend(rows); rejected.update(dropped); labels.extend(r.label for r in rows)
            table_metrics.append(dict(result.get('metrics', {})))
            post_spent[job['expected']['job_id']]=time.perf_counter()-table_began+result.get('metrics',{}).get('shadow_cache_seconds',0.)
        weight = teacher_search_head_weight(self.config)
        actions = (action_summary(table_metrics, self.jobs, self.jobs[0]['search_value']['w'])
                   if self.jobs and weight is not None else None)
        if self.jobs:
            for split, rows in splits.items():
                if not rows:continue
                seal_began=time.perf_counter()
                self.buffer.publish(self.generation, split, rows, model_id=self.jobs[0]['expected']['model_id'], reward_id=self.reward_id)
                seal_seconds=time.perf_counter()-seal_began
                # Charge a shared split seal conservatively to its own tables,
                # without charging unrelated table validation or the other split.
                for job in self.jobs:
                    if job['split']==split:post_spent[job['expected']['job_id']]+=seal_seconds
        self.heldout = splits['heldout']
        # Per-phase/kind/search metrics describe the long (reserved-turn) labels, as
        # before g1491; short-search labels are reported separately.
        from .teacher_labels import label_profile
        total = len(labels); short = [l for l in labels if label_profile(l) == 'fast']
        labels = [l for l in labels if label_profile(l) == 'full']
        self.metrics = dict(tables=dict(Counter(j['split'] for j in self.jobs)), labels=total, long_labels=len(labels),
            reached_reserved_turns=sum(r.get('metrics',{}).get('reserved_turns',0) for r in results),
            root_labels=sum(l['kind']=='root' for l in labels),
            reserved_turns_by_phase={p: 4 * len(self.jobs) for p in PHASES},
            labels_by_kind=dict(Counter(l['kind'] for l in labels)), labels_by_phase=dict(Counter(l['phase'] for l in labels)),
            discarded=dict(rejected), completed_trajectories=summary([l['completed_trajectories'] for l in labels]),
            effective_k=dict(Counter(str(l['effective_k']) for l in labels)), completed_rounds=summary([l['completed_rounds'] for l in labels]),
            deadline_hit_fraction=sum(l['deadline_hit'] for l in labels) / len(labels) if labels else None,
            overrun_seconds=summary([l['overrun_seconds'] for l in labels]),
            max_overrun_seconds=max((l['overrun_seconds'] for l in labels), default=0.),
            fallback_played_fraction=sum(l['fallback_played'] for l in labels) / len(labels) if labels else None,
            collection_metrics=table_metrics,
            scope='Native search/lane/inference metrics supplied by B04; absent metrics are unmeasured. '
                  'Per-phase/kind/search fields count long-search labels; short_labels counts the others')
        if any(j.get('teacher_revision') for j in self.jobs):
            self.metrics['short_labels'] = dict(labels=len(short), by_kind=dict(Counter(l['kind'] for l in short)),
                by_phase=dict(Counter(l['phase'] for l in short)),
                effective_k=dict(Counter(str(l['effective_k']) for l in short)),
                completed_trajectories=summary([l['completed_trajectories'] for l in short]))
            self.metrics['explored_labels'] = dict(long=sum(l.get('explored', False) for l in labels),
                short=sum(l.get('explored', False) for l in short))
        self.metrics['heldout_validation']=validation_plan(self.config,self.state)
        from .teacher_cotton import aggregate
        cotton=aggregate(table_metrics)
        if cotton is not None:self.metrics['cotton_exploration']=cotton
        if actions is not None:self.metrics['teacher_actions']=actions
        if self.jobs and all(by_id[j['expected']['job_id']].get('result') for j in self.jobs):
            from .teacher_cotton import enabled as cotton_enabled
            self.metrics['vp'] = summarize_vp((job['split'], cotton_enabled(job),
                [s['vp'] for s in by_id[job['expected']['job_id']]['result']['seats']]) for job in self.jobs)
        reached=self.metrics['reached_reserved_turns']
        self.metrics['root_label_yield']=self.metrics['root_labels']/reached if reached else None
        self.metrics['root_label_yield_pass']=bool(reached and self.metrics['root_label_yield']>=.95)
        self.admission_seconds = time.perf_counter() - began
        if verifier is not None:
            # Coarse per-table/barrier timing is cheap. Independent admission
            # and durable publication no longer consume the search budget.
            self.metrics['verification_limit_seconds'] = None
            self.metrics['deferred_sealing_seconds'] = self.admission_seconds
            self.metrics['post_admission_per_table_seconds']=post_spent
        if self.search_value is not None:
            values = verifier if hasattr(verifier,'take_value_rows') else None
            self.search_value.admit(results,self.jobs,post_spent=post_spent if verifier is not None else 0.,value_verifier=values)

    def prepare(self, model, ppo_optimizer=None):
        """ppo_optimizer (g5402): the PPO Adam whose step rate a linked soft head follows."""
        from . import training_revision_g3270 as g3270
        settings = g3270.term_settings(self.config)
        from . import training_revision_g3423 as g3423
        soft_settings = g3423.settings(self.config)
        self.soft = None; self.pre_soft = None
        if soft_settings is not None and self.options['mode'] != 'off':
            from .soft_policy import SoftPolicy
            require(settings is not None, 'The soft policy head needs the g3270 teacher revision')
            self.soft = SoftPolicy(model, soft_settings, self.soft_state, generation=self.generation)
            from . import training_revision_g6230 as g6230
            from .training_revision_g5402 import soft_follows_ppo
            # g6230: visited-only targets, and 1e-4 for its first 100 iterations before the PPO link returns.
            self.soft.masked = g6230.masked_soft_targets(self.config)
            rate = g6230.soft_rate(self.config, self.generation)
            if rate is not None:
                self.soft.use_rate(rate, f'g6230-fixed-{rate:g}-until-g{g6230.soft_rate_until(self.config)}')
            elif soft_follows_ppo(self.config):
                require(ppo_optimizer is not None, 'A linked soft head needs the PPO optimizer')
                self.soft.follow(ppo_optimizer)
        # g3270: keep the validation rows' pre-update logits for the policy's movement on them.
        if self.soft is not None and not self.soft.masked:
            # The same logits as policy_logits, plus the soft head's KL from the same forwards.
            self.pre_logits, self.pre_soft = self.soft.heldout(model, self.heldout)
        elif self.soft is not None:
            # g6230 (fix of 2026-11-27): the visited-only soft check before PPO as well, from the same forwards.
            self.pre_logits, self.pre_soft = self.soft.heldout_masked(model, self.heldout, settings['label_scale'])
        else:
            self.pre_logits = policy_logits(model, self.heldout) if settings is not None else None
        self.pre = heldout_measure(model, self.heldout, logits=self.pre_logits)
        self.post_ppo = None
        rows = self.buffer.rows(self.generation) if self.options['mode'] != 'off' else []
        self.term = TeacherTerm(model, self.options, rows, seed=self.config['seed'], generation=self.generation, state=self.weight,
                                revision=settings, soft=self.soft)
        self.term.qualification_context=dict(tables=self.metrics.get('tables',{}),
            labels_by_phase=self.metrics.get('labels_by_phase',{}),
            labels_by_kind=self.metrics.get('labels_by_kind',{}),
            completed_trajectories=self.metrics.get('completed_trajectories'),
            effective_k=self.metrics.get('effective_k'),deadline_hit_fraction=self.metrics.get('deadline_hit_fraction'),
            root_label_yield=self.metrics.get('root_label_yield'),root_label_yield_pass=self.metrics.get('root_label_yield_pass'),
            improvement_by_phase=self.term.improvement_by_phase,
            collection_seconds=self.metrics.get('wall_seconds'),model_id=self.jobs[0]['expected']['model_id'] if self.jobs else None,
            export_hash=self.jobs[0]['expected']['export_hash'] if self.jobs else None)
        return self.term if self.term.active else None

    def after_ppo(self, model):
        """g4475: the soft head's held-out check once more, after the PPO update and before PPG (g6230: the second
        of its two checks, after the one in `prepare`).

        The same validation-table rows and the same forwards as `prepare` and `finish` (no gradients, RNG forked,
        train mode restored), so it changes nothing that trains. The policy's teacher KL comes from the same
        logits at no extra cost. Splits the iteration's soft change into its PPO and PPG parts."""
        self.post_ppo = None
        if getattr(self, 'soft', None) is None: return
        began = time.perf_counter()
        if self.soft.masked:
            # g6230: the one soft held-out check of the iteration, on targets built like the training ones.
            from . import training_revision_g3270 as g3270
            logits, soft = self.soft.heldout_masked(model, self.heldout, g3270.term_settings(self.config)['label_scale'])
        elif getattr(self, 'pre_soft', None) is None: return
        else: logits, soft = self.soft.heldout(model, self.heldout)
        self.post_ppo = dict(soft=soft, policy=heldout_measure(model, self.heldout, logits=logits),
                             seconds=time.perf_counter() - began)

    def finish(self, model, update):
        if self.term is not None: self.weight = self.term.finish()
        post_soft = None
        if self.soft is not None and not self.soft.masked:
            self.soft_state = self.soft.finish()
            post_logits, post_soft = self.soft.heldout(model, self.heldout)
        elif self.soft is not None:
            # g6230: the soft head was measured after PPO only; the policy's own logits as without a soft head.
            self.soft_state = self.soft.finish()
            post_logits = policy_logits(model, self.heldout) if self.pre_logits is not None else None
        else:
            post_logits = policy_logits(model, self.heldout) if self.pre_logits is not None else None
        self.post = heldout_measure(model, self.heldout, logits=post_logits)
        self.receipt, self.ready = readiness(self.receipt, getattr(model, 'utility_calibration', None), self.reward_id,
                                            skip=self.options.get('skip_utility_qualification', False))
        before = self.pre or heldout_measure(model, [])
        changes = {p: self.post[p]['kl'] - before[p]['kl'] if self.post[p]['kl'] is not None and before[p]['kl'] is not None else None for p in before}
        heldout = dict(before=before, after=self.post, delta=changes)
        if self.pre_logits is not None:
            # g3270: the 4 validation tables, measured before they train. An update that ignored the
            # teacher would raise their teacher KL by about the policy's own movement on them.
            moved = movement(self.heldout, self.pre_logits, post_logits)
            heldout.update(movement=moved, pull={p: moved[p] - changes[p] if moved.get(p) is not None and changes.get(p) is not None
                                                 else None for p in moved},
                           scope='Validation tables of this iteration, before they train: teacher KL to the stored search target, '
                                 'policy movement KL(before||after) on the same rows, pull = movement - KL change (estimate)')
        if post_soft is not None:
            # g3423: the soft head on the same rows, against the stored search target softened at T.
            heldout['soft'] = dict(before=self.pre_soft, after=post_soft,
                delta=post_soft['kl'] - self.pre_soft['kl'] if post_soft['kl'] is not None and self.pre_soft['kl'] is not None else None,
                scope='KL(stored target ** (1/T), renormalized || soft head) on the validation tables\' long labels')
            measured = getattr(self, 'post_ppo', None)
            if measured is not None:
                # g4475: the same check after PPO, before PPG; the iteration's change split into its two parts.
                change = lambda a, b: a - b if a is not None and b is not None else None
                mid = measured['soft']
                heldout['soft'].update(after_ppo=mid, ppo_delta=change(mid['kl'], self.pre_soft['kl']),
                                       ppg_delta=change(post_soft['kl'], mid['kl']), after_ppo_seconds=measured['seconds'])
                policy = measured['policy']
                heldout['after_ppo'] = policy
                heldout['ppo_delta'] = {p: change(policy[p]['kl'], before[p]['kl']) for p in before}
        elif self.soft is not None:
            # g6230: measured before and after PPO (not after PPG), against visited-only targets from the current logits.
            measured = getattr(self, 'post_ppo', None)
            change = lambda a, b: a - b if a is not None and b is not None else None
            mid = measured['soft'] if measured else None
            heldout['soft'] = dict(before=self.pre_soft, after_ppo=mid,
                ppo_delta=change((mid or {}).get('kl'), (self.pre_soft or {}).get('kl')),
                after_ppo_seconds=measured['seconds'] if measured else None,
                scope="KL(softmax((current logits + bonus) / T) over visited legal actions || soft head over the same "
                      "actions) on the validation tables' long labels with at least 2 visited actions, before and "
                      "after PPO")
            if measured is not None:
                heldout['after_ppo'] = measured['policy']
                heldout['ppo_delta'] = {p: change(measured['policy'][p]['kl'], before[p]['kl']) for p in before}
        return dict(contract=CONTRACT, mode=self.options['mode'], ready=self.ready, collection=self.metrics,
            term=self.term.report() if self.term else {}, heldout=heldout,
            admission_encoding_seconds=self.admission_seconds, buffer=self.buffer.payload(),
            side_effects=dict(ppo_steps=update.get('optimizer_steps'), kl_guard=update.get('kl_guard'),
                              categorical_probe=update.get('categorical_probe')))

    def payload(self, generation):
        require(generation == self.generation + 1, 'Teacher commit generation differs')
        from .soft_policy import carry
        soft = carry(self.soft_state, generation)
        return dict(contract=CONTRACT, generation=generation, teacher_seed_cursor=self.cursor,
                    readiness=copy.deepcopy(self.receipt), weight=copy.deepcopy(self.weight), buffer=self.buffer.payload(),
                    **({'soft_policy': soft} if soft is not None else {}))

    # g3658: league.commit_cleanups hands over the state it just wrote (no second read of state.json).
    committed_state = True

    def committed(self, generation, *, state=None): self.buffer.committed(generation, state=state)
