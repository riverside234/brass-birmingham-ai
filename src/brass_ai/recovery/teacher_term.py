"""Policy-only KL inside the existing PPO step; no optimizer of its own."""
from collections import Counter
from contextlib import contextmanager
import copy
import time

import numpy as np
import torch

from brass_ai.learning.batching import collate
from brass_ai.learning.contracts import digest, require
from .soft_policy import capture_hidden, single
from .teacher_buffer import sample_indices
from .teacher_labels import PHASES, softmax

ACTOR_PREFIXES = ('state_encoder.', 'entity_encoder.', 'entity_projections.', 'action_encoder.', 'policy_scorer.')
# g3270: one PPO update's label batches (no repeated row, grouped by legal width).
BATCH_CONTRACT = 'teacher-label-batches/1.0.0'
STALENESS_CONTRACT = 'teacher-staleness-sample/1.0.0'


def search_bonus(label, scale):
    """scale * (c1 + most visits) * completed value: the search's preference over the prior.

    The completed values are those of improved_policy (search value where visited, the mixed
    value elsewhere). softmax(stored logits + search_bonus(label, label['c2'])) is the
    exported target; g3270 adds the bonus to the current logits instead.
    """
    logits, counts, means = (np.asarray(label[k], dtype=np.float64) for k in ('logits', 'counts', 'mean_returns'))
    value = float(label['node_value'])
    visited = counts > 0; total = counts.sum()
    mixed = (value + total * np.dot(softmax(logits[visited]), means[visited])) / (1 + total) if total else value
    completed = np.where(visited, means, mixed)
    bonus = scale * (label['c1'] + counts.max()) * completed
    require(np.isfinite(bonus).all(), 'Non-finite teacher search bonus')
    return bonus


def rebuilt_target(logits, bonus):
    """softmax(current logits + bonus), held fixed for the gradient. Padding (-inf logits) stays zero."""
    with torch.no_grad():
        return (logits.detach() + bonus).softmax(-1)


@contextmanager
def policy_context(model):
    device = next(model.parameters()).device
    # Current actor is deterministic. Also protect ordinary minibatch RNG if a
    # future encoder gains stochastic operations; shadow must remain inert.
    devices = [device.index or 0] if device.type == 'cuda' else []
    was_training = model.training
    with torch.random.fork_rng(devices=devices):
        model.eval()
        try: yield
        finally: model.train(was_training)


def _score_policy(model, states, actions, mask):
    # Do not construct any descriptor, outcome, utility or auxiliary graph.
    encoded = model.encode_state(states)
    return model.score_prepared(actions, model.prepare_candidates(encoded)).masked_fill(~mask, -torch.inf)


def forward_policy(model, rows, *, audit=None):
    states, actions, mask, _ = collate([r.encoded for r in rows], next(model.parameters()).device, audit=audit)
    return _score_policy(model, states, actions, mask)


def target_tensor(rows, logits, values=None):
    """One host-to-device transfer, replacing one transfer per teacher row.

    ``values`` replaces the exported targets with other per-action vectors (the g3270 bonus).
    """
    target = np.zeros(tuple(logits.shape), dtype=np.float32)
    for i, row in enumerate(rows):
        vector = row.label['target'] if values is None else values[i]
        target[i, :len(vector)] = vector
    return torch.as_tensor(target, device=logits.device)


def policy_logits(model, rows, *, batches=None, timings=None):
    values = []
    if batches is None:
        batches = (rows[start:start + 64] for start in range(0, len(rows), 64))
    with policy_context(model), torch.no_grad():
        for part in batches:
            if timings is None:
                logits = forward_policy(model, part).float().cpu().numpy()
            else:
                began = time.perf_counter()
                states, actions, mask, _ = collate([r.encoded for r in part], next(model.parameters()).device)
                staged = time.perf_counter()
                output = _score_policy(model, states, actions, mask).float()
                launched = time.perf_counter()
                logits = output.cpu().numpy()
                timings['collate_seconds'] += staged - began
                timings['forward_dispatch_seconds'] += launched - staged
                timings['transfer_and_device_wait_seconds'] += time.perf_counter() - launched
            for r, l in zip(part, logits):
                l = l[:len(r.label['legal_ids'])].copy()
                require(np.isfinite(l).all(), 'Non-finite teacher policy logits')
                values.append(l)
    return values


class StagedRows:
    """g3140 step 1: every teacher row's inputs and target on the training device once per iteration.

    A step gathers its padded batch on the device instead of packing it on the host
    and copying it over. The tensors equal collate()/target_tensor() exactly: zero
    actions, False mask and zero target past each row's legal width. Each row is
    admitted once (InputAudit), as collate would on its first use. ``values`` stages
    other per-action vectors in place of the exported targets (the g3270 search bonus).
    """
    LIMIT_BYTES = 2 * 1024 ** 3

    def __init__(self, rows, device, audit, values=None):
        for r in rows: audit.check(r.encoded)
        self.widths = [len(r.encoded.choice_ids) for r in rows]
        offsets = np.cumsum([1] + self.widths[:-1])
        actions = np.concatenate([np.zeros((1, rows[0].encoded.actions.shape[-1]), np.float32)] + [r.encoded.actions for r in rows])
        vectors = [r.label['target'] for r in rows] if values is None else values
        targets = np.concatenate([np.zeros(1, np.float32)] + [np.asarray(v, np.float32) for v in vectors])
        self.states = torch.as_tensor(np.stack([r.encoded.state for r in rows]), device=device)
        self.actions = torch.as_tensor(np.ascontiguousarray(actions, dtype=np.float32), device=device)
        self.targets = torch.as_tensor(targets, device=device)
        self.offsets = torch.as_tensor(offsets, dtype=torch.int64, device=device)
        self.lengths = torch.as_tensor(self.widths, dtype=torch.int64, device=device)
        self.bytes = sum(t.numel() * t.element_size() for t in (self.states, self.actions, self.targets))

    @classmethod
    def build(cls, rows, device, audit, values=None):
        # Duck-typed inputs, and any label narrower or wider than its encoded choices, keep the host collate path.
        if not rows or any(not isinstance(getattr(r.encoded, 'state', None), np.ndarray)
                           or not isinstance(getattr(r.encoded, 'actions', None), np.ndarray)
                           or len(r.encoded.choice_ids) != len(r.label['target']) for r in rows): return None
        if values is not None and any(len(v) != len(r.label['target']) for r, v in zip(rows, values, strict=True)): return None
        size = sum(len(r.encoded.choice_ids) for r in rows) * rows[0].encoded.actions.shape[-1] * 4
        return cls(rows, device, audit, values) if size <= cls.LIMIT_BYTES else None

    def batch(self, indices):
        width = max(self.widths[i] for i in indices)
        selected = torch.as_tensor(indices, dtype=torch.int64, device=self.states.device)
        position = torch.arange(width, device=selected.device)
        mask = position[None, :] < self.lengths[selected][:, None]
        index = torch.where(mask, self.offsets[selected][:, None] + position[None, :], torch.zeros_like(position)[None, :])
        return self.states[selected], self.actions[index], mask, self.targets[index]


def kl(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    use = p > 0
    return float(np.sum(p[use] * (np.log(p[use]) - np.log(np.maximum(q[use], 1e-300)))))


def summary(values):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return (dict(median=float(np.median(finite)), p10=float(np.quantile(finite, .1)),
                 p90=float(np.quantile(finite, .9))) if finite else dict(median=None, p10=None, p90=None))


def profile(row):
    """'full' for a long-search label, 'fast' for a short-search one (g1491)."""
    return row.label.get('profile', 'full')


def movement(rows, before, after):
    """Mean KL(policy before || policy after) on the same long-label rows, overall and by phase."""
    pairs = [(r, b, a) for r, b, a in zip(rows, before, after, strict=True) if profile(r) == 'full']
    out = {}
    for phase in ('all', *PHASES):
        values = [kl(softmax(b), softmax(a)) for r, b, a in pairs if phase == 'all' or r.label['phase'] == phase]
        out[phase] = float(np.mean(values)) if values else None
    return out


def heldout_measure(model, rows, *, logits=None):
    """Long-label KL by phase, as before g1491; short-search labels under 'short' when present."""
    logits = policy_logits(model, rows) if logits is None else logits; out = {}
    def measure(pairs):
        return dict(rows=len(pairs), kl=float(np.mean([kl(r.label['target'], softmax(l)) for r, l in pairs])) if pairs else None,
            top1=float(np.mean([np.argmax(r.label['target']) == np.argmax(l) for r, l in pairs])) if pairs else None)
    long = [(r, l) for r, l in zip(rows, logits) if profile(r) == 'full']
    for phase in ('all', *PHASES):
        out[phase] = measure([(r, l) for r, l in long if phase == 'all' or r.label['phase'] == phase])
    short = [(r, l) for r, l in zip(rows, logits) if profile(r) == 'fast']
    if short: out['short'] = measure(short)
    return out


def chunks(rows, max_rows=512, max_cells=131072):
    """Index lists in increasing legal width, within the PPO microbatch bounds."""
    order = sorted(range(len(rows)), key=lambda i: (len(rows[i].label['legal_ids']), i))
    result = []; part = []; width = 0
    for i in order:
        wide = max(width, len(rows[i].label['legal_ids']))
        if part and (len(part) + 1 > max_rows or (len(part) + 1) * wide > max_cells):
            result.append(part); part = []; wide = len(rows[i].label['legal_ids'])
        part.append(i); width = wide
    if part: result.append(part)
    return result


def weighted_gradients(model, rows, weights, parameters, *, audit=None, max_cells=131072):
    """Exact gradients of sum_i w_i KL(target_i || policy_i), accumulated over chunks."""
    sums = [None] * len(parameters); total = 0.; per_row = [0.] * len(rows)
    for part in chunks(rows, max_cells=max_cells):
        selected = [rows[i] for i in part]
        logits = forward_policy(model, selected, audit=audit).float()
        target = target_tensor(selected, logits)
        logp = logits.log_softmax(-1).masked_fill(~torch.isfinite(logits), 0.)
        losses = (torch.special.xlogy(target, target) - target * logp).sum(-1)
        loss = (losses * losses.new_tensor([weights[i] for i in part])).sum()
        require(bool(torch.isfinite(loss)), 'Non-finite teacher loss')
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        for k, g in enumerate(gradients):
            if g is not None: sums[k] = g if sums[k] is None else sums[k] + g
        total += float(loss.detach())
        for i, value in zip(part, losses.detach().double().cpu().tolist()): per_row[i] = value
    return total, sums, per_row


def staleness_sample(rows, count, *, seed, generation, age):
    """A fixed sample of one age's rows for the drift measurement; every row when count is None."""
    if count is None or len(rows) <= count: return list(rows)
    key = digest(dict(contract=STALENESS_CONTRACT, seed=seed, generation=generation, age=age))
    picked = np.random.default_rng(int(key[:16], 16)).choice(len(rows), count, replace=False)
    return [rows[i] for i in sorted(picked.tolist())]


class TeacherTerm:
    def __init__(self, model, config, aged_rows, *, seed, generation, state=None, revision=None, soft=None):
        """``revision`` (g3270): rebuilt targets at a label scale, planned no-repeat batches, sampled staleness.

        ``soft`` (S1 of the soft policy plan): a ``soft_policy.SoftPolicy`` trained on the same drawn rows.
        """
        self.config = config; self.mode = config['mode']; self.seed = seed; self.generation = generation
        self.revision = copy.deepcopy(revision); self.soft = soft
        self.batches = None; self.plan_report = None; self.bonus = None; self.position = None
        self.state = copy.deepcopy(state or {})
        self.beta = self.state.get('beta', 1.); self.calibrated = self.state.get('calibrated', False)
        from .teacher_gradient_qualification import restore
        saved_qualification = self.state.get('gradient_qualification')
        self.qualification=restore(saved_qualification,config['term']['share_target'],config['term']['shadow_iterations'])
        self.shadow = self.state.get('shadow_shares', []); self.outside = self.state.get('outside_streak', 0)
        require(np.isfinite(self.beta) and .01 <= self.beta <= 10, 'Invalid teacher beta state')
        target = config['term']['share_target']; old_target = self.state.get('share_target', target)
        if self.qualification is not None and self.qualification['passed']:
            self.beta=self.qualification['beta'];self.calibrated=True
        elif self.mode=='on' and self.qualification is not None:
            raise ValueError(f"{config['term']['shadow_iterations']} full shadow gradient qualifications are required before using fixed beta")
        elif self.mode=='shadow' and self.qualification is not None:
            self.beta=1.;self.calibrated=False
        elif self.mode == 'on' and self.shadow and (self.state.get('mode') == 'shadow' or not self.calibrated):
            values = [v for iteration in self.shadow for v in iteration if v > 0]
            if values:
                self.beta = float(np.clip(target / np.median(values), *config['term']['beta_bounds']))
                self.calibrated = True
        elif self.calibrated and target != old_target:
            self.beta = float(np.clip(self.beta * target / old_target, *config['term']['beta_bounds']))
        from . import adaptive_teacher_beta as adaptive
        self.remeasurement = self.state.get(adaptive.REMEASUREMENT) if self.mode == 'on' else None
        self.remeasure_pending = adaptive.pending(self.state) and self.mode == 'on'
        self.adaptive_enabled = self.mode == 'on' and adaptive.KEY in config
        self.adaptive = None
        self.beta_report = {}
        if self.mode == 'on':
            self.beta = adaptive.accepted(self.beta, self.remeasurement, target)
            if self.remeasure_pending:
                self.shadow, self.outside = [], 0
            if self.adaptive_enabled:
                basis = self.beta_basis()
                prior_basis = None
                if target != old_target:
                    # Retargeting reseals the same gradient observations. Their
                    # committed basis still owns the running beta and norms.
                    prior_basis = [old_target, (saved_qualification or {}).get('sha256'), basis[2]]
                    self.outside = 0
                saved_adaptive = None if self.remeasure_pending else self.state.get(adaptive.KEY)
                self.adaptive = adaptive.restore(saved_adaptive, self.beta, basis, prior_basis=prior_basis)
                self.beta = self.adaptive['beta']
        self.beta_used = 1. if self.mode == 'shadow' else self.beta
        self.rows = []; self.ages = []; self.staleness = {}; self.steps = []; self.seconds = 0.
        self.staged = None; self.staging = dict(staged=False, bytes=0, seconds=0.)
        from .batch_cache import InputAudit
        self.input_audit = InputAudit()
        self.presentations = Counter(); self.used = set(); self.improvement_by_phase = {}
        from .training_revision_g1491 import KEY as REVISION
        # g1491: L = a mean(L_long) + (1-a) mean(L_short); None keeps one long-only pool
        # (every earlier recipe, and the long-only g1490 restart variant).
        self.long_weight = config[REVISION]['labels'].get('long_weight') if REVISION in config else None
        require(self.revision is None or self.long_weight is None, 'g3270 rebuilt targets use long-search labels only')
        require(soft is None or (self.revision is not None and self.long_weight is None),
                'The soft policy head trains on the g3270 long-label path only (no stratified long-and-short path)')
        if self.mode != 'off' and self.revision is None:
            logits = policy_logits(model, [r for _, r in aged_rows])
            for age in sorted({a for a, _ in aged_rows}):
                selected = [(r, l) for (a, r), l in zip(aged_rows, logits) if a == age]
                improve = float(np.mean([kl(r.label['target'], softmax(r.label['logits'])) for r, _ in selected]))
                drift = float(np.mean([kl(softmax(r.label['logits']), softmax(l)) for r, l in selected]))
                skip = drift > config['buffer']['max_staleness_ratio'] * improve
                self.staleness[str(age)] = dict(rows=len(selected), improvement=improve, drift=drift,
                    ratio=drift / improve if improve > 0 else None, skipped=bool(skip))
                if not skip:
                    self.rows.extend(r for r, _ in selected); self.ages.extend([age] * len(selected))
        elif self.mode != 'off':
            # The same per-age rule; the drift is measured on a fixed sample of each age (8 ages, ~10,000 rows).
            groups = {}
            for a, r in aged_rows: groups.setdefault(a, []).append(r)
            probes = {age: staleness_sample(rows, self.revision['staleness_sample_rows'], seed=seed,
                                            generation=generation, age=age) for age, rows in groups.items()}
            ordered = sorted(groups)
            logits = policy_logits(model, [r for age in ordered for r in probes[age]]); cursor = 0
            for age in ordered:
                rows, probe = groups[age], probes[age]
                current = logits[cursor:cursor + len(probe)]; cursor += len(probe)
                improve = float(np.mean([kl(r.label['target'], softmax(r.label['logits'])) for r in rows]))
                drift = float(np.mean([kl(softmax(r.label['logits']), softmax(l)) for r, l in zip(probe, current, strict=True)]))
                skip = drift > config['buffer']['max_staleness_ratio'] * improve
                self.staleness[str(age)] = dict(rows=len(rows), sampled=len(probe), improvement=improve, drift=drift,
                    ratio=drift / improve if improve > 0 else None, skipped=bool(skip))
                if not skip:
                    self.rows.extend(rows); self.ages.extend([age] * len(rows))
            self.bonus = [search_bonus(r.label, self.revision['label_scale']).astype(np.float32) for r in self.rows]
        if self.mode != 'off':
            for phase in PHASES:
                values = [kl(r.label['target'], softmax(r.label['logits'])) for a, r in aged_rows
                          if a == 0 and r.label['phase'] == phase and profile(r) == 'full']
                self.improvement_by_phase[phase] = float(np.mean(values)) if values else None
        self.profiles = [profile(r) for r in self.rows]
        require(self.long_weight is not None or all(p == 'full' for p in self.profiles),
                'Short-search teacher labels need the g1491 recipe')

    def stratified(self, step):
        """Age-weighted draws from each pool; per-row weights give each pool its share of the loss."""
        n = self.config['term']['rows_per_step']; weights = self.config['buffer']['age_weights']
        long = [i for i, p in enumerate(self.profiles) if p == 'full']
        short = [i for i, p in enumerate(self.profiles) if p == 'fast']
        # A missing pool gives the other one the whole loss, so beta keeps its scale.
        if long and short: want, share = round(self.long_weight * n), self.long_weight
        else: want, share = (n, 1.) if long else (0, 0.)
        def draw(pool, count, stream):
            picked = sample_indices([self.ages[i] for i in pool], weights, count, seed=self.seed,
                                    generation=self.generation, step=step, stream=stream)
            return [pool[j] for j in picked]
        picked_long, picked_short = draw(long, want, 'long'), draw(short, n - want, 'short')
        row_weights = ([share / len(picked_long)] * len(picked_long) if picked_long else []) + \
            ([(1 - share) / len(picked_short)] * len(picked_short) if picked_short else [])
        return picked_long + picked_short, row_weights, len(picked_long)

    @property
    def active(self): return self.mode != 'off' and bool(self.rows)

    def plan(self, steps):
        """g3270: fix this PPO update's label batches before its first step.

        Rows are drawn by age weight without replacement (exponential keys divided by the
        weight), so no row repeats inside the update. The drawn rows are sorted by legal
        width and cut into batches of ``rows_per_step``, whose order is then shuffled, so a
        step pads little. A pool smaller than the update's demand gives every row once, in
        batches of ceil(pool / steps).
        """
        self.batches = None; self.plan_report = None
        if self.revision is None or not self.active or type(steps) is not int or steps <= 0: return
        n = self.config['term']['rows_per_step']; weights = self.config['buffer']['age_weights']
        key = digest(dict(contract=BATCH_CONTRACT, seed=self.seed, generation=self.generation, steps=steps, rows=len(self.rows)))
        rng = np.random.default_rng(int(key[:16], 16))
        keys = rng.exponential(size=len(self.rows)) / np.asarray([weights[a] for a in self.ages], dtype=np.float64)
        order = np.argsort(keys, kind='stable')
        demand = steps * n
        chosen, size = (order[:demand], n) if demand < len(order) else (order, -(-len(order) // steps))
        widths = np.asarray([len(self.rows[i].label['legal_ids']) for i in chosen])
        ranked = chosen[np.lexsort((keys[chosen], widths))]
        cut = [ranked[i:i + size].tolist() for i in range(0, len(ranked), size)]
        self.batches = [cut[j] for j in rng.permutation(len(cut)).tolist()]
        width = lambda i: len(self.rows[i].label['legal_ids'])
        cells = sum(max(width(i) for i in b) * len(b) for b in self.batches)
        ages = Counter(self.ages)
        drawn = Counter(self.ages[i] for i in chosen.tolist())
        self.plan_report = dict(contract=BATCH_CONTRACT, planned_steps=steps, rows_per_step=n, pool_rows=len(self.rows),
            demand=demand, drawn_rows=len(chosen), batches=len(self.batches), batch_rows=size,
            padding_efficiency=float(widths.sum() / cells) if cells else None,
            inclusion_by_age={str(a): drawn[a] / ages[a] for a in sorted(ages)},
            scope='Each drawn row is used once in this PPO update; batches stop early if PPO does')

    def beta_basis(self):
        return [self.config['term']['share_target'], (self.qualification or {}).get('sha256'),
                (self.remeasurement or {}).get('measured_generation')]

    def apply(self, model, actor, step):
        if not self.active: return
        if self.batches is not None and step - 1 >= len(self.batches): return  # a small pool is used up
        began = time.perf_counter()
        mix = None
        if self.batches is not None:
            indices = self.batches[step - 1]
        elif self.long_weight is None:
            indices = sample_indices(self.ages, self.config['buffer']['age_weights'], self.config['term']['rows_per_step'],
                                     seed=self.seed, generation=self.generation, step=step)
        else:
            indices, row_weights, long_count = self.stratified(step)
        rows = [self.rows[i] for i in indices]
        parameters = tuple(p for n, p in model.named_parameters() if n.startswith(ACTOR_PREFIXES) and p.requires_grad)
        soft = self.soft if self.long_weight is None else None; soft_seconds = 0.
        with policy_context(model):
            if self.long_weight is None:
                # One padded forward per step, however wide the rows. Splitting wide
                # steps into 32k-cell chunks (g2400) repeated the launch-bound
                # encoder forward/backward per chunk: 20-30 s per PPO pass, not 1-3 s.
                if self.staged is None and not self.staging.get('fallback'):
                    staged_at = time.perf_counter()
                    if self.batches is not None:
                        # g3270: stage only this update's drawn rows, with their search bonus.
                        used = sorted({i for b in self.batches for i in b})
                        self.position = {i: k for k, i in enumerate(used)}
                        self.staged = StagedRows.build([self.rows[i] for i in used], next(model.parameters()).device,
                                                       self.input_audit, values=[self.bonus[i] for i in used])
                    else:
                        self.staged = StagedRows.build(self.rows, next(model.parameters()).device, self.input_audit,
                                                       values=self.bonus)
                    self.staging = dict(staged=self.staged is not None, fallback=self.staged is None,
                        bytes=self.staged.bytes if self.staged else 0, seconds=time.perf_counter() - staged_at)
                # The soft head reads the scorer's hidden layer from this same forward (a hook, no extra pass).
                with capture_hidden(model, soft is not None) as hidden:
                    if self.staged is not None:
                        positions = [self.position[i] for i in indices] if self.position is not None else indices
                        states, actions, mask, target = self.staged.batch(positions)
                        logits = _score_policy(model, states, actions, mask).float()
                    else:
                        logits = forward_policy(model, rows, audit=self.input_audit).float()
                        target = target_tensor(rows, logits, None if self.bonus is None else [self.bonus[i] for i in indices])
                bonus = None
                if self.bonus is not None:
                    # g3270: the staged vector is the search bonus; the target is rebuilt from the current policy.
                    bonus = target
                    target = rebuilt_target(logits, target)
                logp = logits.log_softmax(-1).masked_fill(~torch.isfinite(logits), 0.)
                loss = (torch.special.xlogy(target, target) - target * logp).sum(-1).mean()
                require(bool(torch.isfinite(loss)), 'Non-finite teacher loss')
                if soft is not None:
                    soft_began = time.perf_counter()
                    soft_loss = soft.loss(logits, bonus, single(hidden))
                    soft_seconds += time.perf_counter() - soft_began
                # The teacher's own call, unchanged; it keeps the graph only for a soft loss that reaches the trunk.
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True,
                                                retain_graph=soft is not None and soft.trunk)
                if soft is not None:
                    soft_began = time.perf_counter()
                    # A call of its own: the head is not among `parameters`.
                    head_gradient, soft_gradients = soft.gradients(soft_loss, parameters)
                    soft_seconds += time.perf_counter() - soft_began
            else:
                # 2048 rows do not fit one forward: exact gradient sums over PPO-sized chunks.
                total, gradients, per_row = weighted_gradients(model, rows, row_weights, parameters, audit=self.input_audit)
                loss = torch.tensor(total)
                mean = lambda values: float(np.mean(values)) if values else None
                mix = dict(long_rows=long_count, short_rows=len(rows) - long_count,
                           long_loss=mean(per_row[:long_count]), short_loss=mean(per_row[long_count:]),
                           long_share=sum(row_weights[:long_count]))
        by_parameter = {id(p): g for p, g in zip(parameters, gradients)}
        zero = next(model.parameters()).new_zeros(())
        an, tn, dot = zero, zero, zero
        measured = getattr(actor, 'measure', True)
        if measured:
            for p, a in zip(actor.parameters, actor.actor):
                g = by_parameter.get(id(p))
                an = an + a.square().sum()
                if g is not None: tn = tn + g.square().sum(); dot = dot + (a * g).sum()
        else:
            # A step without the actor pass still reports the teacher's own norm.
            for p in actor.parameters:
                g = by_parameter.get(id(p))
                if g is not None: tn = tn + g.square().sum()
        a, t, d = torch.stack((an.sqrt(), tn.sqrt(), dot)).double().detach().cpu().tolist()
        require(all(np.isfinite(v) for v in (a, t, d)), 'Non-finite teacher gradients')
        raw_share = t / a if measured and a > 0 else None
        if soft is not None:
            soft_began = time.perf_counter()
            # Its weight is 8 x the policy loss already on these rows: 8 beta.
            scale = soft.weight * self.beta_used; soft.beta_weight = scale
            by_soft = {id(p): g for p, g in zip(parameters, soft_gradients)} if soft_gradients is not None else {}
            sn = zero
            for p in actor.parameters:
                g = by_soft.get(id(p))
                if g is not None: sn = sn + g.square().sum()
            sl, sh, st = torch.stack((soft_loss.detach().float(), head_gradient.detach().norm(), sn.sqrt())).double().cpu().tolist()
            require(all(np.isfinite(v) for v in (sl, sh, st)), 'Non-finite soft policy loss or gradients')
            soft_seconds += time.perf_counter() - soft_began
        if self.mode == 'on':
            for p, g in zip(parameters, gradients):
                if g is None: continue
                g = g.detach().mul(self.beta_used)
                if p.grad is None: p.grad = g
                else: p.grad.add_(g)
            if soft is not None:
                soft_began = time.perf_counter()
                for p, g in zip(parameters, soft_gradients or ()):
                    if g is None: continue
                    g = g.detach().mul(scale)
                    if p.grad is None: p.grad = g
                    else: p.grad.add_(g)
                soft.step(head_gradient, sh)
                soft_seconds += time.perf_counter() - soft_began
        if soft is not None:
            soft.record(step=step, loss=sl, head_norm=sh, raw_trunk_norm=st, trunk_norm=st * scale,
                        share=st * scale / a if measured and a > 0 else None)
            soft.seconds += soft_seconds
        self.steps.append(dict(step=step, rows=len(rows), loss=float(loss.detach()), raw_share=raw_share,
            share=raw_share * self.beta_used if raw_share is not None else None,
            cosine=d / (a * t) if measured and a * t > 0 else None, actor_norm=a if measured else None, teacher_norm=t * self.beta_used,
            **(mix or {})))
        if measured and (self.adaptive_enabled or self.remeasure_pending):
            self.steps[-1]['raw_teacher_norm'] = t
        for i in indices: self.presentations[i] += 1; self.used.add(i)
        self.seconds += time.perf_counter() - began

    def finish(self):
        raw = [s['raw_share'] for s in self.steps if s['raw_share'] is not None and s['raw_share'] > 0]
        if self.mode == 'shadow' and raw:
            self.shadow = (self.shadow + [raw])[-self.config['term']['shadow_iterations']:]
        if self.mode == 'on' and not self.calibrated and raw:
            self.beta = float(np.clip(self.config['term']['share_target'] / np.median(raw), *self.config['term']['beta_bounds']))
            self.calibrated = True
        share = summary([s['share'] for s in self.steps])['median']
        if self.mode == 'on' and share is not None:
            target = self.config['term']['share_target']
            self.outside = self.outside + 1 if share < target / 2 or share > target * 2 else 0
        self.state = dict(mode=self.mode, beta=self.beta, calibrated=self.calibrated,
            share_target=self.config['term']['share_target'], shadow_shares=self.shadow, outside_streak=self.outside)
        from .teacher_gradient_qualification import finish as qualify_gradient
        qualify_gradient(self,self.state)
        if self.mode == 'on' and (self.adaptive_enabled or self.remeasurement is not None):
            self.finish_beta()
        return copy.deepcopy(self.state)

    def finish_beta(self):
        from . import adaptive_teacher_beta as adaptive
        target = self.config['term']['share_target']
        pairs = adaptive.paired_norms(self.steps)
        actor = sum(a for a, _ in pairs) / len(pairs) if pairs else None
        teacher = sum(t for _, t in pairs) / len(pairs) if pairs else None
        if self.remeasure_pending:
            self.remeasurement = adaptive.remeasure(self, pairs, self.remeasurement)
        self.beta = adaptive.accepted(self.beta, self.remeasurement, target)
        if self.remeasurement is not None:
            self.state[adaptive.REMEASUREMENT] = self.remeasurement
            self.beta_report[adaptive.REMEASUREMENT] = self.remeasurement
        if self.adaptive_enabled:
            # A new measurement changes the basis and starts an empty EMA.
            controller = adaptive.restore(self.adaptive, self.beta, self.beta_basis())
            if not self.remeasure_pending:
                controller = adaptive.advance(controller, actor, teacher, target)
            self.state[adaptive.KEY] = self.adaptive = controller
            self.beta = controller['beta']
            self.beta_report[adaptive.KEY] = dict(share_target=target, beta_used=self.beta_used, beta_next=self.beta,
                paired_steps=len(pairs), actor_norm_mean=actor, raw_teacher_norm_mean=teacher,
                actor_ema=controller['actor_ema'], raw_teacher_ema=controller['teacher_ema'],
                updates=controller['updates'], ema_share=self.beta_used * controller['teacher_ema'] / controller['actor_ema']
                if controller['updates'] else None)
        self.state['beta'] = self.beta

    def report(self):
        from .adaptive_teacher_beta import accepted

        def counts(key):
            return dict(Counter({str(k): sum(n for i, n in self.presentations.items() if key(i) == k)
                                 for k in {key(i) for i in self.presentations}}))
        mix = {}
        if self.long_weight is not None:
            losses = lambda key: summary([s.get(key) for s in self.steps])
            mix = dict(long_weight=self.long_weight, rows_by_profile=dict(Counter(self.profiles)),
                presentations_by_profile=counts(lambda i: self.profiles[i]),
                distinct_by_profile=dict(Counter(self.profiles[i] for i in self.used)),
                long_loss=losses('long_loss'), short_loss=losses('short_loss'))
        qualified = self.qualification and self.qualification['passed']
        fixed = self.qualification['beta'] if qualified else None
        fixed = accepted(fixed, self.remeasurement, self.config['term']['share_target'])
        if self.soft is not None:
            mix = dict(mix, soft=self.soft.report())
        if self.revision is not None:
            mix = dict(mix, revision=dict(targets='rebuilt-from-current-policy', label_scale=self.revision['label_scale'],
                batches=self.revision['batches'], plan=copy.deepcopy(self.plan_report),
                loss=summary([s['loss'] for s in self.steps]),
                repeated_rows=sum(n > 1 for n in self.presentations.values())))
        return dict(mode=self.mode, beta=self.beta_used, beta_next=self.beta, steps=len(self.steps), rows=len(self.rows), **mix,
            rows_per_step=self.config['term']['rows_per_step'], **self.beta_report,
            fixed_beta=fixed,
            gradient_qualification=copy.deepcopy(self.qualification),
            projected_fixed_share=summary([s['raw_share']*fixed for s in self.steps if s['raw_share'] is not None])
                if fixed is not None else None,
            share=summary([s['share'] for s in self.steps]), raw_share=summary([s['raw_share'] for s in self.steps]),
            cosine=summary([s['cosine'] for s in self.steps]), staleness=self.staleness,
            improvement_by_phase=self.improvement_by_phase, presentations=sum(self.presentations.values()),
            distinct_rows=len(self.used), presentations_by_age=counts(lambda i: self.ages[i]),
            presentations_by_phase=counts(lambda i: self.rows[i].label['phase']),
            distinct_by_age=dict(Counter(str(self.ages[i]) for i in self.used)),
            distinct_by_phase=dict(Counter(self.rows[i].label['phase'] for i in self.used)),
            first_loss=self.steps[0]['loss'] if self.steps else None, last_loss=self.steps[-1]['loss'] if self.steps else None,
            outside_share_warning=self.outside >= 3, term_seconds=self.seconds, staged_rows=copy.deepcopy(self.staging),
            scope='Policy term before PPO clipping/Adam/KL guard; shadow never applies gradients')
