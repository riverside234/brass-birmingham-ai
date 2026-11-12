"""Policy-only KL inside the existing PPO step; no optimizer of its own."""
from collections import Counter
from contextlib import contextmanager
import copy
import time

import numpy as np
import torch

from brass_ai.learning.batching import collate
from brass_ai.learning.contracts import require
from .teacher_buffer import sample_indices
from .teacher_labels import PHASES, softmax

ACTOR_PREFIXES = ('state_encoder.', 'entity_encoder.', 'entity_projections.', 'action_encoder.', 'policy_scorer.')


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


def forward_policy(model, rows):
    states, actions, mask, _ = collate([r.encoded for r in rows], next(model.parameters()).device)
    # Do not construct any descriptor, outcome, utility or auxiliary graph.
    encoded = model.encode_state(states)
    return model.score_prepared(actions, model.prepare_candidates(encoded)).masked_fill(~mask, -torch.inf)


def policy_logits(model, rows):
    values = []
    with policy_context(model), torch.no_grad():
        for start in range(0, len(rows), 64):
            part = rows[start:start + 64]; logits = forward_policy(model, part).float().cpu().numpy()
            for r, l in zip(part, logits):
                l = l[:len(r.label['legal_ids'])].copy()
                require(np.isfinite(l).all(), 'Non-finite teacher policy logits')
                values.append(l)
    return values


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


def heldout_measure(model, rows):
    """Long-label KL by phase, as before g1491; short-search labels under 'short' when present."""
    logits = policy_logits(model, rows); out = {}
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


def weighted_gradients(model, rows, weights, parameters):
    """Exact gradients of sum_i w_i KL(target_i || policy_i), accumulated over chunks."""
    sums = [None] * len(parameters); total = 0.; per_row = [0.] * len(rows)
    for part in chunks(rows):
        selected = [rows[i] for i in part]
        logits = forward_policy(model, selected).float()
        target = torch.zeros_like(logits)
        for j, r in enumerate(selected): target[j, :len(r.label['target'])] = target.new_tensor(r.label['target'])
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


class TeacherTerm:
    def __init__(self, model, config, aged_rows, *, seed, generation, state=None):
        self.config = config; self.mode = config['mode']; self.seed = seed; self.generation = generation
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
            if self.adaptive_enabled:
                basis = self.beta_basis()
                prior_basis = None
                if target != old_target:
                    # Retargeting reseals the same gradient observations. Their
                    # committed basis still owns the running beta and norms.
                    prior_basis = [old_target, (saved_qualification or {}).get('sha256'), basis[2]]
                    self.outside = 0
                self.adaptive = adaptive.restore(self.state.get(adaptive.KEY), self.beta, basis, prior_basis=prior_basis)
                self.beta = self.adaptive['beta']
        self.beta_used = 1. if self.mode == 'shadow' else self.beta
        self.rows = []; self.ages = []; self.staleness = {}; self.steps = []; self.seconds = 0.
        self.presentations = Counter(); self.used = set(); self.improvement_by_phase = {}
        from .training_revision_g1491 import KEY as REVISION
        # g1491: L = a mean(L_long) + (1-a) mean(L_short); None keeps one long-only pool
        # (every earlier recipe, and the long-only g1490 restart variant).
        self.long_weight = config[REVISION]['labels'].get('long_weight') if REVISION in config else None
        if self.mode != 'off':
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

    def beta_basis(self):
        return [self.config['term']['share_target'], (self.qualification or {}).get('sha256'),
                (self.remeasurement or {}).get('measured_generation')]

    def apply(self, model, actor, step):
        if not self.active: return
        began = time.perf_counter()
        mix = None
        if self.long_weight is None:
            indices = sample_indices(self.ages, self.config['buffer']['age_weights'], self.config['term']['rows_per_step'],
                                     seed=self.seed, generation=self.generation, step=step)
        else:
            indices, row_weights, long_count = self.stratified(step)
        rows = [self.rows[i] for i in indices]
        parameters = tuple(p for n, p in model.named_parameters() if n.startswith(ACTOR_PREFIXES) and p.requires_grad)
        with policy_context(model):
            if self.long_weight is None:
                logits = forward_policy(model, rows).float()
                target = torch.zeros_like(logits)
                for i, r in enumerate(rows): target[i, :len(r.label['target'])] = target.new_tensor(r.label['target'])
                logp = logits.log_softmax(-1).masked_fill(~torch.isfinite(logits), 0.)
                loss = (torch.special.xlogy(target, target) - target * logp).sum(-1).mean()
                require(bool(torch.isfinite(loss)), 'Non-finite teacher loss')
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
            else:
                # 2048 rows do not fit one forward: exact gradient sums over PPO-sized chunks.
                total, gradients, per_row = weighted_gradients(model, rows, row_weights, parameters)
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
        if self.mode == 'on':
            for p, g in zip(parameters, gradients):
                if g is None: continue
                g = g.detach().mul(self.beta_used)
                if p.grad is None: p.grad = g
                else: p.grad.add_(g)
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
            outside_share_warning=self.outside >= 3, term_seconds=self.seconds,
            scope='Policy term before PPO clipping/Adam/KL guard; shadow never applies gradients')
