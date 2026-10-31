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


def heldout_measure(model, rows):
    logits = policy_logits(model, rows); out = {}
    for phase in ('all', *PHASES, *(f'strategy_{z}' for z in range(7))):
        pairs = [(r, l) for r, l in zip(rows, logits) if phase == 'all' or r.label['phase'] == phase
                 or phase == 'strategy_'+str(r.label.get('strategy_id',0))]
        out[phase] = dict(rows=len(pairs), kl=float(np.mean([kl(r.label['target'], softmax(l)) for r, l in pairs])) if pairs else None,
            top1=float(np.mean([np.argmax(r.label['target']) == np.argmax(l) for r, l in pairs])) if pairs else None)
    return out


class TeacherTerm:
    def __init__(self, model, config, aged_rows, *, seed, generation, state=None):
        self.config = config; self.mode = config['mode']; self.seed = seed; self.generation = generation
        self.state = copy.deepcopy(state or {})
        self.beta = self.state.get('beta', 1.); self.calibrated = self.state.get('calibrated', False)
        self.shadow = self.state.get('shadow_shares', []); self.outside = self.state.get('outside_streak', 0)
        require(np.isfinite(self.beta) and .01 <= self.beta <= 10, 'Invalid teacher beta state')
        target = config['term']['share_target']; old_target = self.state.get('share_target', target)
        if self.mode == 'on' and self.shadow and (self.state.get('mode') == 'shadow' or not self.calibrated):
            values = [v for iteration in self.shadow for v in iteration if v > 0]
            if values:
                self.beta = float(np.clip(target / np.median(values), *config['term']['beta_bounds']))
                self.calibrated = True
        elif self.calibrated and target != old_target:
            self.beta = float(np.clip(self.beta * target / old_target, *config['term']['beta_bounds']))
        self.beta_used = 1. if self.mode == 'shadow' else self.beta
        self.rows = []; self.ages = []; self.staleness = {}; self.steps = []; self.seconds = 0.
        self.presentations = Counter(); self.used = set(); self.improvement_by_phase = {}
        if self.mode != 'off':
            logits = policy_logits(model, [r for _, r in aged_rows])
            for age in sorted({a for a, _ in aged_rows}):
                selected = [(r, l) for (a, r), l in zip(aged_rows, logits) if a == age]
                improve = float(np.mean([kl(r.label['target'], softmax(r.label['logits'])) for r, _ in selected]))
                drift = float(np.mean([kl(softmax(r.label['logits']), softmax(l)) for r, l in selected]))
                skip = drift > config['buffer']['max_staleness_ratio'] * improve
                self.staleness[str(age)] = dict(rows=len(selected), improvement=improve, drift=drift,
                    ratio=drift / improve if improve > 0 else None, skipped=bool(skip))
                self.staleness[str(age)]['by_strategy']={str(z):dict(rows=len(group),
                    correction_kl=float(np.mean([kl(r.label['target'],softmax(r.label['logits'])) for r,l in group])),
                    policy_drift=float(np.mean([kl(softmax(r.label['logits']),softmax(l)) for r,l in group])))
                    for z in range(7) if (group:=[(r,l) for r,l in selected if r.label.get('strategy_id',0)==z])}
                if not skip:
                    self.rows.extend(r for r, _ in selected); self.ages.extend([age] * len(selected))
            for phase in PHASES:
                values = [kl(r.label['target'], softmax(r.label['logits'])) for a, r in aged_rows if a == 0 and r.label['phase'] == phase]
                self.improvement_by_phase[phase] = float(np.mean(values)) if values else None

    @property
    def active(self): return self.mode != 'off' and bool(self.rows)

    def apply(self, model, actor, step):
        if not self.active: return
        began = time.perf_counter()
        indices = sample_indices(self.ages, self.config['buffer']['age_weights'], self.config['term']['rows_per_step'],
                                 seed=self.seed, generation=self.generation, step=step)
        rows = [self.rows[i] for i in indices]
        parameters = tuple(p for n, p in model.named_parameters() if n.startswith(ACTOR_PREFIXES) and p.requires_grad)
        with policy_context(model):
            logits = forward_policy(model, rows).float()
            target = torch.zeros_like(logits)
            for i, r in enumerate(rows): target[i, :len(r.label['target'])] = target.new_tensor(r.label['target'])
            logp = logits.log_softmax(-1).masked_fill(~torch.isfinite(logits), 0.)
            loss = (torch.special.xlogy(target, target) - target * logp).sum(-1).mean()
            require(bool(torch.isfinite(loss)), 'Non-finite teacher loss')
            gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        by_parameter = {id(p): g for p, g in zip(parameters, gradients)}
        zero = next(model.parameters()).new_zeros(())
        an, tn, dot = zero, zero, zero
        for p, a in zip(actor.parameters, actor.actor):
            g = by_parameter.get(id(p))
            an = an + a.square().sum()
            if g is not None: tn = tn + g.square().sum(); dot = dot + (a * g).sum()
        a, t, d = torch.stack((an.sqrt(), tn.sqrt(), dot)).double().detach().cpu().tolist()
        require(all(np.isfinite(v) for v in (a, t, d)), 'Non-finite teacher gradients')
        raw_share = t / a if a > 0 else None
        if self.mode == 'on':
            for p, g in zip(parameters, gradients):
                if g is None: continue
                g = g.detach().mul(self.beta_used)
                if p.grad is None: p.grad = g
                else: p.grad.add_(g)
        self.steps.append(dict(step=step, rows=len(rows), loss=float(loss.detach()), raw_share=raw_share,
            share=raw_share * self.beta_used if raw_share is not None else None,
            cosine=d / (a * t) if a * t > 0 else None, actor_norm=a, teacher_norm=t * self.beta_used))
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
        archive=copy.deepcopy(self.state.get('archive'))
        self.state = dict(mode=self.mode, beta=self.beta, calibrated=self.calibrated,
            share_target=self.config['term']['share_target'], shadow_shares=self.shadow, outside_streak=self.outside)
        if archive is not None:self.state['archive']=archive
        return copy.deepcopy(self.state)

    def report(self):
        def counts(key):
            return dict(Counter({str(k): sum(n for i, n in self.presentations.items() if key(i) == k)
                                 for k in {key(i) for i in self.presentations}}))
        return dict(mode=self.mode, beta=self.beta_used, beta_next=self.beta, steps=len(self.steps), rows=len(self.rows),
            share=summary([s['share'] for s in self.steps]), raw_share=summary([s['raw_share'] for s in self.steps]),
            cosine=summary([s['cosine'] for s in self.steps]), staleness=self.staleness,
            improvement_by_phase=self.improvement_by_phase, presentations=sum(self.presentations.values()),
            distinct_rows=len(self.used), presentations_by_age=counts(lambda i: self.ages[i]),
            presentations_by_phase=counts(lambda i: self.rows[i].label['phase']),
            presentations_by_strategy=counts(lambda i: self.rows[i].label.get('strategy_id',0)),
            distinct_by_age=dict(Counter(str(self.ages[i]) for i in self.used)),
            distinct_by_phase=dict(Counter(self.rows[i].label['phase'] for i in self.used)),
            first_loss=self.steps[0]['loss'] if self.steps else None, last_loss=self.steps[-1]['loss'] if self.steps else None,
            outside_share_warning=self.outside >= 3, term_seconds=self.seconds,
            scope='Policy term before PPO clipping/Adam/KL guard; shadow never applies gradients')
