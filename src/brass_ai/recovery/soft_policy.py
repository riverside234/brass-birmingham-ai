"""Auxiliary soft policy target (KataGo style): step S1 of ai-research/SOFT_POLICY_TARGET_PLAN.md.

A trainer-owned ``Linear(hidden -> 1)`` without bias reads the policy scorer's GELU layer
(``policy_scorer.1``) and predicts the label target softened at temperature T:

    soft target = softmax((current logits + search bonus) / T)

on the teacher term's long-label rows (S1: no PPO rows). It is not on FourSeatNet, not
exported and never chooses a move. It starts as ``policy_scorer.2.weight / T``, so its first
output is softmax(main logits / T). A bias would only collect rounding noise: adding one
constant to every move changes no softmax, so its gradient is zero.

Gradient path. ``TeacherTerm.apply`` differentiates an explicit tuple of actor parameters and
the head is not on it, so the soft loss gets its own ``autograd.grad`` call: over the head,
and over the actor parameters once the trunk multiplier is above 0. The head steps its own
Adam (from the unweighted soft-loss gradient, clipped) inside the same ``apply``, only when
the teacher term applies gradients. The actor parameters receive 8 x beta x their soft
gradient beside the teacher's, before PPO's clip and Adam.

Warm-up. The trunk multiplier is 0 for the first ``head_only_iterations`` trained iterations:
the hidden layer is detached, so no soft gradient reaches the shared scorer or the trunk.
It then rises by 1/``ramp_iterations`` per trained iteration to 1 (``scale_gradient``).

State. JSON lists of exact float32 values, carried in the teacher phase's payload beside
``weight`` (state.json and the checkpoint hold the same dict): the head, its origin copy,
the Adam moments and step, and the warm-up counter.

g6230 (training_revision_g6230, ``fixed`` set by the teacher phase): the target is fixed per row,
softmax((stored collection logits + search bonus) / T) over the legal actions with visits > 0,
renormalized there (fixed_target); the head's log-softmax runs over the same actions; a row with
at most one visited action has weight 0 and no loss (fixed_loss); the held-out KL uses the same
targets (heldout_fixed), measured once after PPO.
"""
import copy
import math
import time
from contextlib import contextmanager

import numpy as np
import torch

from brass_ai.learning.contracts import digest, keys, require
from brass_ai.learning.local_scoring_heads import scale_gradient

CONTRACT = 'soft-policy-head/1.0.0'
BETAS = (.9, .999)
EPS = 1e-8
STATE_KEYS = ('contract', 'settings_sha256', 'origin', 'generation', 'width', 'head', 'origin_head', 'adam',
              'trained_iterations', 'head_steps', 'identity')


@contextmanager
def capture_hidden(model, enabled=True):
    """The policy scorer's GELU output of each forward run inside the block; None when disabled."""
    if not enabled:
        yield None
        return
    captured = []
    handle = model.policy_scorer[1].register_forward_hook(lambda module, inputs, output: captured.append(output))
    try:
        yield captured
    finally:
        handle.remove()


def single(captured):
    require(captured is not None and len(captured) == 1, 'The soft head needs exactly one policy-scorer forward')
    return captured[0]


def soften(probabilities, temperature):
    """KataGo's soft target of a stored target: p ** (1/T), renormalized; zero stays zero."""
    p = np.asarray(probabilities, dtype=np.float64) ** (1. / temperature)
    return p / p.sum()


def fixed_target(label, bonus, temperature, min_visited=2):
    """g6230: (target, visited, info) for one label row. target = softmax((label logits + bonus) / T) over the
    legal actions with visits > 0, renormalized there and zero elsewhere (float64); visited = 1.0 on those
    actions. info: the visited count, whether the row has a soft loss (at least `min_visited` visited) and the
    mass the unmasked softmax put on unvisited actions."""
    logits = np.asarray(label['logits'], dtype=np.float64); counts = np.asarray(label['counts'])
    bonus = np.asarray(bonus, dtype=np.float64)
    require(logits.shape == counts.shape == bonus.shape and np.isfinite(logits).all() and np.isfinite(bonus).all(),
            'Invalid soft target inputs')
    visited = counts > 0; n = int(visited.sum())
    z = (logits + bonus) / temperature; full = np.exp(z - z.max()); full /= full.sum()
    target = np.zeros(len(logits)); support = np.zeros(len(logits))
    if n:
        part = z[visited]; part = np.exp(part - part.max()); target[visited] = part / part.sum(); support[visited] = 1.
    return target, support, dict(visited=n, eligible=n >= min_visited, dropped=float(full[~visited].sum()))


def multiplier(trained, warmup):
    """Trunk multiplier after `trained` iterations with head steps: 0, 0, .2, .4, .6, .8, 1 (2 + 5)."""
    head_only, ramp = warmup['head_only_iterations'], warmup['ramp_iterations']
    return 0. if trained < head_only else min(1., (trained - head_only + 1) / ramp)


def stage(value):
    return 'head_only' if value == 0 else 'full' if value == 1 else 'ramp'


def identity(state):
    return digest({k: v for k, v in state.items() if k != 'identity'})


def carry(state, generation):
    """A committed state unchanged at a later generation (an iteration without a soft head)."""
    if state is None or state['generation'] == generation:
        return copy.deepcopy(state)
    result = {k: copy.deepcopy(v) for k, v in state.items() if k != 'identity'}
    result['generation'] = generation
    return dict(result, identity=identity(result))


def _floats(tensor):
    values = tensor.detach().float().cpu().tolist()
    require(all(math.isfinite(v) for v in values), 'Non-finite soft policy state')
    return values


def _summary(values):
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return (dict(median=float(np.median(finite)), p10=float(np.quantile(finite, .1)), p90=float(np.quantile(finite, .9)))
            if finite else dict(median=None, p10=None, p90=None))


class SoftPolicy:
    def __init__(self, model, settings, saved, *, generation):
        require(settings['ppo_row_factor'] == 0, 'PPO rows train the soft head only from step S4')
        self.settings = copy.deepcopy(settings); self.generation = generation
        self.temperature = float(settings['temperature']); self.weight = float(settings['weight'])
        self.learning_rate = float(settings['optimizer']['learning_rate'])
        self.clip = float(settings['optimizer']['gradient_clip'])
        scorer = model.policy_scorer
        require(isinstance(scorer[1], torch.nn.GELU) and isinstance(scorer[2], torch.nn.Linear) and scorer[2].out_features == 1,
                'The soft head reads a GELU -> Linear(width, 1) policy scorer')
        self.main = scorer[2].weight; width = scorer[2].in_features
        device = self.main.device
        if saved is None:
            require(generation == settings['from_generation'], 'A fresh soft head starts only at its adoption generation')
            start = (self.main.detach().reshape(-1).float() / self.temperature).clone()
            self.origin = dict(generation=generation)
            head, origin = start.clone(), start.clone()
            m, v, self.t, self.trained, self.total_steps = torch.zeros_like(start), torch.zeros_like(start), 0, 0, 0
            self.fresh = True
        else:
            keys(saved, STATE_KEYS)
            require(saved['contract'] == CONTRACT and saved['identity'] == identity(saved), 'Corrupt soft policy state')
            # At a weight change (g3471, g3528) the committed head was stamped under the settings before it.
            change = self.settings.get('weight_change')
            from .training_revision_g3423 import previous_settings
            stamped = (saved['settings_sha256'] == digest(self.settings) or
                       (change is not None and change['generation'] == generation
                        and saved['settings_sha256'] == digest(previous_settings(self.settings))))
            require(stamped and saved['generation'] == generation
                    and saved['width'] == width, 'Soft policy checkpoint/settings/progress differs')
            tensor = lambda values: torch.tensor(values, dtype=torch.float32, device=device)
            self.origin = copy.deepcopy(saved['origin'])
            head, origin = tensor(saved['head']), tensor(saved['origin_head'])
            m, v = tensor(saved['adam']['exp_avg']), tensor(saved['adam']['exp_avg_sq'])
            self.t, self.trained, self.total_steps = saved['adam']['step'], saved['trained_iterations'], saved['head_steps']
            require(all(len(x) == width for x in (saved['head'], saved['origin_head'], saved['adam']['exp_avg'], saved['adam']['exp_avg_sq']))
                    and type(self.t) is int and self.t == self.total_steps and type(self.trained) is int and self.trained >= 0,
                    'Soft policy state shape/counters differ')
            self.fresh = False
        self.head = head.to(device).requires_grad_(True)
        self.origin_head = origin.to(device); self.m = m.to(device); self.v = v.to(device)
        self.multiplier = multiplier(self.trained, settings['warmup'])
        self.steps = []; self.iteration_steps = 0; self.clipped = 0; self.seconds = 0.; self.beta_weight = None
        self.rate_source = None; self.last_rate = None
        # g6230: fixed visited-only targets (set by the teacher phase), the fixed-rate window and target statistics.
        self.fixed = False; self.rate_note = None; self.skipped_steps = 0; self.targets = []

    def follow(self, optimizer):
        """g5402: step at the PPO LR of the same PPO step (learning_rate.upcoming of the PPO Adam), not the settings' own
        rate. The head's own step runs inside the teacher term's apply, before PPO's scheduled step sets that rate."""
        self.rate_source = optimizer

    def use_rate(self, rate, note):
        """g6230: step at the fixed `rate` this iteration instead of the PPO step rate."""
        self.rate_source = None; self.learning_rate = float(rate); self.rate_note = note

    @property
    def trunk(self):
        """Whether the soft loss reaches the shared scorer and trunk this iteration."""
        return self.multiplier > 0

    # ---- loss and gradients ----------------------------------------------------------------

    def logits(self, hidden, mask):
        h = hidden.float()
        if self.multiplier == 0: h = h.detach()
        elif self.multiplier < 1: h = scale_gradient(h, self.multiplier)
        return torch.matmul(h, self.head).masked_fill(~mask, -torch.inf)

    def loss(self, logits, bonus, hidden):
        """Mean KL(softmax((logits + bonus) / T) || soft head) over the rows; the target is held fixed."""
        mask = torch.isfinite(logits)
        with torch.no_grad():
            base = logits.detach() if bonus is None else logits.detach() + bonus
            target = (base / self.temperature).softmax(-1)
        logq = self.logits(hidden, mask).log_softmax(-1).masked_fill(~mask, 0.)
        return (torch.special.xlogy(target, target) - target * logq).sum(-1).mean()

    def fixed_loss(self, hidden, target, visited, keep):
        """g6230: sum over the step's eligible rows (batch positions `keep`) of KL(fixed target || soft head over the
        visited legal actions), divided by all the step's rows: the other rows have weight 0 and no loss is computed
        for them. None when no row is eligible (the head skips the step)."""
        if not keep: return None
        rows = hidden.shape[0]
        index = torch.as_tensor(keep, dtype=torch.int64, device=hidden.device)
        support = visited.index_select(0, index) > 0
        t = target.index_select(0, index)
        logq = self.logits(hidden.index_select(0, index), support).log_softmax(-1).masked_fill(~support, 0.)
        return (torch.special.xlogy(t, t) - t * logq).sum() / rows

    def gradients(self, loss, parameters):
        """(head gradient, actor-parameter gradients or None at multiplier 0): a call of its own."""
        if not self.trunk:
            return torch.autograd.grad(loss, (self.head,))[0], None
        values = torch.autograd.grad(loss, (self.head, *parameters), allow_unused=True)
        return values[0], values[1:]

    def step(self, gradient, norm):
        """One Adam step of the head (torch's single-tensor Adam, no weight decay) after clipping."""
        g = gradient.detach()
        if norm > self.clip:
            g = g * (self.clip / (norm + 1e-6)); self.clipped += 1
        b1, b2 = BETAS; self.t += 1
        self.m.mul_(b1).add_(g, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(g, g, value=1 - b2)
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        if self.rate_source is not None:
            from .learning_rate import upcoming
            rate = upcoming(self.rate_source, ppo=True)
        else: rate = self.learning_rate
        self.last_rate = rate
        with torch.no_grad():
            self.head.addcdiv_(self.m, (self.v.sqrt() / math.sqrt(c2)).add_(EPS), value=-rate / c1)
        self.total_steps += 1; self.iteration_steps += 1

    def record(self, **values):
        self.steps.append(values)

    # ---- validation tables -----------------------------------------------------------------

    def heldout(self, model, rows):
        """(main logits, soft KL): the policy logits exactly as policy_logits gives them, and the
        head's KL to the stored search target softened at T on the long-label rows, by phase."""
        from .teacher_labels import PHASES
        from .teacher_term import forward_policy, policy_context, profile
        main, measured = [], []
        with policy_context(model), torch.no_grad():
            for start in range(0, len(rows), 64):
                part = rows[start:start + 64]
                with capture_hidden(model) as captured:
                    logits = forward_policy(model, part)
                soft = torch.matmul(single(captured).float(), self.head).masked_fill(~torch.isfinite(logits), -torch.inf)
                soft = soft.log_softmax(-1).double().cpu().numpy(); logits = logits.float().cpu().numpy()
                for r, l, s in zip(part, logits, soft):
                    n = len(r.label['legal_ids'])
                    l = l[:n].copy()
                    require(np.isfinite(l).all(), 'Non-finite teacher policy logits')
                    main.append(l)
                    if profile(r) == 'full':
                        p = soften(r.label['target'], self.temperature); use = p > 0
                        measured.append((r.label['phase'], float(np.sum(p[use] * (np.log(p[use]) - s[:n][use])))))
        result = dict(rows=len(measured), kl=float(np.mean([k for _, k in measured])) if measured else None)
        result['by_phase'] = {phase: float(np.mean(v)) if (v := [k for p, k in measured if p == phase]) else None for phase in PHASES}
        return main, result

    def heldout_fixed(self, model, rows, label_scale):
        """g6230: (main logits, soft KL) with the training targets: on each long label, KL(fixed target || soft head
        over the visited legal actions), rows with fewer than MIN_VISITED visited actions left out; by phase."""
        from .teacher_labels import PHASES
        from .teacher_term import forward_policy, policy_context, profile, search_bonus
        from .training_revision_g6230 import MIN_VISITED
        main, measured, skipped = [], [], 0
        with policy_context(model), torch.no_grad():
            for start in range(0, len(rows), 64):
                part = rows[start:start + 64]
                with capture_hidden(model) as captured:
                    logits = forward_policy(model, part)
                soft = torch.matmul(single(captured).float(), self.head).double().cpu().numpy()
                logits = logits.float().cpu().numpy()
                for r, l, s in zip(part, logits, soft):
                    n = len(r.label['legal_ids'])
                    l = l[:n].copy()
                    require(np.isfinite(l).all(), 'Non-finite teacher policy logits')
                    main.append(l)
                    if profile(r) != 'full': continue
                    target, support, info = fixed_target(r.label, search_bonus(r.label, label_scale), self.temperature,
                                                         MIN_VISITED)
                    if not info['eligible']:
                        skipped += 1; continue
                    use = support > 0; q = s[:n][use]; q = q - q.max(); q = q - np.log(np.exp(q).sum())
                    p = target[use]; positive = p > 0
                    measured.append((r.label['phase'], float(np.sum(p[positive] * (np.log(p[positive]) - q[positive])))))
        result = dict(rows=len(measured), skipped_rows=skipped,
                      kl=float(np.mean([k for _, k in measured])) if measured else None)
        result['by_phase'] = {phase: float(np.mean(v)) if (v := [k for p, k in measured if p == phase]) else None for phase in PHASES}
        return main, result

    # ---- commit and report -----------------------------------------------------------------

    def finish(self):
        """The state committed with the next generation; an iteration with head steps advances the warm-up."""
        state = dict(contract=CONTRACT, settings_sha256=digest(self.settings), origin=copy.deepcopy(self.origin),
            generation=self.generation + 1, width=int(self.head.numel()), head=_floats(self.head),
            origin_head=_floats(self.origin_head), adam=dict(step=self.t, exp_avg=_floats(self.m), exp_avg_sq=_floats(self.v)),
            trained_iterations=self.trained + (1 if self.iteration_steps else 0), head_steps=self.total_steps)
        return dict(state, identity=identity(state))

    def target_report(self):
        """g6230: the fixed targets built this iteration (the rows the step plan drew), and the steps skipped."""
        infos = self.targets; eligible = [i for i in infos if i['eligible']]
        return dict(kind='softmax((collection logits + bonus) / T) over visited legal actions', rows=len(infos),
                    eligible_rows=len(eligible), weight_zero_rows=len(infos) - len(eligible), skipped_steps=self.skipped_steps,
                    visited_actions=_summary([float(i['visited']) for i in eligible]),
                    dropped_unvisited_mass=_summary([i['dropped'] for i in eligible]))

    def report(self):
        with torch.no_grad():
            head = self.head.detach(); main = self.main.detach().reshape(-1).float() / self.temperature
            change, distance = torch.stack(((head - self.origin_head).norm() / self.origin_head.norm(),
                                            (head - main).norm() / main.norm())).double().cpu().tolist()
        losses = [s['loss'] for s in self.steps]
        after = multiplier(self.trained + (1 if self.iteration_steps else 0), self.settings['warmup'])
        return dict(contract=CONTRACT, stage=stage(self.multiplier), multiplier=self.multiplier, next_multiplier=after,
            temperature=self.temperature, weight=self.weight, beta_weight=self.beta_weight,
            trained_iterations=self.trained, fresh=self.fresh, origin=copy.deepcopy(self.origin),
            steps=len(self.steps), head_steps=self.iteration_steps, total_head_steps=self.total_steps, clipped_steps=self.clipped,
            kl=dict(first=losses[0] if losses else None, last=losses[-1] if losses else None, **_summary(losses)),
            share=_summary([s['share'] for s in self.steps]), trunk_norm=_summary([s['trunk_norm'] for s in self.steps]),
            head_gradient_norm=_summary([s['head_norm'] for s in self.steps]),
            head_change=change, distance_from_main=distance, seconds=self.seconds,
            learning_rate=dict(source='ppo-step-rate' if self.rate_source is not None else self.rate_note or 'settings',
                               last=self.last_rate, configured=self.learning_rate),
            **({'fixed_targets': self.target_report()} if self.fixed else {}),
            scope='Soft KL on the teacher term\'s label rows before each step; share = 8 beta x soft trunk norm / PPO actor '
                  'norm on the measured steps (shared state/entity encoders, before clipping); head_change = |w - w0| / |w0| '
                  'from the copy at adoption; distance_from_main = |w - main/T| / |main/T| after the update')
