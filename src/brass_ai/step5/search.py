"""Neural complete-turn proposals and fresh-confirmed next-decision search.

Only permitted observations enter reconstruction. Current-turn plans are fixed
across hidden worlds; rollout seats each receive their own view. No full live
checkpoint is accepted by this API. S2 remains an experiment until equal-budget
evaluation shows a benefit, even when its teacher labels pass correctness checks.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import random
import time
import weakref

import numpy as np

from brass_ai.beliefs import sample_world_setup
from brass_ai.host import RulesHost
from brass_ai.learning.contracts import SOURCE_ROOT, digest, read_json, require
from brass_ai.learning.replay import validate_teacher
from .policy import NeuralPolicy, inference_deadline


@dataclass(frozen=True)
class SearchConfig:
    proposals: int = 12
    finalists: int = 4
    pilot_samples: int = 2
    confirmation_samples: int = 4
    max_plan_choices: int = 96
    max_reply_choices: int = 256
    seconds: float = 10.0
    temperature: float = .1
    turn_interval: int = 4
    max_searches_per_seat: int = 0

    def validate(self):
        require(self.proposals >= 2 and 2 <= self.finalists <= self.proposals,
                'Search needs multiple candidates and finalists')
        require(self.pilot_samples > 0 and self.confirmation_samples >= 2 and
                self.max_plan_choices > 0 and self.max_reply_choices > 0 and
                self.seconds > 0 and self.temperature > 0 and self.turn_interval > 0 and
                self.max_searches_per_seat >= 0, 'Invalid search budget')


class SearchIncomplete(RuntimeError):
    pass


def turn_key(context):
    return tuple(context[k] for k in ('era', 'round', 'turn', 'turn_owner'))


def first_choices(choices, probabilities, width, rng):
    """Preserve action kinds and every available build industry before pruning."""
    families = {}
    for i, choice in enumerate(choices):
        key = (choice['kind'], choice.get('industry') if choice['kind'] == 'build' else None)
        if key not in families or probabilities[i] > probabilities[families[key]]:
            families[key] = i
    selected = list(families.values())
    rest = sorted(set(range(len(choices))) - set(selected))
    # Half remaining slots use top policy choices; half retain stochastic breadth.
    count = max(0, width - len(selected))
    top = sorted(rest, key=lambda i: (-probabilities[i], choices[i]['id']))[:count // 2]
    selected.extend(top)
    rest = [i for i in rest if i not in top]
    while rest and len(selected) < width:
        index = rng.choices(rest, weights=[probabilities[i] + 1e-8 for i in rest], k=1)[0]
        selected.append(index)
        rest.remove(index)
    return selected  # Mandatory families may exceed the requested width.


def confirmed_teacher(choices, plans, values, selected_prefix, confidence, samples, temperature):
    """Conditional continuation labels from fixed plans, never raw visit counts."""
    scores = {}
    prefix = tuple(selected_prefix)
    for plan, value in zip(plans, values):
        if tuple(plan[:len(prefix)]) == prefix and len(plan) > len(prefix):
            next_id = plan[len(prefix)]
            # Multiple proposals for a first action do not inflate its weight.
            scores[next_id] = max(scores.get(next_id, -math.inf), value)
    ids = [c['id'] for c in choices]
    coverage = [identity in scores for identity in ids]
    if sum(coverage) < 2:
        return None
    logits = np.array([scores[i] for i in ids if i in scores], dtype=float) / temperature
    p = np.exp(logits - logits.max()); p /= p.sum()
    iterator = iter(p.tolist())
    teacher = dict(source='fresh-finalist/1.0.0',
                   probabilities=[next(iterator) if c else 0.0 for c in coverage],
                   coverage=coverage, confidence=float(confidence), samples=samples,
                   horizon='complete-current-turn/actual-next-root-decision/neural-v1')
    validate_teacher(teacher, len(choices))
    return teacher


class SearchPolicy:
    kind = 'external'  # Search selections are not on-policy samples of the network.

    def __init__(self, model, host_directory, device='cpu', *, config=None, mono=None,
                 focus='mixed', opponents=None, evaluator=None):
        self.config = config or SearchConfig(); self.config.validate()
        self.base = NeuralPolicy(model, device, exploration=.1, focus=focus, evaluator=evaluator)
        self.rollouts = opponents or [NeuralPolicy(model, device, exploration=.05, focus='neutral', evaluator=evaluator)]
        require(all(isinstance(p, NeuralPolicy) and p.focus == 'neutral' for p in self.rollouts),
                'Use masked frozen neutral-context rollout policies for matched scenarios')
        self.host = RulesHost(host_directory, mono=mono)
        self.composition = read_json(SOURCE_ROOT / 'configs/contracts/v0.2.0/card-composition.json')
        self.policy_id = 'step5-search/' + digest(dict(base=self.base.policy_id, config=asdict(self.config),
                                                     rollouts=[p.policy_id for p in self.rollouts]))
        self.contexts = weakref.WeakKeyDictionary()
        self.telemetry = dict(searches=0, confirmed=0, incomplete=0, proposed=0,
                              pilot_worlds=0, fresh_worlds=0, reply_choices=0)

    def close(self):
        self.host.close()

    def context_features(self, view, rng):
        return self.base.context_features(view, rng)

    @staticmethod
    def check(deadline):
        if time.monotonic() >= deadline:
            raise SearchIncomplete('whole_turn_deadline')

    def reset_world(self, view, seed, deadline):
        self.check(deadline)
        return self.host.reset(1, sample_world_setup(view, self.composition, seed))

    def propose(self, view, choices, rng, deadline):
        self.check(deadline)
        probabilities, _ = self.base.predict([(view, choices, rng)])[0]
        indices = first_choices(choices, probabilities, self.config.proposals, rng)
        setup_seed = rng.randrange(2**31)
        plans = []
        for index in indices:
            context = self.reset_world(view, setup_seed, deadline)
            require([c['id'] for c in self.host.choices()] == [c['id'] for c in choices],
                    'Reconstructed legal choices differ')
            path_rng = random.Random(rng.randrange(2**63))
            self.base.contexts[path_rng] = self.context_features(view, rng)
            path = []
            for depth in range(self.config.max_plan_choices):
                self.check(deadline)
                current = self.host.observe(view['seat'])
                legal = self.host.choices()
                chosen = index if depth == 0 else self.base.select_batch([(current, legal, path_rng)])[0][0]
                path.append(legal[chosen]['id'])
                context = self.host.apply(context, legal[chosen])
                if context['terminal'] or context['actor'] != view['seat'] or turn_key(context) != turn_key(view['context']):
                    break
            else:
                raise SearchIncomplete('incomplete_current_turn')
            if path not in plans:
                plans.append(path)
        self.telemetry['proposed'] += len(plans)
        return plans

    def scenario(self, view, plan, seed, root_context, deadline):
        context = self.reset_world(view, seed, deadline)
        for identity in plan:
            self.check(deadline)
            legal = self.host.choices()
            chosen = next((c for c in legal if c['id'] == identity), None)
            require(chosen is not None, 'Complete root plan changed across unseen worlds')
            context = self.host.apply(context, chosen)
        # Randomness and identities are stable within a scenario and matched
        # across candidates. They depend on search randomness, not live seeds.
        rngs = [random.Random(digest(dict(sample=seed, seat=s))) for s in range(4)]
        assigned = [self.rollouts[random.Random(f'{seed}:{s}:opponent').randrange(len(self.rollouts))]
                    for s in range(4)]
        for _ in range(self.config.max_reply_choices):
            self.check(deadline)
            if context['terminal']:
                return self.host.call('result')['seats'][view['seat']]['utility']
            actor = context['actor']
            allowed = self.host.observe(actor)
            legal = self.host.choices()
            if actor == view['seat']:
                self.base.contexts[rngs[actor]] = root_context
                return float(self.base.predict([(allowed, legal, rngs[actor])])[0][1]['win_share'][actor])
            choice = assigned[actor].select_batch([(allowed, legal, rngs[actor])])[0][0]
            context = self.host.apply(context, legal[choice])
            self.telemetry['reply_choices'] += 1
        raise SearchIncomplete('next_root_decision_not_reached')

    def search(self, view, choices, rng, deadline):
        self.telemetry['searches'] += 1
        plans = self.propose(view, choices, rng, deadline)
        require(plans, 'No complete root turn')
        root_context = self.context_features(view, rng)
        # All plans get the same pilot worlds. Finalists then get unseen worlds.
        pilot = [rng.randrange(2**63) for _ in range(self.config.pilot_samples)]
        scores = [np.mean([self.scenario(view, p, s, root_context, deadline) for s in pilot]) for p in plans]
        self.telemetry['pilot_worlds'] += len(pilot)
        selected = sorted(range(len(plans)), key=lambda i: (-scores[i], plans[i]))[:self.config.finalists]
        finalists = [plans[i] for i in selected]
        fresh = []
        while len(fresh) < self.config.confirmation_samples:
            seed = rng.randrange(2**63)
            if seed not in pilot and seed not in fresh:
                fresh.append(seed)
        samples = [[self.scenario(view, p, s, root_context, deadline) for s in fresh] for p in finalists]
        means = [float(np.mean(x)) for x in samples]
        order = sorted(range(len(finalists)), key=lambda i: (-means[i], finalists[i]))
        confidence = 0.0
        if len(order) >= 2:
            differences = np.asarray(samples[order[0]]) - np.asarray(samples[order[1]])
            error = float(np.std(differences, ddof=1) / math.sqrt(len(fresh)))
            gap = max(0.0, float(np.mean(differences)))
            confidence = min(.9, gap / (gap + 2 * error + .01))
        self.telemetry['fresh_worlds'] += len(fresh)
        self.telemetry['confirmed'] += 1
        return finalists[order[0]], finalists, means, confidence

    def select_batch(self, inputs):
        output = []
        for view, choices, rng in inputs:
            key = turn_key(view['context'])
            state = self.contexts.get(rng)
            if state is None or state['turn'] != key or view['context']['revision'] <= state['revision']:
                turns_seen = 1 if state is None else state['turns_seen'] + 1
                searches_used = 0 if state is None else state['searches_used']
                state = dict(turn=key, revision=-1, deadline=time.monotonic() + self.config.seconds,
                             queue=deque(), prefix=[], searched=False, plans=[], values=[], confidence=0.0,
                             turns_seen=turns_seen, searches_used=searches_used)
                if ((turns_seen - 1) % self.config.turn_interval or
                    (self.config.max_searches_per_seat and searches_used >= self.config.max_searches_per_seat)):
                    state['searched'] = True
                self.contexts[rng] = state
            ids = [c['id'] for c in choices]
            if state['queue'] and state['queue'][0] not in ids:
                state['queue'].clear()
            if not state['queue'] and not state['searched'] and view['context']['phase'] == 'action':
                state['searched'] = True
                state['searches_used'] += 1
                try:
                    with inference_deadline([self.base]+self.rollouts,state['deadline']):
                        plan, plans, values, confidence = self.search(view, choices, rng, state['deadline'])
                    state.update(queue=deque(plan), plans=plans, values=values, confidence=confidence)
                except SearchIncomplete:
                    self.telemetry['incomplete'] += 1
            if state['queue']:
                teacher = confirmed_teacher(choices, state['plans'], state['values'], state['prefix'],
                    state['confidence'], self.config.confirmation_samples, self.config.temperature)
                chosen = state['queue'].popleft()
                state['prefix'].append(chosen)
                output.append((ids.index(chosen), 1.0, teacher))
            else:
                output.append(self.base.select_batch([(view, choices, rng)])[0])
            state['revision'] = view['context']['revision']
        return output
