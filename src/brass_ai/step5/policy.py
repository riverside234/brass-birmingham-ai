"""Frozen neural behavior with explicit exploration and independent seat context."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import time
import uuid
import weakref

import numpy as np

from brass_ai.learning.batching import Evaluator, Request, RequestStopped, accept_response
from brass_ai.learning.contracts import digest, require

INDUSTRIES = ('CottonMill', 'Manufacturer', 'Pottery', 'Brewery', 'IronWorks', 'CoalMine')


@contextmanager
def inference_deadline(policies, deadline):
    """Private planner scope; queue/encoding/inference share its turn clock."""
    unique=list({id(p):p for p in policies}.values())
    previous=[p.request_deadline for p in unique]
    for p in unique: p.request_deadline=deadline
    try: yield
    finally:
        for p,old in zip(unique,previous): p.request_deadline=old


FAMILY_WEIGHTS = dict(build=4, develop=4, network=4, loan=2, sell=2, scout=1)


def policy_identity(model, exploration, temperature, focus, version=2, opening=None):
    fields = dict(model=model, exploration=exploration, temperature=temperature, focus=focus, version=version)
    if version == 3:
        fields['opening'] = opening
    return 'step5-neural/' + digest(fields)


def exploration_distribution(choices, intent=None, version=2, phase=None):
    """Equal action-family exploration; all legal build industries get a group.

    A pottery holdout NEVER changes this distribution or any legal mask. Context
    can give its focus group twice the exploration mass, without changing reward.
    """
    require(type(version) is int and version in (2, 3), 'Unknown exploration distribution')
    if version == 3:
        require(isinstance(phase, str) and phase, 'v3 exploration requires the recorded phase')
    if version == 3 and phase == 'action':
        families = defaultdict(list)
        for i, choice in enumerate(choices):
            families[choice['kind']].append(i)
        require(bool(families), 'No legal candidates')
        if set(families) != {'pass'}:
            families.pop('pass', None)
        total = sum(FAMILY_WEIGHTS.get(k, 1) for k in families)
        out = np.zeros(len(choices), dtype=np.float64)
        for kind, indices in families.items():
            share = FAMILY_WEIGHTS.get(kind, 1) / total
            sub = defaultdict(list)
            for i in indices:
                sub[choices[i].get('industry') if kind in ('build', 'develop') else None].append(i)
            weights = {ind: (2. if intent is not None and ind in INDUSTRIES and
                            intent[INDUSTRIES.index(ind)] > .5 else 1.) for ind in sub}
            for ind, members in sub.items():
                out[members] = share * weights[ind] / sum(weights.values()) / len(members)
        return out
    groups = defaultdict(list)
    for i, choice in enumerate(choices):
        key = (choice['kind'], choice.get('industry') if choice['kind'] in ('build', 'develop') else None)
        groups[key].append(i)
    require(bool(groups), 'No legal candidates')
    weights = {k: 1.0 for k in groups}
    if intent is not None:
        for kind, industry in groups:
            if industry in INDUSTRIES and intent[INDUSTRIES.index(industry)] > .5:
                weights[(kind, industry)] = 2.0
    total = sum(weights.values())
    out = np.zeros(len(choices), dtype=np.float64)
    for key, indices in groups.items():
        out[indices] = weights[key] / total / len(indices)
    return out


def context_for(focus):
    require(focus in ('neutral',) + INDUSTRIES, 'Unknown neural focus')
    if focus == 'neutral':
        return [0.0] * 4, [0.0] * 8
    style = [float(focus in INDUSTRIES[:3]), 0.0,
             float(focus in INDUSTRIES[3:]), 0.0]
    return style, [float(focus == x) for x in INDUSTRIES] + [0.0, 0.0]


class NeuralPolicy:
    kind = 'model'

    def __init__(self, model, device='cpu', *, exploration=.1, temperature=1.0,
                 focus='mixed', batch_size=32, max_candidate_cells=8192, evaluator=None,
                 exploration_version=2, opening=None):
        require(0 <= exploration <= .5 and .1 <= temperature <= 3, 'Invalid behavior distribution')
        require(focus in ('mixed', 'neutral') + INDUSTRIES, 'Invalid context mode')
        self.evaluator = evaluator if evaluator is not None else Evaluator(model, device=device,
            batch_size=batch_size, max_candidate_cells=max_candidate_cells)
        self.owns_evaluator=evaluator is None
        self.exploration, self.temperature, self.focus = exploration, temperature, focus
        require(exploration_version in (2, 3) and (opening is None or exploration_version == 3),
                'Invalid exploration/opening contract')
        self.exploration_version, self.opening = exploration_version, opening
        self.action_counts = weakref.WeakKeyDictionary()
        self.last_behavior = weakref.WeakKeyDictionary()
        self.last_raw = weakref.WeakKeyDictionary()
        self.request_deadline = None
        self.contexts = weakref.WeakKeyDictionary()
        self.utility_remaining = weakref.WeakKeyDictionary()
        self.context_counter, self.context_offset = 0, None
        self.policy_id = policy_identity(self.evaluator.model_id, exploration, temperature, focus,
                                         exploration_version, opening)

    def close(self):
        if self.owns_evaluator:self.evaluator.close()

    def context_features(self, view, rng):
        if rng not in self.contexts:
            focus = self.focus
            if focus == 'mixed':
                # Every twelve new seat trajectories include six neutral seats
                # and one of EACH industry focus. A random initial offset keeps
                # the schedule from always assigning pottery to the same seat.
                if self.context_offset is None:
                    self.context_offset = rng.randrange(12)
                index = (self.context_counter + self.context_offset) % 12
                self.context_counter += 1
                focus = 'neutral' if index % 2 == 0 else INDUSTRIES[index // 2]
            self.contexts[rng] = context_for(focus)
        return self.contexts[rng]

    def prediction_requests(self, inputs):
        requests = []
        for view, choices, rng in inputs:
            if self.request_deadline is not None and time.monotonic()>=self.request_deadline:
                from brass_ai.search.common import SearchIncomplete
                raise SearchIncomplete('inference_deadline')
            style, intent = self.context_features(view, rng)
            requests.append(Request(uuid.uuid4().hex, self.evaluator.model_id, view, choices,
                min(time.monotonic()+120,self.request_deadline) if self.request_deadline is not None else time.monotonic()+120,
                tuple(style), tuple(intent),remaining=self.utility_remaining.get(rng)))
        return requests

    def prediction_outputs(self,requests,responses):
        require(len(requests)==len(responses),'Missing policy inference response')
        outputs = []
        for request, response in zip(requests, responses):
            if self.request_deadline is not None and (request.stopped() or response.get('status') in ('expired','cancelled')):
                from brass_ai.search.common import SearchIncomplete
                raise SearchIncomplete('inference_deadline')
            try:
                accept_response(request, response, request.view['context']['revision'], self.evaluator.model_id)
            except RequestStopped as error:
                # The turn can expire after the precheck above. Only liveness
                # failures end search normally; invalid predictions still fail.
                if self.request_deadline is None:
                    raise
                from brass_ai.search.common import SearchIncomplete
                raise SearchIncomplete('inference_deadline') from error
            raw = np.asarray(response['probabilities'], dtype=np.float64)
            raw = np.power(raw.clip(1e-30), 1 / self.temperature)
            raw /= raw.sum()
            q = exploration_distribution(request.choices, request.intent, self.exploration_version,
                                         request.view['context']['phase'])
            probabilities = (1 - self.exploration) * raw + self.exploration * q
            probabilities /= probabilities.sum()
            outputs.append((probabilities, response))
        return outputs

    def predict(self,inputs):
        requests=self.prediction_requests(inputs)
        return self.prediction_outputs(requests,self.evaluator.predict(requests))

    def select_batch(self, inputs):
        if self.exploration_version == 3:
            requests = self.prediction_requests(inputs)
            responses = self.evaluator.predict(requests)
            output = []
            for (view, choices, rng), request, response in zip(inputs, requests, responses):
                accept_response(request, response, view['context']['revision'], self.evaluator.model_id)
                raw = np.asarray(response['probabilities'], dtype=np.float64)
                raw /= raw.sum()
                output.append(self.sample_training(view, choices, rng, raw, request.intent))
            return output
        output = []
        for (probabilities, _), (_, _, rng) in zip(self.predict(inputs), inputs):
            selected = rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]
            output.append((selected, float(probabilities[selected]), None))
        return output

    def sample_training(self, view, choices, rng, raw, intent):
        phase = view['context']['phase']
        count = self.action_counts.get(rng, 0)
        temperature = (self.opening['temperature'] if self.opening and phase == 'action' and
                       count < self.opening['action_decisions_per_seat'] else self.temperature)
        p = np.power(raw.clip(1e-30), 1 / temperature); p /= p.sum()
        q = exploration_distribution(choices, intent, 3, phase)
        mixed = (1-self.exploration)*p + self.exploration*q
        # Sample the actual mixture component so exploration telemetry is observed,
        # not an estimate inferred from the selected action.
        explore = rng.random() < self.exploration
        selected = rng.choices(range(len(choices)), weights=q if explore else p, k=1)[0]
        self.last_behavior[rng] = dict(temperature=temperature, exploration=self.exploration,
            exploration_version=3, component='exploration' if explore else 'policy',
            action_decision=count if phase == 'action' else None)
        self.last_raw[rng] = raw
        if phase == 'action':
            self.action_counts[rng] = count+1
        return selected, float(mixed[selected]), None


def predict_groups(groups):
    """Submit already-ready leaves together; search statistics are untouched.

    Only the unmodified Step 5 policy opts in. Personality-specific prediction
    overrides and local evaluators keep their established prediction path.
    """
    require(groups,'Missing prediction groups')
    endpoints=[getattr(p.evaluator,'endpoint',None) for p,inputs in groups]
    first=groups[0][0].evaluator
    if (all(type(p) is NeuralPolicy for p,_ in groups) and endpoints[0] is not None and
            all(e is endpoints[0] for e in endpoints) and hasattr(first,'predict_mixed')):
        requests=[p.prediction_requests(inputs) for p,inputs in groups]
        responses=first.predict_mixed([(p.evaluator,r) for (p,_),r in zip(groups,requests)])
        return [p.prediction_outputs(r,out) for (p,_),r,out in zip(groups,requests,responses)]
    return [p.predict(inputs) for p,inputs in groups]
