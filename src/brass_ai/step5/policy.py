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


def exploration_distribution(choices, intent=None):
    """Equal action-family exploration; all legal build industries get a group.

    A pottery holdout NEVER changes this distribution or any legal mask. Context
    can give its focus group twice the exploration mass, without changing reward.
    """
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
                 focus='mixed', batch_size=32, max_candidate_cells=8192, evaluator=None):
        require(0 <= exploration <= .5 and .1 <= temperature <= 3, 'Invalid behavior distribution')
        require(focus in ('mixed', 'neutral') + INDUSTRIES, 'Invalid context mode')
        self.evaluator = evaluator if evaluator is not None else Evaluator(model, device=device,
            batch_size=batch_size, max_candidate_cells=max_candidate_cells)
        self.owns_evaluator=evaluator is None
        self.exploration, self.temperature, self.focus = exploration, temperature, focus
        self.request_deadline = None
        self.contexts = weakref.WeakKeyDictionary()
        self.utility_remaining = weakref.WeakKeyDictionary()
        self.context_counter, self.context_offset = 0, None
        self.policy_id = 'step5-neural/' + digest(dict(model=self.evaluator.model_id,
            exploration=exploration, temperature=temperature, focus=focus, version=2))

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
            q = exploration_distribution(request.choices, request.intent)
            probabilities = (1 - self.exploration) * raw + self.exploration * q
            probabilities /= probabilities.sum()
            outputs.append((probabilities, response))
        return outputs

    def predict(self,inputs):
        requests=self.prediction_requests(inputs)
        return self.prediction_outputs(requests,self.evaluator.predict(requests))

    def select_batch(self, inputs):
        output = []
        for (probabilities, _), (_, _, rng) in zip(self.predict(inputs), inputs):
            selected = rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]
            output.append((selected, float(probabilities[selected]), None))
        return output


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
