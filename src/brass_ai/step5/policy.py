"""Frozen neural behavior with explicit exploration and independent seat context."""
from __future__ import annotations

from collections import defaultdict
import time
import uuid
import weakref

import numpy as np

from brass_ai.learning.batching import Evaluator, Request, accept_response
from brass_ai.learning.contracts import digest, require

INDUSTRIES = ('CottonMill', 'Manufacturer', 'Pottery', 'Brewery', 'IronWorks', 'CoalMine')


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
                 focus='mixed', batch_size=32):
        require(0 <= exploration <= .5 and .1 <= temperature <= 3, 'Invalid behavior distribution')
        require(focus in ('mixed', 'neutral') + INDUSTRIES, 'Invalid context mode')
        self.evaluator = Evaluator(model, device=device, batch_size=batch_size)
        self.exploration, self.temperature, self.focus = exploration, temperature, focus
        self.contexts = weakref.WeakKeyDictionary()
        self.context_counter, self.context_offset = 0, None
        self.policy_id = 'step5-neural/' + digest(dict(model=self.evaluator.model_id,
            exploration=exploration, temperature=temperature, focus=focus, version=2))

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

    def predict(self, inputs):
        requests = []
        for view, choices, rng in inputs:
            style, intent = self.context_features(view, rng)
            requests.append(Request(uuid.uuid4().hex, self.evaluator.model_id, view, choices,
                                    time.monotonic() + 120, tuple(style), tuple(intent)))
        responses = self.evaluator.predict(requests)
        outputs = []
        for request, response in zip(requests, responses):
            accept_response(request, response, request.view['context']['revision'], self.evaluator.model_id)
            raw = np.asarray(response['probabilities'], dtype=np.float64)
            raw = np.power(raw.clip(1e-30), 1 / self.temperature)
            raw /= raw.sum()
            q = exploration_distribution(request.choices, request.intent)
            probabilities = (1 - self.exploration) * raw + self.exploration * q
            probabilities /= probabilities.sum()
            outputs.append((probabilities, response))
        return outputs

    def select_batch(self, inputs):
        output = []
        for (probabilities, _), (_, _, rng) in zip(self.predict(inputs), inputs):
            selected = rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]
            output.append((selected, float(probabilities[selected]), None))
        return output
