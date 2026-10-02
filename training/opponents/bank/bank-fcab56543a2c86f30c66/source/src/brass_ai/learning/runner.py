"""Trusted four-player match coordinator; policies receive detached views only."""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
import random
import time
import uuid

import numpy as np

from brass_ai.host import RulesHost
from brass_ai.runtime_contracts import validate
from .batching import Evaluator, Request, accept_response
from .contracts import CONFIG_DIR, digest, read_json, require
from .replay import split_for, terminal_targets


class RandomPolicy:
    """Legal diagnostic baseline, uniform over kinds then choices within kind."""
    policy_id = 'legal-kind-random/1.0.0'
    kind = 'random'

    def select_batch(self, inputs):
        decisions = []
        for view, choices, rng in inputs:
            groups = {}
            for index, choice in enumerate(choices):
                groups.setdefault(choice['kind'], []).append(index)
            group = groups[rng.choice(sorted(groups))]
            selected = rng.choice(group)
            decisions.append((selected, 1 / len(groups) / len(group), None))
        return decisions


class ModelPolicy:
    kind = 'model'

    def __init__(self, model, device='cpu', batch_size=32, deadline_seconds=30.0):
        require(deadline_seconds > 0, 'Invalid policy deadline')
        self.evaluator = Evaluator(model, device=device, batch_size=batch_size)
        self.policy_id = self.evaluator.model_id
        self.deadline_seconds = deadline_seconds

    def select_batch(self, inputs):
        deadline = time.monotonic() + self.deadline_seconds
        requests = [Request(uuid.uuid4().hex, self.policy_id, view, choices, deadline)
                    for i, (view, choices, _) in enumerate(inputs)]
        responses = self.evaluator.predict(requests)
        decisions = []
        for request, response, (_, _, rng) in zip(requests, responses, inputs):
            accept_response(request, response, request.view['context']['revision'], self.policy_id)
            probabilities = np.asarray(response['probabilities'], dtype=np.float64)
            # random.Random is local to this seat; weights are frozen for a game.
            selected = rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]
            decisions.append((selected, float(probabilities[selected] / probabilities.sum()), None))
        return decisions


def collect_games(host_directory, seeds, policies, *, workers=4, max_decisions=2000, mono=None, sampling_seed=0):
    """Separate process per concurrent game; inference is batched across games.

    Seeds stay in trusted orchestration, outside each policy and replay view.
    The four frozen seat assignments are cyclically rotated across games.
    A failure aborts the collection call; no incomplete game receives targets.
    """
    from pathlib import Path
    host_directory = Path(host_directory)
    pins = read_json(CONFIG_DIR / 'host-1.0.0.json')
    require(hashlib.sha256((host_directory / 'BrassHost.exe').read_bytes()).hexdigest() == pins['host_sha256'],
            'Host build differs from the R1-qualified learning producer')
    require(len(policies) == 4 and all(hasattr(p, 'select_batch') for p in policies), 'Exactly four seat policies required')
    require(seeds and len(set(seeds)) == len(seeds), 'Distinct setup seeds required per collection')
    require(all(type(seed) is int and 0 <= seed <= 2147483647 for seed in seeds), 'Invalid setup seed')
    require(type(workers) is int and workers > 0 and max_decisions > 0, 'Invalid collection limits')
    games = []
    with ExitStack() as stack:
        hosts = [stack.enter_context(RulesHost(host_directory, mono=mono)) for _ in range(min(workers, len(seeds)))]
        for start in range(0, len(seeds), len(hosts)):
            live = []
            for index, seed in enumerate(seeds[start:start + len(hosts)], start):
                host = hosts[index - start]
                context = host.reset(seed)
                group = digest({'rules': pins['rules'], 'setup_seed': seed})
                assigned = [policies[(seat + index) % 4] for seat in range(4)]
                game = {'game_id': uuid.uuid4().hex, 'setup_group': group, 'players': 4,
                        'split': split_for(group), 'opponents': [p.policy_id for p in assigned],
                        'result': None, 'records': []}
                live.append({'host': host, 'context': context, 'game': game, 'policies': assigned,
                    'rngs': [random.Random(digest({'sampling_seed': sampling_seed, 'game': index, 'seat': s})) for s in range(4)]})
            while live:
                groups = {}
                for item in live:
                    require(len(item['game']['records']) < max_decisions, 'Decision cap reached; incomplete game excluded')
                    ctx, host = item['context'], item['host']
                    view, choices = host.observe(ctx['actor']), host.choices()
                    validate('observation', view)
                    validate('choices', choices)
                    require(view['context']['revision'] == ctx['revision'], 'Stale collector view')
                    item['input'] = (view, choices, item['rngs'][ctx['actor']])
                    policy = item['policies'][ctx['actor']]
                    groups.setdefault(id(policy), (policy, []))[1].append(item)
                for policy, items in groups.values():
                    began = time.monotonic()
                    inputs = [item['input'] for item in items]
                    # External policies must implement this permitted-input API;
                    # never pass hosts, checkpoints, seeds or other seat hands.
                    decisions = policy.select_batch(inputs)
                    elapsed = (time.monotonic() - began) / len(items)
                    require(len(decisions) == len(items), 'Policy batch length mismatch')
                    for item, decision in zip(items, decisions):
                        view, choices, _ = item['input']
                        selected, probability, teacher = decision
                        require(type(selected) is int and 0 <= selected < len(choices), 'Policy selected an illegal index')
                        context = item['host'].apply(item['context'], choices[selected])
                        row = {'step': len(item['game']['records']), 'seat': view['seat'],
                            'view': view, 'choices': choices, 'chosen': selected, 'teacher': teacher,
                            'style': [0.0] * 4, 'intent': [0.0] * 8, 'seconds': elapsed,
                            'behavior': {'kind': policy.kind, 'policy_id': policy.policy_id,
                                         'probability': probability, 'temperature': 1.0}}
                        if hasattr(policy, 'context_features'):
                            row['style'],row['intent']=policy.context_features(view,item['input'][2])
                        item['game']['records'].append(row)
                        item['context'] = context
                for item in list(live):
                    if item['context']['terminal']:
                        item['game']['result'] = item['host'].call('result')
                        terminal_targets(item['game']['result'])
                        games.append(item['game'])
                        live.remove(item)
    return games


def match_report(games):
    """Per-policy and per-seat totals; no two-player Elo or strength promotion."""
    policies = {}
    for game in games:
        targets = terminal_targets(game['result'])
        for seat, identity in enumerate(game['opponents']):
            value = policies.setdefault(identity, {'seat_games': 0, 'win_share': 0.0,
                'placement': 0.0, 'vp': 0.0, 'seat_counts': [0] * 4})
            value['seat_games'] += 1
            value['seat_counts'][seat] += 1
            for key in ('win_share', 'placement', 'vp'):
                value[key] += targets[key][seat]
    for values in policies.values():
        for key in ('win_share', 'placement', 'vp'):
            values[key] /= values['seat_games']
    return {'games': len(games), 'players': 4, 'policies': policies,
            'purpose': 'diagnostic; no calibrated difficulty or strength claim'}
