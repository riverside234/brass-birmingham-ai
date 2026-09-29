"""Step 5 parallel host adapter; the R3-pinned serial collector stays immutable."""
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import hashlib
import random
import time
import uuid

from brass_ai.host import RulesHost
from brass_ai.runtime_contracts import validate
from brass_ai.learning.contracts import CONFIG_DIR, digest, read_json, require
from brass_ai.learning.replay import split_for, terminal_targets
from brass_ai.learning.runner import collect_games as serial_collect_games


def collect_games(host_directory, seeds, policies, *, workers=4, max_decisions=2000, mono=None,
                  sampling_seed=0, parallel_hosts=False, rotations=None, sampling_seeds=None):
    """Separate process per concurrent game; inference is batched across games.

    Seeds stay in trusted orchestration, outside each policy and replay view.
    By default assignments rotate cyclically across games. Evaluation can give
    explicit candidate-seat rotations and per-game RNG seeds, independent of
    batch boundaries. Repeated setups require distinct explicit rotations.
    A failure aborts the collection call; no incomplete game receives targets.
    """
    require(type(parallel_hosts) is bool, 'Invalid parallel host setting')
    if not parallel_hosts and rotations is None and sampling_seeds is None:
        return serial_collect_games(host_directory, seeds, policies, workers=workers,
            max_decisions=max_decisions, mono=mono, sampling_seed=sampling_seed)
    from pathlib import Path
    host_directory = Path(host_directory)
    pins = read_json(CONFIG_DIR / 'host-1.0.0.json')
    require(hashlib.sha256((host_directory / 'BrassHost.exe').read_bytes()).hexdigest() == pins['host_sha256'],
            'Host build differs from the R1-qualified learning producer')
    require(len(policies) == 4 and all(hasattr(p, 'select_batch') for p in policies), 'Exactly four seat policies required')
    require(bool(seeds), 'Nonempty setup list required')
    if rotations is None:
        require(len(set(seeds)) == len(seeds), 'Distinct setup seeds required per collection')
    else:
        require(len(rotations) == len(seeds) and all(type(r) is int and 0 <= r < 4 for r in rotations),
                'Invalid explicit seat rotations')
        require(len(set(zip(seeds, rotations))) == len(seeds), 'Duplicate setup/seat evaluation job')
    if sampling_seeds is not None:
        require(len(sampling_seeds) == len(seeds) and all(type(s) is int and s >= 0 for s in sampling_seeds),
                'Invalid per-game sampling seeds')
    require(all(type(seed) is int and 0 <= seed <= 2147483647 for seed in seeds), 'Invalid setup seed')
    require(type(workers) is int and workers > 0 and max_decisions > 0, 'Invalid collection limits')
    games = []
    with ExitStack() as stack:
        hosts = [stack.enter_context(RulesHost(host_directory, mono=mono)) for _ in range(min(workers, len(seeds)))]
        # Each process owns its game/statics. Never issue simultaneous commands
        # to ONE host; only independent hosts run concurrently. Policy inference
        # and per-seat RNG/context updates remain ordered on the coordinator.
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=len(hosts))) if parallel_hosts else None

        def host_map(function, items):
            return list(executor.map(function, items)) if executor else list(map(function, items))

        def observe(item):
            ctx, host = item['context'], item['host']
            return host.observe(ctx['actor']), host.choices()

        def apply(job):
            item, choice = job
            return item['host'].apply(item['context'], choice)

        for start in range(0, len(seeds), len(hosts)):
            live = []
            contexts = host_map(lambda pair: pair[0].reset(pair[1]), zip(hosts, seeds[start:start + len(hosts)]))
            for index, seed in enumerate(seeds[start:start + len(hosts)], start):
                host = hosts[index - start]
                context = contexts[index - start]
                group = digest({'rules': pins['rules'], 'setup_seed': seed})
                offset = index if rotations is None else -rotations[index]
                assigned = [policies[(seat + offset) % 4] for seat in range(4)]
                rng_seed = sampling_seed if sampling_seeds is None else sampling_seeds[index]
                rng_index = index if sampling_seeds is None else 0
                game = {'game_id': uuid.uuid4().hex, 'setup_group': group, 'players': 4,
                        'split': split_for(group), 'opponents': [p.policy_id for p in assigned],
                        'result': None, 'records': []}
                live.append({'host': host, 'context': context, 'game': game, 'policies': assigned,
                    'rngs': [random.Random(digest({'sampling_seed': rng_seed, 'game': rng_index, 'seat': s})) for s in range(4)]})
            while live:
                groups = {}
                for item, (view, choices) in zip(live, host_map(observe, live)):
                    require(len(item['game']['records']) < max_decisions, 'Decision cap reached; incomplete game excluded')
                    ctx, host = item['context'], item['host']
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
                    pending = []
                    for item, decision in zip(items, decisions):
                        view, choices, _ = item['input']
                        selected, probability, teacher = decision
                        require(type(selected) is int and 0 <= selected < len(choices), 'Policy selected an illegal index')
                        pending.append((item, choices[selected]))
                        row = {'step': len(item['game']['records']), 'seat': view['seat'],
                            'view': view, 'choices': choices, 'chosen': selected, 'teacher': teacher,
                            'style': [0.0] * 4, 'intent': [0.0] * 8, 'seconds': elapsed,
                            'behavior': {'kind': policy.kind, 'policy_id': policy.policy_id,
                                         'probability': probability, 'temperature': 1.0}}
                        if hasattr(policy, 'context_features'):
                            row['style'],row['intent']=policy.context_features(view,item['input'][2])
                        item['game']['records'].append(row)
                    for item, context in zip(items, host_map(apply, pending)):
                        item['context'] = context
                for item in list(live):
                    if item['context']['terminal']:
                        item['game']['result'] = item['host'].call('result')
                        terminal_targets(item['game']['result'])
                        games.append(item['game'])
                        live.remove(item)
    return games
