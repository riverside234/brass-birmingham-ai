#!/usr/bin/env python3
"""Play paired PPO self-play openings with frozen weights."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import itertools
import os
from pathlib import Path
from queue import Empty, Queue
import random
import sys
from tempfile import TemporaryDirectory
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

TEMPERATURES = (1.3, 1.5, 2.0)
OPENING_STEPS = 3
SEATS = range(4)
ACTION_KINDS = ('build', 'develop', 'network', 'sell', 'loan', 'scout', 'pass')
COMPOUND_KINDS = ('develop', 'network', 'sell')
MOVE_FIELDS = ('kind', 'target', 'industry', 'merchant')
DEFAULT_RUN = 'artifacts/step5/joint7-v11-2-g1190-restart-3'
DEFAULT_GENERATION = 2090
STUDY_FOLDER = 'opening-temperature-study'


def play_game(session, origin, spec, seed, temperature, max_decisions, stop):
    import numpy as np
    from brass_ai.learning.contracts import digest, require
    from brass_ai.rust_backend.lab_collection import frozen_id
    from brass_ai.rust_backend.lab_perf import admit
    from brass_ai.rust_backend.lab_transport import VERSION

    job = dict(seed=seed, sampling_seed=seed, policies=[spec] * 4)
    job_id = frozen_id(job)
    rngs = [random.Random(digest(dict(sampling_seed=seed, game=0, seat=seat))) for seat in SEATS]
    opening = []
    active = None
    completed_seats = set()
    with session.game(job, Counter(), stop) as (lane, policies):
        lane.begin(job_id, origin, max_decisions)
        while len(completed_seats) < len(SEATS) and not lane.current.get('terminal'):
            require(not stop.is_set(), 'Study interrupted')
            started = time.monotonic()
            decision = lane.current['decision']
            view, choices = admit(decision['view'], decision['choices'])
            decision.update(view=view, choices=choices)
            seat = view['seat']
            policy, rng = policies[seat], rngs[seat]
            forced = len(choices) == 1
            if forced:
                style, intent = policy.context_features(view, rng)
                raw = np.ones(1)
                chosen, probability, teacher = policy.sample_training(view, choices, rng, raw, intent)
            else:
                chosen, probability, teacher = policy.select_batch([(view, choices, rng)])[0]
                style, intent = policy.context_features(view, rng)
                raw = policy.native_raw

            choice = choices[chosen]
            behavior = dict(kind=policy.kind, policy_id=policy.policy_id, probability=probability,
                            **policy.last_behavior[rng])
            index = behavior['action_decision']
            phase = view['context']['phase']
            if index is not None:
                active = None
                if index < OPENING_STEPS:
                    require(phase == 'action' and behavior['temperature'] == temperature,
                            'Opening sampler did not use the requested temperature')
                    active = dict(seat=seat, move=index + 1, step=lane.current['step'],
                                  choice={key: choice[key] for key in MOVE_FIELDS if key in choice},
                                  followups=[])
                    opening.append(active)
            if phase == 'continuation' and active is not None and active['seat'] == seat:
                kind = choice['kind']
                if kind in COMPOUND_KINDS and kind == active['choice']['kind']:
                    active['followups'].append({key: choice[key] for key in MOVE_FIELDS if key in choice})

            row = dict(step=lane.current['step'], seat=seat, view=view, choices=choices, chosen=chosen,
                       teacher=teacher, style=style, intent=intent, seconds=time.monotonic() - started,
                       behavior=behavior)
            evidence = dict(kind='ordinary', model_id=spec['model_id'], policy_id=spec['policy_id'],
                            forced=forced, raw_probabilities=raw.tolist(), feature_version=VERSION,
                            semantic_id=choice['id'])
            lane.append(row, evidence)
            # A third action can still have payments or an additional development before Finish.
            if phase == 'continuation' and active is not None and active['seat'] == seat and choice['kind'] == 'finish':
                if active['move'] == OPENING_STEPS:
                    completed_seats.add(seat)
                active = None

        # Release the opening state without requesting a terminal-game receipt.
        lane.release()

    observed = {(row['seat'], row['move']) for row in opening}
    require(completed_seats == set(SEATS) and len(opening) == len(observed) and
            observed == {(seat, index) for seat in SEATS for index in range(1, OPENING_STEPS + 1)},
            f'Incomplete opening for setup seed {seed} at T={temperature}')
    return dict(seed=seed, temperature=temperature, opening=opening)


def play_batch(seeds, command, collector, execution, cpu, specs, stop, completed):
    from brass_ai.rust_backend.lab_cpu_inference import NoServiceEndpoint
    from brass_ai.rust_backend.lab_session import OrdinarySession
    from brass_ai.rust_backend.lab_transport import Pipe

    with Pipe(command, binary=False, timeout=120) as bootstrap, OrdinarySession(
            ROOT, collector, NoServiceEndpoint(), execution, cpu=cpu) as session:
        for seed in seeds:
            if stop.is_set():
                break
            origin = bootstrap.call('init', seed=seed)
            completed.put([
                play_game(session, origin, specs[temperature], seed, temperature, execution['max_decisions'], stop)
                for temperature in TEMPERATURES
            ])


def move_key(row):
    parts = [tuple(choice.get(key, '') for key in MOVE_FIELDS)
             for choice in (row['choice'], *row['followups'])]
    # Developing the same pair in the opposite order leaves the same industries developed.
    if parts[0][0] == 'develop':
        parts.sort()
    return tuple(parts)


def diversity(rows):
    counts = Counter(map(move_key, rows))
    total = sum(counts.values())
    return dict(samples=total, distinct=len(counts),
                most_common_share=max(counts.values(), default=0) / total if total else None)


def distribution(items, categories=()):
    counts = Counter(items)
    total = sum(counts.values())
    return dict(total=total, categories={
        key: dict(count=counts[key], share=counts[key] / total if total else None)
        for key in sorted(set(counts) | set(categories))
    })


def distributions(rows, board):
    from brass_ai.step5.policy import INDUSTRIES

    slot_to_town = {slot: town for town, slots in board['location_slots'].items() for slot in slots}
    builds = [row['choice'] for row in rows if row['choice']['kind'] == 'build']
    links = [part['target'] for row in rows if row['choice']['kind'] == 'network'
             for part in (row['choice'], *row['followups'])]
    return dict(
        action_types=distribution((row['choice']['kind'] for row in rows), ACTION_KINDS),
        build_industries=distribution((c['industry'] for c in builds), INDUSTRIES),
        build_towns=distribution((slot_to_town[c['target']] for c in builds), board['location_slots']),
        build_slots=distribution((c['target'] for c in builds), slot_to_town),
        build_towns_by_industry={
            industry: distribution((slot_to_town[c['target']] for c in builds if c['industry'] == industry),
                                   board['location_slots'])
            for industry in INDUSTRIES
        },
        network_links=distribution(links, board['link_endpoints']),
    )


def summarize(games, board):
    groups = {}
    first_by_seed_and_seat = {}
    for temperature in TEMPERATURES:
        tables = [game for game in games if game['temperature'] == temperature]
        rows = [row for game in tables for row in game['opening']]
        first = [row for row in rows if row['move'] == 1]
        groups[str(temperature)] = dict(
            games=len(tables),
            first_move_diversity=dict(pooled=diversity(first),
                                      by_seat={str(s): diversity([r for r in first if r['seat'] == s]) for s in SEATS},
                                      first_at_table=diversity([game['opening'][0] for game in tables])),
            all_three_moves=distributions(rows, board),
            by_move={str(index): distributions([r for r in rows if r['move'] == index], board)
                     for index in range(1, OPENING_STEPS + 1)},
        )
        first_by_seed_and_seat[temperature] = {
            (game['seed'], row['seat']): move_key(row)
            for game in tables for row in game['opening'] if row['move'] == 1
        }
    comparisons = {}
    for left, right in itertools.combinations(TEMPERATURES, 2):
        before, after = first_by_seed_and_seat[left], first_by_seed_and_seat[right]
        if before.keys() != after.keys():
            raise ValueError('Temperature groups have different seed/seat coverage')
        changed = sum(before[key] != after[key] for key in before)
        comparisons[f'{left} vs {right}'] = dict(pairs=len(before), changed=changed,
                                                 share=changed / len(before))
    return dict(temperatures=groups, paired_first_move_changes=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=DEFAULT_RUN, help='Read-only training run')
    parser.add_argument('--generation', type=int, default=DEFAULT_GENERATION,
                        help=f'Checkpoint generation (default: {DEFAULT_GENERATION})')
    parser.add_argument('--checkpoint', help='Alternate location for the same verified checkpoint')
    parser.add_argument('--games-per-temperature', type=int, default=320)
    parser.add_argument('--seed-start', type=int, default=20800000, help='Start of the validation seed scan')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--mono', default='mono')
    parser.add_argument('--output', help=f'New report directory (default: results/{STUDY_FOLDER}/g<generation>-<UTC>)')
    args = parser.parse_args()
    if args.games_per_temperature < 1 or args.workers < 1 or not 0 <= args.seed_start < 2**31:
        parser.error('Use positive game and worker counts and a setup seed in [0, 2**31).')

    import torch
    from brass_ai.bank import sha
    from brass_ai.learning.board_topology import topology
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.learning.contracts import project_path, read_json, require
    from brass_ai.recovery.execution import neural_spec
    from brass_ai.rust_backend import lab_cpu_models, lab_cpu_runtime
    from brass_ai.rust_backend.lab_backend import check_build
    from brass_ai.rust_backend.lab_paths import runtime_path
    from brass_ai.step5.league import seeds_for, write_json

    run = project_path(ROOT, args.run)
    state = read_json(run / 'state.json')
    entries = [entry for entry in state['history'] if entry['generation'] == args.generation]
    require(len(entries) == 1, f'The run history does not uniquely identify g{args.generation}')
    entry = entries[0]
    checkpoint = project_path(ROOT, args.checkpoint or entry['path'])
    require(checkpoint.is_file(), f'g{args.generation} checkpoint missing: {checkpoint}. '
            'Restore that checkpoint or supply its archived path with --checkpoint.')
    require(sha(checkpoint) == entry['sha256'], f'g{args.generation} checkpoint SHA-256 mismatch')
    config = state['signature']['config']
    opening = config['opening']
    require(opening['action_decisions_per_seat'] == OPENING_STEPS and opening['seats'] == 'neural',
            'The run config no longer has the three-decision neural opening recipe')
    execution = config['execution']
    build = check_build(state['native_backend']['build'])
    runtime = lab_cpu_runtime.resolve(ROOT)
    command = [str(runtime_path(build['bootstrap']['binary']['path']))]
    if os.name != 'nt':
        command.insert(0, args.mono)
    collector = runtime_path(runtime['collector']['path'])
    seeds, _ = seeds_for(args.seed_start, args.games_per_temperature, 'validation')
    specs = {temperature: neural_spec(entry['model_id'], 0., 'neutral', version=3,
                                     opening=dict(opening, temperature=temperature))
             for temperature in TEMPERATURES}
    board = topology()

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = project_path(ROOT, args.output or f'results/{STUDY_FOLDER}/g{args.generation}-{stamp}')
    require(not output.is_relative_to(run), 'Write study artifacts outside the training run')
    require(not output.exists(), f'Report directory already exists: {output}')
    torch.set_num_threads(2)
    model, checkpoint_data = load_checkpoint(checkpoint)
    require(checkpoint_data['step'] == args.generation and checkpoint_data['model_id'] == entry['model_id'],
            'Loaded checkpoint generation or model ID differs from the run history')
    del checkpoint_data
    scratch = project_path(ROOT, f'artifacts/{STUDY_FOLDER}')
    require(not scratch.is_relative_to(run), 'Write the temporary CPU model outside the training run')
    scratch.mkdir(parents=True, exist_ok=True)
    workers = min(args.workers, len(seeds))
    total_openings = len(seeds) * len(TEMPERATURES)
    print(f'g{args.generation} | {len(seeds)} paired setups | {total_openings} openings | '
          f'{workers} CPU lanes | {output}', flush=True)

    stop = threading.Event()
    completed = Queue()
    games = []
    with (
        TemporaryDirectory(prefix=f'g{args.generation}-', dir=scratch) as temporary,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        bundle = Path(temporary) / 'cpu-model'
        print('Preparing the frozen CPU model', flush=True)
        lab_cpu_models.export_bundle(model.eval(), bundle)
        del model
        manifest = bundle / 'manifest.json'
        cpu = dict(contract=lab_cpu_runtime.CONTRACT,
                   runtime=dict(path=str(runtime_path(runtime['runtime']['path'])), sha256=runtime['runtime']['sha256']),
                   bundles={entry['model_id']: dict(manifest=str(manifest), sha256=sha(manifest))}, model_cache=1)
        tasks = [
            pool.submit(play_batch, seeds[index::workers], command, collector, execution, cpu, specs, stop, completed)
            for index in range(workers)
        ]
        for task in tasks:
            task.add_done_callback(completed.put)
        finished_lanes = 0
        print('Collecting paired openings', flush=True)
        progress_interval = 16 * len(TEMPERATURES)
        try:
            while finished_lanes < workers:
                try:
                    update = completed.get(timeout=1)
                except Empty:
                    continue
                if isinstance(update, list):
                    games.extend(update)
                    count = len(games)
                    if count == len(TEMPERATURES) or count % progress_interval == 0 or count == total_openings:
                        print(f'Finished {count}/{total_openings} openings', flush=True)
                else:
                    update.result()
                    finished_lanes += 1
        except BaseException:
            stop.set()
            raise
    require({(game['seed'], game['temperature']) for game in games}
            == set(itertools.product(seeds, TEMPERATURES)) and len(games) == total_openings,
            'Incomplete or duplicated paired openings')
    games.sort(key=lambda game: (game['seed'], game['temperature']))
    report = dict(checkpoint=dict(path=str(checkpoint), **{k: entry[k] for k in ('generation', 'model_id', 'sha256')}),
                  sampling=dict(setup_seeds=seeds, sampling_seeds=seeds, seed_partition='validation',
                                scope='opening-only', actions_per_seat=OPENING_STEPS, other_decisions_temperature=1.,
                                epsilon=0., seats='all four use the same frozen checkpoint'),
                  network_link_endpoints=board['link_endpoints'], **summarize(games, board))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'opening-actions.json', games, compact=True)
    write_json(output / 'summary.json', report)
    print('Temperature  First-at-table  Pooled  Seat 0  Seat 1  Seat 2  Seat 3  (distinct actions)')
    for temperature, group in report['temperatures'].items():
        first = group['first_move_diversity']
        print(f"{temperature:>11}  {first['first_at_table']['distinct']:>14}  {first['pooled']['distinct']:>6}  "
              + '  '.join(f"{first['by_seat'][str(seat)]['distinct']:>6}" for seat in SEATS))
    print(f"Summary: {output / 'summary.json'}")


if __name__ == '__main__':
    main()
