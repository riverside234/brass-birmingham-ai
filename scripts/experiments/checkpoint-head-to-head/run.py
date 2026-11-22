#!/usr/bin/env python3
"""Compare two saved checkpoints with seat-balanced 2v2 Rust CPU search games."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import partial
from itertools import combinations
import json
import math
import os
from pathlib import Path
from queue import SimpleQueue
import re
import shutil
from statistics import mean
import sys
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from brass_ai.bank import sha
from brass_ai.learning.contracts import digest, project_path, read_json, require


EXPERIMENT = 'checkpoint-head-to-head'
FORMAT = 'brass-checkpoint-head-to-head/1.0.0'
RUN = 'artifacts/step5/joint7-v11-2-g1190-restart-3'
SIMULATIONS, C_SCALE, K, W = 128, 0.10, 8, 0.0
BUDGET = dict(mode='count', full=SIMULATIONS, full_k=K, fast=SIMULATIONS, fast_k=K, c_scale=C_SCALE)
A_SEAT_PAIRS = tuple(combinations(range(4), 2))


def locate_checkpoint(selector, run, state):
    """Resolve a generation from the run's receipts, or a user-supplied file."""
    match = re.fullmatch(r'[gG]?([0-9]+)', selector)
    if match is None:
        path = Path(selector).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Checkpoint does not exist: {path}')
        return path, None

    generation = int(match[1])
    history = [entry for entry in state['history'] if entry['generation'] == generation]
    rating_agents = state.get('ratings', {}).get('agents', {}).values()
    rated = [agent['checkpoint'] for agent in rating_agents
             if agent.get('kind') == 'model' and not agent.get('reference_role')]
    candidates = [*history, *state.get('population', []), *rated]
    for entry in candidates:
        if entry['generation'] != generation or (history and entry['model_id'] != history[-1]['model_id']):
            continue
        path = project_path(ROOT, entry['path'])
        if path.is_file():
            return path, entry
    raise FileNotFoundError(f'Checkpoint g{generation} does not exist in {run}')


def load_model(path, receipt):
    from brass_ai.learning.checkpoint import load_checkpoint

    checksum = sha(path)
    require(receipt is None or checksum == receipt['sha256'], f'Checkpoint bytes changed: {path}')
    model, raw = load_checkpoint(path)
    entry = dict(path=str(path), sha256=checksum, generation=raw['step'], model_id=raw['model_id'],
                 weights_only=raw['optimizer'] is None)
    require(receipt is None or all(entry[key] == receipt[key] for key in ('generation', 'model_id')),
            f'Checkpoint identity differs: {path}')
    require(getattr(model.config, 'utility', None) is not None, f'Checkpoint needs a main utility head: {path}')
    return model.eval(), entry


def replace_hooks(path, replacements):
    text = path.read_text(encoding='utf-8')
    for before, after in replacements:
        require(text.count(before) == 1, f'Checkpoint arena hook changed in {path.name}: {before.strip()}')
        text = text.replace(before, after, 1)
    path.write_text(text, encoding='utf-8', newline='\n')


def patch_worker(directory):
    # The build is private to this experiment. The original worker is a single-model teacher.
    single_model = '''    check(
        job["seats"]
            .as_array()
            .is_some_and(|a| a.len() == 4 && a.iter().all(|v| *v == expected["model_id"])),
        "teacher models differ",
    )?;
'''
    two_models = '''    let seat_models: [String; 4] = serde_json::from_value(job["seats"].clone())?;
    let primary = expected["model_id"].as_str().ok_or("arena primary model")?;
    let opponent = job["opponent_model_id"].as_str().ok_or("arena opponent model")?;
    check(
        all_seat && h2h && request["op"] == "run" && job["teacher_revision"].is_null()
            && job["search_value"]["w"].as_f64() == Some(0.)
            && job["seat_values"].as_array().is_some_and(|values|
                values.iter().all(|value| *value == job["search_value"]))
            && primary != opponent
            && seat_models.iter().filter(|id| id.as_str() == primary).count() == 2
            && seat_models.iter().filter(|id| id.as_str() == opponent).count() == 2,
        "expected two seats per checkpoint with zero search blend",
    )?;
'''
    replace_hooks(directory / 'teacher.rs', (
        (single_model, two_models),
        ('model_id: expected["model_id"].as_str().ok_or("model")?.into(),',
         'model_id: seat_models[seat].clone(),'),
        ('crate::cpu_eval::teacher_evaluate(tensors, self.search_value.as_ref())?',
         'crate::cpu_eval::teacher_evaluate(tensors, self.search_value.as_ref(), &info.key.context.model_id)?'),
        ('"model_id":expected["model_id"],"reward_id":expected["reward_id"],',
         '"model_id":root.context.model_id,"reward_id":expected["reward_id"],'),
        ('"c_scale":turn_work.c_scale}));',
         '"c_scale":turn_work.c_scale,"model_id":root.context.model_id,\n'
         '                    "w":peer.search_value.as_ref().map(|value| &value["w"])}));'),
    ))
    replace_hooks(directory / 'cpu_eval.rs', (
        ('struct Teacher {\n    model: Model,',
         'struct Teacher {\n    model: Model,\n    opponent_model: Model,\n    opponent_positions: u64,'),
        ('    let budget = budget(&block["budget"])?;', '''    let opponent_model = Model::load(&block["opponent_bundle"])?;
    check(
        request["job"]["opponent_model_id"] == opponent_model.model_id.as_str()
            && model.reward_id == opponent_model.reward_id
            && (opponent_model.max_cost - max_cost).abs() < 1e-12,
        "CPU opponent bundle model/reward differs from the frozen job",
    )?;
    let budget = budget(&block["budget"])?;'''),
        ('Some(Teacher {\n            model,',
         'Some(Teacher {\n            model,\n            opponent_model,\n            opponent_positions: 0,'),
        ('    search_value: Option<&Value>,\n) -> Result<(Vec<f64>, [f64; 4])> {',
         '    search_value: Option<&Value>,\n    model_id: &str,\n) -> Result<(Vec<f64>, [f64; 4])> {'),
        ('        let raw = teacher.model.forward(t, &t.remaining)?;', '''        let raw = if teacher.model.model_id == model_id {
            teacher.model.forward(t, &t.remaining)?
        } else {
            check(teacher.opponent_model.model_id == model_id, "unknown arena model")?;
            teacher.opponent_positions += 1;
            teacher.opponent_model.forward(t, &t.remaining)?
        };'''),
        ('"model":t.model.receipt()}),',
         '"model":t.model.receipt(),"opponent_model":t.opponent_model.receipt(),\n'
         '            "opponent_positions":t.opponent_positions}),'),
    ))


def build_worker(build, scratch, toolchain):
    from brass_ai.rust_backend.lab_contract import native_fingerprint
    from brass_ai.rust_backend.lab_paths import runtime_path

    sys.path.insert(0, str(ROOT / 'scripts/experiments/policy-c-scale'))
    from _study import build_binary

    checkout = runtime_path(build['teacher']['path']).parents[3] / 'checkout'
    core = build['teacher_execution_core']
    require((checkout / 'lab-native/src/cpu_eval.rs').is_file() and native_fingerprint(checkout / 'native') == core,
            f'Missing or changed pinned CPU teacher checkout: {checkout}')
    for name in ('native', 'lab-native'):
        shutil.copytree(checkout / name, scratch / name,
                        ignore=shutil.ignore_patterns('target', '.git', '__pycache__'))
    patch_worker(scratch / 'lab-native/src')
    return build_binary(scratch / 'lab-native/Cargo.toml', scratch / 'target', 'brass-lab-teacher', toolchain, teacher=True)


def plan_jobs(entries, reward, games, seed):
    from brass_ai.learning.search_value import binding
    from brass_ai.recovery.search_value_config import fast_profile
    from brass_ai.recovery.teacher_phase import readiness, reserved_rounds
    from brass_ai.recovery.teacher_phase_config import defaults
    from brass_ai.rust_backend.lab_teacher import source_fingerprint
    from brass_ai.step5.league import seeds_for

    a, b = entries
    reward_id = digest(reward)
    generation = a['generation']
    expected = dict(generation=generation, model_id=a['model_id'], export_hash=a['sha256'],
                    reward_id=reward_id, source_fingerprint=source_fingerprint(), epoch=generation)
    config = defaults(reward)
    value = binding(W)
    ready, _ = readiness(None, None, reward_id, skip=True)
    template = dict(kind='teacher_phase', split='heldout', ordinary_training=False, h2h=True,
                    seating='all-seat-search', search=config['search'], fast_search=fast_profile(config['search']),
                    unsearched=config['unsearched'], schedule=config['schedule'], readiness=ready,
                    search_value=value, seat_values=[value] * 4, opponent_model_id=b['model_id'])
    seeds, _ = seeds_for(seed, games // len(A_SEAT_PAIRS), partition='validation')
    jobs = []
    for setup in seeds:
        sampling_seed = int(digest([FORMAT, seed, setup])[:16], 16)
        reservations = {str(seat): reserved_rounds(setup, sampling_seed, seat) for seat in range(4)}
        for a_seats in A_SEAT_PAIRS:
            models = [a['model_id'] if seat in a_seats else b['model_id'] for seat in range(4)]
            job_id = digest([FORMAT, a['sha256'], b['sha256'], setup, sampling_seed, a_seats, BUDGET, W])
            jobs.append(dict(template, expected=dict(expected, job_id=job_id, game_id=f'arena/{job_id}'),
                             seed=setup, sampling_seed=sampling_seed, reservations=reservations, seats=models))
    return jobs


class ArenaTurns:
    def __init__(self, job):
        self.job = job
        self.index = 0

    def verify(self, frame, _received):
        self.index += 1
        row = frame['value_row']
        require(frame['job'] == self.job['expected'] and frame['index'] == self.index
                and frame['profile'] == 'full' and not frame['rows']
                and row['model_id'] == self.job['seats'][row['seat']]
                and row['search_value'] == self.job['search_value'], 'Checkpoint arena turn binding differs')


def score_game(job, result):
    metrics = result['metrics']
    inference = metrics['cpu_inference']
    a_id, b_id = job['expected']['model_id'], job['opponent_model_id']
    require(inference['budget'] == BUDGET and inference['model']['model_id'] == a_id
            and inference['opponent_model']['model_id'] == b_id
            and 0 < inference['opponent_positions'] < inference['positions'], 'Checkpoint arena CPU binding differs')
    searches = metrics['searches']
    require(searches and all(s['profile'] == 'full' and 1 <= s['effective_k'] <= K and not s['deadline_hit']
                            and s['completed'] == s['frozen_n'] == SIMULATIONS
                            and s['c_scale'] == C_SCALE and s['w'] == W and s['model_id'] == job['seats'][s['seat']]
                            for s in searches), f"Search settings or quota differ in game {job['expected']['job_id']}")
    terminal = result['result']
    seats = terminal['seats']
    require(terminal['objective'] == 'win-share/1.0.0' and [s['seat'] for s in seats] == list(range(4))
            and math.isclose(sum(s['utility'] for s in seats), 1.), 'Missing or invalid final scores')
    a, b = ([s for s in seats if job['seats'][s['seat']] == model_id] for model_id in (a_id, b_id))
    a_seats = [s['seat'] for s in a]
    scores_a, scores_b = ([(s['vp'], s['income'], s['cash']) for s in side] for side in (a, b))
    return dict(job_id=job['expected']['job_id'], seed=job['seed'], sampling_seed=job['sampling_seed'],
                a_seats=a_seats, model_ids=job['seats'], seats=seats, win_share_a=sum(s['utility'] for s in a),
                vp_difference=mean(s['vp'] for s in a) - mean(s['vp'] for s in b),
                rank_difference=mean(s['rank'] for s in a) - mean(s['rank'] for s in b),
                cross_checkpoint_outrank_a=mean(1. if x > y else .5 if x == y else 0. for x in scores_a for y in scores_b),
                quota_searches=len(searches), seconds=metrics['elapsed_ns'] / 1e9)


def play(job, *, binary, core, bootstrap, inference, timing, cancel):
    from brass_ai.rust_backend import lab_teacher
    from brass_ai.rust_backend.lab_transport import Pipe

    require(not cancel.is_set(), 'Checkpoint arena cancelled')
    with Pipe(bootstrap, binary=False, timeout=30) as pipe:
        origin = pipe.call('init', seed=job['seed'])
    request = dict(contract=lab_teacher.CONTRACT, op='run', job=job, origin=origin, core_source=core,
                   inference=inference, turn_handoff=lab_teacher.STREAM_TURNS, **timing)
    worker = lab_teacher.Worker(binary, request, None, cancel)
    return score_game(job, worker.collect(job['expected'], core, verifier=ArenaTurns(job)))


def summarize(games, entries, seed):
    import numpy as np

    groups = {}
    for game in games:
        groups.setdefault(game['seed'], []).append(game)
    require(groups and all(len(group) == len(A_SEAT_PAIRS)
                           and {tuple(row['a_seats']) for row in group} == set(A_SEAT_PAIRS) for group in groups.values()),
            'Each setup needs all six distinct seat assignments')
    keys = ('win_share_a', 'vp_difference', 'rank_difference', 'cross_checkpoint_outrank_a')
    means = np.array([[mean(row[key] for row in group) for key in keys] for _, group in sorted(groups.items())])
    summary = dict(zip(keys, means.mean(axis=0).tolist(), strict=True))
    bounds = [None] * len(keys)
    if len(means) > 1:
        draws = np.random.default_rng(seed).integers(0, len(means), size=(10_000, len(means)))
        bounds = np.quantile(means[draws].mean(axis=1), (.025, .975), axis=0).T.tolist()
    summary.update({f'{key}_ci95': bound for key, bound in zip(keys, bounds, strict=True)})
    by_checkpoint = []
    for side, entry in zip(('a', 'b'), entries, strict=True):
        seats = [s for game in games for s in game['seats'] if game['model_ids'][s['seat']] == entry['model_id']]
        by_checkpoint.append(dict(side=side, checkpoint=entry, seat_games=len(seats),
                                  seat_win_share=mean(s['utility'] for s in seats), mean_vp=mean(s['vp'] for s in seats),
                                  mean_rank=mean(s['rank'] for s in seats), top_two_rate=mean(s['rank'] <= 2 for s in seats),
                                  last_place_rate=mean(s['rank'] == 4 for s in seats)))
    interval = summary['win_share_a_ci95']
    winner = None
    if interval is not None and interval[0] > .5:
        winner = 'a'
    elif interval is not None and interval[1] < .5:
        winner = 'b'
    return dict(summary, games=len(games), setup_seeds=len(groups), by_checkpoint=by_checkpoint, winner_side=winner,
                win_share_b=1 - summary['win_share_a'], outright_wins_a=sum(g['win_share_a'] == 1. for g in games),
                outright_wins_b=sum(g['win_share_a'] == 0. for g in games),
                split_wins=sum(0. < g['win_share_a'] < 1. for g in games),
                interval_method='10,000 bootstrap resamples of full six-game setup groups' if interval is not None else None)


def run_games(args, state, models, entries, reward):
    from brass_ai.rust_backend import lab_cpu_models, lab_cpu_runtime, lab_teacher
    from brass_ai.rust_backend.lab_paths import runtime_path
    from brass_ai.rust_backend.lab_shadow import teacher_build
    from brass_ai.step5.league import write_json

    build = lab_cpu_runtime.resolve(ROOT)
    core = build['teacher_execution_core']
    bootstrap = [str(runtime_path(teacher_build(state)['bootstrap']['binary']['path']))]
    if os.name != 'nt':
        bootstrap.insert(0, args.mono)
    jobs = plan_jobs(entries, reward, args.games, args.seed)
    workers = min(len(jobs), args.workers)
    lab_cpu_runtime.raise_open_file_limit()
    cores = lab_cpu_runtime.core_allocation(workers)
    core_queue = SimpleQueue()
    for core_id in cores:
        core_queue.put(core_id)
    pin = (lambda: lab_cpu_runtime.pin_process(core_queue.get_nowait())) if cores else None

    label = f"g{entries[0]['generation']}-vs-g{entries[1]['generation']}"
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = project_path(ROOT, f'artifacts/experiments/{EXPERIMENT}/{label}-{stamp}')
    output.mkdir(parents=True)
    report = dict(format=FORMAT, checkpoints=entries, requested_games=args.games, workers=workers, seed=args.seed,
                  simulations=SIMULATIONS, c_scale=C_SCALE, k=K, w=W, reward=reward, core_sha256=core,
                  exploration='Search selection with no opening or cotton overrides', status='incomplete')
    write_json(output / 'plan.json', dict(report, jobs=[dict(job_id=j['expected']['job_id'], seed=j['seed'],
               sampling_seed=j['sampling_seed'], model_ids=j['seats']) for j in jobs]))
    print(f'{label} | {len(jobs)} games | {workers} workers | N={SIMULATIONS} C_scale={C_SCALE:g} K={K} w={W:g}', flush=True)
    print(f'Results: {output}', flush=True)
    began = monotonic()
    games = []
    cancel = Event()
    try:
        with TemporaryDirectory(prefix='arena-', dir=project_path(ROOT, 'artifacts'),
                                ignore_cleanup_errors=os.name == 'nt') as directory:
            scratch = Path(directory)
            binary = build_worker(build, scratch, args.toolchain)
            report['worker_sha256'] = sha(binary)
            bundles = []
            for side, model, entry in zip(('a', 'b'), models, entries, strict=True):
                destination = scratch / side
                manifest = lab_cpu_models.export_bundle(model, destination)
                require(manifest['model_id'] == entry['model_id'] and manifest['reward_id'] == digest(reward),
                        f'Exported checkpoint identity differs for side {side}')
                path = destination / 'manifest.json'
                bundles.append(dict(manifest=str(path), sha256=sha(path)))
            inference = dict(contract=lab_cpu_runtime.CONTRACT,
                             runtime=dict(path=str(runtime_path(build['runtime']['path'])), sha256=build['runtime']['sha256']),
                             bundle=bundles[0], opponent_bundle=bundles[1], budget=BUDGET)
            calibration = lab_teacher.count_calibration(core, jobs[0])
            timing = dict(estimate_ns=calibration['estimate_ns'], calibration_id=digest(calibration))
            one = partial(play, binary=binary, core=core, bootstrap=bootstrap, inference=inference, timing=timing, cancel=cancel)
            with (output / 'games.jsonl').open('w', encoding='utf-8') as log, \
                    ThreadPoolExecutor(max_workers=workers, initializer=pin) as pool:
                futures = [pool.submit(one, job) for job in jobs]
                try:
                    for task in as_completed(futures):
                        game = task.result()
                        games.append(game)
                        log.write(json.dumps(game, allow_nan=False, separators=(',', ':')) + '\n')
                        log.flush()
                        if len(games) % 10 == 0 or len(games) == len(jobs):
                            print(f'{len(games)}/{len(jobs)} games complete', flush=True)
                except BaseException:
                    cancel.set()
                    for task in futures:
                        task.cancel()
                    raise
        report.update(summarize(games, entries, args.seed), status='complete')
    except BaseException as error:
        report.update(error=str(error), games=len(games))
        raise
    finally:
        write_json(output / 'report.json', dict(report, wall_seconds=monotonic() - began))

    print('Side  Checkpoint   Win share   Mean VP   Mean rank   Top two   Fourth')
    for side in report['by_checkpoint']:
        print(f'{side["side"].upper():>4}  g{side["checkpoint"]["generation"]:<9} {2*side["seat_win_share"]:>9.1%}'
              f'   {side["mean_vp"]:>7.2f}   {side["mean_rank"]:>9.3f}   {side["top_two_rate"]:>7.1%}'
              f'   {side["last_place_rate"]:>6.1%}')
    interval = report['win_share_a_ci95']
    if interval is not None:
        print(f'A win share 95% CI {interval[0]:.1%}–{interval[1]:.1%}')
    print(f'A outranks B {report["cross_checkpoint_outrank_a"]:.1%}')
    verdict = f'{report["winner_side"].upper()} wins' if report['winner_side'] else 'Inconclusive'
    print(f'{verdict}. Report: {output / "report.json"}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=RUN, help='Lab run directory for generation lookup and bootstrap')
    parser.add_argument('--a', '--checkpoint-a', required=True, help='Side A generation (e.g. g4170) or .pt path')
    parser.add_argument('--b', '--checkpoint-b', required=True, help='Side B generation (e.g. g3590) or .pt path')
    parser.add_argument('--games', type=int, default=120, help='Positive multiple of six (default: 120)')
    parser.add_argument('--workers', type=int, default=16, help='Concurrent games (default: 16)')
    parser.add_argument('--seed', type=int, default=45_510_128)
    parser.add_argument('--mono', default='mono')
    parser.add_argument('--toolchain', default='1.99.1')
    args = parser.parse_args()
    if args.games <= 0 or args.games % len(A_SEAT_PAIRS) or args.workers <= 0:
        parser.error('Use a positive multiple of six games and at least one worker.')
    if not 0 <= args.seed < 2**31:
        parser.error('Use a setup seed in [0, 2^31).')
    run = project_path(ROOT, args.run)
    if not (run / 'state.json').is_file():
        parser.error(f'Run state does not exist: {run / "state.json"}')
    state = read_json(run / 'state.json')
    located, missing = [], []
    for side, selector in (('A', args.a), ('B', args.b)):
        try:
            located.append(locate_checkpoint(selector, run, state))
        except FileNotFoundError as error:
            missing.append(f'{side}: {error}')
    if missing:
        parser.error('\n'.join(missing))

    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import torch

    torch.set_num_threads(1)
    try:
        models, entries = zip(*(load_model(path, receipt) for path, receipt in located), strict=True)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if entries[0]['model_id'] == entries[1]['model_id']:
        parser.error('Choose two distinct checkpoints.')
    reward = models[0].config.utility
    if reward != models[1].config.utility:
        parser.error('The two checkpoints must use the same search utility reward contract.')
    run_games(args, state, models, entries, reward)


if __name__ == '__main__':
    main()
