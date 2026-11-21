#!/usr/bin/env python3
"""Compare search scales or simulation counts in a seat-balanced Rust CPU match."""

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
import shutil
import statistics
import sys
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/experiments/policy-c-scale'))
os.environ.setdefault('OMP_NUM_THREADS', '1')

from _study import build_binary, generation_entry
from brass_ai.bank import sha
from brass_ai.learning.contracts import digest, project_path, read_json, require
from brass_ai.learning.search_value import binding
from brass_ai.recovery.teacher_phase import plan_jobs, readiness
from brass_ai.rust_backend import lab_cpu_models, lab_cpu_runtime, lab_teacher
from brass_ai.rust_backend.lab_contract import native_fingerprint
from brass_ai.rust_backend.lab_paths import runtime_path
from brass_ai.rust_backend.lab_transport import Pipe
from brass_ai.step5.league import seeds_for, write_json


RUN = 'artifacts/step5/joint7-v11-2-g1190-restart-3'
GENERATION = 4138
GAMES = 120
WORKERS = 60
SIMULATIONS = 94
K = 8
DEFAULT_SCALES = (0.2, 0.7)
SEAT_ASSIGNMENTS = tuple(combinations(range(4), 2))
SEED = 41_380_200
FORMAT = 'brass-c-scale-head-to-head/2.1.0'
MATCH_NAMESPACE = 'brass-c-scale-head-to-head/1.0.0'  # Preserve setup sampling and game IDs.


class SearchConfig(NamedTuple):
    c_scale: float
    simulations: int
    w: float = 0.

    def label(self, show_weight=False):
        label = f'C_scale {self.c_scale}, n={self.simulations}'
        return f'{label}, w={self.w:g}' if show_weight else label


def patch_worker(source):
    guard = '    let h2h = job["h2h"] == true;\n'
    arena = '''    let scales: [f64; 4] = serde_json::from_value(job["arena_scales"].clone())?;
    let simulations: [u64; 4] = serde_json::from_value(job["arena_simulations"].clone())?;
    let bindings: [Value; 4] = serde_json::from_value(job["seat_values"].clone())?;
    let weights = bindings.each_ref().map(|v| v["w"].as_f64().unwrap_or(f64::NAN));
    let mut configs: [(f64, u64, f64); 4] =
        std::array::from_fn(|seat| (scales[seat], simulations[seat], weights[seat]));
    configs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.total_cmp(&b.2)));
    check(request["op"] == "run" && all_seat && h2h && job["teacher_revision"].is_null()
        && job["search_value"]["w"].as_f64() == Some(0.0)
        && bindings.iter().zip(weights).all(|(v, w)|
            v["protocol"] == job["search_value"]["protocol"]
            && (0.0..=1.0).contains(&w)
            && (w * 100.0 - (w * 100.0).round()).abs() < 1e-8
            && v["value_source"] == if w == 0.0 { "main" } else { "corrected" })
        && scales.iter().all(|scale| scale.is_finite() && *scale > 0.0)
        && simulations.iter().all(|n| (64..=4096).contains(n))
        && configs[0] == configs[1] && configs[1] != configs[2] && configs[2] == configs[3],
        "expected experiment table with two seats per search configuration")?;
'''
    scale_hook = '            turn_work.c_scale = crate::cpu_eval::teacher_c_scale(turn_work.fast);\n'
    count_hook = '                simulations: crate::cpu_eval::teacher_simulations(turn_work.fast, calibration),\n'
    metric_hook = '"c_scale":turn_work.c_scale}));'
    metric = '"c_scale":turn_work.c_scale,"w":peer.search_value.as_ref().map(|v| &v["w"])}));'
    for before, after in ((guard, guard + arena),
                          (scale_hook, '            turn_work.c_scale = scales[seat];\n'),
                          (count_hook, '                simulations: simulations[seat],\n'),
                          (metric_hook, metric)):
        require(source.count(before) == 1, f'CPU teacher experiment hook changed: {before.strip()}')
        source = source.replace(before, after, 1)
    return source


def arena_worker(build, scratch, toolchain):
    # The qualified CPU build already contains the production CPU overlays.
    source = runtime_path(build['teacher']['path']).parents[3] / 'checkout'
    core = build['teacher_execution_core']
    require((source / 'lab-native/src/cpu_eval.rs').is_file() and native_fingerprint(source / 'native') == core,
            f'Missing pinned CPU teacher checkout: {source}')
    for name in ('native', 'lab-native'):
        shutil.copytree(source / name, scratch / name,
                        ignore=shutil.ignore_patterns('target', '.git', '__pycache__'))
    teacher = scratch / 'lab-native/src/teacher.rs'
    teacher.write_text(patch_worker(teacher.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
    return build_binary(scratch / 'lab-native/Cargo.toml', scratch / 'target',
                        'brass-lab-teacher', toolchain, teacher=True)


def plan(state, entry, calibration, games, seed, matchup):
    generation = entry['generation']
    templates, _ = plan_jobs(state['signature']['config'], dict(state, iteration=generation),
                             model_id=entry['model_id'], export_hash=entry['sha256'], epoch=generation,
                             ready=True, native_source=lab_teacher.source_fingerprint())
    template = next((job for job in templates if job.get('seating') == 'all-seat-search'), None)
    require(template is not None, 'Expected an all-seat Rust teacher recipe')
    template.pop('teacher_revision', None)  # Both sides use search selection without extra exploration.
    reward_id = template['expected']['reward_id']
    receipt, _ = readiness(None, calibration, reward_id, skip=True)
    search_value = binding(0.)
    bindings = {config.w: binding(config.w) for config in matchup}
    seeds, _ = seeds_for(seed, games // len(SEAT_ASSIGNMENTS), partition='validation')
    jobs = []
    for setup in seeds:
        sampling_seed = int(digest([MATCH_NAMESPACE, seed, setup])[:16], 16)
        for a_seats in SEAT_ASSIGNMENTS:
            configs = [matchup[0] if seat in a_seats else matchup[1] for seat in range(4)]
            scales = [config.c_scale for config in configs]
            simulations = [config.simulations for config in configs]
            weights = [config.w for config in configs]
            quota = simulations[0] if matchup[0].simulations == matchup[1].simulations else simulations
            identity = [MATCH_NAMESPACE, entry['sha256'], setup, sampling_seed, scales, quota, K]
            if any(weights):
                identity.append(weights)
            key = digest(identity)
            expected = dict(template['expected'], job_id=key, game_id=f'arena/{key}')
            jobs.append(dict(template, expected=expected, seed=setup, sampling_seed=sampling_seed,
                             h2h=True, split='heldout', readiness=receipt, search_value=search_value,
                             seat_values=[bindings[weight] for weight in weights],
                             arena_scales=scales, arena_simulations=simulations))
    return jobs


class ArenaTurns:
    def __init__(self, job):
        self.job = job
        self.index = 0

    def verify(self, frame, _received):
        self.index += 1
        value = frame['value_row']
        require(frame['job'] == self.job['expected'] and frame['index'] == self.index
                and frame['profile'] == 'full' and not frame['rows']
                and value['search_value'] == self.job['seat_values'][value['seat']],
                'Experiment turn has the wrong identity, profile or value blend')


def score_game(job, result, matchup):
    searches = result['metrics']['searches']
    scales = job['arena_scales']
    simulations = job['arena_simulations']
    weights = [value['w'] for value in job['seat_values']]
    configs = list(zip(scales, simulations, weights))
    require(searches and all(search['profile'] == 'full' and 1 <= search['effective_k'] <= K
                            and not search['deadline_hit'] and search['completed'] == search['frozen_n']
                            and (search['c_scale'], search['frozen_n'], search['w']) == configs[search['seat']]
                            for search in searches),
            f"A search used the wrong quota, scale or value blend in game {job['expected']['job_id']}")
    terminal = result['result']
    seats = terminal['seats']
    require(terminal['objective'] == 'win-share/1.0.0' and [s['seat'] for s in seats] == list(range(4))
            and math.isclose(sum(s['utility'] for s in seats), 1.), 'Missing or invalid final scores')
    side_a, side_b = ([s for s in seats if configs[s['seat']] == config] for config in matchup)
    win_share_a = sum(s['utility'] for s in side_a)
    vp_a, vp_b = (statistics.mean(s['vp'] for s in side) for side in (side_a, side_b))
    rank_a, rank_b = (statistics.mean(s['rank'] for s in side) for side in (side_a, side_b))
    scores_a, scores_b = ([(s['vp'], s['income'], s['cash']) for s in side] for side in (side_a, side_b))
    outrank = statistics.mean(1. if a > b else .5 if a == b else 0. for a in scores_a for b in scores_b)
    return dict(job_id=job['expected']['job_id'], seed=job['seed'], sampling_seed=job['sampling_seed'],
                arena_scales=scales, arena_simulations=simulations, arena_weights=weights, seats=seats, win_share_a=win_share_a,
                mean_vp_a=vp_a, mean_vp_b=vp_b, vp_difference=vp_a-vp_b,
                mean_rank_a=rank_a, mean_rank_b=rank_b, rank_difference=rank_a-rank_b,
                cross_scale_outrank_a=outrank,
                quota_searches=len(searches), seconds=result['metrics']['elapsed_ns']/1e9)


def play(job, *, binary, core, bootstrap, inference, timing, cancel, matchup):
    require(not cancel.is_set(), 'Experiment cancelled')
    with Pipe(bootstrap, binary=False, timeout=30) as pipe:
        origin = pipe.call('init', seed=job['seed'])
    request = dict(contract=lab_teacher.CONTRACT, op='run', job=job, origin=origin, core_source=core,
                   inference=inference, turn_handoff=lab_teacher.STREAM_TURNS, **timing)
    native = lab_teacher.Worker(binary, request, None, cancel)
    result = native.collect(job['expected'], core, verifier=ArenaTurns(job))
    return score_game(job, result, matchup)


def summarize(games, seed, matchup):
    import numpy as np

    groups = {}
    for game in games:
        groups.setdefault(game['seed'], []).append(game)
    require(groups and all(len(rows) == len(SEAT_ASSIGNMENTS) for rows in groups.values()),
            'Each setup seed needs all six seat assignments')
    keys = ('win_share_a', 'vp_difference', 'rank_difference', 'cross_scale_outrank_a')
    means = np.array([[statistics.mean(row[key] for row in group) for key in keys]
                      for _, group in sorted(groups.items())])
    samples = np.random.default_rng(seed).integers(0, len(means), size=(10_000, len(means)))
    bounds = np.quantile(means[samples].mean(axis=1), (0.025, 0.975), axis=0)
    share, vp_diff, rank_diff, outrank = means.mean(axis=0).tolist()
    share_ci, vp_ci, rank_ci, outrank_ci = bounds.T.tolist()
    winner_index = 0 if share_ci[0] > .5 else 1 if share_ci[1] < .5 else None
    sides = ('a', 'b')
    winner = None if winner_index is None else matchup[winner_index]

    by_scale = []
    for side, config in zip(sides, matchup):
        seats = [seat for game in games
                 for seat, scale, count, w in zip(game['seats'], game['arena_scales'],
                                                   game['arena_simulations'], game['arena_weights'])
                 if (scale, count, w) == config]
        by_scale.append(dict(side=side, c_scale=config.c_scale, simulations=config.simulations, w=config.w, seat_games=len(seats),
                             seat_win_share=statistics.mean(s['utility'] for s in seats),
                             mean_vp=statistics.mean(s['vp'] for s in seats),
                             mean_rank=statistics.mean(s['rank'] for s in seats),
                             top_two_rate=statistics.mean(s['rank'] <= 2 for s in seats),
                             last_place_rate=statistics.mean(s['rank'] == 4 for s in seats)))
    return dict(games=len(games), setup_seeds=len(groups), by_scale=by_scale,
                winner=None if winner is None else winner.c_scale,
                winner_side=None if winner_index is None else sides[winner_index],
                winner_simulations=None if winner is None else winner.simulations,
                winner_w=None if winner is None else winner.w,
                win_share_a=share, win_share_b=1-share, win_share_a_ci95=share_ci,
                outright_wins_a=sum(g['win_share_a'] == 1. for g in games),
                outright_wins_b=sum(g['win_share_a'] == 0. for g in games),
                split_wins=sum(0. < g['win_share_a'] < 1. for g in games),
                vp_difference=vp_diff, vp_difference_ci95=vp_ci,
                rank_difference=rank_diff, rank_difference_ci95=rank_ci,
                cross_scale_outrank_a=outrank, cross_scale_outrank_a_ci95=outrank_ci,
                interval_method='10,000 bootstrap resamples of full six-game setup-seed groups')


def main(*, compare_weights=False):
    description = 'Compare search residual weights at the same scale and search count.' if compare_weights else __doc__
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--run', default=RUN)
    parser.add_argument('--generation', type=int, default=GENERATION)
    parser.add_argument('--games', type=int, default=GAMES, help=f'Positive multiple of six (default: {GAMES})')
    parser.add_argument('--workers', type=int, default=WORKERS, help=f'Concurrent games (default: {WORKERS})')
    if compare_weights:
        parser.add_argument('--c-scale', type=float, default=0.7, help='Shared C_scale (default: 0.7)')
        parser.add_argument('--simulations', type=int, default=96, help='Shared simulations per search (default: 96)')
        parser.add_argument('--w-a', type=float, default=0.0, help='Search residual weight for side A (default: 0)')
        parser.add_argument('--w-b', type=float, default=1.0, help='Search residual weight for side B (default: 1)')
    else:
        parser.add_argument('--c-scale', type=float, help='Use the same C_scale for both sides')
        parser.add_argument('--c-scale-a', type=float,
                            help=f'C_scale for side A (default: {DEFAULT_SCALES[0]})')
        parser.add_argument('--c-scale-b', type=float,
                            help=f'C_scale for side B (default: {DEFAULT_SCALES[1]})')
        parser.add_argument('--simulations-a', type=int, default=SIMULATIONS,
                            help=f'Simulations per search for side A (default: {SIMULATIONS})')
        parser.add_argument('--simulations-b', type=int, default=SIMULATIONS,
                            help=f'Simulations per search for side B (default: {SIMULATIONS})')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--mono', default='mono')
    parser.add_argument('--toolchain', default='1.99.1')
    args = parser.parse_args()
    if args.games < len(SEAT_ASSIGNMENTS) or args.games % len(SEAT_ASSIGNMENTS) or args.workers < 1:
        parser.error('Use a positive multiple of six games and at least one worker.')
    if not 0 <= args.seed < 2**31:
        parser.error('Use a setup seed in [0, 2^31).')
    if compare_weights:
        scales = [args.c_scale] * 2
        simulations = [args.simulations] * 2
        try:
            weights = [binding(weight)['w'] for weight in (args.w_a, args.w_b)]
        except ValueError:
            parser.error('--w-a and --w-b must be between 0 and 1 in steps of 0.01.')
    else:
        if args.c_scale is None:
            scales = [default if scale is None else scale
                      for scale, default in zip((args.c_scale_a, args.c_scale_b), DEFAULT_SCALES)]
        else:
            if args.c_scale_a is not None or args.c_scale_b is not None:
                parser.error('Use --c-scale by itself or specify --c-scale-a and --c-scale-b separately.')
            scales = [args.c_scale] * 2
        simulations = [args.simulations_a, args.simulations_b]
        weights = [0., 0.]
    if not all(math.isfinite(scale) and scale > 0 for scale in scales):
        parser.error('C_scale must be a finite positive number.')
    if not all(64 <= count <= 4096 for count in simulations):
        parser.error('Each simulation count must be between 64 and 4096.')
    matchup = tuple(SearchConfig(*config) for config in zip(scales, simulations, weights))
    if matchup[0] == matchup[1]:
        parser.error('Sides must have different weights.' if compare_weights
                     else 'Sides must differ in C_scale or simulation count.')

    state = read_json(project_path(ROOT, args.run) / 'state.json')
    entry = generation_entry(state, args.generation)
    checkpoint = project_path(ROOT, entry['path'])
    require(checkpoint.is_file() and sha(checkpoint) == entry['sha256'],
            f'Missing or changed g{args.generation} checkpoint: {checkpoint}')
    from brass_ai.learning.checkpoint import load_checkpoint
    from brass_ai.rust_backend.lab_shadow import teacher_build

    model, raw = load_checkpoint(checkpoint)
    require(raw['step'] == args.generation and raw['model_id'] == entry['model_id'], 'Checkpoint identity differs')
    if compare_weights:
        require(hasattr(model, 'search_value_head'), f'g{args.generation} has no Search Value head')
    model.eval()
    jobs = plan(state, entry, raw['training'].get('utility_calibration'), args.games, args.seed, matchup)
    del raw
    build = lab_cpu_runtime.resolve(ROOT)
    core = build['teacher_execution_core']
    bootstrap = [str(runtime_path(teacher_build(state)['bootstrap']['binary']['path']))]
    if os.name != 'nt':
        bootstrap.insert(0, args.mono)
    lab_cpu_runtime.raise_open_file_limit()
    workers = min(args.workers, len(jobs))
    cores = lab_cpu_runtime.core_allocation(workers)
    core_queue = SimpleQueue()
    for cpu in cores:
        core_queue.put(cpu)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    experiment = 'search-value-head-to-head' if compare_weights else 'c-scale-head-to-head'
    output = ROOT / 'artifacts/experiments' / experiment / f'g{args.generation}-{stamp}'
    output.mkdir(parents=True)
    report_format = 'brass-search-value-head-to-head/1.0.0' if compare_weights else FORMAT
    report = dict(format=report_format, checkpoint=entry, requested_games=args.games, workers=workers, seed=args.seed,
                  simulations=simulations[0] if simulations[0] == simulations[1] else None,
                  simulations_by_side=simulations, k=K, w=None if compare_weights else 0.,
                  weights_by_side=weights, scales=scales, core_sha256=core,
                  exploration='Search selection with no opening or cotton overrides', status='incomplete')
    write_json(output / 'plan.json', dict(report, jobs=[dict(job_id=j['expected']['job_id'], seed=j['seed'],
                                                            sampling_seed=j['sampling_seed'], scales=j['arena_scales'],
                                                            simulations=j['arena_simulations'],
                                                            weights=[value['w'] for value in j['seat_values']])
                                                       for j in jobs]))
    search = f'K={K}' if compare_weights else f'K={K} w=0'
    print(f'g{args.generation}: {args.games} games | {workers} workers | {search} | '
          f'A ({matchup[0].label(compare_weights)}) vs B ({matchup[1].label(compare_weights)})', flush=True)
    print(f'Results: {output}', flush=True)
    began = monotonic()
    games = []
    cancel = Event()
    try:
        with TemporaryDirectory(prefix=f'brass-{experiment}-', dir=ROOT / 'artifacts',
                                ignore_cleanup_errors=os.name == 'nt') as directory:
            scratch = Path(directory)
            binary = arena_worker(build, scratch, args.toolchain)
            report['worker_sha256'] = sha(binary)
            # A private model pack cannot be pruned by ongoing lab training.
            manifest = scratch / 'model/manifest.json'
            lab_cpu_models.export_bundle(model, manifest.parent)
            # The worker uses each seat's count and scale. The shared budget sets K and the watchdog allowance.
            max_count = max(simulations)
            inference = dict(contract=lab_cpu_runtime.CONTRACT,
                             runtime=dict(path=str(runtime_path(build['runtime']['path'])), sha256=build['runtime']['sha256']),
                             bundle=dict(manifest=str(manifest), sha256=sha(manifest)),
                             budget=dict(mode='count', full=max_count, full_k=K, fast=max_count, fast_k=K))
            calibration = lab_teacher.count_calibration(core, jobs[0])
            timing = dict(estimate_ns=calibration['estimate_ns'], calibration_id=digest(calibration))
            one = partial(play, binary=binary, core=core, bootstrap=bootstrap, inference=inference, timing=timing,
                          cancel=cancel, matchup=matchup)
            pin = (lambda: lab_cpu_runtime.pin_process(core_queue.get_nowait())) if cores else None
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
        report.update(summarize(games, args.seed, matchup), status='complete')
    except BaseException as error:
        report.update(error=str(error), games=len(games))
        raise
    finally:
        write_json(output / 'report.json', dict(report, wall_seconds=monotonic()-began))
    lo, hi = report['win_share_a_ci95']
    weight_header = '     w' if compare_weights else ''
    print(f'Side   C_scale    Sims{weight_header}   Win share   Mean VP   Mean rank   Top two   Fourth', flush=True)
    for side in report['by_scale']:
        weight_cell = f'{side["w"]:>6g}' if compare_weights else ''
        print(f'{side["side"].upper():>4}   {str(side["c_scale"]):>7}   {side["simulations"]:>5}{weight_cell}   '
              f'{2*side["seat_win_share"]:>9.1%}   {side["mean_vp"]:>7.2f}   '
              f'{side["mean_rank"]:>9.3f}   {side["top_two_rate"]:>7.1%}   {side["last_place_rate"]:>6.1%}', flush=True)
    print(f'A ({matchup[0].label(compare_weights)}): win share 95% CI {lo:.1%}–{hi:.1%} | '
          f'cross-side outrank {report["cross_scale_outrank_a"]:.1%}', flush=True)
    verdict = 'inconclusive'
    if report['winner_side'] is not None:
        winner = SearchConfig(report['winner'], report['winner_simulations'], report['winner_w'])
        verdict = f'{report["winner_side"].upper()} ({winner.label(compare_weights)}) wins'
    print(f'{verdict}. Report: {output / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
