"""Eight isolated four-player K16/K8 games over six seeds, two tables at once.

Reuses the frozen, previously audited headless rules/observation/ledger runner.
Each move uses the installed search_v1 CPU model and two one-thread sessions.
No installed game, preference, saved-game or training files are modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT/'artifacts/search-v1-k-matches/4s-k8-k16-eight/run_matches.py'
HARNESS_SHA = 'ed7142e1eb2568a7f9bbc1cebbfacf421e344ffbe925642780ad10cb361c84e0'
DEFAULT_SEEDS = (842611, 842612, 842613, 842614, 842615, 842616)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_harness(output):
    if sha(HARNESS) != HARNESS_SHA:
        raise ValueError('Frozen match harness changed')
    for relative in ('artifacts/r2-pythondeps', 'artifacts/r4-pythondeps',
                     'artifacts/r0-pythonlibs', 'artifacts/r0-validation-deps', 'src'):
        sys.path.insert(0, str(ROOT/relative))
    spec = importlib.util.spec_from_file_location('frozen_nn_k_matches', HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT, module.ROOT = output, ROOT
    return module


def schedule(seeds, arms):
    # Six base tables, then swap seats for the first two seeds. Four games of
    # each seating pattern in total; two paired seeds and four single seeds.
    assignments = [(seed, index % 2) for index, seed in enumerate(seeds)]
    assignments += [(seed, 1-index % 2) for index, seed in enumerate(seeds[:2])]
    return [dict(id=f'seed-{seed}-swap-{swap}', seed=seed, pair=seeds.index(seed),
                 swap=swap, seats=[arms[(seat+swap) % 2] for seat in range(4)])
            for seed, swap in assignments]


def prepare(h, out, seeds):
    out.mkdir(parents=True, exist_ok=False)
    installed = ROOT/'BrassNative/alpha-g890'
    native = out/'native'
    shutil.copytree(installed, native)
    original = {p.relative_to(installed).as_posix(): sha(p)
                for p in sorted(installed.rglob('*')) if p.is_file()}
    for name, digest in original.items():
        if sha(native/name) != digest:
            raise ValueError('Native package changed while copying')
    build = h.read(native/'build.json')
    for entry in build['files']:
        if sha(native/entry['path']) != entry['sha256']:
            raise ValueError('Installed native package pin differs')
    for arm in h.ARMS:
        k = h.ARM_K[arm]
        name = 'policy.json' if k == 16 else 'policy-k8.json'
        policy = h.read(native/name)
        if policy['method'] != 'search_v1' or set(policy['k_by_allowance_ms'].values()) != {k}:
            raise ValueError('Expected installed search_v1 K8/K16 policies')
        if policy.get('policy_temperature', 1.0) != 1.0:
            raise ValueError('Tournament requires T=1.0 for both settings')
        h.write(native/(arm+'.json'), policy)
    host = out/'host'
    host.mkdir()
    source = ROOT/'artifacts/training-host/search'
    runtime = h.read(source/'runtime.json')
    for entry in runtime['files']:
        if Path(entry['name']).name != entry['name'] or sha(source/entry['name']) != entry['sha256']:
            raise ValueError('Frozen rules host changed')
        shutil.copyfile(source/entry['name'], host/entry['name'])
    shutil.copyfile(source/'runtime.json', host/'runtime.json')
    paths = [ROOT/'artifacts/patch-installation.json', HARNESS, Path(__file__).resolve()]
    paths += [ROOT/'BrassBirminghamUnity_Data/Managed'/n
              for n in ('Assembly-CSharp.dll', 'BrassPatch.dll', 'BrassLegacyAI.dll')]
    paths += [ROOT/n for n in ('src/brass_ai/host.py', 'src/brass_ai/learning/reward.py',
              'src/brass_ai/learning/contracts.py', 'src/brass_ai/learning/replay.py')]
    snapshot = out/'source'
    for path in paths:
        target = snapshot/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    games = schedule(seeds, h.ARMS)
    assert len(games) == 8 and len({g['seed'] for g in games}) == 6
    assert sum(g['swap'] == 0 for g in games) == 4
    protocol = dict(format='brass-nn-k-tournament/1.0.0',
        created_utc=datetime.now(timezone.utc).isoformat(), games=games, seeds=seeds,
        concurrent_games=2, players_per_game=4, allowance_ms_per_whole_turn=4000,
        requested_k_by_arm=h.ARM_K, search_method='search_v1', sessions_per_search=2,
        compute_threads_per_session=1, policy_temperature=1.0, v205_enabled=False,
        delivery_grace_ms=1000, individual_objective=True, cooperative_teams=False,
        schedule='Six seeds once, first two repeated with swapped K seats; eight tables total',
        engine='Qualified C# real game state; permitted observations only; Rust search/simulation',
        policy_sha256={a: sha(native/(a+'.json')) for a in h.ARMS},
        native_binary_sha256=sha(native/'brass-ai-local.exe'),
        model_manifest_sha256=sha(native/'model/manifest.json'),
        source_sha256=build['source_sha256'], original_installed_native=original,
        protected_files={str(p): sha(p) for p in paths},
        host_runtime_sha256=sha(host/'runtime.json'), script_sha256=HARNESS_SHA,
        statistics='Six unique seeds, two paired seeds; small exploratory screen, no significance claim')
    h.write(out/'protocol.json', protocol)
    return protocol


def summary(h, results):
    rows = []
    for result in results:
        seats = result['result']['seats']
        means = {a: statistics.mean(s['vp'] for s in seats if s['arm'] == a) for a in h.ARMS}
        rows.append(dict(seed=result['game']['seed'], swap=result['game']['swap'],
            scores={a: [s['vp'] for s in seats if s['arm'] == a] for a in h.ARMS},
            k16_minus_k8_mean_vp=means[h.ARMS[1]]-means[h.ARMS[0]],
            winners=[s['arm'] for s in seats if s['rank'] == 1], seconds=result['seconds']))
    arms = {}
    for arm in h.ARMS:
        seats = [s for r in results for s in r['result']['seats'] if s['arm'] == arm]
        shares = [sum(s['utility'] for s in r['result']['seats'] if s['arm'] == arm) for r in results]
        arms[arm] = dict(k=h.ARM_K[arm], seats=len(seats),
            first_place_table_share=sum(shares), outright_table_wins=sum(v == 1 for v in shares),
            shared_table_wins=sum(0 < v < 1 for v in shares),
            mean_vp=statistics.mean(s['vp'] for s in seats), mean_rank=statistics.mean(s['rank'] for s in seats))
    per_seed = [dict(seed=seed, games=sum(r['seed'] == seed for r in rows),
        k16_minus_k8_mean_vp=statistics.mean(r['k16_minus_k8_mean_vp'] for r in rows if r['seed'] == seed))
        for seed in sorted({r['seed'] for r in rows})]
    return dict(passed=True, games=len(results), unique_seeds=len(per_seed),
        settings='search_v1; 4s per whole turn; K16 vs K8; 2 sessions x 1 thread; 2 concurrent games; T1.0',
        arms=arms, game_rows=rows, per_seed=per_seed,
        table_weighted_mean_vp_difference=statistics.mean(r['k16_minus_k8_mean_vp'] for r in rows),
        seed_weighted_mean_vp_difference=statistics.mean(r['k16_minus_k8_mean_vp'] for r in per_seed),
        all_games_terminal=all(r['terminal'] for r in results),
        limitations=['Eight tables over six seeds; only the first two seeds have swapped pairs',
            'Report individual competition, not cooperative teams',
            'Small timed-search screen; no established strength or significance claim'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, help='Fresh directory under artifacts/nn-k-tournaments')
    p.add_argument('--seeds', type=int, nargs=6, default=DEFAULT_SEEDS)
    p.add_argument('--games', type=int, choices=(8,), default=8)
    p.add_argument('--parallel', type=int, choices=(2,), default=2)
    p.add_argument('--prepare-only', action='store_true', help='Freeze inputs/schedule without running any games')
    p.add_argument('--smoke-turns', type=int, choices=(4,), help='Diagnostic only: run one four-player round, never a full table')
    a = p.parse_args()
    seeds = list(a.seeds)
    if len(set(seeds)) != 6 or any(s < 0 or s > 2147483647 for s in seeds):
        p.error('Use six distinct nonnegative Int32 seeds')
    if a.prepare_only and a.smoke_turns:
        p.error('Choose either preparation or a short smoke')
    boundary = ROOT/'artifacts/nn-k-tournaments'
    out = (a.output or boundary/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    if not out.is_relative_to(boundary.resolve()) or out == boundary.resolve():
        p.error('Output must be a new folder under artifacts/nn-k-tournaments')
    h = load_harness(out)
    protocol = prepare(h, out, seeds)
    h.verify_protected(protocol)
    h.note(dict(output=str(out), games=8, unique_seeds=6, concurrent_games=2,
                sessions_per_search=2, threads_per_session=1, allowance_ms=4000))
    if a.prepare_only:
        return
    if a.smoke_turns:
        result = h.play(protocol['games'][0], pilot_turns=a.smoke_turns)
        h.verify_protected(protocol)
        h.write(out/'smoke.json', dict(passed=result['passed'], diagnostic=True, full_games=0,
            turns=result['turns'], protected_files_unchanged=True))
        return
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(h.play, game): game for game in protocol['games']}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(dict(game=futures[future], error=str(error)))
                for pending in futures:
                    pending.cancel()
    h.verify_protected(protocol)
    if failures:
        h.write(out/'run-failure.json', dict(passed=False, completed=len(results), failures=failures))
        raise SystemExit('Tournament incomplete; failed games were not scored as wins. See '+str(out))
    if len(results) != 8 or not all(r['passed'] and r['terminal'] for r in results):
        raise RuntimeError('Eight completed terminal games required')
    report = summary(h, sorted(results, key=lambda r: (r['game']['seed'], r['game']['swap'])))
    report['protected_files_unchanged'] = True
    h.write(out/'summary.json', report)
    h.note(report)
    print('Summary: '+str(out/'summary.json'), flush=True)


if __name__ == '__main__':
    main()
