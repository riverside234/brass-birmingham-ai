#!/usr/bin/env python3
"""Short matched search_v1 vs search_v3 smoke on the frozen local game harness.

One executable and one model; the arms differ only by the policy's ``method``.
Seats swap across each paired seed. A handful of games cannot establish
strength, so the scheduling metrics are the main evidence.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
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
ARMS = ('search_v1_k16', 'search_v3_k16')
PROFILE = 'beta-g1290-neutral/1.0.0'
SEEDS = (913301, 913302, 913303, 913304, 913305, 913306)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(out):
    if sha(HARNESS) != HARNESS_SHA:
        raise ValueError('Frozen match harness changed')
    for relative in ('artifacts/r2-pythondeps', 'artifacts/r4-pythondeps', 'artifacts/r0-pythonlibs',
                     'artifacts/r0-validation-deps', 'src'):
        sys.path.insert(0, str(ROOT/relative))
    spec = importlib.util.spec_from_file_location('frozen_search_v3_matches', HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT, module.ROOT = out, ROOT
    module.ARMS, module.ARM_K, module.PROFILE = ARMS, {a: 16 for a in ARMS}, PROFILE
    # The frozen harness records recall only after whole actions. For a second
    # request in one turn the session owns the recall; an empty list says so.
    search = module.Worker.search

    def delegated(self, value, game_id, generation, remaining_ms):
        key = (game_id, generation)
        if getattr(self, 'last_turn', None) == key:
            value = dict(value, recall=[])
        self.last_turn = key
        return search(self, value, game_id, generation, remaining_ms)
    module.Worker.search = delegated
    return module


def prepare(h, out, package, games):
    out.mkdir(parents=True)
    native = out/'native'
    shutil.copytree(package, native)
    policy = h.read(native/'policy.json')
    if policy['method'] != 'search_v3' or policy.get('policy_temperature', 1.0) != 1.0:
        raise ValueError('Expected the built search_v3 K16 policy')
    for arm in ARMS:
        h.write(native/(arm+'.json'), dict(policy, method=arm.rsplit('_k', 1)[0]))
    host = out/'host'
    host.mkdir()
    source = ROOT/'artifacts/training-host/search'
    runtime = h.read(source/'runtime.json')
    for entry in runtime['files']:
        if Path(entry['name']).name != entry['name'] or sha(source/entry['name']) != entry['sha256']:
            raise ValueError('Frozen rules host changed')
        shutil.copyfile(source/entry['name'], host/entry['name'])
    shutil.copyfile(source/'runtime.json', host/'runtime.json')
    seeds = SEEDS[:(games+1)//2]
    schedule = [dict(id=f'seed-{seed}-swap-{swap}', seed=seed, pair=i, swap=swap, seats=[ARMS[(seat+swap) % 2] for seat in range(4)])
                for i, seed in enumerate(seeds) for swap in (0, 1)][:games]
    protocol = dict(format='brass-search-v3-smoke/1.0.0', created_utc=datetime.now(timezone.utc).isoformat(), games=schedule,
        allowance_ms_per_whole_turn=4000, arms=ARMS, requested_k=16, profile=PROFILE, sessions_per_search=2,
        native_binary_sha256=sha(native/'brass-ai-local.exe'), source_sha256=h.read(native/'build.json')['source_sha256'],
        comparison='One executable and model; only policy.method differs', statistics='Exploratory; no significance claim')
    h.write(out/'protocol.json', protocol)
    return protocol


def searches(out, game):
    rows = []
    with gzip.open(out/game['id']/'trusted-trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            entry = json.loads(line)
            if 'response' not in entry:
                continue
            reply = entry['response']
            owners = reply['replay']['owners']
            active = [o for o in owners if o['quota'] is not None]
            if not active:
                continue
            rows.append(dict(arm=game['seats'][reply['ticket']['actor']], completed=reply['completed'],
                frozen=sum(o['quota']['simulations'] for o in active),
                rounds=min((o['snapshot'] or {}).get('completed_rounds', 0) for o in active),
                full=all((o['snapshot'] or {}).get('reason') == 'quota_complete' for o in active),
                k=max(o['effective_k'] for o in active), provenance=reply['provenance'],
                last_ms=max((o['last_completion_ns'] or 0) for o in active)/1e6))
    return rows


def summarize(h, out, results):
    games = [r['game'] for r in results]
    rows = [row for game in games for row in searches(out, game)]
    arms = {}
    for arm in ARMS:
        mine = [r for r in rows if r['arm'] == arm]
        seats = [s for r in results for s in r['result']['seats'] if s['arm'] == arm]
        shares = [sum(s['utility'] for s in r['result']['seats'] if s['arm'] == arm) for r in results]
        full = [r for r in mine if r['k'] >= 16]
        arms[arm] = dict(searches=len(mine), mean_trajectories=statistics.mean(r['completed'] for r in mine),
            median_trajectories=statistics.median(r['completed'] for r in mine),
            mean_frozen_quota=statistics.mean(r['frozen'] for r in mine),
            mean_rounds_k16=statistics.mean(r['rounds'] for r in full) if full else None,
            finished_all_rounds_k16=sum(r['rounds'] >= 4 for r in full)/len(full) if full else None,
            quota_complete=sum(r['full'] for r in mine)/len(mine),
            mean_last_completion_ms=statistics.mean(r['last_ms'] for r in mine),
            provenance=dict(Counter(r['provenance'] for r in mine)),
            seats=len(seats), mean_vp=statistics.mean(s['vp'] for s in seats), mean_rank=statistics.mean(s['rank'] for s in seats),
            first_place_table_share=sum(shares))
    report = dict(passed=True, games=len(results), arms=arms,
        vp_difference_v3_minus_v1=arms[ARMS[1]]['mean_vp']-arms[ARMS[0]]['mean_vp'],
        trajectory_ratio_v3_over_v1=arms[ARMS[1]]['mean_trajectories']/arms[ARMS[0]]['mean_trajectories'],
        all_terminal=all(r['terminal'] for r in results), seconds=sum(r['seconds'] for r in results),
        limitations=['A few tables only; game outcomes are not significant', 'Scheduling metrics are the main evidence'])
    h.write(out/'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, default=4)
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--package', type=Path, default=ROOT/'artifacts/search-v3/beta-g1290/package')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    out = (args.output or ROOT/'artifacts/search-v3/smoke'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    h = load(out)
    protocol = prepare(h, out, args.package, args.games)
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(h.play, game): game for game in protocol['games']}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(dict(game=futures[future], error=str(error)))
    if failures:
        h.write(out/'run-failure.json', dict(passed=False, completed=len(results), failures=failures))
        raise SystemExit('Smoke incomplete: '+json.dumps(failures)[:600])
    report = summarize(h, out, sorted(results, key=lambda r: r['game']['id']))
    print(json.dumps(report, indent=2))
    print('Summary: '+str(out/'summary.json'))


if __name__ == '__main__':
    main()
