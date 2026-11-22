"""Calibrate the 1.4.63 startup estimate by era stage on the local game's own captured positions.

Positions: the L06 capture's roots from 20 local-rules games (artifacts/rust-backend/20261025-l06-capture: every
seat's view and recall), with each root's settled rounds replayed from the L05 game record at the package's own
reward (no budget spent), as the L10 search fixtures were prepared. Every root that searches is used: action, continuation and
payment roots with more than one choice; debt roots are answered by the debt policy, and forced roots search
nothing. They cover Canal and Rail rounds 1, 3 and 6 for action roots, plus continuation and payment roots across
the game.

`measure`  Beta and Beta-Plus (installed 1.4.62, the same engine as 1.4.63), K 16, 4 s, every position, timed at the
           single 6.36 ms estimate. Cost = receipt to the lane's last completed simulation / completed. The table
           is each stage's maximum, rounded up to a microsecond (the L12 rule). The record also compares one estimate,
           the stage table and per-round estimates on held-out games (calibrate on 10 games, check the other 10).
`verify`   the candidate workers with the table, every position on both models at 4 s (K 16), K 8 and 12 s on the
           action roots. Every search must complete its quota. A stage that overran is raised to the cost it showed
           at its real quota (recorded in calibration.json); then run verify again.
"""
import copy
import gzip
import json
import math
import runpy
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
B = runpy.run_path(str(Path(__file__).with_name('build-search-stage.py')))
sha, read, write, STUDY, PACKAGES, STAGES = (B[k] for k in ('sha', 'read', 'write', 'STUDY', 'PACKAGES', 'STAGES'))
SCRATCH = ROOT / 'tmp/search-stage'
SINGLE_NS = 6_360_000  # 1.4.62
ALLOWANCE_MS = 4000


def stage(era, round_):
    return ('canal' if era == 'CanalEra' else 'rail') + ('_early' if round_ <= 4 else '_late')


def roots(reward):
    """(root id, game, view, recall, ledger, choices) of every capture root, its ledger replayed at `reward`."""
    from brass_ai.learning.reward import CostLedger
    capture = ROOT / 'artifacts/rust-backend'
    found = {}
    with gzip.open(capture / '20261025-l06-capture/roots.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            found[(r['id'].split('-root-')[0], r['state']['context']['revision'])] = r
    result, ledger, previous = [], None, None
    with gzip.open(capture / '20261025-l05-capture-v2/games.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if 'initial' in row:
                ledger = CostLedger(reward); current = row['initial']
            else:
                if 'choice' in row['command']:
                    choice = next(c for c in previous['choices'] if c['id'] == row['command']['choice'])
                    ledger.chosen({k: previous['state'][k] for k in ('context', 'players', 'slots')}, choice)
                current = row['expected']
            ctx = current['state']['context']
            if not ctx['chance']:
                ledger.observe({k: current['state'][k] for k in ('context', 'players', 'slots')})
                root = found.get((row['game'], ctx['revision']))
                if root and ctx['actor'] >= 0 and not ctx['terminal']:
                    actor = root['state']['context']['actor']
                    result.append(dict(id=root['id'], game=row['game'], view=root['views'][actor], recall=root['recall'][actor],
                                       ledger=copy.deepcopy(ledger.snapshot()), choices=len(current['choices'])))
            previous = current
    return result


def positions(package):
    own = read(package / 'warmup.json')
    out = []
    for r in roots(own['reward']):
        x = r['view']['context']
        if x['phase'] == 'debt' or r['choices'] <= 1:
            continue
        # The replayed settled rounds, with no budget spent: the worker compares the remaining budget with its own
        # ledger arithmetic exactly, and a remaining value such as 0.015000000000000001 does not survive its JSON
        # float parsing. The insolvency penalty does not change what a simulation costs, which is all this times.
        cost = own['reward']['insolvency']['max_cost']
        ledger = dict(spent=[0.] * 4, settled=r['ledger']['settled'])
        inp = dict(view=r['view'], recall=r['recall'], context=dict(own['context'], remaining=[cost] * 4),
                   reward=own['reward'], ledger=ledger, sample_seed=0)
        out.append(dict(id=r['id'], game=r['game'], era=x['era'], round=x['round'], phase=x['phase'], stage=stage(x['era'], x['round']),
                        choices=r['choices'], input=inp))
    return out


def worker(package, name):
    Worker = runpy.run_path(str(ROOT / 'tests/RustBackendLocal/test_local.py'))['Worker']
    w = Worker(package, policy=name)
    assert w.hello['data']['measured_readiness']['warmup_passed'], w.hello
    return w


def search(w, p, instance, allowance=ALLOWANCE_MS):
    inp = p['input']
    ticket = dict(game_instance=instance, game_id='calibration', generation=2, revision=inp['view']['context']['revision'],
                  actor=inp['view']['seat'], profile=inp['context']['profile_id'], source_sha256=w.hello['data']['build']['source_sha256'])
    request_id = w.send(dict(op='search', input=inp, ticket=ticket, allowance_ms=allowance, remaining_ms=allowance))
    reply = w.recv(timeout=allowance / 1000 + 30)
    assert reply['id'] == request_id, reply
    if reply['status'] != 'ok':
        return dict(error=reply['data'].get('error'))
    owners = [o for o in reply['data']['replay']['owners'] if o.get('quota') is not None and o['completed']]
    if not owners:
        return None  # nothing to time: a bypassed root
    return dict(completed=[o['completed'] for o in owners], quota=[o['quota']['simulations'] for o in owners],
                estimate_ns=owners[0]['quota']['estimate']['nanoseconds'], source=owners[0]['quota']['estimate']['source'],
                cost_ns=max(o['last_completion_ns'] / o['completed'] for o in owners),
                used=max(o['last_completion_ns'] / o['quota']['available_ns'] for o in owners),
                full=all(o['completed'] == o['quota']['simulations'] for o in owners),
                reply_s=reply['data']['response_ready_ns'] / 1e9)


def measuring_package(model, policy_name, edit, label):
    """A copy of the installed package with one measuring policy, `edit` applied to its K 16 or K 8 policy."""
    source, target = Path(ROOT / 'BrassNative' / model), SCRATCH / 'measure' / model
    if not target.exists():
        shutil.copytree(source, target)
    policy = edit(read(source / policy_name))
    name = 'policy-' + label + '.json'
    write(target / name, policy)
    return target, name


def sweep(packages, rows_for, label, allowance=ALLOWANCE_MS):
    """Interleave the models over the positions; one search each."""
    workers = {m: worker(*packages[m]) for m in packages}
    out = []
    try:
        lists = {m: rows_for(m) for m in packages}
        for i in range(max(len(v) for v in lists.values())):
            for m in packages:
                if i >= len(lists[m]): continue
                p = lists[m][i]
                r = search(workers[m], p, f'{label}-{m}-{p["id"]}-{allowance}', allowance)
                if r is None: continue
                assert 'error' not in r, (m, p['id'], r)
                out.append(dict(r, model=m, id=p['id'], game=p['game'], era=p['era'], round=p['round'], phase=p['phase'],
                                stage=p['stage'], choices=p['choices'], allowance_ms=allowance))
            if i % 50 == 49: print(label, i + 1, 'positions', flush=True)
    finally:
        for w in workers.values(): w.close()
    return out


def up(ns): return int(math.ceil(ns / 1000) * 1000)


def compare(rows):
    """One estimate, the stage table and per-round estimates: calibrate on one half of the games, check the other."""
    games = sorted({r['game'] for r in rows})
    halves = (set(games[0::2]), set(games[1::2]))
    keys = dict(single=lambda r: 'all', stage=lambda r: r['stage'], round=lambda r: f"{r['era']}-{r['round']}")
    result = {}
    for name, key in keys.items():
        overruns = checked = 0; simulations = []
        for fit, test in (halves, halves[::-1]):
            table = defaultdict(float)
            for r in rows:
                if r['game'] in fit: table[key(r)] = max(table[key(r)], r['cost_ns'])
            for r in rows:
                if r['game'] not in test: continue
                estimate = table.get(key(r)) or max(table.values())  # a bucket the fit half never saw: the maximum
                checked += 1; overruns += r['cost_ns'] > estimate
                simulations.append(SINGLE_NS / estimate)
        result[name] = dict(buckets=len({key(r) for r in rows}), held_out_searches=checked,
                            held_out_overruns=overruns, mean_simulations_vs_1462=sum(simulations) / len(simulations))
    return result


def measure():
    installed = read(ROOT / 'artifacts/patch-installation.json')
    assert installed['version'] == '1.4.62-rust-local', installed['version']
    packages = {m: measuring_package(m, 'policy.json', lambda p: p, 'measure-k16') for m in PACKAGES}
    cache = {m: positions(packages[m][0]) for m in PACKAGES}
    rows = sweep(packages, lambda m: cache[m], 'measure')
    by = defaultdict(list)
    for r in rows: by[r['stage']].append(r['cost_ns'])
    table = {s: up(max(by[s])) for s in STAGES}
    rounds = defaultdict(list)
    for r in rows: rounds[f"{r['era']}-{r['round']}"].append(r['cost_ns'])
    STUDY.mkdir(parents=True, exist_ok=True)
    record = dict(format='brass-search-v1-stage-calibration/1.0.0', engine='1.4.62-rust-local (same engine as 1.4.63)',
        positions='L06 capture roots of 20 local-rules games, ledgers replayed from L05; debt and forced roots excluded',
        method='timed K16 searches at 4 s with the single 6.36 ms estimate; cost = receipt to last completed / completed; '
               'stage table = each stage maximum rounded up to a microsecond',
        table_ns=table, fallback_ns=max(SINGLE_NS, *table.values()),
        stage_summary={s: dict(searches=len(by[s]), maximum_ns=max(by[s]), median_ns=sorted(by[s])[len(by[s]) // 2]) for s in STAGES},
        round_summary={k: dict(searches=len(v), maximum_ns=max(v)) for k, v in sorted(rounds.items())},
        held_out_comparison=compare(rows), iterations=[], searches=rows)
    write(STUDY / 'calibration.json', record)
    print(json.dumps({k: record[k] for k in ('table_ns', 'fallback_ns', 'stage_summary', 'round_summary', 'held_out_comparison')}, indent=1), flush=True)


def verify():
    calibration = read(STUDY / 'calibration.json')
    table, fallback = calibration['table_ns'], calibration['fallback_ns']
    target = SCRATCH / 'verify'
    if target.exists(): shutil.rmtree(target)
    B['assemble'](target, table, fallback, 'verification')
    cache = {m: positions(target / 'BrassNative' / m) for m in PACKAGES}
    rows = sweep({m: (target / 'BrassNative' / m, 'policy.json') for m in PACKAGES}, lambda m: cache[m], 'verify')
    actions = {m: [p for p in cache[m] if p['phase'] == 'action'] for m in PACKAGES}
    rows += sweep({m: (target / 'BrassNative' / m, 'policy-k8.json') for m in PACKAGES}, lambda m: actions[m], 'verify-k8')
    rows += sweep({m: (target / 'BrassNative' / m, 'policy.json') for m in PACKAGES}, lambda m: actions[m][::4], 'verify-12s', 12000)
    for r in rows:
        assert r['source'].endswith(r['stage']) and r['estimate_ns'] == table[r['stage']], r  # each root got its stage's estimate
    failed = [r for r in rows if not r['full']]
    summary = dict(table_ns=table, searches=len(rows), full_quota=len(rows) - len(failed), max_used=max(r['used'] for r in rows),
                   by_stage={s: dict(searches=sum(r['stage'] == s for r in rows), full=sum(r['stage'] == s and r['full'] for r in rows),
                                     max_cost_ns=max(r['cost_ns'] for r in rows if r['stage'] == s),
                                     max_used=max(r['used'] for r in rows if r['stage'] == s)) for s in STAGES})
    write(STUDY / 'verification.json', dict(summary, rows=rows))
    print(json.dumps(summary, indent=1), flush=True)
    if failed:
        # Raise each overrunning stage to the cost it showed at its real quota; keep the record of this round.
        raised = dict(table)
        for s in {r['stage'] for r in failed}:
            raised[s] = up(max(r['cost_ns'] for r in rows if r['stage'] == s))
        calibration['iterations'].append(dict(table_ns=table, failed=len(failed), raised_ns=raised))
        calibration.update(table_ns=raised, fallback_ns=max(SINGLE_NS, *raised.values()))
        write(STUDY / 'calibration.json', calibration)
        print('Raised', raised, '- run verify again', flush=True)
        sys.exit(2)


if __name__ == '__main__':
    {'measure': measure, 'verify': verify}[sys.argv[1]]()
