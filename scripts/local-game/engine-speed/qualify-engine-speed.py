"""Qualify the engine-speed rollout: each rebuilt package starts, passes its warm-up and searches, beside the installed one.

The local search runs a fixed simulation quota per lane, set from the time it has and a calibrated cost per
simulation (7.26 ms), and answers as soon as the quota is done. A faster engine therefore answers sooner, and
completes its quota where the installed engine ran out of time; it does not run more simulations.

`qualify` runs the candidate packages and the installed packages on the same positions and allowance, interleaved,
and records each search's reply time, completed simulations and quota. `audit` checks the installed result
afterwards. A handful of single searches, no games.
"""
import json
import runpy
import sys
import time
from pathlib import Path

H = runpy.run_path(str(Path(__file__).with_name('build-engine-speed.py')))
ROOT, STUDY, OUT, PACKAGES, MANAGED, sha, read, write, pins = (
    H[key] for key in ('ROOT', 'STUDY', 'OUT', 'PACKAGES', 'MANAGED', 'sha', 'read', 'write', 'pins'))
EVIDENCE = STUDY / OUT.name.removeprefix('release-')
FORMAT = 'brass-engine-speed-release/1.0.0'
CAPTURE = ROOT / 'artifacts/rust-backend/20261025-l10-capture'
FIXTURES = (0, 8, 17)  # Canal round 1, Rail rounds 1 and 3
ALLOWANCE_MS = 4000


def position(package, index):
    """A captured four-seat position, under this package's own neutral profile."""
    if index is None:
        return read(package / 'warmup.json')
    case = read(CAPTURE / f'brass-{index:03d}-request.json')['cases'][0]
    inp = case['input']; inp.pop('shuffle_tape', None); inp['recall'] = []
    # This package's own context and reward; the insolvency spent so far is the same number of events at its own
    # event cost (capped at its own maximum), so the remaining budget stays consistent with its reward.
    own = read(package / 'warmup.json')
    captured, insolvency = inp['reward']['insolvency'], own['reward']['insolvency']
    spent = [insolvency['max_cost'] if s >= captured['max_cost'] else
             min(round(s / captured['event_cost']) * insolvency['event_cost'], insolvency['max_cost'])
             for s in inp['ledger']['spent']]
    inp['ledger'] = dict(inp['ledger'], spent=spent)
    inp['context'] = dict(own['context'], remaining=[insolvency['max_cost'] - s for s in spent])
    inp['reward'] = own['reward']
    return inp


def search(worker, package, model, index, generation):
    ready = worker.hello['data']
    inp = position(package, index)
    if model == 'alpha-g890':
        inp['context']['profile_id'] = 'alpha-test-g890-neutral/1.0.0'
    ticket = dict(game_instance=f'engine-speed-{model}-{index}', game_id='fixture', generation=generation,
                  revision=inp['view']['context']['revision'], actor=inp['view']['seat'], profile=inp['context']['profile_id'],
                  source_sha256=ready['build']['source_sha256'])
    began = time.perf_counter()
    request_id = worker.send(dict(op='search', input=inp, ticket=ticket, allowance_ms=ALLOWANCE_MS, remaining_ms=ALLOWANCE_MS))
    reply = worker.recv(timeout=30)
    seconds = time.perf_counter() - began
    assert reply['id'] == request_id and reply['status'] == 'ok', reply
    data = reply['data']
    assert data['retired'] and data['actions'] and data['policy_temperature'] == 1.0, reply
    owners = data['replay']['owners']
    return dict(fixture='warmup' if index is None else index, completed=data['completed'],
                quota=sum(o['quota']['simulations'] for o in owners), seconds=seconds,
                reply_seconds=data['response_ready_ns'] / 1e9, actions=data['actions'],
                effective_k=[o['effective_k'] for o in owners])


def start(package):
    Worker = runpy.run_path(str(ROOT / 'tests/RustBackendLocal/test_local.py'))['Worker']
    worker = Worker(package)
    ready = worker.hello['data']
    policy = read(package / 'policy.json')
    assert policy['method'] == 'search_v1' and policy['config']['c2'] == 0.1
    assert ready['build']['source_sha256'] == read(package / 'build.json')['source_sha256']
    assert ready['model_id'] == policy['acceptance']['model_id']
    assert ready['measured_readiness']['warmup_passed'], ready
    return worker


def compare(candidate, installed):
    """The candidate and the installed packages, model by model, the same positions interleaved."""
    results = []
    for model in PACKAGES:
        new_package, old_package = candidate / 'BrassNative' / model, installed / 'BrassNative' / model
        new, old = start(new_package), start(old_package)
        try:
            searches = []
            for generation, index in enumerate((None, *FIXTURES), start=2):
                before = search(old, old_package, model, index, generation)
                after = search(new, new_package, model, index, generation)
                searches.append(dict(fixture=after['fixture'], installed=before, candidate=after,
                                     time_ratio=after['reply_seconds'] / before['reply_seconds']))
                print(model, after['fixture'], f"reply {before['reply_seconds']:.2f} -> {after['reply_seconds']:.2f} s;",
                      f"simulations {before['completed']}/{before['quota']} -> {after['completed']}/{after['quota']}",
                      flush=True)
            results.append(dict(model=model, warmup=new.hello['data']['measured_readiness'],
                                installed_source=old.hello['data']['build']['source_sha256'],
                                candidate_source=new.hello['data']['build']['source_sha256'], searches=searches))
        finally:
            new.close(); old.close()
    return results


def qualify():
    m = read(OUT / 'build-manifest.json')
    assert pins(OUT / 'BrassNative') == m['nativeFiles'] and pins(ROOT / 'BrassNative') == m['baseNative']
    for p in m['sourcePins']:
        assert sha(ROOT / p['path']) == p['sha256'], p['path']
    workers = compare(OUT, ROOT)
    searches = [s for w in workers for s in w['searches']]
    ratios = sorted(s['time_ratio'] for s in searches)
    median = ratios[len(ratios) // 2]
    full = lambda side: sum(s[side]['completed'] == s[side]['quota'] for s in searches)
    assert median < 1, ratios
    write(OUT / 'qualification.json', dict(format=FORMAT, passed=True, manifest_sha256=sha(OUT / 'build-manifest.json'),
        allowance_ms=ALLOWANCE_MS, median_reply_time_ratio=median, searches=len(searches),
        full_quota=dict(installed=full('installed'), candidate=full('candidate')), workers=workers,
        coreTestsSha256=m['coreTestsSha256'], workerTestsSha256=m['workerTestsSha256']))
    print('Qualified', OUT, f'median reply time ratio {median:.2f};', 'full quota (installed, candidate):',
          full('installed'), full('candidate'), 'of', len(searches), flush=True)


def audit():
    m = read(OUT / 'build-manifest.json')
    receipt = read(ROOT / 'artifacts/patch-installation.json')
    assert receipt['version'] == m['version'] and receipt['files'] == m['files']
    assert pins(ROOT / 'BrassNative') == m['nativeFiles']
    for p in m['files']:
        assert sha(ROOT / MANAGED / p['name']) == p['sha256'].lower()
    workers = []
    for model in PACKAGES:
        package = ROOT / 'BrassNative' / model
        worker = start(package)
        try:
            workers.append(dict(model=model, search=search(worker, package, model, None, 2),
                                warmup=worker.hello['data']['measured_readiness']))
            print(model, workers[-1]['search']['completed'], 'simulations', flush=True)
        finally:
            worker.close()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    write(EVIDENCE / 'installed-verification.json', dict(passed=True, version=m['version'], workers=workers,
        qualification_sha256=sha(OUT / 'qualification.json')))


if __name__ == '__main__':
    {'qualify': qualify, 'audit': audit}[sys.argv[1]]()
