"""Qualify release 1.4.62 against the installed 1.4.61: same positions, interleaved, full timed searches.

Every candidate policy (K 16 and K 8 at T 1.0, the live ones) must start and pass its warm-up. Each candidate search
must complete its quota before the work deadline and answer inside the allowance. The record shows the simulations
each side completed. `audit` checks the installed packages afterwards. A handful of single searches, no games.
"""
import runpy
import sys
from pathlib import Path

H = runpy.run_path(str(Path(__file__).with_name('build-search-calibration.py')))
ROOT, STUDY, OUT, PACKAGES, MANAGED, sha, read, write, pins = (
    H[key] for key in ('ROOT', 'STUDY', 'OUT', 'PACKAGES', 'MANAGED', 'sha', 'read', 'write', 'pins'))
C = runpy.run_path(str(Path(__file__).with_name('calibrate-search-v1.py')))
FORMAT = 'brass-search-calibration-release/1.0.0'
POSITIONS = (None, 0, 8, 17)
ALLOWANCE_MS = 4000


def start(package, name):
    w = C['worker'](package, name)
    ready = w.hello['data']
    policy = read(package / name)
    assert ready['build']['source_sha256'] == read(package / 'build.json')['source_sha256']
    assert ready['model_id'] == policy['acceptance']['model_id'] and policy['method'] == 'search_v1'
    return w, policy


def qualify():
    m = read(OUT / 'build-manifest.json')
    assert pins(OUT / 'BrassNative') == m['nativeFiles'] and pins(ROOT / 'BrassNative') == m['baseNative']
    results = []
    for model in PACKAGES:
        for name in ('policy.json', 'policy-k8.json'):
            new_package, old_package = OUT / 'BrassNative' / model, ROOT / 'BrassNative' / model
            (new, policy), (old, _) = start(new_package, name), start(old_package, name)
            assert policy['startup_ns'] == m['startupNs']
            try:
                for index in POSITIONS:
                    tag = f'{model}-{name}-{index}'
                    before = C['search'](old, old_package, model, index, 'installed-' + tag)
                    after = C['search'](new, new_package, model, index, 'candidate-' + tag)
                    assert after['owners'] and all(o['completed'] == o['quota'] for o in after['owners']), after
                    assert all(o['last_completion_ns'] <= o['available_ns'] for o in after['owners']), after
                    assert after['response_ready_ns'] < ALLOWANCE_MS * 1_000_000, after
                    done = lambda r: sum(o['completed'] for o in r['owners'])
                    results.append(dict(model=model, policy=name, position=after['position'], installed=before,
                                        candidate=after, ratio=done(after) / done(before)))
                    print(model, name, after['position'], 'simulations', done(before), '->', done(after),
                          f"reply {before['response_ready_ns'] / 1e9:.2f} -> {after['response_ready_ns'] / 1e9:.2f} s", flush=True)
            finally:
                new.close(); old.close()
    ratios = sorted(r['ratio'] for r in results)
    write(OUT / 'qualification.json', dict(format=FORMAT, passed=True, manifest_sha256=sha(OUT / 'build-manifest.json'),
        allowance_ms=ALLOWANCE_MS, startup_ns=m['startupNs'], previous_startup_ns=m['previousStartupNs'],
        median_simulation_ratio=ratios[len(ratios) // 2], searches=len(results), results=results))
    print('Qualified', OUT, 'median simulations ratio', round(ratios[len(ratios) // 2], 2), flush=True)


def audit():
    m = read(OUT / 'build-manifest.json')
    receipt = read(ROOT / 'artifacts/patch-installation.json')
    assert receipt['version'] == m['version'] and receipt['files'] == m['files']
    assert pins(ROOT / 'BrassNative') == m['nativeFiles']
    for p in m['files']:
        assert sha(ROOT / MANAGED / p['name']) == p['sha256'].lower()
    rows = []
    for model in PACKAGES:
        package = ROOT / 'BrassNative' / model
        w, policy = start(package, 'policy.json')
        try:
            result = C['search'](w, package, model, None, f'audit-{model}')
            assert all(o['completed'] == o['quota'] for o in result['owners']), result
            rows.append(dict(model=model, startup_ns=policy['startup_ns'], search=result))
            print(model, sum(o['completed'] for o in result['owners']), 'simulations', flush=True)
        finally:
            w.close()
    write(STUDY / 'installed-verification.json', dict(passed=True, version=m['version'], workers=rows,
        qualification_sha256=sha(OUT / 'qualification.json')))


if __name__ == '__main__':
    {'qualify': qualify, 'audit': audit}[sys.argv[1]]()
