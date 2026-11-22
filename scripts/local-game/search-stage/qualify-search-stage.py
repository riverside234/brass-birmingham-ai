"""Qualify 1.4.63 against the installed 1.4.62 on the local game's captured positions; audit it after installation.

For each stage, six action roots from different games, on Beta and Beta-Plus, K 16 and K 8, at 4 s, installed and
candidate interleaved. Every candidate search must use its stage's estimate, complete its quota before the work
deadline and answer inside the allowance. The record shows the simulations each side completed.
"""
import runpy
import sys
from collections import defaultdict
from pathlib import Path

H = runpy.run_path(str(Path(__file__).with_name('build-search-stage.py')))
C = runpy.run_path(str(Path(__file__).with_name('calibrate-search-stage.py')))
ROOT, STUDY, OUT, PACKAGES, RETIRED, MANAGED, STAGES = (H[k] for k in ('ROOT', 'STUDY', 'OUT', 'PACKAGES', 'RETIRED', 'MANAGED', 'STAGES'))
sha, read, write, pins = (H[k] for k in ('sha', 'read', 'write', 'pins'))
FORMAT = 'brass-search-stage-release/1.0.0'


def sample(package):
    """Six action roots per stage, each from a different game where possible."""
    chosen = defaultdict(list)
    for p in C['positions'](package):
        if p['phase'] == 'action' and len(chosen[p['stage']]) < 6 and p['game'] not in {q['game'] for q in chosen[p['stage']]}:
            chosen[p['stage']].append(p)
    return [p for s in STAGES for p in chosen[s]]


def qualify():
    m = read(OUT / 'build-manifest.json')
    assert pins(OUT / 'BrassNative') == m['nativeFiles'] and pins(ROOT / 'BrassNative') == m['baseNative']
    table = m['startupNsByStage']
    results = []
    for model in PACKAGES:
        new_package, old_package = OUT / 'BrassNative' / model, ROOT / 'BrassNative' / model
        positions = sample(new_package)
        for name in ('policy.json', 'policy-k8.json'):
            new, old = C['worker'](new_package, name), C['worker'](old_package, name)
            assert new.hello['data']['startup_ns_by_stage'] == table, new.hello['data']
            try:
                for p in positions:
                    tag = f"{model}-{name}-{p['id']}"
                    before = C['search'](old, p, 'installed-' + tag)
                    after = C['search'](new, p, 'candidate-' + tag)
                    assert after['estimate_ns'] == table[p['stage']] and after['source'].endswith(p['stage']), after
                    assert after['full'] and after['used'] <= 1 and after['reply_s'] < C['ALLOWANCE_MS'] / 1000, after
                    results.append(dict(model=model, policy=name, id=p['id'], stage=p['stage'], round=p['round'],
                                        installed=before, candidate=after, ratio=sum(after['completed']) / sum(before['completed'])))
                    print(model, name, p['stage'], p['round'], 'simulations', sum(before['completed']), '->', sum(after['completed']),
                          f"reply {before['reply_s']:.2f} -> {after['reply_s']:.2f} s", flush=True)
            finally:
                new.close(); old.close()
    by = {s: sorted(r['ratio'] for r in results if r['stage'] == s) for s in STAGES}
    medians = {s: v[len(v) // 2] for s, v in by.items()}
    write(OUT / 'qualification.json', dict(format=FORMAT, passed=True, manifest_sha256=sha(OUT / 'build-manifest.json'),
        allowance_ms=C['ALLOWANCE_MS'], startup_ns_by_stage=table, median_simulation_ratio_by_stage=medians,
        searches=len(results), retired_models=list(RETIRED), results=results))
    print('Qualified', OUT, 'median simulations ratio by stage', {s: round(v, 2) for s, v in medians.items()}, flush=True)


def audit():
    m = read(OUT / 'build-manifest.json')
    receipt = read(ROOT / 'artifacts/patch-installation.json')
    assert receipt['version'] == m['version'] and receipt['files'] == m['files']
    assert pins(ROOT / 'BrassNative') == m['nativeFiles']
    assert not any((ROOT / 'BrassNative' / r).exists() for r in RETIRED)
    for p in m['files']:
        assert sha(ROOT / MANAGED / p['name']) == p['sha256'].lower()
    rows = []
    for model in PACKAGES:
        package = ROOT / 'BrassNative' / model
        w = C['worker'](package, 'policy.json')
        try:
            assert w.hello['data']['startup_ns_by_stage'] == m['startupNsByStage']
            for p in sample(package)[::6]:  # one action root per stage
                r = C['search'](w, p, f"audit-{model}-{p['id']}")
                assert r['full'] and r['estimate_ns'] == m['startupNsByStage'][p['stage']], r
                rows.append(dict(model=model, stage=p['stage'], search=r))
                print(model, p['stage'], sum(r['completed']), 'simulations', f"reply {r['reply_s']:.2f} s", flush=True)
        finally:
            w.close()
    write(STUDY / 'installed-verification.json', dict(passed=True, version=m['version'], retired_models=list(RETIRED),
        workers=rows, qualification_sha256=sha(OUT / 'qualification.json')))


if __name__ == '__main__':
    {'qualify': qualify, 'audit': audit}[sys.argv[1]]()
