"""Seal the scoped search_v1 restoration using bounded Unity evidence."""
from pathlib import Path
import runpy
import shutil

B = runpy.run_path(str(Path(__file__).with_name('build-search-v1-default.py')))
ROOT, BASE, OUT = (B[k] for k in ('ROOT', 'BASE', 'OUT'))
read, write, sha, pins = (B[k] for k in ('read', 'write', 'sha', 'pins'))


def main():
    assert not (OUT / 'qualification.json').exists()
    m, old = read(OUT / 'build-manifest.json'), read(BASE / 'build-manifest.json')
    before = {p['path']: p['sha256'] for p in old['sourcePins']}
    changed = []
    for p in m['sourcePins']:
        assert sha(OUT / 'runtime' / p['path']) == p['sha256']
        if before.get(p['path']) != p['sha256']:
            changed.append(p['path'])
            assert sha(ROOT / p['path']) == p['sha256']
    assert changed == B['SCOPE']
    original_source = (BASE / 'runtime' / changed[0]).read_text()
    revised_source = (OUT / 'runtime' / changed[0]).read_text()
    assert revised_source == original_source.replace('Policy["method"]!="search_v2_focus_only"', 'Policy["method"]!="search_v1"')
    for p in m['files']:
        assert sha(OUT / 'Managed' / p['name']) == p['sha256']
        if p['name'] != 'BrassLegacyAI.dll':
            assert p['sha256'] == next(x['sha256'] for x in old['files'] if x['name'] == p['name'])
    assert pins(OUT / 'BrassNative') == m['nativeFiles']
    old_native = {p['path']: p['sha256'] for p in m['baseNative']}
    allowed = {'alpha-g890/build.json'} | {'alpha-g890/' + p for p in B['POLICIES']}
    assert {p['path'] for p in m['nativeFiles'] if old_native[p['path']] != p['sha256']} == allowed
    for name in B['POLICIES']:
        new, previous = read(OUT / 'BrassNative/alpha-g890' / name), read(BASE / 'BrassNative/alpha-g890' / name)
        assert new['method'] == 'search_v1'
        new['method'] = previous['method']
        assert new == previous, 'Only search method may change'
    evidence = OUT / 'evidence'
    evidence.mkdir()
    checks, turns = {}, {}
    for name, scenario, k, temperature in [('live-k16', 'normal', 16, 1.0), ('cancel-k8', 'cancel', 8, 1.5), ('load-k8', 'load', 8, 1.5)]:
        src = OUT.parent / name
        u, t, result = (read(src / f) for f in ('unity.json', 'test-build.json', 'result.json'))
        assert u['passed'] and u['checks'] >= 740 and not u.get('nativeErrors')
        assert t['candidateManifestSha256'] == sha(OUT / 'build-manifest.json')
        assert t['testDllSha256'] == sha(src / 'LocalAiUnityTests.dll')
        for p in t['sources']:
            assert sha(src / 'test-source' / p['path']) == p['sha256']
        assert result['passed'] and result['search_method'] == 'search_v1' and not result['focused']
        assert result['scenario'] == scenario and result['search_k'] == k and result['temperature'] == temperature
        assert result['replies'] and all('search_revision' not in r and r['requested_k'] == k and r['policy_temperature'] == temperature
            and r['sessions'] == 2 and r['threads_per_session'] == 1 for r in result['replies'])
        assert result['delivery_ms'] < 4000
        assert len(result['commits']) == (2 if scenario == 'normal' else 1)
        assert result['commits'][-1]['remaining'] == (0 if scenario == 'normal' else 1)
        checks[name] = u['checks']
        turns[name] = dict(scenario=scenario, requested_k=k, temperature=temperature, delivery_ms=result['delivery_ms'], actions=len(result['commits']))
        shutil.copytree(src, evidence / name)
    write(OUT / 'qualification.json', dict(
        format='brass-search-v1-default/1.0.0', passed=True, version=m['version'],
        scope='local-game-g890/search-v1-default', build_manifest_sha256=sha(OUT / 'build-manifest.json'),
        files=[p for p in pins(OUT) if p['path'].startswith(('Managed/', 'runtime/', 'BrassNative/', 'evidence/')) or p['path'] == 'build-manifest.json'],
        changed_sources=changed, checks=checks, live_turns=turns, search_method='search_v1', v205_enabled=False,
        default_k=16, k_choices=[8, 16], temperature_choices=[1.0, 1.5], default_temperature=1.0,
        native_binary_unchanged=True, model_unchanged=True, final_model=False, lab_deployed=False))
    print(checks, turns)


if __name__ == '__main__':
    main()
