"""Seal the scoped NN selector after short UI/restart and live-turn smokes."""
from pathlib import Path
import runpy
import shutil

B = runpy.run_path(str(Path(__file__).with_name('build-nn-search-choice.py')))
ROOT, BASE, OUT = (B[k] for k in ('ROOT', 'BASE', 'OUT'))
read, write, sha, pins = (B[k] for k in ('read', 'write', 'sha', 'pins'))


def main():
    assert not (OUT / 'qualification.json').exists()
    m = read(OUT / 'build-manifest.json')
    original = read(BASE / 'build-manifest.json')
    before = {p['path']: p['sha256'].lower() for p in original['sourcePins']}
    changed = []
    for p in m['sourcePins']:
        assert sha(OUT / 'runtime' / p['path']) == p['sha256']
        if before.get(p['path']) != p['sha256']:
            changed.append(p['path'])
            assert sha(ROOT / p['path']) == p['sha256']
    assert set(changed) == set(B['SCOPE'])
    for p in m['files']:
        assert sha(OUT / 'Managed' / p['name']) == p['sha256']
        if p['name'] != 'BrassLegacyAI.dll':
            assert p['sha256'] == next(x['sha256'].lower() for x in m['baseFiles'] if x['name'] == p['name'])
    assert pins(OUT / 'BrassNative') == m['nativeFiles']
    old_native = {p['path']: p['sha256'] for p in m['baseNative']}
    for p in m['nativeFiles']:
        if old_native.get(p['path']) != p['sha256']:
            assert p['path'] in ('alpha-g890/build.json', 'alpha-g890/policy-k8.json')
    p16 = read(OUT / 'BrassNative/alpha-g890/policy.json')
    p8 = read(OUT / 'BrassNative/alpha-g890/policy-k8.json')
    assert set(p16['k_by_allowance_ms'].values()) == {16} and set(p8['k_by_allowance_ms'].values()) == {8}
    p8['k_by_allowance_ms'] = p16['k_by_allowance_ms']
    assert p8 == p16 and p16['method'] == 'search_v2_focus_only'
    evidence = OUT / 'evidence'
    evidence.mkdir()
    checks, turns = {}, {}
    for name in ('ui-write-v2', 'ui-read', 'live-k8', 'live-k16'):
        src = OUT.parent / name
        u, t = read(src / 'unity.json'), read(src / 'test-build.json')
        assert u['passed'] and u['checks'] >= 740 and not u.get('nativeErrors')
        assert t['candidateManifestSha256'] == sha(OUT / 'build-manifest.json')
        assert t['testDllSha256'] == sha(src / 'LocalAiUnityTests.dll')
        for p in t['sources']:
            assert sha(src / 'test-source' / p['path']) == p['sha256']
        checks[name] = u['checks']
        if name.startswith('live-'):
            result = read(src / 'result.json')
            k = int(name.split('k')[-1])
            assert result['passed'] and result['search_k'] == k
            assert len(result['commits']) == 2 and result['commits'][-1]['remaining'] == 0
            assert result['delivery_ms'] < 4000
            assert result['replies'] and all(r['requested_k'] == k for r in result['replies'])
            assert all(r['search_revision']['focus']['method'] == 'search_v2_focus_only' for r in result['replies'])
            turns[name] = dict(requested_k=k, delivery_ms=result['delivery_ms'], actions=2)
        shutil.copytree(src, evidence / name)
    images = evidence / 'images'
    images.mkdir()
    for p in (ROOT / 'artifacts/card-tracker-player/artifacts/patch-build').glob('setup-*-largest.png'):
        shutil.copyfile(p, images / p.name)
    shutil.copyfile(ROOT / 'artifacts/card-tracker-player/artifacts/patch-build/setup-preferences.json', evidence / 'restart-preferences.json')
    write(OUT / 'qualification.json', dict(
        format='brass-nn-search-choice/1.0.0', passed=True, version=m['version'],
        scope='local-game-g890/NN-K-selector', build_manifest_sha256=sha(OUT / 'build-manifest.json'),
        files=[p for p in pins(OUT) if p['path'].startswith(('Managed/', 'runtime/', 'BrassNative/', 'evidence/')) or p['path'] == 'build-manifest.json'],
        changed_sources=changed, checks=checks, live_turns=turns,
        search_method='search_v2_focus_only', v205_enabled=False,
        default_k=16, k_choices=[8, 16], final_model=False, lab_deployed=False))
    print(checks, turns)


if __name__ == '__main__':
    main()
