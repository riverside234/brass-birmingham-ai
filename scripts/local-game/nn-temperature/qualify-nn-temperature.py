"""Seal the scoped NN selector after short UI/restart and live-turn smokes."""
from pathlib import Path
import runpy
import shutil
import zipfile

B = runpy.run_path(str(Path(__file__).with_name('build-nn-temperature.py')))
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
            assert p['path'] in ('alpha-g890/build.json','alpha-g890/brass-ai-local.exe','alpha-g890/native-source.zip','alpha-g890/policy-t15.json','alpha-g890/policy-k8-t15.json')
    p16 = read(OUT / 'BrassNative/alpha-g890/policy.json')
    p8 = read(OUT / 'BrassNative/alpha-g890/policy-k8.json')
    assert set(p16['k_by_allowance_ms'].values()) == {16} and set(p8['k_by_allowance_ms'].values()) == {8}
    p8['k_by_allowance_ms'] = p16['k_by_allowance_ms']
    assert p8 == p16 and p16['method'] == 'search_v2_focus_only'
    native=read(OUT.parent/'native-build.json')
    assert native['passed'] and native['source_sha256']==m['native_source_sha256']
    assert sha(OUT/'BrassNative/alpha-g890/brass-ai-local.exe')==native['binary_sha256']
    assert sha(OUT/'BrassNative/alpha-g890/native-source.zip')==native['source_archive_sha256']
    with zipfile.ZipFile(ROOT/'BrassNative/alpha-g890/native-source.zip') as old, zipfile.ZipFile(OUT/'BrassNative/alpha-g890/native-source.zip') as new:
        previous={n:old.read(n) for n in old.namelist() if not n.endswith('/')}
        current={n:new.read(n) for n in new.namelist() if not n.endswith('/')}
    assert not set(previous)-set(current)
    assert {n for n in current if previous.get(n)!=current[n]}==set(native['scope'])
    for base_name,temperature_name in [('policy.json','policy-t15.json'),('policy-k8.json','policy-k8-t15.json')]:
        tempered=read(OUT/'BrassNative/alpha-g890'/temperature_name)
        assert tempered.pop('policy_temperature')==1.5
        assert tempered==read(OUT/'BrassNative/alpha-g890'/base_name)
    evidence = OUT / 'evidence'
    evidence.mkdir()
    for name in ('native-build.json','build.log','temperature-tests.log'):
        shutil.copyfile(OUT.parent/name,evidence/name)
    checks, turns = {}, {}
    for name in ('ui-write', 'ui-read', 'live-k8', 'live-k16'):
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
            assert result['passed'] and result['search_k'] == k and result['temperature'] == 1.5
            assert len(result['commits']) == 2 and result['commits'][-1]['remaining'] == 0
            assert result['delivery_ms'] < 4000
            assert result['replies'] and all(r['requested_k'] == k and r['policy_temperature'] == 1.5 for r in result['replies'])
            assert all(r['search_revision']['focus']['method'] == 'search_v2_focus_only' for r in result['replies'])
            turns[name] = dict(requested_k=k, temperature=1.5, delivery_ms=result['delivery_ms'], actions=2)
        shutil.copytree(src, evidence / name)
    images = evidence / 'images'
    images.mkdir()
    for p in (ROOT / 'artifacts/card-tracker-player/artifacts/patch-build').glob('setup-*-largest.png'):
        shutil.copyfile(p, images / p.name)
    shutil.copyfile(ROOT / 'artifacts/card-tracker-player/artifacts/patch-build/setup-preferences.json', evidence / 'restart-preferences.json')
    write(OUT / 'qualification.json', dict(
        format='brass-nn-temperature/1.0.0', passed=True, version=m['version'],
        scope='local-game-g890/NN-temperature-selector', build_manifest_sha256=sha(OUT / 'build-manifest.json'),
        files=[p for p in pins(OUT) if p['path'].startswith(('Managed/', 'runtime/', 'BrassNative/', 'evidence/')) or p['path'] == 'build-manifest.json'],
        changed_sources=changed, checks=checks, live_turns=turns,
        search_method='search_v2_focus_only', v205_enabled=False,
        default_k=16, k_choices=[8, 16], temperature_choices=[1.0,1.5], default_temperature=1.0, temperature_scope='all_nn_policy_evaluations', native_build_sha256=sha(OUT.parent/'native-build.json'), final_model=False, lab_deployed=False))
    print(checks, turns)


if __name__ == '__main__':
    main()
