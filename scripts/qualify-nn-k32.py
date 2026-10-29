"""Seal the scoped K32 selector from bounded UI/restart and live-turn evidence."""
from pathlib import Path
import runpy
import shutil
import zipfile

B = runpy.run_path(str(Path(__file__).with_name('build-nn-k32.py')))
ROOT, BASE, OUT = (B[k] for k in ('ROOT', 'BASE', 'OUT'))
read, write, sha, pins = (B[k] for k in ('read', 'write', 'sha', 'pins'))


def main():
    assert not (OUT / 'qualification.json').exists()
    m, old = read(OUT / 'build-manifest.json'), read(BASE / 'build-manifest.json')
    before = {p['path']: p['sha256'] for p in old['sourcePins']}
    changed = []
    for pin in m['sourcePins']:
        assert sha(OUT / 'runtime' / pin['path']) == pin['sha256']
        if before.get(pin['path']) != pin['sha256']:
            changed.append(pin['path'])
            assert sha(ROOT / pin['path']) == pin['sha256']
    assert set(changed) == set(B['SCOPE'])
    for pin in m['files']:
        assert sha(OUT / 'Managed' / pin['name']) == pin['sha256']
        if pin['name'] != 'BrassLegacyAI.dll':
            assert pin['sha256'] == next(p['sha256'] for p in old['files'] if p['name'] == pin['name'])
    assert pins(OUT / 'BrassNative') == m['nativeFiles']
    previous_native = {p['path']: p['sha256'] for p in m['baseNative']}
    changed_native = {p['path'] for p in m['nativeFiles'] if previous_native.get(p['path']) != p['sha256']}
    assert changed_native == {'alpha-g890/build.json', 'alpha-g890/policy-k32.json', 'alpha-g890/policy-k32-t15.json',
        'alpha-g890/brass-ai-local.exe','alpha-g890/native-source.zip'}
    assert len(m['nativeFiles']) == len(previous_native) + 2
    for base_name, name in [('policy.json', 'policy-k32.json'), ('policy-t15.json', 'policy-k32-t15.json')]:
        new = read(OUT / 'BrassNative/alpha-g890' / name)
        old_policy = read(BASE / 'BrassNative/alpha-g890' / base_name)
        assert new['method'] == old_policy['method'] == 'search_v1'
        assert set(new['k_by_allowance_ms'].values()) == {32}
        new['k_by_allowance_ms'] = old_policy['k_by_allowance_ms']
        assert new == old_policy, 'Only root K mapping may change'
    for name in ('model/model.onnx', 'model/onnxruntime.dll'):
        assert sha(OUT / 'BrassNative/alpha-g890' / name) == sha(BASE / 'BrassNative/alpha-g890' / name)
    native=read(OUT.parent/'native-build.json')
    assert native['passed'] and native['source_sha256']==m['native_source_sha256']
    assert sha(OUT/'BrassNative/alpha-g890/brass-ai-local.exe')==native['binary_sha256']
    with zipfile.ZipFile(BASE/'BrassNative/alpha-g890/native-source.zip') as a, zipfile.ZipFile(OUT/'BrassNative/alpha-g890/native-source.zip') as b:
        old_source={n:a.read(n) for n in a.namelist() if not n.endswith('/')}
        new_source={n:b.read(n) for n in b.namelist() if not n.endswith('/')}
    assert set(old_source)==set(new_source)
    assert {n for n in old_source if old_source[n]!=new_source[n]}==set(native['scope'])
    evidence = OUT / 'evidence'
    evidence.mkdir()
    checks, turns = {}, {}
    for name in ('ui-write-final', 'ui-read-final', 'live-t10-final', 'live-t15-final'):
        directory = OUT.parent / name
        unity, test = read(directory / 'unity.json'), read(directory / 'test-build.json')
        assert unity['passed'] and unity['checks'] >= 740 and not unity.get('nativeErrors')
        assert test['candidateManifestSha256'] == sha(OUT / 'build-manifest.json')
        assert test['testDllSha256'] == sha(directory / 'LocalAiUnityTests.dll')
        for pin in test['sources']:
            assert sha(directory / 'test-source' / pin['path']) == pin['sha256']
        checks[name] = unity['checks']
        if name.startswith('live-'):
            result = read(directory / 'result.json')
            temperature = 1.0 if name == 'live-t10-final' else 1.5
            assert result['passed'] and result['search_method'] == 'search_v1' and not result['focused']
            assert result['search_k'] == 32 and result['temperature'] == temperature
            assert result['delivery_ms'] < 4000 and len(result['commits']) == 2 and result['commits'][-1]['remaining'] == 0
            assert result['replies'] and all(r['requested_k'] == 32 and r['policy_temperature'] == temperature
                and r['sessions'] == 2 and r['threads_per_session'] == 1 and 'search_revision' not in r for r in result['replies'])
            assert any(owner['effective_k'] == 32 for r in result['replies'] for owner in r['replay']['owners'])
            turns[name] = dict(search_k=32, temperature=temperature, delivery_ms=result['delivery_ms'], actions=2)
        shutil.copytree(directory, evidence / name)
    images = evidence / 'images'
    images.mkdir()
    for image in (ROOT / 'artifacts/card-tracker-player/artifacts/patch-build').glob('setup-*-largest.png'):
        shutil.copyfile(image, images / image.name)
    shutil.copyfile(ROOT / 'artifacts/card-tracker-player/artifacts/patch-build/setup-preferences.json', evidence / 'restart-preferences.json')
    write(OUT / 'qualification.json', dict(format='brass-nn-k32/1.0.0', passed=True, version=m['version'],
        scope='local-game-g890/K32-selector', build_manifest_sha256=sha(OUT / 'build-manifest.json'),
        files=[p for p in pins(OUT) if p['path'].startswith(('Managed/', 'runtime/', 'BrassNative/', 'evidence/')) or p['path'] == 'build-manifest.json'],
        changed_sources=changed, checks=checks, live_turns=turns, search_method='search_v1', v205_enabled=False,
        default_k=16, k_choices=[8,16,32], allowances_ms=[4000,8000,10000,12000],
        temperature_choices=[1.0,1.5], default_temperature=1.0, native_binary_unchanged=False,
        k32_host_reserve_ns=1200000000, native_build_sha256=sha(OUT.parent/'native-build.json'),
        native_core_unchanged=True, model_unchanged=True, final_model=False, lab_deployed=False))
    print(checks, turns)


if __name__ == '__main__':
    main()
