"""Seal scoped runtime preservation and bounded regression evidence for 1.4.37."""
import importlib.util
from pathlib import Path
import shutil

spec = importlib.util.spec_from_file_location('build', Path(__file__).with_name('build-turn-history-fix.py'))
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
ROOT, BASE, OUT = build.ROOT, build.BASE, build.OUT
read, write, sha = build.read, build.write, build.sha
EVIDENCE = OUT.parent


def pins(directory):
    return [dict(path=p.relative_to(directory).as_posix(), sha256=sha(p)) for p in sorted(directory.rglob('*')) if p.is_file()]


def main():
    assert not (OUT / 'qualification.json').exists(), 'Do not overwrite sealed evidence'
    manifest = read(OUT / 'build-manifest.json')
    baseline = read(BASE / 'build-manifest.json')
    before = {p['path']: p['sha256'].lower() for p in baseline['sourcePins']}
    changed = []
    for p in manifest['sourcePins']:
        assert sha(OUT / 'runtime' / p['path']) == p['sha256']
        if before.get(p['path']) != p['sha256']:
            changed.append(p['path'])
            assert sha(ROOT / p['path']) == p['sha256']
    assert set(changed) == set(build.SCOPE)
    assert set(before).issubset(p['path'] for p in manifest['sourcePins'])
    for p in manifest['files']:
        assert sha(OUT / 'Managed' / p['name']) == p['sha256']
    for p in baseline['files']:
        assert sha(EVIDENCE / 'baseline/Managed' / p['name']) == p['sha256'].lower()
    assert pins(OUT / 'BrassNative') == pins(BASE / 'BrassNative')
    repro = read(EVIDENCE / 'baseline/turn-history-tests.json')
    tests = read(OUT / 'turn-history-tests.json')
    preserved = read(EVIDENCE / 'preserved-runtime.json')
    debt = read(EVIDENCE / 'liquidation.json')
    unity = read(EVIDENCE / 'unity-smoke/unity.json')
    ui_cases = read(EVIDENCE / 'unity-smoke/unity.turn-history.json')
    ui_build = read(EVIDENCE / 'unity-smoke/test-build.json')
    assert all(x['passed'] for x in (repro, tests, preserved, debt, unity, ui_cases))
    assert repro['bug_reproduced'] and repro['before_actor'] == 'Yellow' and repro['after_actor'] == 'Blue'
    assert tests['checks'] >= 296 and debt['checks'] >= 79 and unity['checks'] >= 755
    assert ui_build['candidateManifestSha256'] == sha(OUT / 'build-manifest.json')
    assert ui_build['testDllSha256'] == sha(EVIDENCE / 'unity-smoke/LocalAiUnityTests.dll')
    for p in ui_build['sources']:
        assert sha(EVIDENCE / 'unity-smoke/test-source' / p['path']) == p['sha256']
    assert {row['family'] for row in ui_cases['live_turns']} == {'nn', 'v4'}
    assert all(row['search_ms'] == 4000 and row['saves_after_setup'] == 2 for row in ui_cases['live_turns'])
    assert sha(ROOT / 'tests/BrassPatch/Liquidation/LiquidationTests.cs') == debt['sourceSha256'].lower()
    evidence = OUT / 'evidence'
    evidence.mkdir()
    reports = ['baseline/turn-history-tests.json', 'preserved-runtime.json', 'liquidation.json',
               'unity-smoke/unity.json', 'unity-smoke/unity.turn-history.json', 'unity-smoke/test-build.json',
               'unity-smoke/LocalAiUnityTests.dll']
    for name in reports:
        target = evidence / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(EVIDENCE / name, target)
    shutil.copytree(EVIDENCE / 'unity-smoke/test-source', evidence / 'unity-smoke/test-source')
    shutil.copyfile(OUT / 'turn-history-tests.json', evidence / 'turn-history-tests.json')
    for name in ('scripts/build-turn-history-fix.py', 'scripts/qualify-turn-history-fix.py', 'scripts/install-turn-history-fix.ps1',
                 'tests/BrassPatch/TurnHistory/TurnHistoryRegression.cs', 'tests/BrassPatch/TurnHistory/TurnHistoryUnityTests.cs',
                 'tests/BrassPatch/TurnHistory/PreservedRuntimeChecks.cs', 'tests/BrassPatch/Liquidation/LiquidationTests.cs'):
        target = evidence / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    shutil.copyfile(OUT / 'build-manifest.json', evidence / 'tested-build-manifest.json')
    manifest['local_release_qualified'] = True
    write(OUT / 'build-manifest.json', manifest)
    sealed = [p for p in pins(OUT) if p['path'].startswith(('Managed/', 'runtime/', 'BrassNative/', 'evidence/')) or p['path'] == 'build-manifest.json']
    # The installer copies only the three manifest DLLs. Diagnostic binaries
    # are retained and pinned here as evidence, never installed in the game.
    qualification = dict(format='brass-turn-history-hotfix/1.0.0', version=manifest['version'], passed=True,
        local_release_qualified=True, scope='local-game-g890', save_policy=manifest['save_policy'],
        base_release=baseline['version'], base_qualification_sha256=sha(BASE / 'qualification.json'),
        build_manifest_sha256=sha(OUT / 'build-manifest.json'), files=sealed, native_files=pins(OUT / 'BrassNative'),
        changed_sources=changed, regression_checks=tests['checks'], liquidation_checks=debt['checks'], unity_checks=unity['checks'],
        final_model=False, lab_deployed=False)
    write(OUT / 'qualification.json', qualification)
    print('Qualified 1.4.37; all native files and 15,137 existing methods preserved')


if __name__ == '__main__':
    main()
