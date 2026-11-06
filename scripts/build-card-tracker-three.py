"""Build, smoke-test and verify three-player Card Tracker over installed 1.4.52."""
from pathlib import Path
import json
import os
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
HELPERS = runpy.run_path(str(ROOT / 'scripts/fix-sell-replay-speed.py'))
sha, read, write, pins, compile_cs = (HELPERS[key] for key in ('sha', 'read', 'write', 'pins', 'compile_cs'))
STUDY = ROOT / 'artifacts/card-tracker-three/20261105'
BASE = ROOT / 'artifacts/nn-delivery/20261105/release-1.4.52'
OUT = STUDY / 'release-1.4.53'
NAMES = ('Assembly-CSharp.dll', 'BrassPatch.dll', 'BrassLegacyAI.dll')
SCOPE = ['src/BrassPatch/CardTracker/' + name for name in (
    'CardTrackerProjection.cs', 'CardTrackerEntryPoints.cs', 'CardTrackerOverlay.cs')]


def build():
    assert not OUT.exists(), 'Preserve build evidence'
    installed = read(ROOT / 'artifacts/patch-installation.json')
    baseline = read(BASE / 'build-manifest.json')
    assert installed['version'] == baseline['version'] == '1.4.52-rust-local'
    for file in installed['files']:
        assert sha(ROOT / 'BrassBirminghamUnity_Data/Managed' / file['name']) == file['sha256'] == sha(BASE / 'Managed' / file['name'])
    shutil.copytree(BASE / 'runtime', OUT / 'runtime')
    shutil.copytree(BASE / 'Managed', OUT / 'Managed')
    for name in SCOPE:
        shutil.copyfile(ROOT / name, OUT / 'runtime' / name)
    managed = OUT / 'Managed'
    references = [path for path in managed.glob('*.dll')
                  if path.name not in ('BrassPatch.dll', 'BrassLegacyAI.dll') and 'Tests' not in path.name and 'Regression' not in path.name]
    compile_cs(managed / 'BrassPatch.dll', sorted((OUT / 'runtime/src/BrassPatch').rglob('*.cs')), references, ['/nostdlib+'])
    write(OUT / 'build-manifest.json', dict(version='1.4.53-rust-local', baseVersion=installed['version'],
        stockSha256=baseline['stockSha256'], baseFiles=installed['files'],
        files=[dict(name=name, sha256=sha(managed / name)) for name in NAMES],
        nativeFiles=pins(ROOT / 'BrassNative'), sourceScope=SCOPE, sourcePins=pins(OUT / 'runtime'),
        installed_baseline_receipt_sha256=sha(ROOT / 'artifacts/patch-installation.json'), tracker_player_counts=[3, 4]))
    write(STUDY / 'protected-before.json', HELPERS['protected']())
    print('Built three-player Card Tracker 1.4.53', flush=True)


def smoke(label, players):
    target = STUDY / label
    assert not target.exists(), 'Preserve smoke evidence'
    target.mkdir(parents=True)
    game = ROOT if label.startswith('installed') else ROOT / 'artifacts/card-tracker-player'
    managed = game / 'BrassBirminghamUnity_Data/Managed'
    if game != ROOT:
        for name in NAMES:
            shutil.copyfile(OUT / 'Managed' / name, managed / name)
    source = target / 'CardTrackerNativeTests.cs'
    shutil.copyfile(ROOT / 'tests/BrassPatch/CardTracker/CardTrackerNativeTests.cs', source)
    test_dll = target / 'LocalAiUnityTests.dll'
    compile_cs(test_dll, [source], [path for path in managed.glob('*.dll')
               if 'Tests' not in path.name and 'Regression' not in path.name], ['/nostdlib+'])
    staging = game / 'artifacts/patch-build'
    touched = [staging / 'Managed/LocalAiUnityTests.dll', staging / 'local-ai-native.json']
    previous = {}
    for path in touched:
        if path.exists():
            backup = target / ('prior-' + path.name)
            shutil.copyfile(path, backup)
            previous[path] = backup
    environment = {key: value for key, value in os.environ.items() if not key.startswith('BRASS_')}
    environment.update(BRASS_TRACKER_PLAYERS=str(players), BRASS_TRACKER_IMAGES=str(target))
    try:
        shutil.copyfile(test_dll, touched[0])
        write(touched[1], dict(passed=False, status='running'))
        process = subprocess.run([str(game / 'BrassBirminghamUnity.exe'), '-batchmode', '-screen-width', '1920',
            '-screen-height', '1200', '-screen-fullscreen', '0', '-logFile', str(target / 'unity.log'), '--brass-local-ai-test'],
            cwd=game, env=environment, timeout=90, creationflags=subprocess.CREATE_NO_WINDOW)
        shutil.copyfile(touched[1], target / 'unity.json')
        report = read(target / 'unity.json')
        write(target / 'test-build.json', dict(files=[dict(name=name, sha256=sha(managed / name)) for name in NAMES],
            source_sha256=sha(source), test_dll_sha256=sha(test_dll), players=players))
        print(json.dumps({key: report.get(key) for key in ('passed', 'players', 'checks', 'failures', 'error')}), flush=True)
        assert process.returncode == 0 and report['passed'] and not report['nativeErrors']
    finally:
        for path in touched:
            if path in previous:
                shutil.copyfile(previous[path], path)
            elif path.exists():
                path.unlink()


def preservation():
    target = STUDY / 'preservation'
    target.mkdir()
    for source in (ROOT / 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll', OUT / 'Managed/Newtonsoft.Json.dll'):
        shutil.copyfile(source, target / source.name)
    source = target / 'Preservation.cs'
    checker = (ROOT / 'tests/BrassPatch/CardTracker/CardTrackerLogButtonPreservedRuntimeChecks.cs').read_text()
    checker = checker.replace('return name=="BrassPatch.CardTracker.CardTrackerOverlay"||name.StartsWith("BrassPatch.CardTracker.CardTrackerOverlay/",StringComparison.Ordinal);',
                              'return name.StartsWith("BrassPatch.CardTracker.",StringComparison.Ordinal);')
    source.write_text(checker, encoding='utf-8')
    executable = target / 'Preservation.exe'
    compile_cs(executable, [source], list(target.glob('*.dll')))
    subprocess.run([str(executable), str(BASE / 'Managed'), str(OUT / 'Managed'), str(target / 'result.json')], check=True)


def qualify():
    manifest = read(OUT / 'build-manifest.json')
    baseline = {pin['path']: pin['sha256'] for pin in read(BASE / 'build-manifest.json')['sourcePins']}
    assert {pin['path'] for pin in manifest['sourcePins'] if baseline.get(pin['path']) != pin['sha256']} == set(SCOPE)
    for name in SCOPE:
        assert sha(ROOT / name) == sha(OUT / 'runtime' / name)
    assert read(STUDY / 'preservation/result.json')['passed']
    reports = []
    for label in ('candidate-3', 'candidate-4'):
        report = read(STUDY / label / 'unity.json')
        assert report['passed'] and not report['nativeErrors']
        assert read(STUDY / label / 'test-build.json')['files'] == manifest['files']
        reports.append(dict(label=label, checks=report['checks'], sha256=sha(STUDY / label / 'unity.json')))
    assert read(STUDY / 'protected-before.json') == HELPERS['protected']()
    assert manifest['nativeFiles'] == pins(ROOT / 'BrassNative')
    visual = read(STUDY / 'visual-review.json')
    assert visual['passed'] and visual['resolutions'] == ['1920x1200', '1920x1080']
    for screenshot in visual['screenshots']:
        assert sha(STUDY / screenshot['path']) == screenshot['sha256']
    write(OUT / 'qualification.json', dict(format='brass-card-tracker-three-release/1.0.0', passed=True,
        build_manifest_sha256=sha(OUT / 'build-manifest.json'), evidence=reports, tracker_player_counts=[3, 4],
        visual_review_sha256=sha(STUDY / 'visual-review.json')))
    print('Qualified three- and four-player tracker', flush=True)


def upgrade():
    fixture = STUDY / 'install-rehearsal'
    assert not fixture.exists(), 'Preserve installation rehearsal evidence'
    managed = fixture / 'BrassBirminghamUnity_Data/Managed'
    managed.mkdir(parents=True)
    for name in NAMES:
        shutil.copyfile(BASE / 'Managed' / name, managed / name)
    shutil.copytree(ROOT / 'BrassNative', fixture / 'BrassNative')
    prior = read(ROOT / 'artifacts/patch-installation.json')
    prior.update(gameRoot=str(fixture), saveDataPaths=[str(fixture / 'saves')])
    record = fixture / 'artifacts/patch-installation.json'
    record.parent.mkdir()
    write(record, prior)
    original = sha(record)
    installer = ROOT / 'scripts/install-card-tracker-three.ps1'
    command = ['pwsh', '-NoProfile', '-File', str(installer), '-GameRoot', str(fixture)]
    for action, extra, success in [
            ('Install', [], True), ('Rollback', [], True),
            ('Install', ['-TestInterrupt'], False), ('Recover', [], True)]:
        result = subprocess.run(command + ['-Action', action] + extra, capture_output=True, text=True)
        (fixture / (action + ('-interrupt' if extra else '') + '.log')).write_text(
            result.stdout + result.stderr, encoding='utf-8')
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        if not success:
            assert 'TEST INTERRUPT' in result.stderr
        if action in ('Rollback', 'Recover'):
            assert sha(record) == original
            for file in prior['files']:
                assert sha(managed / file['name']) == file['sha256']
    write(STUDY / 'upgrade-test.json', dict(passed=True, install_rollback=True,
        interrupted_recovery=True, installer_sha256=sha(installer)))
    print('Installer and recovery passed', flush=True)


def audit():
    manifest = read(OUT / 'build-manifest.json')
    receipt_path = ROOT / 'artifacts/patch-installation.json'
    receipt = read(receipt_path)
    assert receipt['version'] == manifest['version'] and receipt['files'] == manifest['files']
    for file in manifest['files']:
        assert sha(ROOT / 'BrassBirminghamUnity_Data/Managed' / file['name']) == file['sha256']
    report = read(STUDY / 'installed-3/unity.json')
    assert report['passed'] and not report['nativeErrors']
    assert read(STUDY / 'installed-3/test-build.json')['files'] == manifest['files']
    assert read(STUDY / 'protected-before.json') == HELPERS['protected']()
    assert pins(ROOT / 'BrassNative') == manifest['nativeFiles']
    verification = STUDY / 'installed-verification.json'
    write(verification, dict(passed=True, version=receipt['version'], checks=report['checks'], players=[3, 4],
        displays=['1920x1200', '1920x1080'], user_saves_unchanged=True, native_unchanged=True,
        installed_report_sha256=sha(STUDY / 'installed-3/unity.json')))
    receipt.update(cardTrackerThreeInstalledSmoke='passed', cardTrackerThreeVerification=str(verification),
                   cardTrackerThreeVerificationSha256=sha(verification))
    temporary = receipt_path.with_suffix('.tracker.tmp')
    write(temporary, receipt)
    temporary.replace(receipt_path)
    print(json.dumps(read(verification)), flush=True)


if __name__ == '__main__':
    if sys.argv[1] in ('build', 'preservation', 'qualify', 'upgrade', 'audit'):
        globals()[sys.argv[1]]()
    else:
        smoke(sys.argv[1], int(sys.argv[2]))
