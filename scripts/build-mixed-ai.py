"""Build and verify the per-seat AI / 2x animation overlay on installed 1.4.50."""
from pathlib import Path
import json
import os
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/mixed-ai/20261102'
BASE = ROOT / 'artifacts/nn-il/20261102/release-1.4.50'
OUT = STUDY / 'release-1.4.51'
H = runpy.run_path(str(ROOT / 'scripts/fix-sell-replay-speed.py'))
sha, read, write, pins, compile_cs = (H[k] for k in ('sha', 'read', 'write', 'pins', 'compile_cs'))
SCOPE = ['src/BrassLocalAI/' + name for name in (
    'Policy/MixedGameState.cs', 'Policy/LocalGameState.cs', 'Policy/LocalSetupPreferences.cs', 'Policy/AiSelectionCatalog.cs',
    'Native/NativeGameState.cs', 'Native/NativeInput.cs', 'Native/NativeModel.cs', 'Native/NativeProcess.cs',
    'Unity/LocalMenu.cs', 'Unity/LocalTurnBridge.cs', 'Unity/AiSelectionPanel.cs')] + ['src/BrassPatch.Weaver/AnimationDefaults.cs']
NAMES = ('Assembly-CSharp.dll', 'BrassPatch.dll', 'BrassLegacyAI.dll')


def build():
    assert not OUT.exists(), 'Preserve existing build evidence'
    old = read(BASE / 'build-manifest.json'); installed = read(ROOT / 'artifacts/patch-installation.json')
    assert installed['version'] == old['version'] == '1.4.50-rust-local' and read(BASE / 'qualification.json')['passed']
    for pin in installed['files']:
        assert sha(ROOT / 'BrassBirminghamUnity_Data/Managed' / pin['name']) == pin['sha256'] == sha(BASE / 'Managed' / pin['name'])
    shutil.copytree(BASE / 'runtime', OUT / 'runtime'); shutil.copytree(BASE / 'Managed', OUT / 'Managed')
    shutil.copytree(ROOT / 'BrassNative', OUT / 'BrassNative')
    for name in SCOPE:
        dst = OUT / 'runtime' / name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dst)
    tools = OUT / 'tools'; tools.mkdir()
    cecil = ROOT / 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll'
    shutil.copyfile(cecil, tools / cecil.name)
    compile_cs(tools / 'AnimationDefaults.exe', [OUT / 'runtime/src/BrassPatch.Weaver/AnimationDefaults.cs'], [cecil])
    subprocess.run([str(tools / 'AnimationDefaults.exe'), str(BASE / 'Managed/Assembly-CSharp.dll'), str(OUT / 'Managed/Assembly-CSharp.dll')], check=True)
    rsp = (BASE / 'legacy.rsp').read_text(encoding='utf-8-sig').replace(str(BASE), str(OUT))
    rsp += '\n"' + str(OUT / 'runtime/src/BrassLocalAI/Policy/MixedGameState.cs') + '"\n'
    (OUT / 'legacy.rsp').write_text(rsp, encoding='utf-8-sig')
    subprocess.run([str(H['CSC']), '@' + str(OUT / 'legacy.rsp')], check=True)
    write(OUT / 'build-manifest.json', dict(version='1.4.51-rust-local', baseVersion=old['version'], stockSha256=old['stockSha256'],
        baseFiles=installed['files'], files=[dict(name=n, sha256=sha(OUT / 'Managed' / n)) for n in NAMES],
        nativeFiles=pins(ROOT / 'BrassNative'), sourceScope=SCOPE, sourcePins=pins(OUT / 'runtime'),
        installed_baseline_receipt_sha256=sha(ROOT / 'artifacts/patch-installation.json'),
        mixed_ai=True, animation_default=2, search_method='search_v1', v205_enabled=False))
    write(STUDY / 'protected-before.json', H['protected']())
    print('Built 1.4.51 per-seat AI + 2x animations', flush=True)


def smoke(label, installed=False):
    target = STUDY / label
    assert not target.exists(), 'Use a new evidence label'
    target.mkdir(parents=True)
    game = ROOT if installed else ROOT / 'artifacts/card-tracker-player'
    managed = game / 'BrassBirminghamUnity_Data/Managed'
    manifest = read(OUT / 'build-manifest.json')
    if not installed:
        for n in NAMES: shutil.copyfile(OUT / 'Managed' / n, managed / n)
        # This isolated player already holds the three qualified model packages.
        assert pins(game / 'BrassNative') == manifest['nativeFiles']
    tests = ['tests/BrassLocalAI/LocalAiUnityTests.cs', 'tests/BrassLocalAI/AiExecutionResourceTests.cs',
        'tests/RustBackendLocal/NativeLocalUnityTests.cs', 'tests/RustBackendLocal/NativeMatchedUnityTests.cs',
        'tests/RustBackendLocal/NativeValidationLifecycleTests.cs', 'tests/RustBackendLocal/NativeReleaseUnityTests.cs',
        'tests/RustBackendLocal/ReleaseSegmentUnityTests.cs', 'tests/BrassLocalAI/MixedAiUnityTests.cs']
    sources = []
    for name in tests:
        dst = target / 'test-source' / name; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, dst); sources.append(dst)
    dll = target / 'LocalAiUnityTests.dll'
    compile_cs(dll, sources, [p for p in managed.glob('*.dll') if 'Tests' not in p.name and 'Regression' not in p.name], ['/nostdlib+'])
    staging = game / 'artifacts/patch-build'
    touched = [staging / 'Managed/LocalAiUnityTests.dll', staging / 'local-ai-native.json']
    prior = {}
    for p in touched:
        if p.exists():
            saved = target / ('prior-' + p.name); shutil.copyfile(p, saved); prior[p] = saved
    env = dict(os.environ)
    for key in ('BRASS_IL_UI_TEST', 'BRASS_BETA_UI_TEST', 'BRASS_INDUSTRY_FOCUS_TEST'):
        env.pop(key, None)
    mode = 'live' if 'live' in label else 'read' if 'read' in label else 'write'
    env.update(BRASS_MIXED_CASE=mode, BRASS_MIXED_PREFERENCES=str(STUDY / 'preferences.json'),
        BRASS_MIXED_FIXTURE=str(ROOT / 'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures/live-92781-CanalEra-2.json'))
    try:
        shutil.copyfile(dll, touched[0]); write(touched[1], dict(passed=False, status='running'))
        result = subprocess.run([str(game / 'BrassBirminghamUnity.exe'), '-batchmode', '-screen-width', '1920', '-screen-height', '1080',
            '-screen-fullscreen', '0', '-logFile', str(target / 'unity.log'), '--brass-local-ai-test'], cwd=game, env=env,
            timeout=60, creationflags=subprocess.CREATE_NO_WINDOW)
        shutil.copyfile(touched[1], target / 'unity.json')
        u = read(target / 'unity.json')
        write(target / 'test-build.json', dict(testedFiles=[dict(name=n, sha256=sha(managed / n)) for n in NAMES],
            sources=pins(target / 'test-source'), testDllSha256=sha(dll), mode=mode))
        print(json.dumps(dict(label=label, passed=u['passed'], checks=u.get('checks'), error=u.get('error'))), flush=True)
        assert result.returncode == 0 and u['passed'] and not u['nativeErrors']
    finally:
        for p in touched:
            if p in prior: shutil.copyfile(prior[p], p)
            elif p.exists(): p.unlink()


def qualify():
    m = read(OUT / 'build-manifest.json')
    original = {p['path']: p['sha256'] for p in read(BASE / 'build-manifest.json')['sourcePins']}
    assert {p['path'] for p in m['sourcePins'] if original.get(p['path']) != p['sha256']} == set(SCOPE)
    for p in m['sourcePins']: assert sha(OUT / 'runtime' / p['path']) == p['sha256']
    for name in SCOPE: assert sha(ROOT / name) == sha(OUT / 'runtime' / name)
    assert pins(ROOT / 'BrassNative') == pins(OUT / 'BrassNative') == m['nativeFiles']
    assert read(STUDY / 'protected-before.json') == H['protected']()
    subprocess.run([shutil.which('pwsh'), '-NoProfile', '-File', str(ROOT / 'tests/BrassLocalAI/verify-mixed-ai-overlay.ps1')], check=True)
    reports = []
    for label in ('ui-write', 'ui-read', 'live'):
        u = read(STUDY / label / 'unity.json'); t = read(STUDY / label / 'test-build.json')
        assert u['passed'] and not u['nativeErrors'] and t['testedFiles'] == m['files']
        reports.append(dict(label=label, checks=u['checks'], sha256=sha(STUDY / label / 'unity.json')))
    write(OUT / 'qualification.json', dict(format='brass-mixed-ai-release/1.0.0', passed=True,
        build_manifest_sha256=sha(OUT / 'build-manifest.json'), evidence=reports, native_unchanged=True))
    print('Qualified mixed AI candidate', flush=True)


def upgrade():
    fixture = STUDY / 'install-rehearsal'
    assert not fixture.exists(), 'Preserve installer rehearsal'
    managed = fixture / 'BrassBirminghamUnity_Data/Managed'; managed.mkdir(parents=True)
    for n in NAMES: shutil.copyfile(BASE / 'Managed' / n, managed / n)
    shutil.copytree(ROOT / 'BrassNative', fixture / 'BrassNative')
    record = fixture / 'artifacts/patch-installation.json'; record.parent.mkdir()
    prior = read(ROOT / 'artifacts/patch-installation.json'); prior['gameRoot'] = str(fixture); prior['saveDataPaths'] = [str(fixture / 'saves')]
    write(record, prior); original = sha(record)
    command = [shutil.which('pwsh'), '-NoProfile', '-File', str(ROOT / 'scripts/install-mixed-ai.ps1'), '-GameRoot', str(fixture)]
    for action, extra, success in [('Install', [], True), ('Rollback', [], True), ('Install', ['-TestInterrupt'], False), ('Recover', [], True)]:
        result = subprocess.run(command + ['-Action', action] + extra, capture_output=True, text=True)
        (fixture / (action + ('-interrupt' if extra else '') + '.log')).write_text(result.stdout + result.stderr, encoding='utf-8')
        assert (result.returncode == 0) == success, result.stdout + result.stderr
        if not success: assert 'TEST INTERRUPT' in result.stderr
        if action in ('Rollback', 'Recover'):
            assert sha(record) == original
            for p in prior['files']: assert sha(managed / p['name']) == p['sha256']
    write(STUDY / 'upgrade-test.json', dict(passed=True, install_rollback=True, interrupted_recovery=True,
        installer_sha256=sha(ROOT / 'scripts/install-mixed-ai.ps1'), qualification_sha256=sha(OUT / 'qualification.json')))
    print('Installer, rollback and interrupted recovery passed', flush=True)


def audit():
    manifest = read(OUT / 'build-manifest.json'); record = ROOT / 'artifacts/patch-installation.json'; installed = read(record)
    assert installed['version'] == manifest['version'] and installed['files'] == manifest['files']
    for p in installed['files']: assert sha(ROOT / 'BrassBirminghamUnity_Data/Managed' / p['name']) == p['sha256']
    assert pins(ROOT / 'BrassNative') == manifest['nativeFiles']
    assert read(STUDY / 'protected-before.json') == H['protected']()
    for name in SCOPE: assert sha(ROOT / name) == sha(OUT / 'runtime' / name)
    reports = []
    for label in ('installed-read', 'installed-live'):
        u = read(STUDY / label / 'unity.json'); t = read(STUDY / label / 'test-build.json')
        assert u['passed'] and not u['nativeErrors'] and t['testedFiles'] == manifest['files']
        assert sha(STUDY / label / 'LocalAiUnityTests.dll') == t['testDllSha256']
        assert pins(STUDY / label / 'test-source') == t['sources']
        reports.append(dict(label=label, checks=u['checks'], sha256=sha(STUDY / label / 'unity.json')))
    upgrade = read(STUDY / 'upgrade-test.json')
    assert upgrade['passed'] and upgrade['installer_sha256'] == sha(ROOT / 'scripts/install-mixed-ai.ps1')
    assert upgrade['qualification_sha256'] == sha(OUT / 'qualification.json')
    evidence = OUT / 'evidence'; evidence.mkdir(exist_ok=True)
    for size in ('1920x1080', '1920x1200', '1280x720'):
        name = 'mixed-ai-' + size + '.png'
        shutil.copyfile(ROOT / 'artifacts/card-tracker-player/artifacts/patch-build' / name, evidence / name)
    verification = STUDY / 'installed-verification.json'
    write(verification, dict(passed=True, version=installed['version'], native_unchanged=True,
        user_saves_and_history_unchanged=True, scope='short functional smoke tests', ui_primary=['1920x1080', '1920x1200'],
        installed=reports, screenshots=pins(evidence), game_overlay=read(STUDY / 'game-overlay-verification.json'),
        live_turns=read(STUDY / 'installed-live/unity.json')['turns'], installer=upgrade))
    installed.update(mixedAiInstalledSmoke='passed/mixed-live-turns-save-load-cancellation-ui',
        mixedAiInstalledVerification=str(verification), mixedAiInstalledVerificationSha256=sha(verification))
    temporary = record.with_suffix('.mixed.tmp'); write(temporary, installed); temporary.replace(record)
    print('Installed 1.4.51 verified; model packages and user saves/history unchanged', flush=True)


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage in ('build', 'qualify', 'upgrade', 'audit'): globals()[stage]()
    else: smoke(stage, stage.startswith('installed-'))
