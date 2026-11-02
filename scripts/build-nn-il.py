"""Package and smoke-test IL over the installed 1.4.49 replay-speed fix."""
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import zipfile

H = runpy.run_path(str(Path(__file__).with_name('nn-il.py')))
ROOT, STUDY, BASE, OUT, BUNDLE, CHECKPOINT, BACKEND, PROFILE, SCOPE = (
    H[k] for k in ('ROOT', 'STUDY', 'BASE', 'OUT', 'BUNDLE', 'CHECKPOINT', 'BACKEND', 'PROFILE', 'SCOPE'))
sha, read, write, pins = (H[k] for k in ('sha', 'read', 'write', 'pins'))
SCOPE = SCOPE + ['src/BrassLocalAI/Unity/LocalTurnBridge.cs']
LEGACY = BASE
BASE = ROOT / 'artifacts/sell-replay-speed/20261101/release-1.4.49'
OUT = STUDY / 'release-1.4.50'


def native():
    out = STUDY / 'native-build'
    assert not out.exists(), 'Preserve native build evidence'
    out.mkdir(parents=True)
    support = runpy.run_path(str(ROOT / 'scripts/qualify-rust-scaffold.py'))
    env, cargo = support['tool_environment']()
    scope = ['native/brass-ai-worker/Cargo.toml', 'native/brass-ai-worker/src/local_model.rs']
    checkout = out / 'n'
    with zipfile.ZipFile(ROOT / 'BrassNative/beta-g1290/native-source.zip') as z:
        assert all((checkout / n).resolve().is_relative_to(checkout.resolve()) for n in z.namelist())
        z.extractall(checkout)
    formatter = ROOT / 'artifacts/native-toolchain/rustup/toolchains/1.99.1-x86_64-pc-windows-gnu/bin/rustfmt.exe'
    for name in scope:
        shutil.copyfile(ROOT / name, checkout / name)
        if name.endswith('.rs'):
            subprocess.run([str(formatter), '--edition', '2021', '--config', 'skip_children=true', str(checkout / name)], env=env, check=True)
    env.update(CARGO_TARGET_DIR=str(ROOT / 'artifacts/nn-temperature/target'), CARGO_INCREMENTAL='0')
    checks = []
    for label, args in (
        ('build', ['build', '--release', '--bin', 'brass-ai-local', '--features', 'local-il']),
        ('il-worker-tests', ['test', '--release', '-p', 'brass-ai-worker', '--lib', '--features', 'local-il'])):
        log = out / (label + '.log')
        with log.open('wb') as f:
            subprocess.run([str(cargo), *args, '--offline', '--locked'], cwd=checkout / 'native', env=env,
                           stdout=f, stderr=subprocess.STDOUT, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        checks.append(dict(stage=label, passed=True, log_sha256=sha(log)))
        print(log.read_text()[-1200:], flush=True)
    support['native_fingerprint'].__globals__['NATIVE'] = checkout / 'native'
    identity = support['native_fingerprint']()
    shutil.copyfile(Path(env['CARGO_TARGET_DIR']) / 'x86_64-pc-windows-gnu/release/brass-ai-local.exe', out / 'brass-ai-local.exe')
    with zipfile.ZipFile(out / 'native-source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(checkout.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(checkout).as_posix())
    write(out / 'native-build.json', dict(passed=True, source_sha256=identity, scope=scope, checks=checks,
        binary_sha256=sha(out / 'brass-ai-local.exe'), source_archive_sha256=sha(out / 'native-source.zip'), feature='local-il'))


def package():
    assert not OUT.exists(), 'Preserve candidate'
    installed = read(ROOT / 'artifacts/patch-installation.json')
    old = read(BASE / 'build-manifest.json')
    assert installed['version'] == old['version'] == '1.4.49-rust-local' and read(BASE / 'qualification.json')['passed']
    assert pins(ROOT / 'BrassNative') == old['nativeFiles']
    for p in old['files']:
        assert sha(ROOT / 'BrassBirminghamUnity_Data/Managed' / p['name']) == p['sha256']
    native = read(STUDY / 'native-build/native-build.json')
    model = read(STUDY / 'model/manifest.json')
    assert native['passed'] and model['checkpoint_sha256'] == CHECKPOINT
    for folder in ('runtime', 'Managed'):
        shutil.copytree(BASE / folder, OUT / folder)
    shutil.copytree(ROOT / 'BrassNative', OUT / 'BrassNative')
    for name in SCOPE:
        shutil.copyfile(ROOT / name, OUT / 'runtime' / name)
    rsp = OUT / 'legacy.rsp'
    rsp.write_text((LEGACY / 'legacy.rsp').read_text(encoding='utf-8-sig').replace(str(LEGACY), str(OUT)), encoding='utf-8-sig')
    subprocess.run([str(ROOT / 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'), '@' + str(rsp)], check=True)
    il = OUT / 'BrassNative/il-v11-2'
    shutil.copytree(OUT / 'BrassNative/alpha-g890', il)
    for name in ('brass-ai-local.exe', 'native-source.zip'):
        shutil.copyfile(STUDY / 'native-build' / name, il / name)
    for p in (STUDY / 'model').iterdir():
        if p.is_file():
            shutil.copyfile(p, il / 'model' / p.name)
    shutil.copyfile(BUNDLE / 'manifest.json', il / 'model/checkpoint-manifest.json')
    warm = read(il / 'warmup.json')
    warm['context'].update(model_id=model['model_id'], profile_id=PROFILE, style=[0.] * 4, intent=[0.] * 8)
    write(il / 'warmup.json', warm)
    acceptance = dict(version='brass-local-utility-acceptance/1.0.0', scope='part1-local-il-test', decision='user-2026-11-02',
        model_id=model['model_id'], export_sha256=model['export_sha256'], checkpoint_sha256=CHECKPOINT,
        utility_calibration=None, final_model=False)
    for name in ('policy.json', 'policy-k8.json', 'policy-k32.json', 'policy-t15.json', 'policy-k8-t15.json', 'policy-k32-t15.json'):
        policy = read(il / name)
        policy.update(backend=BACKEND, model_manifest_sha256=sha(il / 'model/manifest.json'),
                      acceptance=acceptance, warmup_sha256=sha(il / 'warmup.json'))
        assert policy['method'] == 'search_v1'
        write(il / name, policy)
    build = read(il / 'build.json')
    build.update(backend=BACKEND, scope='local-il-test', source_sha256=native['source_sha256'])
    for p in build['files']:
        p['sha256'] = sha(il / p['path'])
    write(il / 'build.json', build)
    (il / 'README.txt').write_text(
        'IL v11.2 imitation checkpoint. Neutral context, search_v1, V205 disabled.\n'
        'Original policy/trunk/official heads unchanged; no fitting or calibration.\n'
        'Search utility = win + 0.1*(4-placement)/3 + 0.001*min(VP,180), zero residual.\n'
        'The inference extension has its own derived model ID; original identity is in model/export-receipt.json.\n'
        'Two independent ONNX Runtime CPU sessions, one thread per session.\n'
        'Rebuild native/: cargo build --locked --release --bin brass-ai-local --features local-il --target x86_64-pc-windows-gnu\n'
        'Sources: native-source.zip and model/export-source.zip; dependency notices retained.\n', encoding='utf-8')
    old.update(version='1.4.50-rust-local', baseVersion=installed['version'], baseFiles=installed['files'], baseNative=pins(ROOT / 'BrassNative'),
        sourceScope=SCOPE, sourcePins=pins(OUT / 'runtime'),
        files=[dict(name=p['name'], sha256=sha(OUT / 'Managed' / p['name'])) for p in old['files']], nativeFiles=pins(OUT / 'BrassNative'),
        installed_baseline_receipt_sha256=sha(ROOT / 'artifacts/patch-installation.json'), il_model_id=model['model_id'],
        il_checkpoint_sha256=CHECKPOINT, il_export_sha256=model['export_sha256'], il_native_source_sha256=native['source_sha256'],
        il_native_build_sha256=sha(il / 'build.json'), nn_ui_choices=['alpha-test', 'beta', 'il'], native_unchanged=False,
        search_method='search_v1', v205_enabled=False)
    write(OUT / 'build-manifest.json', old)
    shutil.copyfile(ROOT / 'artifacts/patch-installation.json', STUDY / 'before-installation.json')
    write(STUDY / 'user-data-before.json', dict(files=[dict(path=str(p), sha256=sha(p))
        for directory in installed['saveDataPaths'] for p in Path(directory).rglob('*') if p.is_file()]))
    print(OUT, flush=True)


def unity(label):
    d = STUDY / label
    assert not d.exists(), 'Preserve smoke evidence'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for name in ('BRASS_BETA_UI_TEST', 'BRASS_IL_UI_TEST', 'BRASS_INDUSTRY_FOCUS_TEST', 'BRASS_L14_TRACE'):
        env.pop(name, None)
    command = [shutil.which('pwsh'), '-NoProfile', '-File', str(ROOT / 'tests/RustBackendLocal/run-unity.ps1'),
        '-BuildDirectory', str(OUT.relative_to(ROOT)), '-EvidenceDirectory', str(d.relative_to(ROOT)), '-MaxSeconds', '60']
    if label.startswith('ui-'):
        env['BRASS_IL_UI_TEST'] = label[3:]
        command += ['-IlUi']
    elif label == 'legacy-ui':
        command += ['-SetupPreferences', 'write']
    else:
        scenario = 'cancel' if label == 'cancel-il' else 'normal'
        model = label[5:] if label.startswith('live-') else 'il'
        env.update(BRASS_SEARCH_V2_RESULT=str(d / 'result.json'),
            BRASS_SEARCH_V2_EVENTS=str(ROOT / 'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures/live-92781-CanalEra-2.json'),
            BRASS_SEARCH_V2_MS='4000', BRASS_SEARCH_K='16', BRASS_NN_TEMPERATURE='1.0', BRASS_NN_FOCUS='neutral',
            BRASS_NN_MODEL=model, BRASS_SEARCH_V2_SCENARIO=scenario, BRASS_SEARCH_HOLD_UNTIL_MS='0')
        command += ['-SearchV2']
    with (STUDY / (label + '.log')).open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=85)
    u = read(d / 'unity.json')
    assert u['passed'] and not u['nativeErrors']
    if label.startswith(('live-', 'cancel-')):
        r = read(d / 'result.json')
        assert r['passed'] and r['nn_model'] == model and r['search_method'] == 'search_v1'
        manifest = read(OUT / 'BrassNative' / dict(il='il-v11-2', beta='beta-g1290', **{'alpha-test': 'alpha-g890'})[model] / 'model/manifest.json')
        assert all(x['model_id'] == manifest['model_id'] and x['export_sha256'] == manifest['export_sha256']
                   and x['sessions'] == 2 and x['threads_per_session'] == 1 for x in r['replies'])
        assert r['commits'][-1]['remaining'] == (0 if scenario == 'normal' else 1)
        if scenario == 'normal':
            assert len(r['commits']) == 2 and r['turn_clock_ms'] < 5000
    print(label, u['checks'], 'checks', flush=True)


def qualify():
    assert not (OUT / 'qualification.json').exists()
    m = read(OUT / 'build-manifest.json')
    previous = {p['path']: p['sha256'] for p in read(BASE / 'build-manifest.json')['sourcePins']}
    assert {p['path'] for p in m['sourcePins'] if previous.get(p['path']) != p['sha256']} == set(SCOPE)
    for p in m['sourcePins']:
        assert sha(OUT / 'runtime' / p['path']) == p['sha256']
    for p in SCOPE:
        assert (ROOT / p).read_bytes() == (OUT / 'runtime' / p).read_bytes()
    assert pins(OUT / 'BrassNative') == m['nativeFiles']
    for p in m['baseNative']:
        assert p in m['nativeFiles'] and sha(ROOT / 'BrassNative' / p['path']) == p['sha256']
    for p in m['files']:
        assert sha(OUT / 'Managed' / p['name']) == p['sha256']
        if p['name'] != 'BrassLegacyAI.dll':
            assert p in m['baseFiles']
    receipt = read(STUDY / 'model/export-receipt.json')
    assert sha(BUNDLE / 'checkpoint.pt') == CHECKPOINT and receipt['passed']
    assert receipt['official_heads_exact'] and receipt['zero_residual'] and receipt['training_steps'] == 0
    for p in receipt['source_pins']:
        assert sha(ROOT / p['path']) == p['sha256']
    with zipfile.ZipFile(ROOT / 'BrassNative/beta-g1290/native-source.zip') as before, \
            zipfile.ZipFile(STUDY / 'native-build/native-source.zip') as after:
        a, b = {n: before.read(n) for n in before.namelist()}, {n: after.read(n) for n in after.namelist()}
        assert sorted(n for n in a.keys() | b.keys() if a.get(n) != b.get(n)) == [
            'native/brass-ai-worker/Cargo.toml', 'native/brass-ai-worker/src/local_model.rs']
    native = read(STUDY / 'native-build/native-build.json')
    assert native['passed'] and native['feature'] == 'local-il'
    evidence = []
    for label in ('ui-write', 'ui-read', 'legacy-ui', 'live-il', 'cancel-il', 'live-beta', 'live-alpha-test'):
        d = STUDY / label
        t, u = read(d / 'test-build.json'), read(d / 'unity.json')
        assert u['passed'] and not u['nativeErrors'] and t['candidateManifestSha256'] == sha(OUT / 'build-manifest.json')
        assert t['testDllSha256'] == sha(d / 'LocalAiUnityTests.dll')
        for p in t['sources']:
            assert sha(d / 'test-source' / p['path']) == p['sha256']
        evidence.append(dict(label=label, checks=u['checks'], unity_sha256=sha(d / 'unity.json')))
    images = OUT / 'evidence'
    images.mkdir()
    shutil.copyfile(ROOT / 'artifacts/card-tracker-player/artifacts/patch-build/il-1080-largest.png', images / 'il-1080-largest.png')
    write(OUT / 'qualification.json', dict(format='brass-nn-il-release/1.0.0', passed=True, version=m['version'],
        build_manifest_sha256=sha(OUT / 'build-manifest.json'), il_native_build_receipt_sha256=sha(STUDY / 'native-build/native-build.json'),
        export_receipt_sha256=sha(STUDY / 'model/export-receipt.json'), il_model_id=m['il_model_id'], il_checkpoint_sha256=CHECKPOINT,
        existing_packages_unchanged=True, engine_search_inference_sources_unchanged=True,
        official_heads_exact=True, zero_residual=True, training_steps=0,
        evidence=evidence, files=pins(OUT), scope='Short functional smokes; no strength claim.'))
    print('Qualified additive IL package', flush=True)


if __name__ == '__main__':
    stage = sys.argv[1]
    assert stage in ('native', 'package', 'ui-write', 'ui-read', 'legacy-ui', 'live-il', 'cancel-il', 'live-beta', 'live-alpha-test', 'qualify')
    unity(stage) if stage.startswith(('ui-', 'live-', 'cancel-')) or stage == 'legacy-ui' else globals()[stage]()
