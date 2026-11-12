"""Verify opening-temperature searches and the NN panel in an isolated Unity player."""
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

H = runpy.run_path(str(Path(__file__).with_name('build-nn-opening-temperature.py')))
ROOT, STUDY, OUT, MANAGED, PACKAGES, sha, read, write, pins = (
    H[key] for key in ('ROOT', 'STUDY', 'OUT', 'MANAGED', 'PACKAGES', 'sha', 'read', 'write', 'pins'))
EVIDENCE = STUDY / 'beta-plus'
PLAYER = EVIDENCE / 'player'
FORMAT = 'brass-nn-opening-temperature-release/1.0.0'


def native(game):
    Worker = runpy.run_path(str(ROOT / 'tests/RustBackendLocal/test_local.py'))['Worker']
    results = []
    for model in PACKAGES:
        package = game / 'BrassNative' / model
        policy = read(package / 'policy.json')
        assert policy['config']['c2'] == 0.1 and policy['method'] == 'search_v1'
        worker = Worker(package)
        try:
            ready = worker.hello['data']
            assert ready['policy_temperature'] == 1.0 and ready['request_policy_temperature'] is True
            assert ready['build']['source_sha256'] == read(package / 'build.json')['source_sha256']
            assert ready['model_id'] == policy['acceptance']['model_id']
            inp = read(package / 'warmup.json')
            if model == 'alpha-g890':
                inp['context']['profile_id'] = 'alpha-test-g890-neutral/1.0.0'
            ticket = dict(game_instance='opening-qualification', game_id='fixture', revision=inp['view']['context']['revision'],
                actor=inp['view']['seat'], profile=inp['context']['profile_id'], source_sha256=ready['build']['source_sha256'])
            observed = []
            for generation, selected in ((1, 1.3), (1, 1.5), (2, 1.5), (3, None)):
                operation = dict(op='search', input=inp, ticket=dict(ticket, generation=generation), allowance_ms=4000, remaining_ms=2000)
                if selected is not None:
                    operation['policy_temperature'] = selected
                request_id = worker.send(operation)
                reply = worker.recv(timeout=15)
                assert reply['id'] == request_id, reply
                data = reply['data']
                if generation == 1 and selected == 1.5:
                    assert reply['status'] == 'error' and 'turn policy temperature changed' in data['error'], reply
                    continue
                assert reply['status'] == 'ok' and data['retired'] and data['actions'], reply
                assert data['policy_temperature'] == (1.0 if selected is None else selected), reply
                observed.append(data['policy_temperature'])
            results.append(dict(model=model, temperatures=observed, warmup=ready['measured_readiness']))
            print(model, observed, flush=True)
        finally:
            worker.close()
    return results


def stage_player():
    PLAYER.mkdir(parents=True, exist_ok=True)
    for name in ('BrassBirminghamUnity.exe', 'UnityCrashHandler64.exe', 'UnityPlayer.dll'):
        shutil.copyfile(ROOT / name, PLAYER / name)

    def link_asset(source, dest):
        if os.path.exists(dest) and os.path.samefile(source, dest):
            return dest
        try:
            os.link(source, dest)
        except OSError:
            shutil.copyfile(source, dest)
        return dest

    shutil.copytree(ROOT / 'MonoBleedingEdge', PLAYER / 'MonoBleedingEdge', copy_function=link_asset, dirs_exist_ok=True)
    data = ROOT / 'BrassBirminghamUnity_Data'
    shutil.copytree(data, PLAYER / data.name, ignore=shutil.ignore_patterns('Managed'), copy_function=link_asset, dirs_exist_ok=True)
    shutil.copytree(OUT / 'Managed', PLAYER / MANAGED, dirs_exist_ok=True)
    shutil.copytree(OUT / 'BrassNative', PLAYER / 'BrassNative', dirs_exist_ok=True)
    staging = PLAYER / 'artifacts/patch-build/Managed'
    staging.mkdir(parents=True, exist_ok=True)
    sources = ['tests/BrassLocalAI/' + name + '.cs' for name in (
        'LocalAiUnityTests', 'AiExecutionResourceTests', 'MixedAiUnityTests', 'OpeningTemperatureUnityTests')]
    sources += ['tests/RustBackendLocal/' + name + '.cs' for name in (
        'NativeLocalUnityTests', 'NativeMatchedUnityTests', 'NativeValidationLifecycleTests',
        'NativeReleaseUnityTests', 'ReleaseSegmentUnityTests')]
    H['HELPERS']['compile_cs'](staging / 'LocalAiUnityTests.dll', [ROOT / name for name in sources],
        [p for p in (PLAYER / MANAGED).glob('*.dll') if 'Tests' not in p.name and 'Regression' not in p.name], ['/nostdlib+'])


def unity(mode):
    staging = PLAYER / 'artifacts/patch-build'
    target = EVIDENCE / mode
    target.mkdir(exist_ok=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith('BRASS_')}
    env['BRASS_OPENING_TEMPERATURE_TEST'] = mode
    result = staging / 'local-ai-native.json'
    write(result, dict(passed=False))
    process = subprocess.run([str(PLAYER / 'BrassBirminghamUnity.exe'), '-batchmode', '-screen-fullscreen', '0',
        '-screen-width', '1920', '-screen-height', '1080', '-logFile', str(target / 'unity.log'), '--brass-local-ai-test'],
        cwd=PLAYER, env=env, timeout=80, creationflags=subprocess.CREATE_NO_WINDOW)
    shutil.copyfile(result, target / 'unity.json')
    evidence = read(result)
    assert process.returncode == 0 and evidence['passed'] and not evidence['nativeErrors'], evidence['error']
    for path in evidence['images']:
        image = Path(path)
        if image.name.startswith('opening-temperature-'):
            shutil.copyfile(image, target / image.name)
    print(mode, evidence['checks'], 'Unity checks', flush=True)


def qualify():
    m = read(OUT / 'build-manifest.json')
    assert pins(OUT / 'BrassNative') == m['nativeFiles']
    stage_player()
    for mode in ('ui', 'live'):
        unity(mode)
    workers = native(OUT)
    assert pins(PLAYER / 'BrassNative') == m['nativeFiles']
    for p in m['files']:
        assert sha(OUT / 'Managed' / p['name']) == sha(PLAYER / MANAGED / p['name']) == p['sha256']
    for p in m['sourcePins']:
        assert sha(ROOT / p['path']) == p['sha256']
    write(OUT / 'qualification.json', dict(format=FORMAT, passed=True,
        manifest_sha256=sha(OUT / 'build-manifest.json'), workers=workers, ui=pins(EVIDENCE / 'ui'), live=pins(EVIDENCE / 'live'),
        nativeTestsSha256=m['nativeTestsSha256']))


def audit():
    m = read(OUT / 'build-manifest.json')
    receipt = read(ROOT / 'artifacts/patch-installation.json')
    assert receipt['version'] == m['version'] and receipt['files'] == m['files']
    assert pins(ROOT / 'BrassNative') == m['nativeFiles']
    for p in m['files']:
        assert sha(ROOT / MANAGED / p['name']) == p['sha256']
    write(EVIDENCE / 'installed-verification.json', dict(passed=True, version=m['version'], workers=native(ROOT),
        qualification_sha256=sha(OUT / 'qualification.json')))


if __name__ == '__main__':
    {'qualify': qualify, 'audit': audit}[sys.argv[1]]()
