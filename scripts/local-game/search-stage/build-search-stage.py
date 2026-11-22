"""Local release 1.4.63: search_v1 startup estimate by era stage; Beta and Beta-Plus only.

Built from the repository's native/ tree, the same tree whose brass-ai-core the training CPU teacher copies, so an
engine improvement reaches both the local game and training in one pass; a local release only needs a rebuild.
What differs between them stays in their own layers:
- training (lab CPU teacher): fixed-count searches, long and short mixed, budgets from the recipe;
- local game (brass-ai-worker local.rs): one time-based search per turn, sized by the startup estimate, which
  only the local game calibrates. 1.4.63 adds `startup_ns_by_stage` (canal_early, canal_late, rail_early,
  rail_late: rounds 1-4 and 5-8 of each era; native/brass-ai-worker/src/local_startup.rs); `startup_ns` stays the
  fallback and bounds the table.
Only beta-g1290 (feature local-beta) and beta2-g1190 (local-beta2) are rebuilt. alpha-g890 and il-v11-2 leave
BrassNative, so the game lists them as "Model runtime is not installed"; their files go to the install's recovery
backup, so a rollback restores them.

`binaries` builds the two workers from native/ and runs the core and worker unit tests. `release` assembles the
release from the calibrated table (calibrate-search-stage.py measure).
"""
import json
import runpy
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SPEED = runpy.run_path(str(ROOT / 'scripts/local-game/engine-speed/build-engine-speed.py'))
sha, read, write, pins, MANAGED, TOOLCHAIN = (SPEED[k] for k in ('sha', 'read', 'write', 'pins', 'MANAGED', 'TOOLCHAIN'))
STUDY = ROOT / 'artifacts/search-stage'
BASE_VERSION = '1.4.62-rust-local'
BASE = ROOT / 'artifacts/search-calibration/release-1.4.62'
VERSION = '1.4.63-rust-local'
WORKERS = STUDY / 'native-build-1.4.63'
OUT = STUDY / 'release-1.4.63'
PACKAGES = {'beta-g1290': 'local-beta', 'beta2-g1190': 'local-beta2'}
RETIRED = ('alpha-g890', 'il-v11-2')
POLICIES = ('policy.json', 'policy-k8.json', 'policy-k32.json', 'policy-t15.json', 'policy-k8-t15.json', 'policy-k32-t15.json')
STAGES = ('canal_early', 'canal_late', 'rail_early', 'rail_late')
NATIVE = ROOT / 'native'


def tools():
    support = runpy.run_path(str(ROOT / 'scripts/qualify-rust-scaffold.py'))
    env, cargo = support['tool_environment']()
    return support, env, cargo


def binaries():
    assert not WORKERS.exists(), 'Preserve existing build evidence'
    support, env, cargo = tools()
    checkout = WORKERS / 'n'
    shutil.copytree(NATIVE, checkout / 'native', ignore=shutil.ignore_patterns('target', '__pycache__'))
    support['native_fingerprint'].__globals__['NATIVE'] = checkout / 'native'
    source_hash = support['native_fingerprint']()
    support['native_fingerprint'].__globals__['NATIVE'] = NATIVE
    assert support['native_fingerprint']() == source_hash, 'The copy differs from native/'
    with zipfile.ZipFile(WORKERS / 'native-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(checkout.rglob('*')):
            if file.is_file():
                archive.write(file, file.relative_to(checkout).as_posix())
    target = ROOT / 'artifacts/nn-temperature/target'
    env.update(CARGO_TARGET_DIR=str(target), CARGO_INCREMENTAL='0')
    for package, feature in PACKAGES.items():
        print('Building', package, flush=True)
        with (WORKERS / (package + '.log')).open('wb') as log:
            subprocess.run([str(cargo), 'build', '--release', '--offline', '--locked', '--bin', 'brass-ai-local',
                            '--features', feature], cwd=checkout / 'native', env=env, check=True,
                           stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        shutil.copyfile(target / 'x86_64-pc-windows-gnu/release/brass-ai-local.exe', WORKERS / (package + '.exe'))
    print('Testing the core and the worker', flush=True)
    tests = [('core-tests.log', ['-p', 'brass-ai-core', '--lib'])]
    tests += [(f'worker-tests-{f}.log', ['-p', 'brass-ai-worker', '--lib', '--features', f, 'local']) for f in PACKAGES.values()]
    for name, args in tests:
        with (WORKERS / name).open('wb') as log:
            subprocess.run([str(cargo), 'test', '--release', '--offline', '--locked', *args], cwd=checkout / 'native', env=env,
                           check=True, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    write(WORKERS / 'build.json', dict(source_sha256=source_hash, source='native/', packages=list(PACKAGES),
          workers={p: sha(WORKERS / (p + '.exe')) for p in PACKAGES}))
    print('Built workers', WORKERS, source_hash, flush=True)


def stage_policy(policy, table, fallback, source):
    """`policy` with the stage table; everything else as installed."""
    before = {k: v for k, v in policy.items() if k not in ('startup_ns', 'startup_source')}
    assert 'startup_ns_by_stage' not in policy
    result = dict(policy, startup_ns=fallback, startup_source=source, startup_ns_by_stage={s: table[s] for s in STAGES})
    assert {k: v for k, v in result.items() if k not in ('startup_ns', 'startup_source', 'startup_ns_by_stage')} == before
    return result


def assemble(out, table, fallback, source):
    """BrassNative for 1.4.63 at `out`: the two rebuilt packages with the stage table; no alpha or IL."""
    built = read(WORKERS / 'build.json')
    for package in PACKAGES:
        directory = out / 'BrassNative' / package
        shutil.copytree(ROOT / 'BrassNative' / package, directory)
        shutil.copyfile(WORKERS / (package + '.exe'), directory / 'brass-ai-local.exe')
        shutil.copyfile(WORKERS / 'native-source.zip', directory / 'native-source.zip')
        for name in POLICIES:
            write(directory / name, stage_policy(read(directory / name), table, fallback, source))
        listed = read(directory / 'build.json')
        listed['source_sha256'] = built['source_sha256']
        for pin in listed['files']:
            if pin['path'] == 'brass-ai-local.exe' or pin['path'] in POLICIES:
                pin['sha256'] = sha(directory / pin['path'])
            else:
                assert pin['sha256'] == sha(directory / pin['path']), pin['path']
        write(directory / 'build.json', listed)


def release():
    assert not OUT.exists(), 'Preserve existing build evidence'
    receipt = ROOT / 'artifacts/patch-installation.json'
    installed = read(receipt)
    receipt_hash = sha(receipt)
    assert installed['version'] == BASE_VERSION, installed['version']
    base_native = pins(ROOT / 'BrassNative')
    assert base_native == read(BASE / 'build-manifest.json')['nativeFiles'], 'The installed packages differ from 1.4.62'
    for file in installed['files']:
        assert sha(ROOT / MANAGED / file['name']) == file['sha256'].lower(), file['name']
    calibration = read(STUDY / 'calibration.json')
    table, fallback = calibration['table_ns'], calibration['fallback_ns']
    assert set(table) == set(STAGES) and fallback >= max(table.values())
    # The full verification sweep was stopped at the user's request (2026-11-21); qualify-search-stage.py checks
    # every candidate search's quota on each stage's action roots instead.
    source = f'1.4.63 era-stage calibration maximum; {sha(STUDY / "calibration.json")}'
    assemble(OUT, table, fallback, source)
    native = pins(OUT / 'BrassNative')
    removed = sorted(p['path'] for p in base_native if p['path'].split('/')[0] in RETIRED)
    assert {p['path'] for p in native} == {p['path'] for p in base_native} - set(removed)
    changed = sorted(p['path'] for p in native if p not in base_native)
    expected = sorted(f'{p}/{n}' for p in PACKAGES for n in ('brass-ai-local.exe', 'native-source.zip', 'build.json', *POLICIES))
    assert changed == expected, changed
    assert sha(receipt) == receipt_hash and pins(ROOT / 'BrassNative') == base_native
    write(OUT / 'build-manifest.json', dict(version=VERSION, baseVersion=BASE_VERSION, stockSha256=installed['stockSha256'],
        files=installed['files'], baseNative=base_native, nativeFiles=native, removedNative=removed, retiredModels=list(RETIRED),
        installedBaselineReceiptSha256=receipt_hash, nativeSourceSha256=read(WORKERS / 'build.json')['source_sha256'],
        startupNsByStage=table, startupNs=fallback, previousStartupNs=6360000, startupSource=source,
        calibrationSha256=sha(STUDY / 'calibration.json'), verification='qualification (full sweep stopped by the user)', coreTestsSha256=sha(WORKERS / 'core-tests.log'),
        workerTests={f: sha(WORKERS / f'worker-tests-{f}.log') for f in PACKAGES.values()},
        sourcePins=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in sorted([
            *HERE.glob('*.py'), *HERE.glob('*.ps1'), NATIVE / 'brass-ai-worker/src/local.rs',
            NATIVE / 'brass-ai-worker/src/local_startup.rs', NATIVE / 'brass-ai-worker/src/lib.rs'])]))
    print('Built', OUT, 'table', table, 'fallback', fallback, flush=True)


if __name__ == '__main__':
    {'binaries': binaries, 'release': release}[sys.argv[1]]()
