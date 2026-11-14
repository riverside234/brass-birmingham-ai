"""Build the local opening-temperature overlay from the installed game."""
import runpy
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'artifacts/nn-opening-temperature'
BASE_VERSION = '1.4.58-rust-local'
BASE = STUDY / 'release-1.4.58'
VERSION = '1.4.59-rust-local'
OUT = STUDY / 'release-1.4.59'
PACKAGES = {'alpha-g890': None, 'beta-g1290': 'local-beta', 'beta2-g1190': 'local-beta2', 'il-v11-2': 'local-il'}
MANAGED = 'BrassBirminghamUnity_Data/Managed'
MANAGED_SCOPE = ['src/BrassLocalAI/Native/NativePackage.cs', 'src/BrassLocalAI/Policy/LocalSetupPreferences.cs']
NATIVE_SCOPE = ['native/brass-ai-worker/src/local_temperature.rs']
HELPERS = runpy.run_path(str(ROOT / 'scripts/fix-sell-replay-speed.py'))
sha, read, write, pins = (HELPERS[key] for key in ('sha', 'read', 'write', 'pins'))


def build_workers():
    support = runpy.run_path(str(ROOT / 'scripts/qualify-rust-scaffold.py'))
    env, cargo = support['tool_environment']()
    out = STUDY / 'native-build-1.4.59'
    checkout = out / 'n'
    checkout.mkdir(parents=True)
    original = ROOT / 'BrassNative/beta2-g1190/native-source.zip'
    with zipfile.ZipFile(original) as source:
        assert all((checkout / name).resolve().is_relative_to(checkout.resolve()) for name in source.namelist())
        source.extractall(checkout)
    formatter = ROOT / 'artifacts/native-toolchain/rustup/toolchains/1.99.1-x86_64-pc-windows-gnu/bin/rustfmt.exe'
    for name in NATIVE_SCOPE:
        shutil.copyfile(ROOT / name, checkout / name)
        subprocess.run([str(formatter), '--edition', '2021', '--config', 'skip_children=true', str(checkout / name)],
                       env=env, check=True)
    support['native_fingerprint'].__globals__['NATIVE'] = checkout / 'native'
    source_hash = support['native_fingerprint']()
    with zipfile.ZipFile(out / 'native-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(checkout.rglob('*')):
            if file.is_file():
                archive.write(file, file.relative_to(checkout).as_posix())

    target = ROOT / 'artifacts/nn-temperature/target'
    env.update(CARGO_TARGET_DIR=str(target), CARGO_INCREMENTAL='0')
    for package, feature in PACKAGES.items():
        print('Building', package, flush=True)
        args = ['build', '--release', '--offline', '--locked', '--bin', 'brass-ai-local']
        if feature:
            args += ['--features', feature]
        with (out / (package + '.log')).open('wb') as log:
            subprocess.run([str(cargo), *args], cwd=checkout / 'native', env=env, check=True,
                           stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        shutil.copyfile(target / 'x86_64-pc-windows-gnu/release/brass-ai-local.exe', out / (package + '.exe'))
    with (out / 'worker-tests.log').open('wb') as log:
        subprocess.run([str(cargo), 'test', '--release', '--offline', '--locked', '-p', 'brass-ai-worker',
                        '--lib', '--features', 'local-beta2', 'local_temperature'], cwd=checkout / 'native', env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    return out, source_hash


def build():
    assert not OUT.exists(), 'Preserve existing build evidence'
    receipt = ROOT / 'artifacts/patch-installation.json'
    installed = read(receipt)
    receipt_hash = sha(receipt)
    base_version = installed['version']
    assert base_version == BASE_VERSION and installed['nnCScale'] == 0.1
    baseline = read(BASE / 'build-manifest.json')
    base_native = pins(ROOT / 'BrassNative')
    assert sha(BASE / 'qualification.json') == installed['nnOpeningTemperatureQualificationSha256'].lower()
    assert baseline['nativeFiles'] == base_native
    for file in installed['files']:
        name = file['name']
        assert sha(ROOT / MANAGED / name) == file['sha256'].lower() == sha(BASE / 'Managed' / name)
    for pin in baseline['sourcePins']:
        if pin['path'].startswith('src/'):
            assert sha(BASE / 'runtime' / pin['path']) == pin['sha256']
    for directory in ('Managed', 'runtime'):
        shutil.copytree(BASE / directory, OUT / directory)
    for name in MANAGED_SCOPE:
        shutil.copyfile(ROOT / name, OUT / 'runtime' / name)
    rsp = OUT / 'legacy.rsp'
    rsp.write_text((BASE / rsp.name).read_text(encoding='utf-8-sig').replace(str(BASE), str(OUT)), encoding='utf-8-sig')
    subprocess.run([str(HELPERS['CSC']), '@' + str(rsp)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    print('Built managed AI panel and turn bridge', flush=True)

    shutil.copytree(ROOT / 'BrassNative', OUT / 'BrassNative')
    workers, source_hash = build_workers()
    native_tests = sha(workers / 'worker-tests.log')
    changed_native = set()
    for package in PACKAGES:
        directory = OUT / 'BrassNative' / package
        shutil.copyfile(workers / (package + '.exe'), directory / 'brass-ai-local.exe')
        shutil.copyfile(workers / 'native-source.zip', directory / 'native-source.zip')
        build_list = read(directory / 'build.json')
        build_list['source_sha256'] = source_hash
        for pin in build_list['files']:
            if pin['path'] == 'brass-ai-local.exe':
                pin['sha256'] = sha(directory / pin['path'])
            else:
                assert pin['sha256'] == sha(directory / pin['path']), pin['path']
        write(directory / 'build.json', build_list)
        changed_native.update(f'{package}/{name}' for name in ('brass-ai-local.exe', 'native-source.zip', 'build.json'))
    native = pins(OUT / 'BrassNative')
    assert {p['path'] for p in native} == {p['path'] for p in base_native}
    assert {p['path'] for p in native if p not in base_native} == changed_native
    files = [dict(name=p['name'], sha256=sha(OUT / 'Managed' / p['name'])) for p in installed['files']]
    assert {p['name'] for p in files if p not in installed['files']} == {'BrassLegacyAI.dll'}
    assert sha(receipt) == receipt_hash and pins(ROOT / 'BrassNative') == base_native
    source_paths = sorted({pin['path'] for pin in baseline['sourcePins']} | set(MANAGED_SCOPE) | set(NATIVE_SCOPE))
    source_pins = [dict(path=p, sha256=sha(ROOT / p)) for p in source_paths]
    write(OUT / 'build-manifest.json', dict(version=VERSION, baseVersion=base_version,
        stockSha256=installed['stockSha256'], baseFiles=installed['files'], files=files,
        baseNative=base_native, nativeFiles=native, installedBaselineReceiptSha256=receipt_hash,
        nativeSourceSha256=source_hash, nativeTestsSha256=native_tests, sourcePins=source_pins,
        openingTemperatureChoices=[1.0, 1.5, 2.0], openingMovesPerSeat=3, regularTemperature=1.0, cScale=0.1))
    print('Built', OUT, flush=True)


if __name__ == '__main__':
    build()
