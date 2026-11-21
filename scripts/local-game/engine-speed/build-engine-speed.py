"""Build the local engine-speed rollout: the four installed AI packages rebuilt with the teacher's faster search core.

Native files only (brass-ai-local.exe, native-source.zip and build.json of each package). The installed native
source is the pre-change core; this replaces exactly the three core files that carry the exact decision reuse
(ai-research/ENGINE_SEARCH_SPEED.md) with the repository's, and leaves every other source file, the models, the
policies and the managed patch untouched.
"""
import runpy
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'artifacts/engine-speed'
BASE_VERSION = '1.4.60-rust-local'
VERSION = '1.4.61-rust-local'
OUT = STUDY / 'release-1.4.61'
PACKAGES = {'alpha-g890': None, 'beta-g1290': 'local-beta', 'beta2-g1190': 'local-beta2', 'il-v11-2': 'local-il'}
CHANGED = ('brass-ai-local.exe', 'native-source.zip', 'build.json')
MANAGED = 'BrassBirminghamUnity_Data/Managed'
CORE_SCOPE = ['native/brass-ai-core/src/search/brass.rs', 'native/brass-ai-core/src/search/reward.rs',
              'native/brass-ai-core/src/state/observation.rs']
# The installed copies of CORE_SCOPE are the repository's files before the engine change (commit 6739595e),
# up to rustfmt layout and line endings.
PRE_CHANGE = '6739595e'
HELPERS = runpy.run_path(str(ROOT / 'scripts/local-game/sell-replay-speed/fix-sell-replay-speed.py'))
sha, read, write, pins = (HELPERS[key] for key in ('sha', 'read', 'write', 'pins'))
TOOLCHAIN = ROOT / 'artifacts/native-toolchain/rustup/toolchains/1.99.1-x86_64-pc-windows-gnu/bin'


def formatted(text, env, scratch):
    """rustfmt layout with LF endings, to compare sources that differ only in layout."""
    scratch.write_text(text.replace('\r\n', '\n'), encoding='utf-8', newline='\n')
    subprocess.run([str(TOOLCHAIN / 'rustfmt.exe'), '--edition', '2021', '--config', 'skip_children=true', str(scratch)],
                   env=env, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    return scratch.read_text(encoding='utf-8').replace('\r\n', '\n')


def build_workers():
    support = runpy.run_path(str(ROOT / 'scripts/qualify-rust-scaffold.py'))
    env, cargo = support['tool_environment']()
    out = STUDY / 'native-build-1.4.61'
    checkout = out / 'n'
    checkout.mkdir(parents=True)
    sources = {sha(ROOT / 'BrassNative' / package / 'native-source.zip') for package in PACKAGES}
    assert len(sources) == 1, 'The four packages must share one installed native source'
    with zipfile.ZipFile(ROOT / 'BrassNative/beta2-g1190/native-source.zip') as source:
        assert all((checkout / name).resolve().is_relative_to(checkout.resolve()) for name in source.namelist())
        source.extractall(checkout)
    for name in CORE_SCOPE:
        installed = (checkout / name).read_text(encoding='utf-8')
        before = subprocess.run(['git', '-C', str(ROOT), 'show', f'{PRE_CHANGE}:{name}'], check=True,
                                capture_output=True).stdout.decode('utf-8')
        assert formatted(installed, env, out / 'installed.rs') == formatted(before, env, out / 'before.rs'), name
        shutil.copyfile(ROOT / name, checkout / name)
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
    # The core's unit tests include search::brass::decision_cache_tests: every reused decision is rebuilt and
    # compared in full, and the same searches with reuse off give identical reports.
    print('Testing the core', flush=True)
    with (out / 'core-tests.log').open('wb') as log:
        subprocess.run([str(cargo), 'test', '--release', '--offline', '--locked', '-p', 'brass-ai-core', '--lib'],
                       cwd=checkout / 'native', env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    with (out / 'worker-tests.log').open('wb') as log:
        subprocess.run([str(cargo), 'test', '--release', '--offline', '--locked', '-p', 'brass-ai-worker',
                        '--lib', '--features', 'local-beta2', 'local'], cwd=checkout / 'native', env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    return out, source_hash


def build():
    assert not OUT.exists(), 'Preserve existing build evidence'
    receipt = ROOT / 'artifacts/patch-installation.json'
    installed = read(receipt)
    receipt_hash = sha(receipt)
    assert installed['version'] == BASE_VERSION, installed['version']
    base_native = pins(ROOT / 'BrassNative')
    for package in PACKAGES:
        listed = read(ROOT / 'BrassNative' / package / 'build.json')
        assert listed['source_sha256'] == installed['nnOpeningTemperatureNativeSourceSha256'], package
        for pin in listed['files']:
            assert sha(ROOT / 'BrassNative' / package / pin['path']) == pin['sha256'], (package, pin['path'])
    for file in installed['files']:
        assert sha(ROOT / MANAGED / file['name']) == file['sha256'].lower(), file['name']

    shutil.copytree(ROOT / 'BrassNative', OUT / 'BrassNative')
    workers, source_hash = build_workers()
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
        changed_native.update(f'{package}/{name}' for name in CHANGED)
    native = pins(OUT / 'BrassNative')
    assert {p['path'] for p in native} == {p['path'] for p in base_native}
    assert {p['path'] for p in native if p not in base_native} == changed_native
    assert sha(receipt) == receipt_hash and pins(ROOT / 'BrassNative') == base_native
    source_pins = [dict(path=p, sha256=sha(ROOT / p)) for p in CORE_SCOPE]
    write(OUT / 'build-manifest.json', dict(version=VERSION, baseVersion=BASE_VERSION,
        stockSha256=installed['stockSha256'], files=installed['files'], baseNative=base_native, nativeFiles=native,
        installedBaselineReceiptSha256=receipt_hash, baseNativeSourceSha256=installed['nnOpeningTemperatureNativeSourceSha256'],
        nativeSourceSha256=source_hash, coreTestsSha256=sha(workers / 'core-tests.log'),
        workerTestsSha256=sha(workers / 'worker-tests.log'), sourcePins=source_pins, preChange=PRE_CHANGE))
    print('Built', OUT, flush=True)


if __name__ == '__main__':
    build()
