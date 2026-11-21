"""Build release 1.4.62: the installed 1.4.61 packages with the re-calibrated search_v1 startup estimate.

Only each package's six policy files (`startup_ns`, `startup_source`) and the build.json pins of those files change.
The binaries, native source, models, warm-up roots, K mappings, temperatures and the managed patch are unchanged.
The estimate is the full-quota calibration maximum (calibrate-search-v1.py full), rounded up to a whole microsecond,
and needs a verification at that estimate in which every search completed its quota.
"""
import runpy
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SPEED = runpy.run_path(str(ROOT / 'scripts/local-game/engine-speed/build-engine-speed.py'))
sha, read, write, pins, MANAGED = (SPEED[key] for key in ('sha', 'read', 'write', 'pins', 'MANAGED'))
PACKAGES = list(SPEED['PACKAGES'])
STUDY = ROOT / 'artifacts/search-calibration'
BASE_VERSION = '1.4.61-rust-local'
BASE = ROOT / 'artifacts/engine-speed/release-1.4.61'
VERSION = '1.4.62-rust-local'
OUT = STUDY / 'release-1.4.62'
POLICIES = ('policy.json', 'policy-k8.json', 'policy-k32.json', 'policy-t15.json', 'policy-k8-t15.json', 'policy-k32-t15.json')
CHANGED = (*POLICIES, 'build.json')
PREVIOUS_NS = 7262519


def estimate():
    """Stage 2's maximum lane cost at the real quota, rounded up to a microsecond (the L12 rule), which needs stage 1
    behind it and a full verification at that value."""
    calibration = STUDY / 'full-scale.json'
    record = read(calibration)
    assert record['stage1_sha256'] == sha(STUDY / 'calibration.json')
    measured = record['summary']['maximum_ns']
    value = int(-(-measured // 1000) * 1000)
    verification = read(STUDY / f'verification-{value}.json')['summary']
    assert verification['full_quota'] == verification['searches'] > 0, 'Every verification search must complete its quota'
    assert 1_000_000 <= value < PREVIOUS_NS and value >= measured, value
    return value, sha(calibration), sha(STUDY / f'verification-{value}.json'), measured


def build():
    assert not OUT.exists(), 'Preserve existing build evidence'
    receipt = ROOT / 'artifacts/patch-installation.json'
    installed = read(receipt)
    receipt_hash = sha(receipt)
    assert installed['version'] == BASE_VERSION, installed['version']
    base = read(BASE / 'build-manifest.json')
    base_native = pins(ROOT / 'BrassNative')
    assert base_native == base['nativeFiles'], 'The installed packages differ from release 1.4.61'
    for file in installed['files']:
        assert sha(ROOT / MANAGED / file['name']) == file['sha256'].lower(), file['name']
    value, calibration_hash, verification_hash, measured = estimate()
    source = f'engine-speed 1.4.61 full-quota calibration maximum; {calibration_hash}'
    shutil.copytree(ROOT / 'BrassNative', OUT / 'BrassNative')
    changed = set()
    for package in PACKAGES:
        directory = OUT / 'BrassNative' / package
        for name in POLICIES:
            policy = read(directory / name)
            assert policy['startup_ns'] == PREVIOUS_NS and policy['method'] == 'search_v1', (package, name)
            before = {k: v for k, v in policy.items() if k not in ('startup_ns', 'startup_source')}
            policy.update(startup_ns=value, startup_source=source)
            write(directory / name, policy)
            assert {k: v for k, v in read(directory / name).items() if k not in ('startup_ns', 'startup_source')} == before
        listed = read(directory / 'build.json')
        for pin in listed['files']:
            if pin['path'] in POLICIES:
                pin['sha256'] = sha(directory / pin['path'])
            else:
                assert pin['sha256'] == sha(directory / pin['path']), pin['path']
        write(directory / 'build.json', listed)
        changed.update(f'{package}/{name}' for name in CHANGED)
    native = pins(OUT / 'BrassNative')
    assert {p['path'] for p in native} == {p['path'] for p in base_native}
    assert {p['path'] for p in native if p not in base_native} == changed
    assert sha(receipt) == receipt_hash and pins(ROOT / 'BrassNative') == base_native
    write(OUT / 'build-manifest.json', dict(version=VERSION, baseVersion=BASE_VERSION, stockSha256=installed['stockSha256'],
        files=installed['files'], baseNative=base_native, nativeFiles=native, installedBaselineReceiptSha256=receipt_hash,
        nativeSourceSha256=base['nativeSourceSha256'], previousStartupNs=PREVIOUS_NS, startupNs=value, measuredMaximumNs=measured,
        startupSource=source, calibrationSha256=calibration_hash, verificationSha256=verification_hash,
        sourcePins=[dict(path=p, sha256=sha(ROOT / p)) for p in (
            'scripts/local-game/search-calibration/calibrate-search-v1.py',
            'scripts/local-game/search-calibration/build-search-calibration.py')]))
    print('Built', OUT, 'startup_ns', PREVIOUS_NS, '->', value, flush=True)


if __name__ == '__main__':
    build()
