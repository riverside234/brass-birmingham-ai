"""Reuse the qualified C# candidate and native builder for release 1.4.44."""
from pathlib import Path
import argparse
import importlib.util
import runpy
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
B = runpy.run_path(str(ROOT / 'scripts/build-host-delivery.py'))
STUDY, read, write, sha, pins = (B[k] for k in ('STUDY','read','write','sha','pins'))
HOST = STUDY / 'candidate'
NATIVE = STUDY / 'reserve08-native'
OUT = STUDY / 'release-1.4.44'


def native():
    assert not NATIVE.exists(), 'Preserve native build evidence'
    spec = importlib.util.spec_from_file_location('scoped_native_builder', ROOT / 'scripts/build-nn-k32-native.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUT = NATIVE
    module.main()


def package():
    assert not OUT.exists(), 'Preserve release inputs'
    q = read(HOST / 'qualification.json')
    assert q['passed'] and q['native_unchanged']
    for pin in q['files']:
        assert sha(HOST / pin['path']) == pin['sha256']
    n = read(NATIVE / 'native-build.json')
    assert n['passed'] and sha(NATIVE / 'brass-ai-local.exe') == n['binary_sha256']
    m = read(HOST / 'build-manifest.json')
    assert sha(ROOT / 'artifacts/patch-installation.json') == m['installed_baseline_receipt_sha256']
    with zipfile.ZipFile(HOST / 'BrassNative/alpha-g890/native-source.zip') as old, zipfile.ZipFile(NATIVE / 'native-source.zip') as new:
        a = {p:old.read(p) for p in old.namelist() if not p.endswith('/')}
        b = {p:new.read(p) for p in new.namelist() if not p.endswith('/')}
    assert set(a) == set(b)
    assert {p for p in a if a[p] != b[p]} == set(n['scope'])
    assert b'const K32_HOST_RESERVE_NS: u64 = 800_000_000;' in b['native/brass-ai-worker/src/local.rs']
    for name in ('runtime','Managed','BrassNative'):
        shutil.copytree(HOST / name, OUT / name)
    (OUT / 'legacy.rsp').write_text((HOST / 'legacy.rsp').read_text(encoding='utf-8-sig').replace(str(HOST),str(OUT)),encoding='utf-8-sig')
    package = OUT / 'BrassNative/alpha-g890'
    for name in ('brass-ai-local.exe','native-source.zip'):
        shutil.copyfile(NATIVE / name, package / name)
    build = read(package / 'build.json')
    for pin in build['files']:
        if pin['path'] == 'brass-ai-local.exe':
            pin['sha256'] = sha(package / pin['path'])
        else:
            assert sha(package / pin['path']) == pin['sha256']
    build['source_sha256'] = n['source_sha256']
    write(package / 'build.json', build)
    m.update(version='1.4.44-rust-local', nativeFiles=pins(OUT / 'BrassNative'),
        native_build_sha256=sha(package / 'build.json'),native_source_sha256=n['source_sha256'],
        native_source_scope=n['scope'],k32_host_reserve_ns=800000000,
        host_qualification_sha256=sha(HOST / 'qualification.json'),native_build_receipt_sha256=sha(NATIVE / 'native-build.json'))
    write(OUT / 'build-manifest.json',m)
    print(OUT,flush=True)


def qualify():
    assert not (OUT / 'qualification.json').exists()
    m = read(OUT / 'build-manifest.json')
    assert m['host_qualification_sha256'] == sha(HOST / 'qualification.json')
    assert m['native_build_receipt_sha256'] == sha(NATIVE / 'native-build.json')
    turns = {}
    for case in ('canal-k32','rail-k32','rail-branching-k32','canal-k16'):
        d = STUDY / ('reserve08-' + case)
        r = read(d / 'result.json')
        receipt = read(d / 'receipt.json')
        assert r['passed'] and receipt['passed'] and receipt['build_manifest_sha256'] == sha(OUT / 'build-manifest.json')
        assert r['search_method'] == 'search_v1' and not r['focused'] and r['allowance_ms'] == 4000
        assert r['search_k'] == (16 if case.endswith('k16') else 32)
        assert len(r['commits']) == 2 and r['commits'][-1]['remaining'] == 0 and r['delivery_ms'] < 4000
        assert r['readiness']['k32_host_delivery_reserve_ns'] == 800000000
        assert r['replies'] and all(p['sessions'] == 2 and p['threads_per_session'] == 1 and p['response_ready_ns'] < p['deadline_ns'] for p in r['replies'])
        assert all(p['requested_k'] == r['search_k'] for p in r['replies'])
        if r['search_k'] == 32 and case != 'rail-k32':
            assert any(owner['effective_k'] == 32 for p in r['replies'] for owner in p['replay']['owners'])
        turns[case] = dict(k=r['search_k'],delivery_ms=r['delivery_ms'],checks=receipt['checks'],result_sha256=sha(d / 'result.json'))
    old = read(HOST / 'build-manifest.json')
    assert old['files'] == m['files'] and old['sourcePins'] == m['sourcePins']
    previous = {p['path']:p['sha256'] for p in old['nativeFiles']}
    assert {p['path'] for p in m['nativeFiles'] if previous[p['path']] != p['sha256']} == {
        'alpha-g890/build.json','alpha-g890/brass-ai-local.exe','alpha-g890/native-source.zip'}
    for pin in m['files']:
        assert sha(OUT / 'Managed' / pin['name']) == pin['sha256']
    assert pins(OUT / 'BrassNative') == m['nativeFiles']
    write(OUT / 'qualification.json',dict(format='brass-host-delivery-release/1.0.0',passed=True,version=m['version'],
        scope='local-game-g890/host-delivery/reduced-K32-reserve',build_manifest_sha256=sha(OUT / 'build-manifest.json'),
        host_qualification_sha256=m['host_qualification_sha256'],native_build_receipt_sha256=m['native_build_receipt_sha256'],
        k32_minimum_delivery_reserve_ns=800000000,search_method='search_v1',v205_enabled=False,
        two_sessions_one_thread_each=True,shared_rust_core_unchanged=True,live_turns=turns,files=pins(OUT),
        smoke_receipts=[dict(path=str((STUDY/('reserve08-'+c)/'receipt.json').relative_to(ROOT)),sha256=sha(STUDY/('reserve08-'+c)/'receipt.json')) for c in turns]))
    print(turns,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('native','package','qualify'))
    globals()[parser.parse_args().stage]()
