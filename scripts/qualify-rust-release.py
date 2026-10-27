"""L15 bounded, isolated acceptance work. Never installs or admits lab training.

Each invocation runs one small group. Evidence is append-only; use a fresh label
after a failed attempt. The seal distinguishes implemented smoke coverage from
the still-open A11 release gates.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import platform
from pathlib import Path
import runpy
import shutil
import subprocess
import struct
import sys
import time
import zipfile
import winreg

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'artifacts/rust-backend/20261026-l15'
SUPPORT = runpy.run_path(str(ROOT / 'scripts/qualify-rust-scaffold.py'))
PYTHON = sys.executable


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def pin(path):
    path = Path(path).resolve()
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))


def check_pin(entry):
    path = (ROOT / entry['path']).resolve()
    assert path.is_relative_to(ROOT) and sha(path) == entry['sha256'], entry['path']


def installed_pins():
    record = ROOT / 'artifacts/patch-installation.json'
    paths = [record]
    for entry in read(record)['files']:
        path = ROOT / 'BrassBirminghamUnity_Data/Managed' / entry['name']
        assert sha(path) == entry['sha256'].lower()
        paths.append(path)
    paths += [p for p in (ROOT / 'BrassNative').rglob('*') if p.is_file()]
    return [pin(p) for p in paths]


def run(out, label, command, env=None, cwd=ROOT, timeout=90):
    directory = out / label
    directory.mkdir(parents=True, exist_ok=False)
    print(label, flush=True)
    started = time.monotonic()
    with (directory / 'run.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen([str(v) for v in command], cwd=cwd, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        timed_out = False
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, timeout=15)
            process.wait(timeout=15)
            code = process.returncode
    write(directory / 'command.json', dict(command=[str(v) for v in command], cwd=str(cwd),
          timeout_seconds=timeout, elapsed_seconds=time.monotonic()-started,
          exit_code=code, timed_out=timed_out, passed=code == 0 and not timed_out))
    if code or timed_out:
        raise RuntimeError(f'{label} failed; retained {directory / "run.log"}')


def freeze(out):
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'protocol.json', dict(format='brass-l15-smoke-protocol/1.0.0',
          created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
          provider='CPUExecutionProvider', runtime='1.22.1', rust_binding='2.0.0-rc.10',
          sessions=2, threads_per_session=1, matched_allowance_ms=4000,
          allowances_ms=[4000, 8000, 10000, 12000],
          complete_game_smoke=dict(nn=['nn-human', 'nn-ai'], v4=['v4-human', 'v4-two', 'v4-three'],
              nn_remaining_ms=450, v4_decisions='scripted Pass settlement control',
              distinction='NN complete games use real live bridge and old-save remaining-clock adapter. Not full-allowance strength matches.'),
          maximum_test_group_seconds=90, long_sessions=False,
          release_gate='All A11 evidence required. Smoke pass alone never admits Part 2.',
          save_policy='Initial creation, then once after human End; zero AI saves.',
          fallback='Stop NN table; no single-seat V4 substitution. Exact installation rollback is tested only in sandbox.'))
    write(out / 'freeze.json', dict(installed=installed_pins(),
          native_source_before=SUPPORT['native_fingerprint'](),
          baseline=pin(ROOT / 'artifacts/rust-backend/20261026-l14-candidate-v5/build-manifest.json'),
          l14=pin(ROOT / 'artifacts/rust-backend/20261026-l14-smoke/final-checks.json'),
          provider_decision=pin(ROOT / 'ai-research/RUST_BACKEND_L14_5_REPORT.md')))


def environment():
    env, cargo = SUPPORT['tool_environment']()
    env['CARGO_INCREMENTAL'] = '0'
    env['CARGO_BUILD_JOBS'] = '2'
    env['CARGO_TARGET_DIR'] = str(ROOT / 'artifacts/rust-native-target')
    env.pop('PROPTEST_RNG_SEED', None)
    return env, cargo


def build(out, label):
    env, cargo = environment()
    run(out, label+'-fmt', [cargo, 'fmt', '--all', '--', '--check'], env, ROOT/'native')
    run(out, label+'-clippy', [cargo, 'clippy', '--offline', '-p', 'brass-ai-worker', '--all-targets',
        '--target', 'x86_64-pc-windows-gnu', '--', '-D', 'warnings'], env, ROOT/'native', 300)
    run(out, label+'-compile', [cargo, 'build', '--offline', '--release', '--bin', 'brass-ai-local',
        '--target', 'x86_64-pc-windows-gnu'], env, ROOT/'native', 300)
    run(out, label+'-package', [PYTHON, ROOT/'scripts/build-rust-local-package.py', '--out', out/'candidate',
        '--binary', ROOT/'artifacts/rust-native-target/x86_64-pc-windows-gnu/release/brass-ai-local.exe'], env, timeout=90)
    package = out/'candidate/BrassNative/alpha-g890'
    worker = runpy.run_path(str(ROOT/'tests/RustBackendLocal/test_local.py'))['Worker'](package)
    try:
        hello = worker.hello['data']
        assert hello['build']['source_sha256'] == SUPPORT['native_fingerprint']()
        assert hello['provider'] == 'CPUExecutionProvider'
        write(out/'readiness.json', hello)
    finally:
        worker.close()


def powershell():
    value = shutil.which('pwsh') or shutil.which('powershell')
    assert value
    return value


def unity(out, label, case):
    args = [powershell(), '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        ROOT/'tests/RustBackendLocal/run-unity.ps1', '-BuildDirectory', (out/'candidate').relative_to(ROOT),
        '-EvidenceDirectory', (out/(label+'-evidence')).relative_to(ROOT), '-MaxSeconds', '90']
    args += ['-Quick'] if case == 'lifecycle' else ['-ReleaseCase', case]
    run(out, label, args, timeout=110)


def native(out, label):
    env, cargo = environment()
    package = out/'candidate/BrassNative/alpha-g890'
    env['BRASS_LOCAL_PACKAGE'] = str(package)
    env['BRASS_ONNX_MANIFEST'] = str(package/'model/manifest.json')
    env['BRASS_ONNX_GOLDEN'] = str(ROOT/'artifacts/rust-backend/20261025-l09-qualified-v2/golden.json')
    run(out, label+'-process', [PYTHON, ROOT/'tests/RustBackendLocal/test_local.py', '-v'], env)
    run(out, label+'-golden', [cargo, 'test', '--offline', '--release', '-p', 'brass-ai-inference', '--test', 'runtime',
        '--target', 'x86_64-pc-windows-gnu', '--', '--ignored', '--nocapture'], env, ROOT/'native')
    run(out, label+'-worker', [cargo, 'test', '--offline', '--release', '-p', 'brass-ai-worker', '--lib',
        '--target', 'x86_64-pc-windows-gnu'], env, ROOT/'native')


def contracts(out, label):
    run(out, label+'-preservation', [powershell(), '-NoProfile', '-File', ROOT/'tests/RustBackendLocal/check-preservation.ps1',
        '-BuildDirectory', (out/'candidate').relative_to(ROOT)])
    run(out, label+'-checkpoint', [powershell(), '-NoProfile', '-File', ROOT/'tests/RustBackendLocal/check-l14.ps1',
        '-BuildDirectory', (out/'candidate').relative_to(ROOT), '-StudyDirectory',
        'artifacts/rust-backend/20261026-l14-study-v2', '-Label', 'l15-'+label])


def seal(out):
    assert not (out/'acceptance.json').exists(), 'Sealed evidence is immutable'
    frozen = read(out/'freeze.json')
    for entry in frozen['installed']:
        check_pin(entry)
    build_manifest = read(out/'candidate/build-manifest.json')
    assert build_manifest['native_source_sha256'] == SUPPORT['native_fingerprint']()
    for entry in build_manifest['sourcePins']:
        assert sha(out/'candidate/runtime'/entry['path']) == entry['sha256']
        if entry['path'] in build_manifest['sourceScope']:
            assert sha(ROOT/entry['path']) == entry['sha256']
    for entry in build_manifest['files']:
        assert sha(out/'candidate/Managed'/entry['name']) == entry['sha256']
    package = out/'candidate/BrassNative/alpha-g890'
    for entry in read(package/'build.json')['files']:
        assert sha(package/entry['path']) == entry['sha256']
    assert read(out/'readiness.json')['provider'] == 'CPUExecutionProvider'
    assert read(package/'model/manifest.json')['runtime_version'] == '1.22.1'
    assert read(package/'model/manifest.json')['binding_version'] == '2.0.0-rc.10'
    baseline_build = ROOT/'artifacts/rust-backend/20261026-l14-candidate-v5'
    baseline_archive = baseline_build/'BrassNative/alpha-g890/native-source.zip'
    with zipfile.ZipFile(baseline_archive) as archive:
        source_hash = hashlib.sha256()
        entries = [n for n in archive.namelist() if Path(n).suffix in ('.rs','.toml','.lock','.json')]
        for name in sorted(entries,key=lambda n:tuple(n.split('/'))):
            relative = name.removeprefix('native/').encode()
            data = archive.read(name)
            source_hash.update(struct.pack('<Q',len(relative))+relative+struct.pack('<Q',len(data))+data)
        assert source_hash.hexdigest() == read(baseline_build/'build-manifest.json')['native_source_sha256']
        unchanged = [n for n in archive.namelist() if n.startswith(('native/brass-ai-core/','native/brass-ai-inference/src/','native/vendor/'))]
        assert all((ROOT/name).read_bytes() == archive.read(name) for name in unchanged)
    write(out/'core-preservation.json',dict(passed=True,files=len(unchanged),baseline_archive=pin(baseline_archive),
        scope='Rules, search mathematics, features, CPU inference implementation and upstream vendor files unchanged from L14 v5.'))
    cases = {}
    for path in sorted(out.glob('*-evidence/unity.release.json')):
        result = read(path)
        if result.get('passed') and read(path.parent/'unity.json')['passed'] and (path.parent/'test-build.json').exists():
            tested = read(path.parent/'test-build.json')
            assert tested['candidateManifestSha256'] == sha(out/'candidate/build-manifest.json')
            assert tested['testDllSha256'] == sha(path.parent/'LocalAiUnityTests.dll')
            for entry in tested['sources']:
                assert sha(path.parent/'test-source'/entry['path']) == entry['sha256']
            cases[result['mode']] = pin(path)
    assert set(cases) == {'nn-human', 'nn-ai', 'v4-human', 'v4-two', 'v4-three', 'failures'}, cases
    commands = [p for p in out.glob('*/command.json') if read(p)['passed']]
    required = ['-fmt', '-clippy', '-compile', '-package', '-process', '-golden', '-worker', '-preservation', '-checkpoint']
    assert all(any(p.parent.name.endswith(suffix) for p in commands) for suffix in required)
    lifecycle = [p for p in out.glob('*-evidence/unity.json') if read(p).get('passed') and (p.parent/'unity.native-turns.json').exists()]
    assert lifecycle, 'A current full lifecycle smoke is required'
    assert all(not read(p).get('nativeErrors') for p in lifecycle)
    contract = ROOT/'artifacts/rust-backend/20261026-l14-study-v2/l15-contracts-01'
    contract_results = [contract/'validation-events.json', contract/'checkpoint-tests.json']
    assert all(read(p)['passed'] for p in contract_results)
    for path in contract_results:
        shutil.copyfile(path, out/path.name)
    deployment = out/'deployment/receipt.json'
    assert read(deployment)['passed']
    source_paths = [ROOT/'scripts/qualify-rust-release.py', ROOT/'scripts/test-rust-release-deployment.ps1',
        ROOT/'scripts/replay-native-failure.py', ROOT/'tests/RustBackendLocal/NativeReleaseUnityTests.cs',
        ROOT/'tests/RustBackendLocal/run-unity.ps1', ROOT/'tests/BrassLocalAI/LocalAiUnityTests.cs',
        ROOT/'tests/RustBackendLocal/test_local.py', ROOT/'scripts/build-rust-local-package.py']
    source_paths = sorted(set(source_paths + [p for p in (ROOT/'tests/RustBackendLocal').iterdir()
        if p.suffix in ('.cs','.py','.ps1')]))
    with zipfile.ZipFile(out/'release-sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
        for path in (out/'candidate/runtime').rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(out/'candidate/runtime').as_posix())
    with zipfile.ZipFile(out/'native-candidate-windows-x64.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in (out/'candidate/BrassNative').rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(out/'candidate').as_posix())
    open_gates = [
        'L14 actual Unity managed allocation totals unavailable; failed profiler self-test retained.',
        'L14 statistical strength and population latency tails deferred under short-test instruction.',
        'L15 full-allowance complete NN and V4 search matches deferred; 450 ms remaining-clock NN games and scripted V4 controls are mechanical smoke only.',
        'Support evidence covers this Windows x64 machine; no broader CPU or lab qualification.']
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        processor = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    prior = ['20261025-l04-qualified','20261025-l05-qualified-v4','20261025-l06-qualified-v3',
        '20261025-l07-qualified-v4','20261025-l08-qualified-v3','20261025-l09-qualified-v2',
        '20261025-l10-qualified','20261025-l11-qualified-v2','20261025-l12-qualified-v2','20261026-l13-qualified']
    manifest = dict(format='brass-local-l15-acceptance/1.0.0', smoke_passed=True,
        local_release_qualified=False, part2_admitted=False, installed=False,
        provider=dict(requested='CPUExecutionProvider', effective='CPUExecutionProvider', runtime='1.22.1', rust_binding='2.0.0-rc.10',
            override='CPU is the sole admitted provider; no OpenVINO code/dependencies in this package.',
            fallback='Unavailable/incompatible CPU runtime stops NN play; whole-table installation rollback is separate.'),
        support_matrix=dict(os='Windows x64', target='x86_64-pc-windows-gnu', nn='Standard four-seat UI tables require a human; all-AI harness is diagnostic only',
            v4='Existing two/three/four-seat settings; scripted complete-game controls plus preserved release qualification',
            sessions=2, compute_threads_each=1, timings_ms=[4000,8000,10000,12000]),
        measured_host=dict(processor=processor, os=platform.platform()),
        candidate=pin(out/'candidate/build-manifest.json'), package=pin(package/'build.json'), readiness=pin(out/'readiness.json'),
        core_preservation=pin(out/'core-preservation.json'),
        model=pin(package/'model/manifest.json'), protocol=pin(out/'protocol.json'),
        complete_games=cases, lifecycle=[pin(p) for p in lifecycle], deployment=pin(deployment),
        source_pins=[pin(p) for p in source_paths], source_archive=pin(out/'release-sources.zip'),
        native_archive=pin(out/'native-candidate-windows-x64.zip'),
        inherited_evidence=[frozen['l14'], frozen['provider_decision']], open_gates=open_gates,
        historical_parity=[pin(ROOT/'artifacts/rust-backend'/name/'qualification.json') for name in prior],
        issue_disposition=[
            dict(id='L15-001',status='fixed',issue='Root preparation incorrectly canceled inside delivery reserve',
                regression='pair_harness::tests::delivery_reserve_can_prepare_bounded_completion_without_search_or_nn_work; LocalTests.test_delivery_reserve_completes_without_search_at_all_difficulties',
                before_replay=pin(out/'reserve-replay-before.json')),
            dict(id='L15-H01',status='harness_fixed',issue='Complete-game driver must await asynchronous liquidation'),
            dict(id='L15-H02',status='harness_scoped',issue='All-AI research fixture has no human terminal reviewer; supported UI still requires one human'),
            dict(id='L15-H03',status='harness_fixed',issue='Mono can expose a null exception stack during filtering; use exact message and terminal-state guard')],
        failures=[pin(p) for p in out.glob('*/command.json') if not read(p)['passed']])
    manifest['files'] = [pin(p) for p in sorted(out.rglob('*')) if p.is_file()]
    write(out/'acceptance.json', manifest)
    audit(out)


def audit(out):
    record = read(out/'acceptance.json')
    assert record['smoke_passed'] and not record['local_release_qualified'] and not record['part2_admitted']
    for entry in record['files'] + record['source_pins']:
        check_pin(entry)
    write(out/'final-verification.json', dict(passed=True, acceptance=pin(out/'acceptance.json'),
        files_verified=len(record['files']), local_release_qualified=False, part2_admitted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze','build','unity','native','contracts','seal','audit'])
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--label', default='run-01')
    parser.add_argument('--case', choices=['nn-human','nn-ai','v4-human','v4-two','v4-three','failures','lifecycle'])
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(ROOT/'artifacts/rust-backend')
    assert args.label and all(c.isalnum() or c in '-_' for c in args.label)
    if args.mode not in ('freeze','audit'):
        assert (out/'freeze.json').exists() and not (out/'acceptance.json').exists()
    if args.mode == 'unity':
        assert args.case
        unity(out,args.label,args.case)
    elif args.mode in ('build','native','contracts'):
        globals()[args.mode](out,args.label)
    else:
        globals()[args.mode](out)
