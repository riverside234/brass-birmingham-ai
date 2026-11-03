#!/usr/bin/env python3
"""B01 Linux/CUDA qualification. No training, source adoption or game deployment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from brass_ai.learning.contracts import read_json, require
from brass_ai.rust_backend.lab_contract import (B01_CONTRACT, SOURCE, admit, extract_new, native_fingerprint,
                                               pin, sha, verify_inputs)

# These reference fixtures live outside native/ in the installed source tree,
# so the immutable native-only source.zip does not carry them. Pin their LF
# bytes separately; never patch the archived Rust source or skip its tests.
TEST_FIXTURES = {
    'tests/GumbelISMCTS/fixtures/session-v1.json': '1f97002a7ac7fdb14006279640cc7536d9d00e62278922c81421d22b60473bfd',
    'tests/GumbelISMCTS/fixtures/coordinator-v1.json': 'f4b86853ce60570a90fd624f87bbaaa8d8b1b326578e2f427aac4894ca8ec96f',
    'tests/GumbelISMCTS/fixtures/reuse-v1.json': '237ec3ca50af2730978aee0201c82d915f90aa9fb2bdee182540d59e2d8e6f54',
}
TEST_PATHS = ('src', 'tests/R2', 'tests/Recovery', 'tests/FeatureEngineering', 'tests/RustBackendLab')


def python_environment(environ=None):
    """Give qualification subprocesses the complete, checkout-local fixture path."""
    env = dict(os.environ if environ is None else environ)
    env['PYTHONPATH'] = os.pathsep.join(str(ROOT / name) for name in TEST_PATHS)
    return env


def check_compile_inputs(native):
    """Resolve every literal include in the sealed Rust tree before compilation."""
    native = Path(native).resolve()
    checkout = native.parent
    includes = []
    pattern = re.compile(r'\binclude(?:_str|_bytes)?!\s*\(\s*"([^"\\]+)"\s*,?\s*\)')
    for source in sorted(native.rglob('*.rs')):
        if 'target' in source.relative_to(native).parts:
            continue
        for match in pattern.finditer(source.read_text(encoding='utf-8')):
            path = (source.parent / match[1]).resolve()
            require(path.is_relative_to(checkout), 'Compile input escapes detached checkout: ' + match[1])
            name = path.relative_to(checkout).as_posix()
            require(path.is_file(), 'Missing Rust compile input: ' + name)
            includes.append(dict(source=source.relative_to(checkout).as_posix(), input=name, sha256=sha(path)))
    require(includes, 'No Rust compile inputs found')
    return includes


def prepare_test_inputs(native, source_root=ROOT):
    """Hydrate the detached test tree without changing the pinned native core."""
    native = Path(native).resolve()
    source_root = Path(source_root).resolve()
    require(native != (source_root / 'native').resolve(), 'Test inputs require a detached checkout')
    require(native_fingerprint(native) == SOURCE, 'Test inputs require the sealed B01 native core')
    staged = {}
    # Validate the complete supplement before writing any fixture.
    for name, expected in TEST_FIXTURES.items():
        source = source_root / name
        require(source.is_file(), 'Missing B01 test fixture: ' + name)
        data = source.read_bytes().replace(b'\r\n', b'\n')
        require(hashlib.sha256(data).hexdigest() == expected, 'Changed B01 test fixture: ' + name)
        destination = (native.parent / name).resolve()
        require(destination.is_relative_to(native.parent) and not destination.is_relative_to(native),
                'Test fixture must remain outside the native core: ' + name)
        require(not destination.exists() or (destination.is_file() and destination.read_bytes() == data),
                'Conflicting detached test fixture: ' + name)
        staged[destination] = data
    for destination, data in staged.items():
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(data)
    includes = check_compile_inputs(native)
    require(native_fingerprint(native) == SOURCE, 'Native core changed while staging test inputs')
    return dict(contract='brass-lab-native-test-inputs/1.0.0', base_source=SOURCE,
                fixtures=dict(TEST_FIXTURES), includes=includes)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def command(args, *, cwd, env, log, timeout):
    """Bounded process group, including compiler children; retain complete logs."""
    began = time.monotonic()
    print('B01:', log.stem, flush=True)
    with log.open('wb') as stream:
        process = subprocess.Popen([str(x) for x in args], cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    require(code == 0, 'B01 command failed; see ' + str(log))
    return dict(argv=[str(x) for x in args], seconds=time.monotonic()-began, log=pin(log))


def python_checks(out, env):
    """Run the exact lab CPU gates before starting a native build."""
    checks = [command([sys.executable, '-m', 'unittest', 'discover', '-s',
                       ROOT / 'tests/RustBackendLab', '-p', 'test_b01*.py', '-v'],
                      cwd=ROOT, env=env, log=out / 'adapter-tests.log', timeout=60)]
    checks.append(command([sys.executable, '-m', 'unittest', '-v',
        'test_inference_deadline_race', 'test_inference_failure_review',
        'test_followup_inference_review', 'test_inference_performance'],
        cwd=ROOT / 'tests/Recovery', env=env, log=out / 'existing-inference-tests.log', timeout=90))
    return checks


def source_inventory():
    # The lab runs the current Python trainer/evaluator with the pinned native
    # archive. Identify that combination separately from the saved old producer.
    files = []
    for folder in ('src', 'configs', 'scripts', *TEST_PATHS[1:], 'training/runtime'):
        for p in sorted((ROOT / folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in (
                    '.py', '.json', '.cs', '.toml', '.dll', '.exe', '.so'):
                files.append(dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)))
    return files


def qualify(args, out):
    require(platform.system() == 'Linux', 'B01 qualification must run on the lab Linux host')
    require(Path(args.run).is_dir() and (Path(args.run) / 'state.json').is_file(), 'Select an existing committed run')
    run = Path(args.run).resolve()
    require(not out.is_relative_to(run) and not run.is_relative_to(out), 'Evidence must be separate from training run')
    require(args.bundle_sha256.lower() == sha(args.bundle), 'Transferred B01 bundle hash mismatch')
    profile = read_json(args.profile); admit(profile)
    write(out / 'profile.json', profile)
    sources = source_inventory(); write(out / 'python-runtime-sources.json', sources)
    extract_new(args.bundle, out / 'inputs')
    local = verify_inputs(out / 'inputs', profile); write(out / 'local-entry.json', local)
    extract_new(out / 'inputs/source.zip', out / 'checkout')
    checkout = out / 'checkout/native'
    require(native_fingerprint(checkout) == SOURCE, 'Extracted shared core differs from selected source')
    write(out / 'native-test-inputs.json', prepare_test_inputs(checkout))
    env = python_environment()
    env.update(CARGO_INCREMENTAL='0', PROPTEST_RNG_SEED='890', PROPTEST_CASES='16')
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # Override the archived Windows build target without editing hashed sources.
    for key in list(env):
        if key in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_TARGET', 'CARGO_TARGET_DIR') or (
                key.startswith('CARGO_TARGET_') and key.endswith('_RUSTFLAGS')):
            env.pop(key)
    version = subprocess.check_output(['rustc', '+1.99.1', '-vV'], env=env, text=True, timeout=15)
    target = next(line.split(': ', 1)[1] for line in version.splitlines() if line.startswith('host: '))
    require('linux' in target, 'Rust compiler host is not Linux')
    write(out / 'toolchain.json', dict(rustc=version, target=target,
          cargo=subprocess.check_output(['cargo', '+1.99.1', '--version'], env=env, text=True, timeout=15).strip(),
          compiler_environment={k: v for k, v in env.items() if k in ('CC', 'CXX', 'AR', 'RUSTC_WRAPPER')
                                or k.startswith('CARGO_TARGET_')}, platform_patch=None))
    import torch
    from brass_ai.native.protocol import NativeWorker
    from brass_ai.recovery.inference import validate_inference_devices
    from brass_ai.recovery.retention import run_lock
    from brass_ai.rust_backend.lab_snapshot import capture, verify_unchanged
    devices = validate_inference_devices(args.device)
    # Fixed fp32 parity settings, recorded as diagnostics, never a recipe edit.
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    commands = []
    with run_lock(run):
        snapshot, model, raw = capture(ROOT, run, devices=devices)
        snapshot['diagnostic_precision'] = dict(dtype='fp32', tf32=False, deterministic=True, cpu_threads=1)
        snapshot['extra_runtime_manifests'] = [pin(p) for p in args.runtime_manifest]
        write(out / 'lab-state.json', snapshot)
        try:
            commands.extend(python_checks(out, env))
            # Build once, with an explicit native Linux target and isolated target dir.
            common = ['--manifest-path', checkout / 'Cargo.toml', '--target', target,
                      '--target-dir', out / 'target', '--locked']
            cargo = ['cargo', '+1.99.1']
            build = cargo + ['build', '--release', '-p', 'brass-ai-worker', '--bin', 'brass-ai-worker',
                             '--bin', 'gumbel-pair', *common]
            if args.offline:
                build.append('--offline')
            commands.append(command(build, cwd=checkout, env=env, log=out / 'build.log', timeout=args.build_timeout))
            # Compile tests may need dev dependencies on the first lab build.
            tests = cargo + ['test', '--release', '-p', 'brass-ai-core', '--lib']
            for name in ('contracts', 'state', 'rules', 'privacy', 'simulation', 'features', 'search', 'sessions', 'cooperative'):
                tests += ['--test', name]
            tests += common
            if args.offline:
                tests += ['--offline']
            tests += ['--', '--test-threads=2', '--skip', 'both_one_and_neither_owner_hit_without_sharing_counts_or_root_credit',
                      '--skip', 'reused_owner_rejects_changed_value_sampler_information_and_cancellation']
            commands.append(command(tests, cwd=checkout, env=env, log=out / 'core-tests.log', timeout=args.build_timeout))
            tests = cargo + ['test', '--release', '-p', 'brass-ai-worker', '--lib', *common, '--offline', '--', '--test-threads=2']
            commands.append(command(tests, cwd=checkout, env=env, log=out / 'worker-tests.log', timeout=args.build_timeout))
            require(native_fingerprint(checkout) == SOURCE, 'Native source changed during build')
            binary = out / 'target' / target / 'release/brass-ai-worker'
            fixtures = read_json(out / 'inputs/fixtures.json')
            second_entry = next((p for p in snapshot['population'] if p['entry']['model_id'] != raw['model_id']), None)
            require(second_entry is not None, 'B01 mixed-model test needs a distinct committed population/anchor model')
            with NativeWorker(binary, timeout=10) as worker:
                hello = worker.hello
                require(hello['build']['source_sha256'] == SOURCE and hello['build']['target'] == target
                        and hello['build']['opt_level'] == '3'
                        and hello['build']['cargo_lock_sha256'] == sha(checkout / 'Cargo.lock'), 'Linux worker build identity differs')
                write(out / 'linux-build.json', dict(binary=pin(binary), handshake=hello,
                      search_binary=pin(out / 'target' / target / 'release/gumbel-pair'), commands=commands))
                for row in fixtures['transitions']:
                    worker.import_state(0, row['before']['state'])
                    actual = worker.apply_choice(0, row['before']['state']['context']['revision'], row['command']['choice'])
                    require(all(actual[k] == row['expected'][k] for k in ('state', 'choices', 'chance')),
                            'Portable staged transition mismatch: ' + row['root'])
            # CUDA and private reply lanes live in a killable diagnostic process.
            # A stuck device cannot hold the training lock indefinitely.
            commands.append(command([sys.executable, '-m', 'brass_ai.rust_backend.lab_smoke',
                '--worker', binary, '--fixtures', out / 'inputs/fixtures.json', '--state', out / 'lab-state.json',
                '--second', second_entry['file']['path'], '--profile', out / 'profile.json', '--out', out,
                *[arg for device in devices for arg in ('--device', device)]],
                cwd=ROOT, env=env, log=out / 'gpu-smoke.log', timeout=120 * len(devices)))
            gpu_reports = [pin(out / f'gpu-{index}.json') for index in range(len(devices))]
            for report in gpu_reports:
                require(read_json(report['path'])['passed'] is True, 'GPU smoke did not pass')
            write(out / 'commands.json', commands)
            require(source_inventory() == sources, 'Python/runtime sources changed during B01')
            for original in snapshot['extra_runtime_manifests']:
                require(pin(original['path']) == original, 'Runtime manifest changed during B01')
            verify_unchanged(snapshot)
            return dict(contract=B01_CONTRACT, passed=True, milestone='B01',
                        method=profile['method'], v205_enabled=False, completed_utc=datetime.now(timezone.utc).isoformat(),
                        profile=pin(out / 'profile.json'), local_entry=pin(out / 'local-entry.json'),
                        test_inputs=pin(out / 'native-test-inputs.json'),
                        lab_state=pin(out / 'lab-state.json'), python_sources=pin(out / 'python-runtime-sources.json'),
                        linux_build=pin(out / 'linux-build.json'), gpu=gpu_reports,
                        commands=pin(out / 'commands.json'),
                        fixture_roots=len(fixtures['roots']), staged_transitions=len(fixtures['transitions']),
                        training_started=False, lab_deployed=False, teacher_admitted=False,
                        next='B02 ordinary collection; B04 whole-search/teacher integration is not certified by B01')
        finally:
            verify_unchanged(snapshot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Existing committed training run, paused at a clean boundary')
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--bundle-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True, help='New directory under artifacts/rust-backend, outside the training run')
    parser.add_argument('--device', action='append', required=True, help='Explicit CUDA device, e.g. cuda:0; repeat for each device')
    parser.add_argument('--profile', type=Path, default=ROOT / 'configs/search/lab-search-v1.json')
    parser.add_argument('--runtime-manifest', type=Path, action='append', default=[])
    parser.add_argument('--offline', action='store_true', help='Require a populated pinned Cargo cache')
    parser.add_argument('--build-timeout', type=int, default=900, help='Compiler/test-command watchdog; no gameplay sessions')
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.is_relative_to(ROOT / 'artifacts/rust-backend') and out != ROOT / 'artifacts/rust-backend',
            'Evidence must be in a specific artifacts/rust-backend directory')
    require(0 < args.build_timeout <= 1800, 'Invalid compiler watchdog')
    out.mkdir(parents=True, exist_ok=False)
    try:
        result = qualify(args, out)
    except Exception as error:
        write(out / 'qualification.json', dict(contract=B01_CONTRACT, passed=False,
              milestone='B01', error=f'{type(error).__name__}: {error}', lab_deployed=False))
        raise
    write(out / 'qualification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
