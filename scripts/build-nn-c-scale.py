#!/usr/bin/env python3
"""Set the Gumbel c_scale (search constant c2) to 1.0 in every installed NN policy.

Builds a new candidate of the BrassNative packages over the exact installed
release, then qualifies it on real workers: each policy must load, pass its
warm-up search and report the same search settings as before. Only the policy
files and the build lists that pin them change. Managed code, models, ONNX
Runtime and the Rust workers are untouched.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'artifacts/nn-c-scale/20261109'
OUT = STUDY/'release-1.4.54'
BASE_VERSION = '1.4.53-rust-local'
VERSION = '1.4.54-rust-local'
PACKAGES = ('alpha-g890', 'beta-g1290', 'il-v11-2')
POLICIES = ('policy.json', 'policy-k8.json', 'policy-k32.json', 'policy-t15.json', 'policy-k8-t15.json', 'policy-k32-t15.json')
C_SCALE = 1.0
PROTOCOL = 'brass-local-native/1.0.0'
FORMAT = 'brass-nn-c-scale-release/1.0.0'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def pins(directory):
    return [dict(path=p.relative_to(directory).as_posix(), sha256=sha(p))
            for p in sorted(Path(directory).rglob('*')) if p.is_file()]


def set_c_scale(path):
    """Edit the one number, so every other byte of the policy stays as qualified."""
    text = path.read_text(encoding='utf-8')
    updated, count = re.subn(r'("c2":\s*)0\.1\b', lambda m: m.group(1)+'1.0', text)
    assert count == 1, f'{path}: expected one c2 of 0.1, found {count}'
    path.write_text(updated, encoding='utf-8')
    value = read(path)
    assert value['config']['c2'] == C_SCALE and value['config']['c1'] == 50.
    return value


def build():
    assert not OUT.exists(), 'Preserve the existing candidate'
    installed = read(ROOT/'artifacts/patch-installation.json')
    assert installed['version'] == BASE_VERSION, 'Install this over the exact release it was built from'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    base = pins(ROOT/'BrassNative')
    shutil.copytree(ROOT/'BrassNative', OUT/'BrassNative')
    for package in PACKAGES:
        directory = OUT/'BrassNative'/package
        for name in POLICIES:
            set_c_scale(directory/name)
        build_list = read(directory/'build.json')
        for entry in build_list['files']:
            if entry['path'] in POLICIES:
                entry['sha256'] = sha(directory/entry['path'])
            else:
                assert entry['sha256'] == sha(directory/entry['path']), entry['path']
        write(directory/'build.json', build_list)
    write(OUT/'build-manifest.json', dict(version=VERSION, baseVersion=BASE_VERSION, stockSha256=installed['stockSha256'],
        baseFiles=installed['files'], files=installed['files'], baseNative=base, nativeFiles=pins(OUT/'BrassNative'),
        changedNative=[f'{package}/{name}' for package in PACKAGES for name in (*POLICIES, 'build.json')],
        installed_baseline_receipt_sha256=sha(ROOT/'artifacts/patch-installation.json'),
        search_method='search_v1', v205_enabled=False, c1=50., c2=C_SCALE, nativeBuildSha256=installed.get('nativeBuildSha256')))
    print('Built', OUT, flush=True)


def frame(stream):
    header = stream.read(4)
    if len(header) != 4:
        raise RuntimeError('Worker closed its output')
    length = struct.unpack('<I', header)[0]
    assert 1 <= length <= 1024*1024, length
    return json.loads(stream.read(length))


def send(stream, value):
    data = json.dumps(value, separators=(',', ':')).encode()
    stream.write(struct.pack('<I', len(data))+data)
    stream.flush()


def start_worker(policy):
    executable = policy.parent/'brass-ai-local.exe'
    process = subprocess.Popen([str(executable), str(policy), sha(policy)], cwd=policy.parent,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    # A hung worker must fail this check, never block it forever.
    watchdog = threading.Timer(90, process.kill)
    watchdog.start()
    return process, watchdog


def stop_worker(process, watchdog):
    try:
        if process.poll() is None:
            send(process.stdin, dict(protocol=PROTOCOL, id=2, request=dict(op='shutdown')))
            stopped = frame(process.stdout)
            assert stopped['status'] == 'stopped', stopped
            process.stdin.close()
            assert process.wait(timeout=30) == 0
    finally:
        watchdog.cancel()
        if process.poll() is None:
            process.kill()
            process.wait()


def worker_readiness(policy):
    process, watchdog = start_worker(policy)
    try:
        hello = frame(process.stdout)
        assert hello['id'] == 0 and hello['status'] == 'ready' and hello['protocol'] == PROTOCOL, hello
        readiness = hello['data']
        send(process.stdin, dict(protocol=PROTOCOL, id=1, request=dict(op='health')))
        health = frame(process.stdout)
        assert health['status'] == 'health' and health['data']['busy'] is False, health
    finally:
        stop_worker(process, watchdog)
    assert readiness['measured_readiness']['warmup_passed'] is True, readiness
    return readiness


SETTINGS = ('backend', 'sessions', 'threads_per_session', 'provider', 'k_by_allowance_ms', 'policy_temperature',
            'delivery_grace_ms', 'model_id', 'export_sha256')


def qualify():
    assert not (OUT/'qualification.json').exists(), 'Preserve the existing qualification'
    manifest = read(OUT/'build-manifest.json')
    assert pins(OUT/'BrassNative') == manifest['nativeFiles'], 'Candidate changed since it was built'
    results = []
    for package in PACKAGES:
        for name in POLICIES:
            old_policy, new_policy = ROOT/'BrassNative'/package/name, OUT/'BrassNative'/package/name
            old_value, new_value = read(old_policy), read(new_policy)
            assert old_value['config']['c2'] == .1 and new_value['config']['c2'] == C_SCALE
            old_value['config']['c2'] = C_SCALE
            assert old_value == new_value, f'{package}/{name}: only c2 may differ'
            before, after = worker_readiness(old_policy), worker_readiness(new_policy)
            for key in SETTINGS:
                assert before.get(key) == after.get(key), f'{package}/{name}: {key} changed'
            results.append(dict(package=package, policy=name, sha256=sha(new_policy), warmup_passed=True,
                                warmup_nn_positions=after['measured_readiness']['native_nn_positions'], k_by_allowance_ms=after['k_by_allowance_ms'],
                                policy_temperature=after['policy_temperature']))
            print(f'{package}/{name}: ready, warm-up passed', flush=True)
    for package in PACKAGES:
        build_list = read(OUT/'BrassNative'/package/'build.json')
        for entry in build_list['files']:
            assert entry['sha256'] == sha(OUT/'BrassNative'/package/entry['path']), entry['path']
    write(OUT/'qualification.json', dict(format=FORMAT, passed=True, version=VERSION, baseVersion=BASE_VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(), c1=50., c2=C_SCALE, search_method='search_v1', v205_enabled=False,
        build_manifest_sha256=sha(OUT/'build-manifest.json'), files=pins(OUT/'BrassNative'), workers=results,
        scope='Every policy of every installed NN package loads on its real worker and passes the warm-up search; other search settings are unchanged'))
    print('Qualified', len(results), 'policies', flush=True)


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if command in ('build', 'all'):
        build()
    if command in ('qualify', 'all'):
        qualify()
