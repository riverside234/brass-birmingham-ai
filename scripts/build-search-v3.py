#!/usr/bin/env python3
"""Rebuild the three local NN packages with search_v3 rolling halving quotas.

Each package keeps its own frozen source archive and model. Only the audited
search_v3 patches are applied. Nothing under BrassNative is modified here.
"""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
OUT = ROOT/'artifacts/search-v3'
PACKAGES = dict([('alpha-g890', None), ('beta-g1290', 'local-beta'), ('il-v11-2', 'local-il')])
POLICIES = ('policy.json', 'policy-k8.json', 'policy-k32.json', 'policy-t15.json', 'policy-k8-t15.json', 'policy-k32-t15.json')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def extract(archive, checkout):
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            if not (checkout/name).resolve().is_relative_to(checkout.resolve()):
                raise ValueError('Unsafe source archive member: '+name)
        source.extractall(checkout)


def cargo(support, arguments, cwd, log, env):
    _, binary = support['tool_environment']()
    with Path(log).open('wb') as stream:
        result = subprocess.run([str(binary), *arguments, '--offline', '--locked'], cwd=cwd, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise SystemExit(Path(log).read_text(errors='replace')[-4000:])


def build_package(support, name, feature, shared_target, *, tests):
    import search_v3_patch
    base = OUT/name
    if base.exists():
        raise ValueError('Preserve build evidence: '+str(base))
    checkout = base/'n'
    checkout.mkdir(parents=True)
    installed = ROOT/'BrassNative'/name
    extract(installed/'native-source.zip', checkout)
    search_v3_patch.apply(checkout/'native')
    env, _ = support['tool_environment']()
    env['CARGO_TARGET_DIR'] = str(shared_target)
    env['CARGO_INCREMENTAL'] = '0'
    features = ['--features', feature] if feature else []
    cargo(support, ['build', '--release', '--bin', 'brass-ai-local', *features], checkout/'native', base/'build.log', env)
    records = []
    if tests:
        # Golden count-mode traces must still pass; their fixtures live outside the archive.
        shutil.copytree(ROOT/'tests/GumbelISMCTS/fixtures', checkout/'tests/GumbelISMCTS/fixtures')
        cargo(support, ['test', '--release', '-p', 'brass-ai-core'], checkout/'native', base/'core-tests.log', env)
        cargo(support, ['test', '--release', '-p', 'brass-ai-worker', '--lib', *features], checkout/'native', base/'worker-tests.log', env)
        records = [support['sha'](base/'core-tests.log'), support['sha'](base/'worker-tests.log')]
    binary = Path(shared_target)/'x86_64-pc-windows-gnu/release/brass-ai-local.exe'
    package = base/'package'
    shutil.copytree(installed, package)
    shutil.copyfile(binary, package/'brass-ai-local.exe')
    (package/'native-source.zip').unlink()
    with zipfile.ZipFile(package/'native-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((checkout/'native').rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(checkout).as_posix())
    for policy_name in POLICIES:
        policy = read(package/policy_name)
        if policy['method'] != 'search_v1':
            raise ValueError('Expected search_v1 policy: '+policy_name)
        policy['method'] = 'search_v3'
        write(package/policy_name, policy)
    g = support['native_fingerprint'].__globals__
    g['NATIVE'] = checkout/'native'
    source = support['native_fingerprint']()
    build = read(package/'build.json')
    build['source_sha256'] = source
    for pin in build['files']:
        pin['sha256'] = support['sha'](package/pin['path'])
    write(package/'build.json', build)
    return dict(package=name, feature=feature, source_sha256=source, binary_sha256=support['sha'](package/'brass-ai-local.exe'),
                source_archive_sha256=support['sha'](package/'native-source.zip'), tests=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', nargs='+', choices=tuple(PACKAGES), default=list(PACKAGES))
    parser.add_argument('--skip-tests', action='store_true')
    args = parser.parse_args()
    support = runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'))
    OUT.mkdir(parents=True, exist_ok=True)
    results = [build_package(support, name, PACKAGES[name], OUT/'target'/name, tests=not args.skip_tests) for name in args.packages]
    write(OUT/'native-build.json', dict(passed=True, packages=results, method='search_v3', toolchain='1.99.1',
                                        profile='release/locked/offline/x86-64'))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
