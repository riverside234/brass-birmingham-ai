"""Pinned, offline build/freeze helpers for the isolated search-v2 study."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = runpy.run_path(str(ROOT / 'scripts/qualify-rust-scaffold.py'))
sha, read, write, receipt = (BASE[k] for k in ('sha', 'read', 'write', 'receipt'))
OUT = ROOT / 'artifacts/search-v2/20261028'


def freeze():
    target = OUT / 'control'
    target.mkdir(parents=True, exist_ok=False)
    sources = []
    with zipfile.ZipFile(target / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for folder in ('native', 'src/BrassLocalAI', 'tests/RustBackend', 'tests/GumbelISMCTS', 'scripts'):
            for p in sorted((ROOT / folder).rglob('*')):
                if p.is_file() and p.suffix in ('.rs', '.toml', '.lock', '.json', '.cs', '.py', '.ps1') and '__pycache__' not in p.parts:
                    archive.write(p, p.relative_to(ROOT)); sources.append(receipt(p))
    package = ROOT / 'BrassNative/alpha-g890'
    shutil.copytree(package, target / 'BrassNative/alpha-g890')
    shutil.copyfile(ROOT / 'artifacts/patch-installation.json', target / 'installation.json')
    managed = [receipt(ROOT / 'BrassBirminghamUnity_Data/Managed' / name) for name in
               ('Assembly-CSharp.dll', 'BrassPatch.dll', 'BrassLegacyAI.dll')]
    for name in ('gumbel-pair.exe', 'gumbel-session.exe', 'gumbel-trace.exe'):
        p = ROOT / 'artifacts/rust-native-target/x86_64-pc-windows-gnu/release' / name
        if p.exists(): shutil.copyfile(p, target / name)
    write(target / 'manifest.json', dict(version='search-v2-control/1',
          git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          native_source_sha256=BASE['native_fingerprint'](), sources=sources, managed=managed,
          package=[receipt(p) for p in sorted(package.rglob('*')) if p.is_file()],
          copied=[receipt(p) for p in sorted(target.rglob('*')) if p.is_file()]))
    print(target)


def cargo(args, label):
    env, binary = BASE['tool_environment']()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / (label + '.log')).open('wb') as log:
        result = subprocess.run([str(binary), *args, '--offline', '--locked'], cwd=ROOT / 'native',
                                env=env, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    print((OUT / (label + '.log')).read_text()[-10000:])
    if result.returncode: raise SystemExit(result.returncode)


def build_control():
    control = OUT / 'control'; checkout = control / 'checkout'; checkout.mkdir(exist_ok=False)
    with zipfile.ZipFile(control / 'source.zip') as z:
        for name in z.namelist():
            if name.startswith(('native/', 'tests/GumbelISMCTS/')):
                destination = (checkout / name).resolve()
                assert destination.is_relative_to(checkout.resolve())
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(z.read(name))
    env, binary = BASE['tool_environment']()
    env['CARGO_TARGET_DIR'] = str(ROOT / 'artifacts/rust-native-target')
    with (OUT / 'build-frozen-control.log').open('wb') as log:
        subprocess.run([str(binary), 'build', '--release', '--bin', 'gumbel-pair', '--offline', '--locked'],
            cwd=checkout/'native',env=env,stdout=log,stderr=subprocess.STDOUT,check=True,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    target=control/'rebuilt-gumbel-pair.exe'
    shutil.copyfile(ROOT/'artifacts/rust-native-target/x86_64-pc-windows-gnu/release/gumbel-pair.exe',target)
    write(control/'rebuilt-control.json',dict(binary=receipt(target),source_archive=receipt(control/'source.zip'),
        control=receipt(control/'manifest.json'),toolchain=BASE['TOOLCHAIN'],profile='release/x86-64/locked/offline'))
    print(target)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['freeze', 'build', 'test', 'build-control'])
    p.add_argument('--label', default='build'); a = p.parse_args()
    if a.action == 'freeze': freeze()
    elif a.action == 'build-control': build_control()
    elif a.action == 'build': cargo(['build', '--release', '-p', 'brass-ai-worker', '--bin', 'gumbel-pair', '--bin', 'gumbel-session', '--bin', 'brass-ai-local'], a.label)
    else: cargo(['test', '-p', 'brass-ai-core', '-p', 'brass-ai-worker'], a.label)
