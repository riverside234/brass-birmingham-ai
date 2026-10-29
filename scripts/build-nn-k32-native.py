"""Rebuild only the local native worker from its installed frozen source."""
from pathlib import Path
import json
import os
import runpy
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/nn-k32'
SCOPE=['native/brass-ai-worker/src/local.rs','native/brass-ai-worker/src/pair.rs']
SUPPORT=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    env,cargo=SUPPORT['tool_environment']()
    formatter=ROOT/'artifacts/native-toolchain/rustup/toolchains/1.99.1-x86_64-pc-windows-gnu/bin/rustfmt.exe'
    subprocess.run([str(formatter),'--edition','2021','--config','skip_children=true',
                    str(ROOT/SCOPE[0]),str(ROOT/SCOPE[1])],check=True,env=env)
    checkout=OUT/'n'
    assert not checkout.exists(),'Preserve the frozen build inputs'
    archive=ROOT/'BrassNative/alpha-g890/native-source.zip'
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            destination=(checkout/name).resolve()
            assert destination.is_relative_to(checkout.resolve())
        z.extractall(checkout)
    for name in SCOPE:shutil.copyfile(ROOT/name,checkout/name)
    env['CARGO_TARGET_DIR']=str(ROOT/'artifacts/nn-temperature/target')
    env['CARGO_INCREMENTAL']='0'
    records=[]
    for label,args in [
        ('build',['build','--release','--bin','brass-ai-local']),
        ('host-reserve-tests',['test','--release','-p','brass-ai-worker','--lib','host_reserve_tests']),
    ]:
        log=OUT/(label+'.log')
        print(label,flush=True)
        with log.open('wb') as f:
            result=subprocess.run([str(cargo),*args,'--offline','--locked'],cwd=checkout/'native',env=env,
                                  stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        print(log.read_text()[-4500:],flush=True)
        if result.returncode:raise SystemExit(result.returncode)
        records.append(dict(stage=label,passed=True,log_sha256=SUPPORT['sha'](log)))
    g=SUPPORT['native_fingerprint'].__globals__
    g['NATIVE']=checkout/'native'
    source_sha=SUPPORT['native_fingerprint']()
    binary=Path(env['CARGO_TARGET_DIR'])/'x86_64-pc-windows-gnu/release/brass-ai-local.exe'
    shutil.copyfile(binary,OUT/'brass-ai-local.exe')
    with zipfile.ZipFile(OUT/'native-source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(checkout.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(checkout).as_posix())
    SUPPORT['write'](OUT/'native-build.json',dict(passed=True,source_sha256=source_sha,
        binary_sha256=SUPPORT['sha'](OUT/'brass-ai-local.exe'),source_archive_sha256=SUPPORT['sha'](OUT/'native-source.zip'),
        baseline_source_archive_sha256=SUPPORT['sha'](archive),scope=SCOPE,checks=records,
        toolchain='1.99.1',profile='release/locked/offline/x86-64'))


if __name__=='__main__':main()
