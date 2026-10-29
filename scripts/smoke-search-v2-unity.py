"""One bounded isolated Unity turn; never copies into the installed game."""
import argparse
import os
import runpy
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
T=runpy.run_path(str(ROOT/'scripts/search-v2-tools.py'))
read,write,sha,receipt=(T[k] for k in ('read','write','sha','receipt'))


def main(a):
    if a.method=='search_v2':assert a.build and not a.profile, 'V205 smoke requires its explicit isolated build'
    out=T['OUT']/a.label;assert not out.exists();out.mkdir()
    source=T['OUT']/'local-candidate' if a.method=='search_v2_focus_only' else ROOT/'artifacts/setup-preferences/candidate'
    if a.profile:source=T['OUT']/('candidate-profile-build' if a.method=='search_v2_focus_only' else 'control-profile-build')
    if a.build:source=a.build.resolve();assert not a.profile
    build=out/'build';build.mkdir()
    for folder in ('Managed','BrassNative'):shutil.copytree(source/folder,build/folder)
    package=build/'BrassNative/alpha-g890';policy=read(package/'policy.json')
    if a.setting=='8s_k8':policy['k_by_allowance_ms']['8000']=8
    write(package/'policy.json',policy)
    native=read(package/'build.json')
    for pin in native['files']:pin['sha256']=sha(package/pin['path'])
    write(package/'build.json',native)
    manifest=read(source/'build-manifest.json');manifest.update(diagnostic_only=True,
        source_candidate=receipt(source/'build-manifest.json'),native_build_sha256=sha(package/'build.json'))
    write(build/'build-manifest.json',manifest)
    fixture=ROOT/'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures/live-92781-CanalEra-2.json'
    env=dict(os.environ,BRASS_SEARCH_V2_RESULT=str(out/'result.json'),BRASS_SEARCH_V2_EVENTS=str(fixture),
        BRASS_SEARCH_V2_MS='4000' if a.setting=='4s_k8' else '8000',BRASS_SEARCH_V2_SCENARIO=a.scenario)
    if a.profile:env['BRASS_L14_TRACE']=str(out/'stage-trace.json')
    command=['pwsh','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ROOT/'tests/RustBackendLocal/run-unity.ps1'),
        '-BuildDirectory',str(build.relative_to(ROOT)),'-EvidenceDirectory',str((out/'evidence').relative_to(ROOT)),
        '-SearchV2','-MaxSeconds','60']
    with (out/'runner.log').open('wb') as log:
        r=subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=100,
                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    write(out/'receipt.json',dict(method=a.method,setting=a.setting,scenario=a.scenario,fixture=receipt(fixture),
        build=receipt(build/'build-manifest.json'),exit_code=r.returncode,
        result=receipt(out/'result.json') if (out/'result.json').exists() else None))
    print((out/'runner.log').read_text()[-2000:])
    assert r.returncode==0
    row=read(out/'result.json');print({k:v for k,v in row.items() if k not in ('input','replies','readiness')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--setting',choices=['4s_k8','8s_k8','8s_k16'],required=True)
    p.add_argument('--method',choices=['search_v1','search_v2_focus_only','search_v2'],default='search_v2_focus_only')
    p.add_argument('--scenario',choices=['normal','cancel','load'],default='normal');p.add_argument('--label',required=True)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--build',type=Path)
    main(p.parse_args())
