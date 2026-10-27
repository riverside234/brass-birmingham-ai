"""Build and qualify L13 in an isolated Windows player; never installs it.

The evidence directory seals only after all checks pass. L14/L15 stay pending.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')

def qualify(out,resume=False):
    out=out.resolve()
    assert out.is_relative_to(ROOT/'artifacts/rust-backend') and not (out/'qualification.json').exists()
    out.mkdir(parents=True,exist_ok=True)
    support=runpy.run_path(str(ROOT/'scripts/qualify-rust-scaffold.py'))
    env,cargo=support['tool_environment']();env.pop('PROPTEST_RNG_SEED',None);env['CARGO_INCREMENTAL']='0'
    source=support['native_fingerprint']();commands=[]
    if resume:
        build=read(out/'build-manifest.json');assert build['native_source_sha256']==source
        for pin in build['sourcePins']:
            assert sha(out/'runtime'/pin['path'])==pin['sha256']
            if pin['path'] in build['sourceScope']:assert sha(ROOT/pin['path'])==pin['sha256']
        for pin in build['files']:assert sha(out/'Managed'/pin['name'])==pin['sha256']
        commands=read(out/'progress.json')['commands']
        for command in commands:
            if command['exit_code']:
                log=out/command['log'];retained=log.with_name(log.stem+'-failed-'+str(len(commands))+'.log')
                log.rename(retained);command['log']=retained.name
    powershell=shutil.which('pwsh') or shutil.which('powershell')
    assert powershell,'PowerShell runtime unavailable'
    def run(name,args,cwd=ROOT,environment=None):
        if resume and any(c['name']==name and c['exit_code']==0 for c in commands):
            print(name+' (recorded pass retained)',flush=True);return
        print(name,flush=True)
        start=datetime.datetime.now(datetime.timezone.utc).isoformat()
        with (out/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([str(x) for x in args],cwd=cwd,env=environment or env,stdout=log,stderr=subprocess.STDOUT)
        commands.append(dict(name=name,args=[str(x) for x in args],cwd=str(cwd),started_utc=start,exit_code=result.returncode,log=name+'.log'))
        write(out/'progress.json',dict(step='L13',status='running' if result.returncode==0 else 'failed',commands=commands))
        if result.returncode:raise RuntimeError(name+' failed; see '+str(out/(name+'.log')))
    def rust(name,*args):run(name,[cargo,*args],ROOT/'native')
    rust('fmt','fmt','--all','--','--check')
    rust('clippy','clippy','--offline','--workspace','--all-targets','--target','x86_64-pc-windows-gnu','--','-D','warnings')
    rust('debug-tests','test','--offline','--workspace','--target','x86_64-pc-windows-gnu')
    rust('release-tests','test','--offline','--release','--workspace','--target','x86_64-pc-windows-gnu')
    rust('release-build','build','--offline','--release','--bin','brass-ai-local','--target','x86_64-pc-windows-gnu')
    run('package-build',[sys.executable,ROOT/'scripts/build-rust-local-package.py','--out',out,'--binary',ROOT/'artifacts/rust-native-target/x86_64-pc-windows-gnu/release/brass-ai-local.exe'])
    package=out/'BrassNative/alpha-g890'
    env['BRASS_ONNX_MANIFEST']=str(package/'model/manifest.json')
    env['BRASS_ONNX_GOLDEN']=str(ROOT/'artifacts/rust-backend/20261025-l09-qualified-v2/golden.json')
    rust('onnx-golden','test','--offline','--release','-p','brass-ai-inference','--test','runtime','--target','x86_64-pc-windows-gnu','--','--ignored','--nocapture')
    env['BRASS_LOCAL_PACKAGE']=str(package)
    run('process-tests',[sys.executable,ROOT/'tests/RustBackendLocal/test_local.py','-v'])
    run('transport-replay',[sys.executable,ROOT/'scripts/verify-rust-local.py','--package',package,'--out',out/'transport-replay'])
    relative=out.relative_to(ROOT).as_posix()
    run('unity-tests',[powershell,'-NoProfile','-ExecutionPolicy','Bypass','-File',ROOT/'tests/RustBackendLocal/run-unity.ps1','-BuildDirectory',relative])
    run('preservation-tests',[powershell,'-NoProfile','-ExecutionPolicy','Bypass','-File',ROOT/'tests/RustBackendLocal/check-preservation.ps1','-BuildDirectory',relative])
    unity=read(out/'unity.json');assert unity['passed'] and not unity['nativeErrors']
    turns=read(out/'unity-turns.json');assert {r['difficulty'] for r in turns}=={'easy','mid','hard','max'}
    assert read(out/'transport-replay/verification.json')['status']=='passed'
    assert read(out/'preserved-runtime.json')['passed']
    assert source==support['native_fingerprint']()
    before=read(ROOT/'artifacts/rust-backend/l13-preservation-before.json')
    for pin in before:assert sha(ROOT/pin['path'])==pin['sha256'],pin['path']
    write(out/'preservation.json',dict(passed=True,files=before))
    # Capture exact startup readiness independently of the local acceptance.
    worker_module=runpy.run_path(str(ROOT/'tests/RustBackendLocal/test_local.py'))
    worker=worker_module['Worker'](package)
    try:hello=worker.hello['data']
    finally:worker.close()
    assert hello['build']['source_sha256']==source and hello['measured_readiness']['warmup_passed']
    write(out/'readiness.json',hello)
    build=read(out/'build-manifest.json');build['measured_readiness']=hello['measured_readiness'];write(out/'build-manifest.json',build)
    images=[]
    for name in ('l13-alpha-panel.png','l13-alpha-table.png'):
        path=ROOT/'artifacts/card-tracker-player/artifacts/patch-build'/name
        shutil.copyfile(path,out/name);images.append(dict(path=name,sha256=sha(out/name)))
    with zipfile.ZipFile(out/'alpha-test-native-windows-x64.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((out/'BrassNative').rglob('*')):
            if p.is_file():archive.write(p,p.relative_to(out).as_posix())
    executed=[ROOT/'scripts'/n for n in ('qualify-rust-local.py','build-rust-local-package.py','verify-rust-local.py')]
    executed+=list((ROOT/'tests/RustBackendLocal').glob('*'))+[ROOT/'tests/BrassLocalAI/LocalAiUnityTests.cs',ROOT/'tests/BrassLocalAI/AiExecutionResourceTests.cs']
    pins=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in executed if p.is_file()]
    write(out/'source-pins.json',pins)
    with zipfile.ZipFile(out/'integration-source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for pin in pins:archive.write(ROOT/pin['path'],pin['path'])
        for p in sorted((out/'runtime').rglob('*')):
            if p.is_file():archive.write(p,p.relative_to(out/'runtime').as_posix())
    write(out/'progress.json',dict(step='L13',status='complete',commands=commands))
    files=[dict(path=p.relative_to(out).as_posix(),sha256=sha(p)) for p in sorted(out.rglob('*'))
           if p.is_file() and p.name not in ('qualification.json','final-verification.json')]
    result=dict(format='brass-local-l13-qualification/1.0.0',passed=True,step='L13',
        scope='local-alpha-test-integration',qualified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        native_source_sha256=source,build_manifest_sha256=sha(out/'build-manifest.json'),
        package_build_sha256=sha(package/'build.json'),files=files,images=images,
        acceptance=hello['utility_acceptance'],measured_readiness=hello['measured_readiness'],
        unity_checks=unity['checks'],shadow_turns=turns,l14_performance_quality='pending',l15_full_games='pending',installed=False)
    write(out/'qualification.json',result)
    audit(out)
    print('L13 passed: '+str(out),flush=True)

def audit(out):
    out=out.resolve();q=read(out/'qualification.json');assert q['passed'] and q['step']=='L13'
    for pin in q['files']:
        p=(out/pin['path']).resolve();assert p.is_relative_to(out) and sha(p)==pin['sha256'],pin['path']
    assert sha(out/'build-manifest.json')==q['build_manifest_sha256']
    assert sha(out/'BrassNative/alpha-g890/build.json')==q['package_build_sha256']
    write(out/'final-verification.json',dict(passed=True,qualification_sha256=sha(out/'qualification.json'),verified_files=len(q['files'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--audit',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.audit:audit(args.out)
    else:qualify(args.out,args.resume)
