#!/usr/bin/env python3
"""B02 additive Linux collector build and bounded isolated lab qualification."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from brass_ai.learning.contracts import read_json,require,digest
from brass_ai.rust_backend.lab_contract import B01_CONTRACT,METHOD,SOURCE,admit,native_fingerprint,pin,sha,verify_inputs

def b01_tools():
    spec=importlib.util.spec_from_file_location('b01_qualification',ROOT/'scripts/qualify-rust-lab.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def check_shared_dependencies(core_lock, collector_lock):
    def registry(path):
        packages=tomllib.loads(Path(path).read_text(encoding='utf-8'))['package']
        return {(p['name'],p['version'],p.get('source'),p.get('checksum')) for p in packages if p.get('source')}
    require(registry(collector_lock)<=registry(core_lock),
            'Collector registry dependencies differ from the qualified B01 lock')


def entry(path, run, device):
    """No bypass flag: development can precede B01, lab qualification cannot."""
    path=Path(path).resolve(); value=read_json(path)
    require(value.get('contract')==B01_CONTRACT and value.get('passed') is True
            and value.get('milestone')=='B01' and value.get('v205_enabled') is False
            and value.get('method')==METHOD,'B01 qualification is Pending, failed or from an older profile; complete search_v1 B01 first')
    for key in ('profile','local_entry','lab_state','linux_build','python_sources','commands'):
        require(pin(value[key]['path'])==value[key], 'Changed B01 '+key)
    admit(read_json(value['profile']['path']))
    verify_inputs(path.parent/'inputs',read_json(value['profile']['path']))
    state=read_json(value['lab_state']['path'])
    require(Path(state['state_file']['path']).parent==Path(run).resolve(),'B01 belongs to another run')
    from brass_ai.rust_backend.lab_snapshot import verify_unchanged
    verify_unchanged(state)
    qualified=[]
    for item in value['gpu']:
        require(pin(item['path'])==item,'Changed B01 device receipt')
        receipt=read_json(item['path']);require(receipt['passed'] is True,'B01 device failed');qualified.append(receipt['device'])
    require(device in qualified,'Requested GPU lacks a passing B01 receipt')
    checkout=path.parent/'checkout/native'
    require(native_fingerprint(checkout)==SOURCE,'B01 shared core changed')
    build=read_json(value['linux_build']['path'])
    require(pin(build['binary']['path'])==build['binary'] and build['handshake']['build']['source_sha256']==SOURCE,'B01 Linux binary changed')
    return state,checkout


def build_bootstrap(out, env, command, mono):
    from brass_ai.recovery.host_runtime import local_scoring_host
    source=local_scoring_host(ROOT);manifest=read_json(source/'runtime.json')
    target=out/'bootstrap';target.mkdir()
    for item in manifest['files']:
        require(sha(source/item['name'])==item['sha256'],'Reference host pin differs')
        shutil.copyfile(source/item['name'],target/item['name'])
    shutil.copyfile(source/'runtime.json',target/'runtime.json')
    code=ROOT/'src/BrassLabBootstrap/Bootstrap.cs'
    compiler=ROOT/'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
    require(compiler.is_file(),'Pinned C# compiler missing; restore existing patch toolchain')
    args=['/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',
          f'/out:"{target / "BrassLabBootstrap.exe"}"',f'/reference:"{target / "BrassHost.exe"}"']
    args += [f'/reference:"{p}"' for p in sorted(target.glob('*.dll'))]+[f'"{code}"']
    rsp=target/'build.rsp';rsp.write_text('\n'.join(args),encoding='utf-8-sig')
    prefix=[str(compiler)] if os.name=='nt' else [mono,str(compiler)]
    command([*prefix,'@'+str(rsp)],cwd=ROOT,env=env,log=out/'bootstrap-build.log',timeout=120)
    return dict(binary=pin(target/'BrassLabBootstrap.exe'),source=pin(code),compiler=pin(compiler),reference=pin(source/'runtime.json'),
                dependencies=[pin(target/i['name']) for i in manifest['files']])


def qualify(args,out):
    require(platform.system()=='Linux','Run B02 qualification on the lab Linux host')
    base=b01_tools();env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'src'),str(ROOT/'tests/Recovery'),env.get('PYTHONPATH','')])
    env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    state,core=entry(args.b01,args.run,args.device)
    from brass_ai.recovery.retention import run_lock
    from brass_ai.rust_backend.lab_snapshot import verify_unchanged
    source=base.source_inventory()+[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
        for p in sorted((ROOT/'lab-native').rglob('*')) if p.is_file() and p.suffix in ('.rs','.toml','.lock')]
    with run_lock(args.run):
        verify_unchanged(state)
        try:
            checkout=out/'checkout';checkout.mkdir()
            shutil.copytree(core,checkout/'native')
            shutil.copytree(ROOT/'lab-native',checkout/'lab-native',ignore=shutil.ignore_patterns('target'))
            require(native_fingerprint(checkout/'native')==SOURCE,'Staged shared core differs')
            check_shared_dependencies(checkout/'native/Cargo.lock',checkout/'lab-native/Cargo.lock')
            for key in list(env):
                if key in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_TARGET','CARGO_TARGET_DIR') or (key.startswith('CARGO_TARGET_') and key.endswith('_RUSTFLAGS')):env.pop(key)
            version=subprocess.check_output(['rustc','+1.99.1','-vV'],text=True,env=env,timeout=15)
            target=next(x.split(': ',1)[1] for x in version.splitlines() if x.startswith('host: '))
            require('linux' in target,'Linux toolchain required')
            common=['--manifest-path',checkout/'lab-native/Cargo.toml','--target',target,'--target-dir',out/'target','--locked']
            if args.offline:common.append('--offline')
            commands=[]
            for operation in ('build','test'):
                commands.append(base.command(['cargo','+1.99.1',operation,'--release',*common],cwd=checkout,env=env,
                    log=out/(operation+'.log'),timeout=900))
            commands.append(base.command([sys.executable,'-m','unittest','discover','-s','tests/RustBackendLab','-v'],
                cwd=ROOT,env=env,log=out/'adapter-tests.log',timeout=90))
            bootstrap=build_bootstrap(out,env,base.command,args.mono)
            executable=out/'target'/target/'release/brass-lab-collector'
            identity=dict(contract='brass-lab-b02-build/1.0.0',shared_core=SOURCE,collector=pin(executable),
                bootstrap=bootstrap,source=source,rustc=version,cargo_lock=pin(checkout/'lab-native/Cargo.lock'),
                b01=pin(args.b01),search_method=METHOD,search_enabled=False,v205_enabled=False)
            base.write(out/'build.json',identity)
            commands.append(base.command([sys.executable,'-m','brass_ai.rust_backend.lab_b02_smoke',
                '--run',Path(args.run).resolve(),'--state',read_json(args.b01)['lab_state']['path'],
                '--build',out/'build.json','--out',out,'--device',args.device,'--mono',args.mono],
                cwd=ROOT,env=env,log=out/'collection-smoke.log',timeout=args.smoke_timeout))
            smoke=read_json(out/'collection-smoke.json')
            require(smoke['passed'] is True and smoke['branch_exit_qualified'] is True
                    and not smoke['missing_table_kinds'],'B02 smoke/coverage gate failed')
            verify_unchanged(state)
            current=base.source_inventory()+[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
                for p in sorted((ROOT/'lab-native').rglob('*')) if p.is_file() and p.suffix in ('.rs','.toml','.lock')]
            require(current==source and native_fingerprint(checkout/'native')==SOURCE,'Sources changed during qualification')
            base.write(out/'commands.json',commands)
            return dict(contract='brass-lab-b02/2.0.0',milestone='B02',passed=True,b01=pin(args.b01),
                method=METHOD,v205_enabled=False,search_enabled=False,
                build=pin(out/'build.json'),smoke=pin(out/'collection-smoke.json'),training_admitted=False,
                lab_deployed=False,qualification_scope=smoke['scope'],next='B03 strict producer/replay admission')
        finally:verify_unchanged(state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--b01',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--mono',default='mono')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--offline',action='store_true')
    p.add_argument('--smoke-timeout',type=int,default=240,help='Bounded short coverage smoke, maximum 600 seconds')
    a=p.parse_args();a.out=a.out.resolve()
    require(a.out.is_relative_to(ROOT/'artifacts/rust-backend') and a.out!=ROOT/'artifacts/rust-backend','Evidence must be under artifacts/rust-backend')
    require(not a.out.is_relative_to(a.run.resolve()) and not a.run.resolve().is_relative_to(a.out),'Evidence cannot overlap training run')
    require(0<a.smoke_timeout<=600,'Invalid smoke watchdog');a.out.mkdir(parents=True,exist_ok=False)
    write=b01_tools().write
    try:result=qualify(a,a.out)
    except Exception as error:
        write(a.out/'qualification.json',dict(contract='brass-lab-b02/2.0.0',milestone='B02',passed=False,error=str(error),lab_deployed=False));raise
    write(a.out/'qualification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
