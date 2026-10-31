#!/usr/bin/env python3
"""Build B04 exports and run exactly two ready, frozen shadow tables on the lab."""
import argparse
from datetime import datetime,timezone
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def module(name):
    spec=importlib.util.spec_from_file_location(name.replace('-','_'),ROOT/'scripts'/name)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path);p.add_argument('--b02',type=Path);p.add_argument('--b03',type=Path)
    p.add_argument('--expected-generation',type=int);p.add_argument('--checkpoint-sha256')
    p.add_argument('--device',default='cuda:0');p.add_argument('--mono',default='mono');p.add_argument('--out',type=Path)
    p.add_argument('--offline',action='store_true');p.add_argument('--fixtures-only',action='store_true')
    p.add_argument('--teacher-profile',choices=('legacy','domino'),default='legacy')
    p.add_argument('--transfer-manifest',type=Path)
    a=p.parse_args()
    from brass_ai.learning.contracts import require,project_path,read_json
    from brass_ai.step5.league import write_json,implementation_id
    from brass_ai.recovery.retention import run_lock,owned
    from brass_ai.rust_backend.lab_transfer import verify
    from brass_ai.rust_backend.lab_contract import pin,native_fingerprint,SOURCE
    from brass_ai.rust_backend.lab_teacher_overlay import apply,source_fingerprint
    from brass_ai.rust_backend.lab_b04_smoke import entry
    from brass_ai.rust_backend.lab_snapshot import verify_unchanged
    require(a.fixtures_only or (a.run and a.b02 and a.b03 and a.expected_generation is not None and a.checkpoint_sha256),
            'Supply --run, --b02, --b03, --expected-generation and --checkpoint-sha256')
    require(not a.fixtures_only or not any((a.run,a.b02,a.b03,a.expected_generation,a.checkpoint_sha256)),
            'CPU fixture scope cannot claim lab qualification')
    out=project_path(ROOT,a.out or ('artifacts/rust-backend/'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-b04'))
    require(out.is_relative_to(ROOT/'artifacts/rust-backend') and not out.exists(),'Use a new evidence directory under artifacts/rust-backend')
    if a.run:
        _,a.run=owned(ROOT,a.run)
        require(not out.is_relative_to(a.run) and not a.run.is_relative_to(out),'Evidence cannot overlap production')
    out.mkdir(parents=True)
    report=dict(contract='brass-lab-b04/1.0.0',milestone='B04',passed=False,method='search_v1',v205_enabled=False,
                lab_qualification='Pending',production_deployed=False)
    base=module('qualify-rust-lab.py');b02_tools=module('qualify-rust-lab-b02.py')
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/n) for n in ('src','tests/R2','tests/Recovery','tests/FeatureEngineering','tests/RustBackendLab')),
             CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        report['transfer']=verify('b04',a.transfer_manifest)
        report['teacher_profile']=a.teacher_profile
        print('B04: short dispatch/inference fixtures',flush=True)
        report['fixtures']=base.command([sys.executable,'-m','unittest','test_b04',*(['test_teacher_diversity'] if a.teacher_profile=='domino' else []),'-v'],cwd=ROOT,
            env=dict(env,CUDA_VISIBLE_DEVICES=''),log=out/'fixtures.log',timeout=180)
        if a.fixtures_only:report['passed']=True;report['scope']='CPU fixtures only; native/Linux/CUDA gates remain Pending';return
        require(platform.system()=='Linux','Run full B04 qualification on the Linux/CUDA lab server')
        with run_lock(a.run):
            snapshot=entry(a.b02,a.b03,a.run,a.device)
            require(snapshot['generation']==a.expected_generation and snapshot['checkpoint']['sha256']==a.checkpoint_sha256,
                    'Expected committed generation/checkpoint differs')
            try:
                old=read_json(read_json(a.b02)['build']['path']);b01=Path(old['b01']['path']).parent
                checkout=out/'checkout';checkout.mkdir()
                shutil.copytree(b01/'checkout/native',checkout/'native')
                require(native_fingerprint(checkout/'native')==SOURCE,'B01 core differs')
                overlay=apply(checkout/'native');write_json(out/'overlay.json',overlay)
                shutil.copytree(ROOT/'lab-native',checkout/'lab-native',ignore=shutil.ignore_patterns('target'))
                fixture=checkout/'tests/GumbelISMCTS/fixtures/session-v1.json';fixture.parent.mkdir(parents=True)
                shutil.copyfile(ROOT/'tests/GumbelISMCTS/fixtures/session-v1.json',fixture)
                descriptor=checkout/'configs/learning/domino-descriptor-1.0.0.json';descriptor.parent.mkdir(parents=True)
                shutil.copyfile(ROOT/'configs/learning/domino-descriptor-1.0.0.json',descriptor)
                b02_tools.check_shared_dependencies(checkout/'native/Cargo.lock',checkout/'lab-native/Cargo.lock')
                for k in list(env):
                    if k in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_TARGET','CARGO_TARGET_DIR') or (k.startswith('CARGO_TARGET_') and k.endswith('_RUSTFLAGS')):env.pop(k)
                version=subprocess.check_output(['rustc','+1.99.1','-vV'],text=True,env=env,timeout=15)
                target=next(line.split(': ',1)[1] for line in version.splitlines() if line.startswith('host: '))
                require('linux' in target,'Linux target required')
                common=['--manifest-path',checkout/'lab-native/Cargo.toml','--features','teacher','--release','--target',target,
                        '--target-dir',out/'target','--locked','-j','2']
                if a.offline:common+=['--offline']
                commands=[]
                for op in ('build','test'):
                    print('B04: native '+op,flush=True)
                    commands.append(base.command(['cargo','+1.99.1',op,*common],cwd=ROOT,env=env,log=out/(op+'.log'),timeout=900))
                binary=out/'target'/target/'release/brass-lab-teacher'
                if a.teacher_profile=='domino':
                    core=['--manifest-path',checkout/'native/Cargo.toml','-p','brass-ai-core','--lib','domino',
                        '--release','--target',target,'--target-dir',out/'target','--locked','-j','2']
                    if a.offline:core+=['--offline']
                    commands.append(base.command(['cargo','+1.99.1','test',*core],cwd=ROOT,env=env,
                        log=out/'core-objective-tests.log',timeout=900))
                    commands.append(base.command([sys.executable,'-m','unittest',
                        'test_teacher_diversity.DiversityTeacher.test_native_descriptor_exact_in_four_phases',
                        'test_teacher_diversity.DiversityTeacher.test_native_descriptor_populated_public_state_all_seats',
                        'test_teacher_diversity.DiversityTeacher.test_native_intrinsic_probe_and_real_service',
                        'test_b04.B04.test_native_count_probe_targets_and_actual_model_logits','-v'],
                        cwd=ROOT,env=dict(env,BRASS_DOMINO_NATIVE=str(binary),BRASS_B04_BINARY=str(binary),
                            BRASS_B04_NATIVE=str(checkout/'native'),BRASS_B04_CORE=overlay['core_source']),
                        log=out/'native-python-parity.log',timeout=90))
                build=dict(contract='brass-lab-b04-build/1.0.0',binary=pin(binary),core_source=overlay['core_source'],
                    base_source=SOURCE,producer=source_fingerprint(),implementation=implementation_id(),bootstrap=old['bootstrap'],
                    b02=pin(a.b02),b03=pin(a.b03),rustc=version,overlay=pin(out/'overlay.json'),lock=pin(checkout/'lab-native/Cargo.lock'))
                write_json(out/'build.json',build)
                print('B04: GPU calibration and two 4s/K16 shadow tables',flush=True)
                command=[sys.executable,'-m','brass_ai.rust_backend.lab_b04_smoke','--run',a.run,
                    '--b02',a.b02.resolve(),'--b03',a.b03.resolve(),'--build',out/'build.json','--out',out,
                    '--expected-generation',str(a.expected_generation),'--checkpoint-sha256',a.checkpoint_sha256,
                    '--device',a.device,'--mono',a.mono,'--teacher-profile',a.teacher_profile]
                commands.append(base.command(command,cwd=ROOT,env=dict(env,BRASS_B04_PARENT_LOCK=(a.run/'.active-run.json').read_text()),
                    log=out/'lab-smoke.log',timeout=300))
                report['lab']=read_json(out/'lab-checks.json');report['passed']=report['lab']['passed']
                report['lab_qualification']='Passed for this exact source/model/device' if report['passed'] else 'Pending: readiness deferred'
                report['build']=pin(out/'build.json');write_json(out/'commands.json',commands)
                report['commands']=pin(out/'commands.json');verify('b04',a.transfer_manifest)
            finally:verify_unchanged(snapshot)
    except Exception as e:report['error']=str(e);raise
    finally:write_json(out/'qualification.json',report);print(out/'qualification.json',flush=True)


if __name__=='__main__':main()
