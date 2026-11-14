#!/usr/bin/env python3
"""Build the opt-in CPU-inference workers. Training state is never changed.

Compiles a detached copy of lab-native with the ONNX Runtime evaluator against
the same sealed search cores as the GPU binaries, pins the Linux runtime, then
runs the focused regressions. The qualified GPU binaries are left untouched.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import traceback
import urllib.request
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
VERSION='1.22.1'
ARCHIVE=f'https://github.com/microsoft/onnxruntime/releases/download/v{VERSION}/onnxruntime-linux-x64-{VERSION}.tgz'
MEMBER=f'onnxruntime-linux-x64-{VERSION}/lib/libonnxruntime.so.{VERSION}'
TEST_PATHS=('src','scripts','tests/R2','tests/Recovery','tests/FeatureEngineering','tests/RustBackendLab')


def module(name):
    spec=importlib.util.spec_from_file_location(name.replace('-','_'),ROOT/'scripts'/name)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def locate_runtime(explicit,out):
    """The Python wheel may lack the shared library, so use the official archive."""
    target=out/'runtime';target.mkdir(parents=True)
    if explicit:
        source=Path(explicit).resolve()
        if not source.is_file():raise ValueError('Missing ONNX Runtime library: '+str(source))
        origin='explicit'
    else:
        import onnxruntime
        capi=Path(onnxruntime.__file__).parent/'capi'
        found=sorted([*capi.glob(f'libonnxruntime.so.{VERSION}'),*capi.glob('onnxruntime.dll')])
        if found:source,origin=found[0],'python-wheel'
        else:
            archive=out/'onnxruntime.tgz'
            print('Downloading the official ONNX Runtime archive: '+ARCHIVE,flush=True)
            with urllib.request.urlopen(ARCHIVE,timeout=300) as response,archive.open('wb') as stream:
                shutil.copyfileobj(response,stream)
            with tarfile.open(archive) as bundle:
                data=bundle.extractfile(bundle.getmember(MEMBER)).read()
            source=out/'libonnxruntime-download.so';source.write_bytes(data);origin=ARCHIVE
    name=source.name if source.name.startswith('onnxruntime') else f'libonnxruntime.so.{VERSION}'
    destination=target/name;shutil.copyfile(source,destination)
    return destination.resolve(),origin


def compile_worker(out,*,teacher,zero_cost,toolchain,offline,focused=False):
    """Detached checkout: sealed core, then the same overlays as the GPU build."""
    from brass_ai.learning.contracts import require
    from brass_ai.rust_backend.lab_contract import SOURCE,extract_new,native_fingerprint
    from brass_ai.rust_backend.lab_cpu_overlay import apply as apply_cpu
    from brass_ai.rust_backend.lab_cpu_overlay import apply_teacher_performance
    from brass_ai.rust_backend.lab_teacher_overlay import apply as apply_teacher
    base=module('qualify-rust-lab.py');b02=module('qualify-rust-lab-b02.py')
    env=dict(base.python_environment())
    for key in list(env):
        if key in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_TARGET','CARGO_TARGET_DIR') or (key.startswith('CARGO_TARGET_') and key.endswith('_RUSTFLAGS')):
            env.pop(key)
    out.mkdir(parents=True)
    extract_new(ROOT/'distribution/rust-lab-g1250/inputs.zip',out/'inputs')
    extract_new(out/'inputs/source.zip',out/'checkout')
    base.prepare_test_inputs(out/'checkout/native')
    core=SOURCE
    if teacher:
        core=apply_teacher(out/'checkout/native')['core_source']
        if zero_cost:
            from brass_ai.rust_backend.lab_reward_runtime import apply as apply_reward
            core=apply_reward(out/'checkout/native')['core_source']
    require(native_fingerprint(out/'checkout/native')==core,'CPU worker core source differs')
    execution_core=core
    if teacher:
        from brass_ai.rust_backend.lab_cotton_overlay import apply as apply_cotton
        apply_cotton(out/'checkout/native')
        execution_core=apply_teacher_performance(out/'checkout/native')
    overlay=apply_cpu(out/'checkout/lab-native')
    b02.check_shared_dependencies(out/'checkout/native/Cargo.lock',out/'checkout/lab-native/Cargo.lock')
    cargo=['cargo','+'+toolchain];version=subprocess.check_output(['rustc','+'+toolchain,'-vV'],text=True,env=env,timeout=15)
    target=next(x.split(': ',1)[1] for x in version.splitlines() if x.startswith('host: '))
    common=['--manifest-path',out/'checkout/lab-native/Cargo.toml','--release','--target',target,
            '--target-dir',out/'target','--locked','-j','8']
    if teacher:common+=['--features','teacher']
    if offline:common+=['--offline']
    name='brass-lab-teacher' if teacher else 'brass-lab-collector'
    for op in ('test','build'):
        extra=['--bin',name] if op=='build' else (['--lib','cpu_'] if focused else [])
        base.command([*cargo,op,*common,*extra],cwd=ROOT,env=env,log=out/(op+'.log'),timeout=900)
    if teacher:
        core_args=[*cargo,'test','--manifest-path',out/'checkout/native/Cargo.toml','-p','brass-ai-core',
            '--lib','search::session::cotton','--release','--target',target,'--target-dir',out/'target','--locked','-j','8']
        if offline:core_args+=['--offline']
        base.command(core_args,cwd=ROOT,env=env,log=out/'cotton-core-test.log',timeout=900)
    return dict(binary=out/'target'/target/'release'/(name+('.exe' if os.name=='nt' else '')),test_log=out/'test.log',
                build_log=out/'build.log',core=core,execution_core=execution_core,overlay=overlay,rustc=version)


def regressions(out,*,teacher,collector,runtime,fixtures,core,native,focused=False):
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/x) for x in TEST_PATHS),
        BRASS_CPU_TEACHER=str(teacher),BRASS_CPU_COLLECTOR=str(collector),BRASS_CPU_RUNTIME=str(runtime),
        BRASS_CPU_FIXTURES=str(fixtures),BRASS_CPU_CORE=core,BRASS_CPU_NATIVE=str(native))
    with (out/'regressions.log').open('w') as log:
        cases=(['test_teacher_cotton','test_training_revision_g1491','test_cpu_inference.Overlay',
                'test_cpu_inference.NativeWorkers.test_retuned_budget_reaches_long_search_and_its_labels']
               if focused else ['test_cpu_inference','test_teacher_cotton'])
        subprocess.run([sys.executable,'-B','-m','unittest',*cases,'-v'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180 if focused else 900)


def prepare(args,evidence):
    from brass_ai.bank import read
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.recovery.retention import owned,run_lock
    from brass_ai.rust_backend.lab_backend import check_build,validate_state
    from brass_ai.rust_backend.lab_contract import pin
    from brass_ai.rust_backend.lab_cpu_overlay import fingerprint
    from brass_ai.rust_backend.lab_cpu_runtime import BUILD,CONTRACT,SELECTOR,predecessor_builds,resolve
    from brass_ai.rust_backend.lab_perf_runtime import resolve as resolve_ordinary
    from brass_ai.rust_backend.lab_reward_runtime import ZERO_CORE
    from brass_ai.rust_backend.lab_shadow import teacher_build
    from brass_ai.step5.league import write_json
    root,run=owned(ROOT,args.run);selected=root/SELECTOR
    with run_lock(run):
        state=read(run/'state.json');backend=validate_state(state)
        require(state['iteration']==args.expected_generation and backend is not None and backend['mode']=='rust',
                'CPU preparation requires the expected Rust checkpoint')
        entry=state['history'][-1]
        require(pin(project_path(root,entry['path']))['sha256']==entry['sha256'],'Committed checkpoint changed')
        teacher=teacher_build(state);gpu=resolve_ordinary(root,check_build(backend['build']))
        source=fingerprint(root)
        if selected.is_file() and read(selected).get('overlay_source')==source and read(selected).get('teacher_core')==teacher['core_source']:
            result=resolve(root,teacher_core=teacher['core_source'])
            print('Reused verified CPU inference build: '+str(selected),flush=True)
            return dict(reused=True,runtime=result['receipt'],generation=state['iteration'],training_changed=False)
    runtime,origin=locate_runtime(args.onnxruntime_library,evidence)
    collector=compile_worker(evidence/'collector',teacher=False,zero_cost=False,toolchain=args.toolchain,offline=args.offline,focused=args.focused_regressions)
    worker=compile_worker(evidence/'teacher',teacher=True,zero_cost=teacher['core_source']==ZERO_CORE,toolchain=args.toolchain,offline=args.offline,focused=args.focused_regressions)
    require(worker['core']==teacher['core_source'],'CPU teacher core differs from the selected GPU teacher')
    fixtures=Path(gpu['b01']['path']).parent/'inputs/fixtures.json'
    if not args.skip_regressions:
        regressions(evidence,teacher=worker['binary'],collector=collector['binary'],runtime=runtime,fixtures=fixtures,
                    core=worker['execution_core'],native=evidence/'teacher/checkout/native',focused=args.focused_regressions)
    with run_lock(run):
        current=read(run/'state.json')
        require(current['iteration']==args.expected_generation and current['history'][-1]==entry and fingerprint(root)==source,
                'Run or source changed during CPU preparation')
        logs=evidence/'build-log.txt'
        logs.write_bytes(b''.join(Path(p).read_bytes() for p in (collector['build_log'],worker['build_log'])))
        tests=evidence/'test-log.txt'
        tests.write_bytes(b''.join(Path(p).read_bytes() for p in (collector['test_log'],worker['test_log'],evidence/'teacher/cotton-core-test.log')))
        value=dict(contract=BUILD,inference=CONTRACT,overlay_source=source,runtime=pin(runtime),runtime_origin=origin,
            runtime_version=VERSION,collector=pin(collector['binary']),teacher=pin(worker['binary']),
            collector_core=collector['core'],teacher_core=worker['core'],teacher_execution_core=worker['execution_core'],tests=pin(tests),build_log=pin(logs),
            rustc=worker['rustc'],regressions=None if args.skip_regressions else pin(evidence/'regressions.log'),
            training_changed=False)
        if selected.is_file():
            from brass_ai.rust_backend.lab_backend import checked_pin
            previous_pin=pin(selected)
            previous=checked_pin(previous_pin)
            if (previous.get('contract')==BUILD and previous.get('inference')==CONTRACT
                    and previous.get('collector_core')==collector['core']):
                value['previous_collectors']=list(dict.fromkeys(
                    (*predecessor_builds(previous),previous_pin['sha256'])))
        write_json(selected,value)
        result=resolve(root,teacher_core=teacher['core_source'])
    return dict(reused=False,runtime=result['receipt'],generation=state['iteration'],training_changed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--expected-generation',type=int,required=True)
    p.add_argument('--onnxruntime-library',help=f'Path to libonnxruntime.so.{VERSION}, for machines without download access')
    p.add_argument('--toolchain',default='1.99.1');p.add_argument('--offline',action='store_true')
    p.add_argument('--skip-regressions',action='store_true',help='Build only; not for a run about to train')
    p.add_argument('--focused-regressions',action='store_true',help='Short cotton/revision regressions, no games or benchmarks')
    args=p.parse_args()
    from brass_ai.step5.league import write_json
    from rust_lab_results import export
    evidence=ROOT/'artifacts/rust-backend'/('cpu-inference-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8])
    evidence.mkdir(parents=True)
    report=dict(passed=False,scope='CPU inference build and focused regressions; no training')
    try:
        report.update(prepare(args,evidence),passed=True)
        print('CPU inference ready: '+str(report['runtime']['path'])+'; training unchanged.',flush=True)
    except BaseException as error:
        report.update(error=str(error),traceback=traceback.format_exc());raise
    finally:
        write_json(evidence/'qualification.json',report)
        print('Git-visible results: '+str(export(evidence)),flush=True)
        print('Original lab evidence: '+str(evidence),flush=True)


if __name__=='__main__':main()
