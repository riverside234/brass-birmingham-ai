#!/usr/bin/env python3
"""B04.5 g1250-only shadow qualification. Never repeats B01–B04 or adopts training."""
import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def module(name):
    spec = importlib.util.spec_from_file_location(name.replace('-','_'),ROOT/'scripts'/name)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def verify_bundle():
    import hashlib
    from brass_ai.learning.contracts import read_json, require
    from brass_ai.rust_backend.lab_contract import sha
    from brass_ai.step5.league import implementation_id
    manifest = read_json(ROOT/'distribution/rust-lab-g1250/manifest.json')
    require(manifest['implementation'] == implementation_id(), 'B04.5 checkout differs from the bundled implementation')
    for name, expected in manifest['files'].items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT), 'B04.5 bundle path escaped')
        data = path.read_bytes()
        if path.suffix in manifest['canonical_text_suffixes']: data = data.replace(b'\r\n',b'\n')
        require(hashlib.sha256(data).hexdigest() == expected, 'B04.5 bundle file differs: '+name)
    return manifest


def build(out, *, offline=False, mono='mono'):
    from brass_ai.learning.contracts import read_json, require
    from brass_ai.rust_backend.lab_contract import extract_new, pin
    from brass_ai.rust_backend.lab_teacher_overlay import apply, source_fingerprint
    from brass_ai.step5.league import write_json, implementation_id
    base, b02 = module('qualify-rust-lab.py'), module('qualify-rust-lab-b02.py')
    env = base.python_environment()
    for key in list(env):
        if key in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_TARGET','CARGO_TARGET_DIR') or (key.startswith('CARGO_TARGET_') and key.endswith('_RUSTFLAGS')):
            env.pop(key)
    extract_new(ROOT/'distribution/rust-lab-g1250/inputs.zip',out/'inputs')
    extract_new(out/'inputs/source.zip',out/'checkout')
    base.prepare_test_inputs(out/'checkout/native')
    overlay = apply(out/'checkout/native')
    shutil.copytree(ROOT/'lab-native',out/'checkout/lab-native',ignore=shutil.ignore_patterns('target'))
    b02.check_shared_dependencies(out/'checkout/native/Cargo.lock',out/'checkout/lab-native/Cargo.lock')
    version = subprocess.check_output(['rustc','+1.99.1','-vV'],text=True,env=env,timeout=15)
    target = next(x.split(': ',1)[1] for x in version.splitlines() if x.startswith('host: '))
    common = ['--manifest-path',out/'checkout/lab-native/Cargo.toml','--features','teacher','--release',
              '--target',target,'--target-dir',out/'target','--locked','-j','2']
    if offline: common += ['--offline']
    for op in ('test','build'):
        base.command(['cargo','+1.99.1',op,*common],cwd=ROOT,env=env,log=out/(op+'.log'),timeout=900)
    bootstrap = b02.build_bootstrap(out,env,base.command,mono)
    name = 'brass-lab-teacher'+('.exe' if os.name == 'nt' else '')
    result = dict(contract='brass-lab-b045-build/1.0.0', binary=pin(out/'target'/target/'release'/name),
        core_source=overlay['core_source'],producer=source_fingerprint(),implementation=implementation_id(),
        bootstrap=bootstrap,overlay=overlay,rustc=version,fixtures=pin(out/'inputs/fixtures.json'))
    write_json(out/'build.json',result)
    return result


def main():
    from brass_ai.learning.contracts import read_json, require, project_path
    from brass_ai.step5.league import write_json
    from brass_ai.rust_backend.lab_snapshot import capture, verify_unchanged
    from brass_ai.recovery.retention import run_lock
    from rust_lab_results import export
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--mono',default='mono')
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--fixtures-only',action='store_true'); p.add_argument('--offline',action='store_true')
    p.add_argument('--build',type=Path,help='Reuse only a matching B04.5 build with verified runtime pins')
    args = p.parse_args()
    token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-g1250-b045-'+uuid.uuid4().hex[:8]
    out = ROOT/'artifacts/rust-backend'/token; out.mkdir(parents=True,exist_ok=False)
    report = dict(milestone='B04.5',passed=False,lab_qualification='Pending',production_modified=False,previous_qualifications_repeated=False)
    base = module('qualify-rust-lab.py'); env = base.python_environment()
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        report['bundle'] = verify_bundle()['implementation']
        base.command([sys.executable,'-m','unittest','test_b045','test_b045_failures','test_b04','test_collection_throughput','test_result_export','-q'],
                     cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),log=out/'fixtures.log',timeout=120)
        if args.fixtures_only:
            report.update(passed=True,scope='CPU fixtures only; lab Pending'); return
        require(platform.system()=='Linux', 'Run full B04.5 on the Linux CUDA lab')
        run = project_path(ROOT,args.run)
        with run_lock(run):
            snapshot,_,_ = capture(ROOT,run,devices=[args.device])
            reference = read_json(ROOT/'distribution/rust-lab-g1250/training-reference.json')
            require(snapshot['generation']==1250 and snapshot['checkpoint']['sha256']==reference['checkpoint']['sha256']
                    and snapshot['model_id']==reference['checkpoint']['model_id']
                    and snapshot['config_sha256']==reference['config_sha256'], 'Pause at the exact recorded g1250 checkpoint/config')
            write_json(out/'snapshot.json',snapshot)
            try:
                if args.build:
                    from brass_ai.rust_backend.lab_b045_smoke import verify_build
                    built=verify_build(read_json(args.build)); write_json(out/'build.json',built)
                else: built=build(out,offline=args.offline,mono=args.mono)
                # Short scripted full-game fixtures catch later-turn clock and
                # terminal-state bugs that fixed-root timing probes cannot.
                import json
                base.command([sys.executable,'-m','unittest','test_b045_failures.NativeScriptedTables','-q'],
                    cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='',BRASS_B045_BINARY=built['binary']['path'],
                        BRASS_B045_CORE=built['core_source'],
                        BRASS_B045_BOOTSTRAP=json.dumps([args.mono,built['bootstrap']['binary']['path']])),
                    log=out/'scripted-tables.log',timeout=45)
                base.command([sys.executable,'-m','brass_ai.rust_backend.lab_b045_smoke','--out',out,
                    '--device',args.device,'--mono',args.mono],cwd=ROOT,
                    env=dict(env,BRASS_B045_PARENT_LOCK=(run/'.active-run.json').read_text()),log=out/'lab-smoke.log',timeout=240)
                report.update(read_json(out/'lab-checks.json'))
            finally: verify_unchanged(snapshot)
    except BaseException as error:
        report['error'] = str(error)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        write_json(out/'qualification.json',report)
        destination = export(out)
        print('Git-visible results: '+str(destination),flush=True)
        print('Original lab evidence: '+str(out),flush=True)


if __name__ == '__main__': main()
