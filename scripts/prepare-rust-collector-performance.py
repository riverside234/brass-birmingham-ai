#!/usr/bin/env python3
"""Build only the ordinary transport upgrade; retain qualified teacher/search."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def prepare(args,evidence):
    from brass_ai.bank import read
    from brass_ai.learning.contracts import require,project_path
    from brass_ai.recovery.retention import owned,run_lock
    from brass_ai.rust_backend.lab_backend import validate_state,check_build
    from brass_ai.rust_backend.lab_contract import pin
    from brass_ai.rust_backend.lab_perf_runtime import CONTRACT,fingerprint,resolve,predecessor
    from brass_ai.step5.league import write_json
    import rust_lab_b07_retry as retry
    root,run=owned(ROOT,args.run)
    with run_lock(run):
        state=read(run/'state.json');backend=validate_state(state)
        require(state['iteration']==args.expected_generation and backend['mode']=='rust','Stale generation or backend')
        checkpoint=pin(project_path(root,state['history'][-1]['path']))
        require(state['history'][-1]['generation']==state['iteration'] and checkpoint['sha256']==state['history'][-1]['sha256'],
                'Committed collector checkpoint changed')
        if state.get('signature',{}).get('config',{}).get('teacher_phase',{}).get('mode')=='shadow':
            from brass_ai.rust_backend.lab_shadow import teacher_build
            teacher_build(state)
        original=check_build(backend['build']);source=fingerprint()
        selected=root/'artifacts/rust-backend/collector-performance/build.json'
        selected_pin=pin(selected) if selected.exists() else None
        if selected.is_file() and read(selected).get('ordinary_source')==source:
            resolve(root,original)
            print('Reused verified ordinary runtime: '+str(selected),flush=True)
            return dict(reused=True,runtime=pin(selected),generation=state['iteration'])
        previous=[]
        if selected_pin is not None:
            old=predecessor(selected_pin,original)
            retained=evidence/'previous-ordinary-build.json';retained.write_bytes(selected.read_bytes())
            require(pin(retained)['sha256']==selected_pin['sha256'],'Previous runtime changed during retention')
            previous=[pin(retained),*old.get('previous_runtimes',[])]
            require(len(previous)<=8,'Previous collector history exceeds bound')
    out=evidence/'runtime'
    # Existing builder extracts the sealed B01 core and original bootstrap;
    # search/teacher binaries are never rebuilt or overwritten here.
    # A newly compiled binary must not inherit the old "native bytes unchanged"
    # source_reuse receipt. The legacy verifier would otherwise take its old
    # teacher-only reuse branch and reject this intentional ordinary upgrade.
    compile_inputs={k:v for k,v in original.items() if k not in ('source_reuse','_receipt_sha256')}
    built=retry.build_ordinary(out,compile_inputs,offline=args.offline)
    env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/x) for x in
        ('src','scripts','tests/R2','tests/Recovery','tests/FeatureEngineering','tests/RustBackendLab')),
        BRASS_B07_COLLECTOR=built['collector']['path'],
        BRASS_B07_FIXTURES=str(Path(original['b01']['path']).parent/'inputs/fixtures.json'))
    with (out/'performance-wire.log').open('w') as log:
        subprocess.run([sys.executable,'-B','-m','unittest','test_collector_performance.NativeWire',
                        'test_collector_performance_tail.NativeTail','-v'],
                       cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
    with run_lock(run):
        current=read(run/'state.json')
        require(current['iteration']==args.expected_generation and current['history'][-1]==state['history'][-1] and fingerprint()==source and
                pin(checkpoint['path'])==checkpoint,'Run or source changed during build')
        require((pin(selected) if selected.exists() else None)==selected_pin,'Collector runtime changed during build')
        result=dict(built,contract=CONTRACT,ordinary_source=source,parent_build_sha256=original['_receipt_sha256'],
                    performance_wire='ordinary-collector-fast/1',completion_export='ordinary-export-pages/1',
                    performance_tests=pin(out/'performance-wire.log'),previous_runtimes=previous)
        destination=ROOT/'artifacts/rust-backend/collector-performance/build.json'
        write_json(destination,result);resolve(ROOT,original)
    print('Prepared ordinary runtime: '+str(destination),flush=True)
    print('Qualified teacher binary, shadow share target and checkpoint are unchanged.',flush=True)
    return dict(reused=False,runtime=pin(destination),generation=state['iteration'],checkpoint=state['history'][-1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--expected-generation',required=True,type=int)
    p.add_argument('--offline',action='store_true')
    args=p.parse_args()
    from brass_ai.step5.league import write_json
    from rust_lab_results import export
    evidence=ROOT/'artifacts/rust-backend'/('collector-performance-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8])
    evidence.mkdir(parents=True,exist_ok=False)
    report=dict(passed=False,scope='Ordinary wire build/short regressions only; no training or teacher rebuild',
                expected_generation=args.expected_generation)
    try:
        report.update(prepare(args,evidence),passed=True)
    except BaseException as error:
        report.update(error=str(error),traceback=traceback.format_exc());raise
    finally:
        write_json(evidence/'qualification.json',report)
        try:print('Git-visible results: '+str(export(evidence)),flush=True)
        except Exception as error:
            if report['passed']:raise
            print('Result export also failed: '+str(error),file=sys.stderr,flush=True)
        print('Original lab evidence: '+str(evidence),flush=True)


if __name__=='__main__':main()
