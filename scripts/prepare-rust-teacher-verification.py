#!/usr/bin/env python3
"""Prepare only the teacher adapter's unbounded verification handshake."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def prepare(args,evidence):
    from brass_ai.bank import read
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.recovery.retention import owned,run_lock
    from brass_ai.rust_backend.lab_backend import validate_state
    from brass_ai.rust_backend.lab_contract import pin
    from brass_ai.rust_backend.lab_shadow import teacher_build
    from brass_ai.rust_backend.lab_teacher_runtime import POLICY,SELECTOR,fingerprint,resolve
    from brass_ai.step5.league import write_json
    import rust_lab_b07_retry as retry
    root,run=owned(ROOT,args.run);selected=root/SELECTOR
    with run_lock(run):
        state=read(run/'state.json');backend=validate_state(state)
        require(backend is not None and backend['mode']=='rust','Teacher preparation requires the adopted Rust backend')
        require(state['iteration']==args.expected_generation,'Stale teacher preparation generation')
        checkpoint=state['history'][-1]
        require(checkpoint['generation']==state['iteration'] and
            pin(project_path(root,checkpoint['path']))['sha256']==checkpoint['sha256'], 'Committed checkpoint changed')
        original=teacher_build(state,prepared=False);source=fingerprint()
        previous=pin(selected) if selected.exists() else None
        if selected.exists() and read(selected).get('verification_source')==source and read(selected).get('verification_policy')==POLICY:
            result=resolve(root,original)
            print('Reused verified teacher adapter: '+str(selected),flush=True)
            return dict(reused=True,runtime=result['receipt'],generation=state['iteration'])
        if selected.exists():
            retained=evidence/'previous-teacher-runtime.json';retained.write_bytes(selected.read_bytes())
            require(pin(retained)['sha256']==previous['sha256'],'Previous teacher selector changed')
    built=retry.build_teacher(evidence/'runtime',original,offline=args.offline)
    with run_lock(run):
        current=read(run/'state.json')
        require(current['iteration']==args.expected_generation and current['history'][-1]==checkpoint and
            fingerprint()==source and pin(project_path(root,checkpoint['path']))['sha256']==checkpoint['sha256'],
            'Run/checkpoint/source changed during teacher build')
        require((pin(selected) if selected.exists() else None)==previous,'Teacher selector changed during build')
        write_json(selected,dict(built,verification_policy=POLICY,verification_source=source,
                                original_teacher_receipt=original['receipt']))
        result=resolve(root,original)
    return dict(reused=False,runtime=result['receipt'],generation=state['iteration'],
                scope='Uncapped post-search handoff/verification; search core/quota/selection and checkpoint unchanged')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--expected-generation',type=int,required=True);p.add_argument('--offline',action='store_true')
    args=p.parse_args()
    from brass_ai.step5.league import write_json
    from rust_lab_results import export
    out=ROOT/'artifacts/rust-backend'/('teacher-verification-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True);report=dict(passed=False)
    try:report.update(prepare(args,out),passed=True)
    except BaseException as error:
        report.update(error=str(error),traceback=traceback.format_exc());raise
    finally:
        write_json(out/'qualification.json',report)
        print('Git-visible results: '+str(export(out)),flush=True)
        print('Original lab evidence: '+str(out),flush=True)


if __name__=='__main__':main()
