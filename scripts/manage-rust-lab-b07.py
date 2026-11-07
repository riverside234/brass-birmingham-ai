#!/usr/bin/env python3
"""Explicit B07 adoption/reference switch and next normal commit verification."""
import argparse
from datetime import datetime,timezone
import importlib.util
from pathlib import Path
import sys
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import require,project_path
    from brass_ai.step5.league import write_json
    from brass_ai.recovery.retention import run_lock
    from brass_ai.rust_backend import lab_backend as backend,lab_continuation as continuation
    from rust_lab_results import export
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('adopt','reference','verify-next','verify-rollback'))
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--qualification',type=Path)
    p.add_argument('--expected-generation',type=int,required=True)
    args=p.parse_args();run=project_path(ROOT,args.run)
    out=ROOT/'artifacts/rust-backend'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-b07-'+args.action+'-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True);report=dict(action=args.action,passed=False,training_advanced=False)
    try:
        spec=importlib.util.spec_from_file_location('b07_manager_verify',ROOT/'scripts/qualify-rust-lab-b07.py')
        qualifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(qualifier);qualifier.verify()
        with run_lock(run):
            state=read(run/'state.json')
            require(state['iteration']==args.expected_generation,'Committed generation changed; inspect before acting')
            if args.action=='adopt':
                require(args.qualification is not None,'Supply original passing B07 qualification.json')
                report.update(backend.adopt(ROOT,run,project_path(ROOT,args.qualification),out/'rollback'))
            elif args.action=='reference':report.update(backend.reference_switch(ROOT,run))
            elif args.action=='verify-rollback':
                value=backend.validate_state(state,check_runtime=False)
                require(value is not None,'No saved backend rollback point')
                fork=backend.rollback_fork(ROOT,value)
                restored=continuation.restore(fork,device='cpu',allow_cuda_topology_change=True)
                report.update(rollback='Passed full saved checkpoint/optimizer/dependency restore',
                    restoration=restored.report,production_modified=False)
            else:
                from brass_ai.learning.checkpoint import load_checkpoint
                value=backend.validate_state(state);require(value and value['mode']=='rust','Rust backend is not active')
                require(state['iteration']==value['adopted_generation']+1,'Verify immediately after the next normal G-to-G+1 commit')
                entry=state['history'][-1];path=project_path(ROOT,entry['path']);require(sha(path)==entry['sha256'],'Checkpoint changed')
                _,raw=load_checkpoint(path);backend.validate_state(state,raw)
                require(raw['training'].get('native_backend')==value,'Backend was not persisted in the next checkpoint')
                iteration=read(run/f"iteration-{value['adopted_generation']:04d}"/'report.json')
                require(iteration['native_backend']['mode']=='rust' and
                        iteration['native_backend']['collection']['native_families']>0,'Next iteration did not use the native collector')
                incoming=continuation.preflight(ROOT,run,generation=state['iteration'],checkpoint_sha256=entry['sha256'])
                fork=continuation.create_fork(ROOT,out/'restart',incoming)
                restored=continuation.restore(fork,device='cpu',allow_cuda_topology_change=True)
                continuation.verify_dependencies(ROOT,incoming)
                report.update(next_normal_commit_restart='Passed',generation=state['iteration'],
                    restoration=restored.report,production_modified=False)
            report['passed']=True
    except BaseException as error:
        report.update(error=str(error),traceback=traceback.format_exc());raise
    finally:
        write_json(out/'qualification.json',report)
        print('Git-visible results: '+str(export(out)),flush=True)
        print('Original lab evidence: '+str(out),flush=True)


if __name__=='__main__':main()
