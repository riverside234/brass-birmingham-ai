#!/usr/bin/env python3
"""Short B03 fixtures, sealed B02 replay admission and full joint7 restoration."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
TEST_PATHS=('src','tests/R2','tests/Recovery','tests/FeatureEngineering','tests/RustBackendLab')


def text_sha(path):return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def verify_transfer(supplement=None):
    if supplement is not None:
        from brass_ai.rust_backend.lab_transfer import verify
        return verify('b03',supplement)
    from brass_ai.learning.contracts import require
    from brass_ai.step5.league import implementation_id
    manifest=ROOT/'distribution/rust-lab-b03/transfer-manifest.json'
    receipt=json.loads(manifest.read_text(encoding='utf-8'))
    require(receipt['contract']=='brass-lab-b03-transfer/1.0.0','Unknown B03 transfer')
    for name,expected in receipt['files'].items():
        path=(ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and text_sha(path)==expected,'Changed B03 file: '+name)
    require(implementation_id()==receipt['implementation'],'B03 checkout differs from the reviewed source; update the checkout, do not edit the review hash')
    return dict(manifest_sha256=text_sha(manifest),implementation=receipt['implementation'],files=len(receipt['files']))


def lab_check(args,out):
    import torch
    from brass_ai.learning.contracts import require
    from brass_ai.step5.league import write_json
    from brass_ai.rust_backend import lab_continuation as continuation,lab_b03_smoke as replay
    from brass_ai.rust_backend.lab_snapshot import verify_unchanged
    from brass_ai.rust_backend.lab_contract import pin
    parent = json.loads(os.environ.get('BRASS_B03_PARENT_LOCK', '{}'))
    require(parent.get('pid') == os.getppid() and
            json.loads((args.run / '.active-run.json').read_text(encoding='utf-8')) == parent,
            'Lab child requires its qualification parent to hold the production run lock')
    require(platform.system()=='Linux' and torch.device(args.device).type=='cuda' and torch.cuda.is_available(),
            'Full B03 qualification requires the lab Linux/CUDA environment; use --fixtures-only for CPU development')
    torch.set_num_threads(2)
    snapshot,jobs,families,producer=replay.entry(args.b02,args.run,args.device)
    incoming=continuation.preflight(ROOT,args.run,generation=args.expected_generation,
        checkpoint_sha256=args.checkpoint_sha256,devices=[args.device])
    require(snapshot['state_file']==incoming['snapshot']['state_file'] and snapshot['checkpoint']==incoming['snapshot']['checkpoint'],
            'B02 and B03 bind different committed run states')
    write_json(out/'continuation-preflight.json',incoming)
    try:
        fork=continuation.create_fork(ROOT,out/'continuation',incoming)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            restored=continuation.restore(fork,device=args.device,allow_cuda_topology_change=args.allow_cuda_topology_change)
            write_json(out/'restoration.json',restored.report)
        admitted=replay.run(snapshot,jobs,families,producer,out/'admitted-replay')
        write_json(out/'replay-admission.json',admitted)
        il=continuation.check_completed_il(ROOT,args.il_run) if args.il_run else dict(status='Not requested',scope='Optional fresh IL handoff; separate from established joint7 continuation')
        write_json(out/'il-handoff.json',il)
        continuation.verify_dependencies(ROOT,incoming);verify_unchanged(incoming['snapshot'])
        return dict(passed=True,preflight=pin(out/'continuation-preflight.json'),restoration=pin(out/'restoration.json'),
            replay=pin(out/'replay-admission.json'),diagnostic_fork=pin(fork.directory/'diagnostic-fork.json'),
            il=pin(out/'il-handoff.json'),new_games=0,optimizer_steps=0,source_run_modified=False,
            next='B04 teacher exports and B05 fixed-data update/commit; production adoption remains B07')
    finally:
        continuation.verify_dependencies(ROOT,incoming);verify_unchanged(incoming['snapshot'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path);p.add_argument('--b02',type=Path)
    p.add_argument('--expected-generation',type=int);p.add_argument('--checkpoint-sha256')
    p.add_argument('--device',default='cuda:0');p.add_argument('--il-run',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--fixtures-only',action='store_true')
    p.add_argument('--allow-cuda-topology-change',action='store_true',help='Existing explicit logical-GPU RNG compatibility route; recorded in the result')
    p.add_argument('--lab-child',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--transfer-manifest',type=Path,help='Explicit sealed B04 supplement')
    a=p.parse_args()
    from brass_ai.learning.contracts import require,project_path
    from brass_ai.recovery.retention import owned,run_lock
    from brass_ai.step5.league import write_json
    from brass_ai.rust_backend.lab_admission import hash64
    if a.lab_child:
        value=lab_check(a,a.out);write_json(a.out/'lab-checks.json',value);return
    require(a.fixtures_only or (a.run is not None and a.b02 is not None and type(a.expected_generation) is int
        and a.expected_generation>0 and hash64(a.checkpoint_sha256)),
        'Supply --run, --b02, --expected-generation and --checkpoint-sha256, or --fixtures-only')
    require(not a.fixtures_only or not any((a.run,a.b02,a.expected_generation,a.checkpoint_sha256,a.il_run)),
            'Fixture mode cannot claim a lab run qualification')
    out=project_path(ROOT,a.out or ('artifacts/rust-backend/'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-b03'))
    require(out.is_relative_to(ROOT/'artifacts/rust-backend') and not out.exists(),'Use a new evidence directory under artifacts/rust-backend')
    if a.run:
        _,a.run=owned(ROOT,a.run)
        require(not out.is_relative_to(a.run) and not a.run.is_relative_to(out),'Evidence cannot overlap production')
    out.mkdir(parents=True);start=time.monotonic()
    report=dict(contract='brass-lab-b03/1.0.0',milestone='B03',passed=False,method='search_v1',v205_enabled=False,
        lab_qualification='Pending',scope='CPU fixture development' if a.fixtures_only else 'Replay admission and full-state restoration; no games or training steps')
    try:
        report['transfer']=verify_transfer(a.transfer_manifest)
        env=dict(os.environ,PYTHONPATH=os.pathsep.join(str(ROOT/n) for n in TEST_PATHS),CUBLAS_WORKSPACE_CONFIG=':4096:8')
        print('B03: bounded CPU fixtures',flush=True)
        with (out/'fixtures.log').open('wb') as log:
            test=subprocess.run([sys.executable,'-m','unittest','test_b03','-v'],cwd=ROOT,
                env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,timeout=180)
        require(test.returncode==0,'B03 fixtures failed; see fixtures.log')
        report['fixtures']=dict(passed=True,sha256=text_sha(out/'fixtures.log'))
        if not a.fixtures_only:
            # Parent owns the run lock until child/device work exits, including timeout cleanup.
            module_spec=importlib.util.spec_from_file_location('b03_bounded',ROOT/'scripts/qualify-rust-lab.py')
            bounded=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(bounded)
            command=[sys.executable,str(Path(__file__)), '--lab-child','--run',str(a.run),'--b02',str(a.b02.resolve()),
                     '--expected-generation',str(a.expected_generation),'--checkpoint-sha256',a.checkpoint_sha256,
                     '--device',a.device,'--out',str(out)]
            if a.il_run:command+=['--il-run',str(a.il_run)]
            if a.allow_cuda_topology_change:command+=['--allow-cuda-topology-change']
            with run_lock(a.run):
                print('B03: existing B02 replay and isolated full-state restoration',flush=True)
                child_env=dict(env,BRASS_B03_PARENT_LOCK=(a.run/'.active-run.json').read_text(encoding='utf-8'))
                report['command']=bounded.command(command,cwd=ROOT,env=child_env,log=out/'lab-checks.log',timeout=300)
                report['lab']=json.loads((out/'lab-checks.json').read_text(encoding='utf-8'))
                require(report['lab']['passed'] is True,'Lab B03 failed')
            report['lab_qualification']='Passed for this exact committed checkpoint'
        report['passed']=True
    except Exception as e:
        report['error']=str(e);raise
    finally:
        report['seconds']=time.monotonic()-start;write_json(out/'qualification.json',report)
        print(str(out/'qualification.json'),flush=True)


if __name__=='__main__':main()
