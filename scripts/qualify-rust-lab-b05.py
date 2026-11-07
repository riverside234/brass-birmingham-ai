#!/usr/bin/env python3
"""B05 only: fixed-data updates and restore. No games or production adoption."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
DIRECTORY=ROOT/'distribution/rust-lab-b05-g1250'
TESTS=['test_b05','test_b045','test_b025.PPO','test_b025.Buffer','test_b025.Resume','test_b025.Lifecycle',
       'test_b03.Admission.test_native_reference_targets_features_and_head_metadata_match',
       'test_b03.Continuation.test_production_source_commit_crash_chain_is_reviewed',
       'test_b03.Continuation.test_reference_and_native_critic_ages_coexist_without_relabelling',
       'test_ppg_g210.Restored.test_32_step_kl_checks_pending_loss_state_and_restores_adams_and_companion',
       'test_ppg_g210.Restored.test_kl_cadence_spans_epochs_and_checks_short_tails',
       'test_ppg_g210.Restored.test_loss_rejection_restores_last_interval_and_no_SE_tolerance',
       'test_training_g1247','test_result_export']


def verify_bundle():
    from brass_ai.learning.contracts import require
    from brass_ai.step5.league import implementation_id
    manifest=json.loads((DIRECTORY/'manifest.json').read_text())
    require(manifest['implementation']==implementation_id(),'B05 checkout differs from the reviewed bundle')
    for name,expected in manifest['files'].items():
        p=(ROOT/name).resolve();require(p.is_relative_to(ROOT),'B05 bundle path escaped')
        value=p.read_bytes()
        if p.suffix in manifest['canonical_text_suffixes']:value=value.replace(b'\r\n',b'\n')
        require(hashlib.sha256(value).hexdigest()==expected,'B05 bundle file differs: '+name)
    return manifest


def main():
    from brass_ai.learning.contracts import require,project_path
    from brass_ai.step5.league import write_json
    from brass_ai.recovery.retention import run_lock
    from rust_lab_results import export
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--device',default='cuda:0');p.add_argument('--fixtures-only',action='store_true')
    p.add_argument('--allow-cuda-topology-change',action='store_true')
    p.add_argument('--lab-child',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--out',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args();args.run=project_path(ROOT,args.run)
    if args.lab_child:
        reference=json.loads((DIRECTORY/'reference.json').read_text())
        import torch
        require(platform.system()=='Linux' and torch.device(args.device).type=='cuda' and torch.cuda.is_available(),'Linux/CUDA lab required')
        lock=(args.run/'.active-run.json').read_text()
        require(lock==os.environ.get('BRASS_B05_PARENT_LOCK') and json.loads(lock)['pid']==os.getppid(),'Missing parent run lock')
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from brass_ai.rust_backend.lab_b05_smoke import run
        write_json(args.out/'lab-checks.json',run(args,reference));return
    token=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-g1250-b05-'+uuid.uuid4().hex[:8]
    out=ROOT/'artifacts/rust-backend'/token;out.mkdir(parents=True,exist_ok=False)
    report=dict(milestone='B05',passed=False,lab_qualification='Pending',production_modified=False,new_games=0,
                previous_qualifications_repeated=False)
    spec=importlib.util.spec_from_file_location('b05_base',ROOT/'scripts/qualify-rust-lab.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    env=base.python_environment();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    try:
        reference=json.loads((DIRECTORY/'reference.json').read_text())
        report['prerequisites']=reference['prerequisites']
        report['implementation']=verify_bundle()['implementation']
        base.command([sys.executable,'-m','unittest',*TESTS,'-q'],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),
                     log=out/'fixtures.log',timeout=180)
        if args.fixtures_only:
            report.update(passed=True,scope='CPU fixtures only; Linux/CUDA qualification Pending');return
        require(platform.system()=='Linux','Run full B05 on the lab; --fixtures-only is available locally')
        with run_lock(args.run):
            command=[sys.executable,str(Path(__file__)),'--lab-child','--run',str(args.run),'--out',str(out),'--device',args.device]
            if args.allow_cuda_topology_change:command.append('--allow-cuda-topology-change')
            base.command(command,cwd=ROOT,env=dict(env,BRASS_B05_PARENT_LOCK=(args.run/'.active-run.json').read_text()),
                         log=out/'lab-checks.log',timeout=180)
            report.update(json.loads((out/'lab-checks.json').read_text()))
    except BaseException as error:
        report.update(error=str(error),traceback=traceback.format_exc());raise
    finally:
        write_json(out/'qualification.json',report)
        print('Git-visible results: '+str(export(out)),flush=True)
        print('Original lab evidence: '+str(out),flush=True)


if __name__=='__main__':main()
