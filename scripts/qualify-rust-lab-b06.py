#!/usr/bin/env python3
"""B06 only: bounded matched collection and separately measured teacher overhead."""
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
DIRECTORY=ROOT/'distribution/rust-lab-b06-g1250'
INPUTS=DIRECTORY/'evidence'
TESTS=['test_b06','test_b05_retry','test_b03.Admission',
       'test_b04.B04.test_gpu_wait_cancellation_drains_slot_and_does_not_expose_logits',
       'test_b04.B04.test_wrong_model_reply_retires_owner',
       'test_b04.B04.test_cancel_between_native_actions_discards_unpublished_labels','test_result_export']
LIMITS={'ordinary':360,'teacher':150,'kernels':180}


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--mono',default='mono')
    parser.add_argument('--fixtures-only',action='store_true')
    parser.add_argument('--phase',choices=tuple(LIMITS),help=argparse.SUPPRESS)
    parser.add_argument('--out',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if not args.fixtures_only and args.device!='cuda:0':parser.error('B06 uses the pinned logical learner cuda:0')
    return args


def verify_bundle():
    from brass_ai.learning.contracts import require,read_json
    from brass_ai.step5.league import implementation_id
    manifest=read_json(DIRECTORY/'manifest.json')
    require(manifest['implementation']==implementation_id(),'B06 checkout differs from reviewed implementation')
    for name,expected in manifest['files'].items():
        path=(ROOT/name).resolve();require(path.is_relative_to(ROOT),'B06 bundle path escaped')
        data=path.read_bytes()
        if path.suffix in manifest['canonical_text_suffixes']:data=data.replace(b'\r\n',b'\n')
        require(hashlib.sha256(data).hexdigest()==expected,'B06 bundle file differs: '+name)
    return manifest


def verify_runtime(reference):
    """Reuse exact original binaries, not old Python/source admission rules."""
    from brass_ai.learning.contracts import require,read_json
    from brass_ai.rust_backend.lab_contract import pin,SOURCE
    from brass_ai.rust_backend.lab_teacher_overlay import source_fingerprint
    ordinary=read_json(INPUTS/'ordinary-build.json');teacher=read_json(INPUTS/'teacher-build.json')
    require(ordinary['shared_core']==SOURCE and ordinary['search_enabled'] is False and
            ordinary['search_method']=='search_v1' and ordinary['v205_enabled'] is False,'Original collector scope differs')
    require(teacher['producer']==source_fingerprint(),'Original teacher native source differs')
    for name,expected in reference['ordinary_native_source_review'].items():
        require(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected,
                'Ordinary native source differs from qualified binary: '+name)
    pins=[ordinary['collector'],teacher['binary']]
    for build in (ordinary,teacher):pins += [build['bootstrap']['binary'],*build['bootstrap']['dependencies']]
    for item in pins:
        require(Path(item['path']).is_file() and pin(item['path'])==item,
            'Restore original qualified lab runtime (do not substitute/rebuild silently): '+item['path'])
    return ordinary,teacher


def child_command(args,phase,out):
    return [sys.executable,str(Path(__file__)),'--phase',phase,'--run',str(args.run),
            '--out',str(out),'--device',args.device,'--mono',args.mono]


def kernels(args,snapshot,reference):
    """Current g1250 model/Adams/PPG recipe; explicitly bounded authored inputs."""
    from unittest.mock import patch
    from brass_ai.rust_backend import lab_continuation as continuation
    from brass_ai.rust_backend.lab_b05_smoke import ordinary_step
    from brass_ai.recovery import updates,value_phase
    from brass_ai.step5.league import write_json
    from rust_lab_b06_metrics import Meter
    incoming=continuation.preflight(ROOT,args.run,generation=1250,
        checkpoint_sha256=reference['checkpoint_sha256'],devices=[args.device])
    try:
        with Meter(args.device) as meter:
            with meter.phase('full_fork_and_restore'):
                fork=continuation.create_fork(ROOT,args.out/'continuation',incoming)
                restored=continuation.restore(fork,device=args.device,allow_cuda_topology_change=True)
            write_json(args.out/'restoration.json',restored.report)
            original_ppo,original_fit=updates.ppo_update,value_phase.fit
            def ppo(*a,**kw):
                with meter.phase('ordinary_ppo'):return original_ppo(*a,**kw)
            def ppg(*a,**kw):
                with meter.phase('ordinary_ppg'):return original_fit(*a,**kw)
            with patch.object(updates,'ppo_update',ppo),patch.object(value_phase,'fit',ppg),meter.phase('component_update_commit_reload'):
                result=ordinary_step(restored,snapshot['state']['signature']['config'],args.out)
            return dict(passed=True,measurement=meter.report(),result=result,new_games=0,
                scope='Authored 145 actor / 64 critic / 16 validation rows, actual g1250 recipe/model/Adams. Kernel phase sample, not full-iteration throughput')
    finally:continuation.verify_dependencies(ROOT,incoming)


def run_child(args,reference):
    from brass_ai.learning.contracts import require,read_json
    from brass_ai.step5.league import write_json
    from brass_ai.rust_backend.lab_snapshot import verify_unchanged
    report={};snapshot=None
    try:
        import torch
        from rust_lab_b06_metrics import measurement_preflight
        require(platform.system()=='Linux' and torch.cuda.is_available(),'Linux/CUDA lab required')
        report['environment']=measurement_preflight()
        lock=(args.run/'.active-run.json').read_text()
        require(lock==os.environ.get('BRASS_B06_PARENT_LOCK') and json.loads(lock)['pid']==os.getppid(),'Missing parent run lock')
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        snapshot=read_json(args.out/'snapshot.json');ordinary,teacher=verify_runtime(reference)
        if args.phase=='ordinary':
            from rust_lab_b06_ordinary import run
            report=run(args,snapshot,ordinary,reference)
        elif args.phase=='teacher':
            from rust_lab_b06_teacher import run
            report=run(args,snapshot,teacher,INPUTS)
        else:report=kernels(args,snapshot,reference)
    except BaseException as error:
        report.update(passed=False,error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
        raise
    finally:
        try:
            if snapshot is not None:verify_unchanged(snapshot)
        except BaseException as error:
            report.update(passed=False,source_verification_error=str(error))
            raise
        finally:write_json(args.out/(args.phase+'-checks.json'),report)


def main():
    from brass_ai.learning.contracts import require,read_json,project_path
    from brass_ai.step5.league import write_json
    from brass_ai.recovery.retention import run_lock
    from brass_ai.rust_backend.lab_snapshot import capture,verify_unchanged
    from rust_lab_results import export
    args=arguments();args.run=project_path(ROOT,args.run)
    if args.phase:
        run_child(args,read_json(DIRECTORY/'reference.json'));return
    out=ROOT/'artifacts/rust-backend'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-g1250-b06-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    report=dict(milestone='B06',passed=False,lab_qualification='Pending',production_modified=False,
                previous_qualifications_repeated=False,whole_iteration_speedup=None,new_teacher_tables=0)
    spec=importlib.util.spec_from_file_location('b06_base',ROOT/'scripts/qualify-rust-lab.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    env=base.python_environment();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    env['PYTHONPATH']=str(ROOT/'scripts')+os.pathsep+env['PYTHONPATH']
    active='bundle'
    try:
        report['implementation']=verify_bundle()['implementation'];reference=read_json(DIRECTORY/'reference.json')
        report['prerequisites']=reference['prerequisites']
        active='environment'
        from rust_lab_b06_metrics import measurement_preflight
        report['environment']=measurement_preflight()
        write_json(out/'environment.json',report['environment'])
        active='fixtures'
        base.command([sys.executable,'-m','unittest',*TESTS,'-q'],cwd=ROOT,
            env=dict(env,CUDA_VISIBLE_DEVICES=''),log=out/'fixtures.log',timeout=120)
        if args.fixtures_only:
            report.update(passed=True,scope='CPU fixtures only; Linux/CUDA B06 Pending');return
        active='runtime'
        require(platform.system()=='Linux','Full B06 requires Linux/CUDA; use --fixtures-only locally')
        verify_runtime(reference)
        active='snapshot'
        with run_lock(args.run):
            snapshot,_,_=capture(ROOT,args.run,devices=[args.device])
            require(snapshot['generation']==1250 and snapshot['checkpoint']['sha256']==reference['checkpoint_sha256']
                    and snapshot['model_id']==reference['model_id'] and snapshot['config_sha256']==reference['config_sha256'],
                    'B06 needs preserved exact g1250 state/config; never rewind or relabel the live run')
            write_json(out/'snapshot.json',snapshot)
            try:
                for phase,timeout in LIMITS.items():
                    active=phase;print('B06: '+phase,flush=True)
                    base.command(child_command(args,phase,out),cwd=ROOT,
                        env=dict(env,BRASS_B06_PARENT_LOCK=(args.run/'.active-run.json').read_text()),
                        log=out/(phase+'.log'),timeout=timeout)
                    detail=read_json(out/(phase+'-checks.json'));require(detail['passed'],'B06 '+phase+' failed')
                    report[phase]=dict(passed=True,path=str(out/(phase+'-checks.json')))
                ordinary=read_json(out/'ordinary-checks.json')
                report.update(passed=True,lab_qualification='Passed for bounded matched measurement/correctness scope',
                    performance=ordinary['performance'],natural_branch_observed=ordinary['natural_branch_observed'],
                    new_ordinary_parent_games=2*ordinary['matched']['work']['families'],
                    new_ordinary_branch_games=2*ordinary['matched']['work']['branches'],
                    performance_acceptance='Review measured phase costs and coverage before B07; no automatic production adoption')
            finally:verify_unchanged(snapshot)
    except BaseException as error:
        report.update(passed=False,error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc(),failed_phase=active)
        if active and (out/(active+'-checks.json')).is_file():
            detail=read_json(out/(active+'-checks.json'))
            if detail.get('passed') is False:report['failure']=detail
        raise
    finally:
        write_json(out/'qualification.json',report)
        print('Git-visible results: '+str(export(out)),flush=True)
        print('Original lab evidence: '+str(out),flush=True)


if __name__=='__main__':main()
