#!/usr/bin/env python3
"""B07 bounded optimization/canary qualification. Production adoption is explicit."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import traceback
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
DIRECTORY=ROOT/'distribution/rust-lab-b07-g1250'
PHASES={'ordinary':360,'canary-1':210,'canary-2':210,'continuation':180}


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def verify():
    from brass_ai.learning.contracts import require
    from brass_ai.step5.league import implementation_id
    manifest=json.loads((DIRECTORY/'manifest.json').read_text())
    require(manifest['implementation']==implementation_id(),'B07 checkout differs from sealed source')
    for name,expected in manifest['files'].items():
        path=(ROOT/name).resolve();require(path.is_relative_to(ROOT),'Unsafe B07 bundle path')
        data=path.read_bytes()
        if path.suffix in manifest['text']:data=data.replace(b'\r\n',b'\n')
        require(hashlib.sha256(data).hexdigest()==expected,'B07 file differs: '+name)
    reference=json.loads((DIRECTORY/'reference.json').read_text())
    old=ROOT/'distribution/rust-lab-b06-g1250/evidence'
    for label,name in [('B02','b02-qualification.json'),('B04.5','b045-qualification.json'),('B05','b05-qualification.json')]:
        path=old/name
        require(hashlib.sha256(path.read_bytes()).hexdigest()==reference['prerequisites'][label]['sha256'] and
                json.loads(path.read_text())['passed'],'Missing passing prerequisite: '+label)
    path=DIRECTORY/'evidence/b06-qualification.json'
    baseline=json.loads(path.read_text())
    require(hashlib.sha256(path.read_bytes()).hexdigest()==reference['b06']['sha256'] and baseline['passed'] and
            baseline['implementation']==reference['b07_source_review']['from_implementation'],
            'B06 baseline/source review differs')
    require(reference['b07_source_review']['to_implementation']==manifest['implementation'],'B07 source review differs')
    return manifest


def child(args):
    import torch
    from brass_ai.learning.contracts import require,read_json
    from brass_ai.step5.league import write_json
    from brass_ai.rust_backend.lab_snapshot import verify_unchanged
    from rust_lab_b06_metrics import measurement_preflight
    import rust_lab_b07_smoke as smoke
    old=module('b07_b06',ROOT/'scripts/qualify-rust-lab-b06.py')
    report={};snapshot=None
    try:
        reference=read_json(DIRECTORY/'reference.json')
        report['prerequisites']=dict(reference['prerequisites'],B06=reference['b06'])
        require(platform.system()=='Linux' and torch.cuda.is_available(),'Linux/CUDA lab required')
        lock=(args.run/'.active-run.json').read_text()
        require(lock==os.environ.get('BRASS_B07_PARENT_LOCK') and json.loads(lock)['pid']==os.getppid(),'Missing parent run lock')
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        report['environment']=measurement_preflight(args.device)
        write_json(args.out/(args.phase+'-environment.json'),report['environment'])
        snapshot=read_json(args.out/'snapshot.json');ordinary,teacher=old.verify_runtime(reference)
        if args.phase=='ordinary':report.update(smoke.ordinary(args,snapshot,ordinary,reference))
        elif args.phase.startswith('canary-'):
            report.update(smoke.canary(args,snapshot,teacher,old.INPUTS,int(args.phase[-1])))
        else:report.update(old.kernels(args,snapshot,reference))
    except BaseException as error:
        report.update(passed=False,error=str(error),traceback=traceback.format_exc());raise
    finally:
        try:
            if snapshot is not None:verify_unchanged(snapshot)
        except BaseException as error:
            report.update(passed=False,source_verification_error=str(error));raise
        finally:write_json(args.out/(args.phase+'-checks.json'),report)


def main():
    from brass_ai.learning.contracts import require,read_json,project_path
    from brass_ai.step5.league import write_json
    from brass_ai.rust_backend.lab_contract import pin
    from brass_ai.rust_backend.lab_snapshot import capture,verify_unchanged
    from brass_ai.recovery.retention import run_lock
    from rust_lab_results import export
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--device',default='cuda:0');p.add_argument('--mono',default='mono')
    p.add_argument('--fixtures-only',action='store_true')
    p.add_argument('--phase',choices=tuple(PHASES),help=argparse.SUPPRESS)
    p.add_argument('--out',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args();args.run=project_path(ROOT,args.run)
    require(args.device=='cuda:0','B07 uses the pinned logical learner cuda:0')
    if args.phase:return child(args)
    out=ROOT/'artifacts/rust-backend'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-g1250-b07-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    report=dict(milestone='B07',passed=False,lab_qualification='Pending',production_modified=False,
        production_adoption='Pending',next_normal_commit_restart='Pending',previous_qualifications_repeated=False)
    base=module('b07_base',ROOT/'scripts/qualify-rust-lab.py');active='bundle'
    env=base.python_environment();env['PYTHONPATH']=str(ROOT/'scripts')+os.pathsep+env['PYTHONPATH'];env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    try:
        report['implementation']=verify()['implementation']
        reference=read_json(DIRECTORY/'reference.json')
        report['prerequisites']=dict(reference['prerequisites'],B06=reference['b06'])
        active='environment'
        from rust_lab_b06_metrics import measurement_preflight
        report['environment']=measurement_preflight()
        active='fixtures'
        tests=['test_b07','test_b06','test_b05_retry','test_b03.Admission','test_b04','test_b045','test_result_export',
               'test_branch_scheduler','test_audit_retention']
        base.command([sys.executable,'-m','unittest',*tests,'-q'],cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=''),log=out/'fixtures.log',timeout=180)
        if args.fixtures_only:
            report.update(passed=True,scope='CPU fixtures only; B07 lab and adoption Pending');return
        active='runtime'
        require(platform.system()=='Linux','B07 benchmark/canary requires Linux/CUDA')
        module('b07_runtime',ROOT/'scripts/qualify-rust-lab-b06.py').verify_runtime(reference)
        with run_lock(args.run):
            active='snapshot';snapshot,_,_=capture(ROOT,args.run,devices=[args.device])
            require(snapshot['generation']==reference['generation'] and snapshot['checkpoint']['sha256']==reference['checkpoint_sha256']
                    and snapshot['config_sha256']==reference['config_sha256'],'Preserve the qualified g1250 boundary; refresh gates for a later checkpoint')
            write_json(out/'snapshot.json',snapshot)
            report.update(generation=snapshot['generation'],checkpoint_sha256=snapshot['checkpoint']['sha256'],config_sha256=snapshot['config_sha256'])
            try:
                for phase,timeout in PHASES.items():
                    active=phase
                    command=[sys.executable,str(Path(__file__)),'--phase',phase,'--run',str(args.run),'--out',str(out),'--device',args.device,'--mono',args.mono]
                    base.command(command,cwd=ROOT,env=dict(env,BRASS_B07_PARENT_LOCK=(args.run/'.active-run.json').read_text()),
                                 log=out/(phase+'.log'),timeout=timeout)
                    detail=read_json(out/(phase+'-checks.json'));require(detail['passed'],'B07 '+phase+' failed/deferred')
                report['ordinary']=read_json(out/'ordinary-checks.json')
                report['canary']=dict(passed=True,passes=2,tables=4,teacher_mode='shadow',blend_weight=0,
                    receipts=[pin(out/(phase+'-checks.json')) for phase in ('canary-1','canary-2')])
                report['continuation']=read_json(out/'continuation-checks.json')
                report['ordinary_build']=pin(ROOT/'distribution/rust-lab-b06-g1250/evidence/ordinary-build.json')
                useful=report['ordinary']['performance']['useful_accepted_work_gain']
                report.update(passed=True,lab_qualification='Passed bounded canary/correctness scope',
                    adoption_ready=useful,performance_acceptance='Useful matched gain' if useful else 'Pending further optimization; adoption blocked',
                    whole_iteration_speedup=None,full_scale_teacher_enablement=False)
            finally:verify_unchanged(snapshot)
    except BaseException as error:
        report.update(passed=False,error=str(error),failed_phase=active,traceback=traceback.format_exc())
        if (out/(active+'-checks.json')).exists():report['failure']=read_json(out/(active+'-checks.json'))
        raise
    finally:
        write_json(out/'qualification.json',report)
        print('Git-visible results: '+str(export(out)),flush=True)
        print('Original lab evidence: '+str(out),flush=True)


if __name__=='__main__':main()
