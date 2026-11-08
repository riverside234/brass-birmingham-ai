#!/usr/bin/env python3
"""Continue the same Rust run with branching off and 32-table shadow gradients."""
import argparse
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def run_training(argv,log):
    """Tee the real child traceback to exported evidence; retain Ctrl-C cleanup."""
    import os,signal
    with Path(log).open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(argv,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf-8',errors='replace',bufsize=1,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            for line in process.stdout:
                stream.write(line);stream.flush();print(line,end='',flush=True)
            code=process.wait()
            if code:raise subprocess.CalledProcessError(code,argv)
        except BaseException:
            if process.poll() is None:
                import psutil
                try:children=psutil.Process(process.pid).children(recursive=True)
                except psutil.NoSuchProcess:children=[]
                try:
                    if os.name=='posix':process.send_signal(signal.SIGINT)
                    else:process.terminate()
                    process.wait(timeout=15)
                except (OSError,subprocess.TimeoutExpired):
                    for child in reversed(children):
                        try:child.kill()
                        except psutil.NoSuchProcess:pass
                    if process.poll() is None:process.kill()
                    process.wait()
            raise
        finally:process.stdout.close()


def prepare(root,run_path,expected,iterations,share_target=.25):
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.recovery.collector_runtime import configured,from_config,recipe
    from brass_ai.recovery.retention import owned,run_lock
    from brass_ai.rust_backend.lab_backend import validate_state
    from brass_ai.rust_backend.lab_shadow import teacher_build
    from brass_ai.rust_backend.lab_shadow_config import configure,is_shadow,archive_pending
    from brass_ai.step5.league import write_json
    root,run=owned(root,run_path)
    require(type(expected) is int and expected>=1257 and type(iterations) is int and iterations>0,'Invalid continuation boundary')
    with run_lock(run):
        state=read(run/'state.json');validate_state(state)
        require(expected<=state['iteration']<=expected+iterations,'Run is outside the requested continuation boundary')
        entry=state['history'][-1]
        require(entry['generation']==state['iteration'] and sha(project_path(root,entry['path']))==entry['sha256'],'Committed checkpoint changed')
        teacher_build(state)
        import copy
        before=recipe(state['signature']['config'])
        if is_shadow(before):
            config=copy.deepcopy(before);config['teacher_phase']['term']['share_target']=share_target
        else:config=configure(before,share_target)
        require(is_shadow(config),'Invalid shadow share target')
        require(config==before or state['iteration']==expected,'Change shadow settings at the requested starting generation')
        tag=str(share_target).replace('.','')
        output=root/'configs/step5'/(run.name+'-rust-shadow32-no-branch-share'+tag+'.json')
        existing=read(output) if output.exists() else state['signature']['config']
        require(not output.exists() or recipe(existing)==config,'Existing shadow config differs')
        config=configured(config,from_config(existing))
        if recipe(config)!=before:archive_pending(root,run,state)
        write_json(output,config)
        return state,output,expected+iterations-state['iteration']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--expected-generation',type=int,required=True)
    p.add_argument('--iterations',type=int,default=2)
    p.add_argument('--share-target',type=float,choices=(.25,.4,.5),default=.25)
    p.add_argument('--device',default='cuda');p.add_argument('--mono',default='mono')
    args=p.parse_args()
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',',1)[0]
    from datetime import datetime,timezone
    import uuid
    from brass_ai.bank import read
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.step5.league import write_json
    from rust_lab_results import export
    from brass_ai.rust_backend.lab_replay_compat import execution
    out=ROOT/'artifacts/rust-backend'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-shadow-training-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True);report=dict(passed=False,training_passed=False,teacher_mode='shadow',branching=0.,
        tables=dict(train=28,held_out=4),share_target=args.share_target,expected_generation=args.expected_generation)
    try:
        report['execution_source']=execution()
        state,config,remaining=prepare(ROOT,args.run,args.expected_generation,args.iterations,args.share_target)
        from brass_ai.recovery.collector_runtime import from_config
        report['collector_runtime']=from_config(read(config))
        print('Branching=0%; teacher=shadow; tables=28+4; teacher gradients are diagnostic only.',flush=True)
        print('Collector runtime: '+str(report['collector_runtime'])+' (training config)',flush=True)
        if remaining:
            argv=[sys.executable,'-u',str(ROOT/'scripts/step5.py'),'--root',str(ROOT),'--config',str(config),
                '--device',args.device,'--mono',args.mono,'train','--run',args.run,'--expected-generation',str(state['iteration']),
                '--iterations',str(remaining),'--imitation-run',state['signature']['imitation_source']['bc_run'],'--resume-reviewed-source']
            report['training_command']=argv;write_json(out/'qualification.json',report)
            run_training(argv,out/'training.log')
        state=read(project_path(ROOT,args.run)/'state.json')
        require(state['iteration']==args.expected_generation+args.iterations,'Requested training boundary did not commit')
        qualification=state.get('teacher_phase',{}).get('weight',{}).get('gradient_qualification')
        report.update(training_passed=True,generation=state['iteration'],checkpoint=state['history'][-1],
            gradient_qualification=qualification,passed=bool(qualification and qualification['passed']),
            teacher_mode=state['signature']['config']['teacher_phase']['mode'])
        require(report['teacher_mode']=='shadow','Continuation unexpectedly enabled teacher gradients')
        print('Shadow gradient qualification: '+('Passed; fixed beta='+str(qualification['beta']) if report['passed'] else
            'Pending; inspect readiness, table coverage and finite gradient measurements. Shadow remains active.'),flush=True)
    except BaseException as error:
        report.update(error=str(error),traceback=traceback.format_exc());raise
    finally:
        log=out/'training.log'
        if log.exists():
            from brass_ai.rust_backend.lab_contract import pin
            report['training_log']=pin(log)
            if not report['training_passed']:
                with log.open('rb') as stream:
                    stream.seek(max(0,log.stat().st_size-16384))
                    report['training_error_tail']=stream.read().decode('utf-8',errors='replace')
        write_json(out/'qualification.json',report)
        print('Git-visible results: '+str(export(out)),flush=True)
        print('Original lab evidence: '+str(out),flush=True)


if __name__=='__main__':main()
