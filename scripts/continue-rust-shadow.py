#!/usr/bin/env python3
"""Continue the same Rust run with branching off and shadow or qualified teaching."""
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


def inference_choice(args):
    """Parse --inference: keep the recorded choice, or select CPU/GPU explicitly."""
    if args.inference=='keep':return 'keep'
    if args.inference=='gpu':return None
    from brass_ai.recovery.collector_runtime import cpu_block,validate_cpu
    budget=dict(mode='timed') if args.teacher_budget=='timed' else dict(mode='count',full=args.teacher_full,fast=args.teacher_fast)
    block=cpu_block(workers=args.workers,teacher_budget=budget,pinning='none' if args.no_pinning else 'one-lane-per-cpu')
    validate_cpu(block)
    return block


def prepare(root,run_path,expected,iterations,share_target=.25,*,mode='shadow',revise_stages=False,inference='keep'):
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.recovery.collector_runtime import configured,from_config,recipe
    from brass_ai.recovery.retention import owned,run_lock
    from brass_ai.rust_backend.lab_backend import validate_state
    from brass_ai.rust_backend.lab_shadow import teacher_build
    from brass_ai.rust_backend.lab_shadow_config import configure,is_shadow,archive_pending,qualification_upgrade,preserves_pending
    from brass_ai.rust_backend.lab_teacher_activation import configure as enable,is_on,authorize
    from brass_ai.step5.league import write_json
    root,run=owned(root,run_path)
    require(type(expected) is int and expected>=1257 and type(iterations) is int and iterations>0 and mode in ('shadow','on'),'Invalid continuation boundary')
    with run_lock(run):
        state=read(run/'state.json');validate_state(state)
        require(expected<=state['iteration']<=expected+iterations,'Run is outside the requested continuation boundary')
        entry=state['history'][-1]
        require(entry['generation']==state['iteration'] and sha(project_path(root,entry['path']))==entry['sha256'],'Committed checkpoint changed')
        teacher_build(state)
        import copy
        before=recipe(state['signature']['config'])
        from brass_ai.learning.post_transfer_reward import revise as revise_reward,REVISION,is_revised
        from brass_ai.recovery.transfer_resume import instruction
        moved=instruction(root,run)
        pending_reward=bool(moved and moved.get('reward_revision')==REVISION)
        working=revise_reward(before) if pending_reward else before
        if mode=='on':
            config=enable(working,share_target,revise_stages=revise_stages);authorize(state,config)
        elif is_shadow(before):
            config=copy.deepcopy(before);config['teacher_phase']['term']['share_target']=share_target
            config['teacher_phase']['term']['shadow_iterations']=4
        else:config=configure(before,share_target)
        if revise_stages and mode=='shadow':
            from brass_ai.recovery.search_value_schedule import revise
            config=revise(config)
        require(is_on(config) if mode=='on' else is_shadow(config),'Invalid teacher continuation recipe')
        diagnostic_only=qualification_upgrade(before,config)
        require(config==before or diagnostic_only or state['iteration']==expected,'Change shadow settings at the requested starting generation')
        tag=str(share_target).replace('.','')
        prefix='-rust-teacher-on-stage-a-share' if mode=='on' else '-rust-shadow32-no-branch-share'
        if 'schedule' in config['teacher_phase']['search_value']:
            prefix='-rust-teacher-on-stages-v2-share' if mode=='on' else '-rust-shadow-stages-v2-share'
        if is_revised(config):prefix=prefix.replace('-share','-no-insolvency-share')
        output=root/'configs/step5'/(run.name+prefix+tag+'.json')
        existing=read(output) if output.exists() else state['signature']['config']
        require(not output.exists() or recipe(existing)==config or qualification_upgrade(recipe(existing),config),'Existing shadow config differs')
        settings=from_config(existing)
        if inference is None:settings.pop('native_inference',None)
        elif inference!='keep':settings['native_inference']=inference
        if settings.get('native_inference') is not None:
            # Fail before any config write if the CPU workers are not prepared.
            from brass_ai.rust_backend.lab_cpu_runtime import resolve as resolve_cpu
            resolve_cpu(root,teacher_core=teacher_build(state)['core_source'])
        config=configured(config,settings)
        if recipe(config)!=before and not diagnostic_only:
            target=dict(state['signature'],config=recipe(config))
            if preserves_pending(state,target):
                require(not list(run.glob(f"generation-{state['iteration']+1:04d}.pt*")),
                    'Uncommitted checkpoint exists; recover it before enabling teacher training')
            else:archive_pending(root,run,state)
        write_json(output,config)
        return state,output,expected+iterations-state['iteration']


def main(*,default_mode='shadow',default_stage_revision=False):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--expected-generation',type=int,required=True)
    p.add_argument('--mode',choices=('shadow','on'),default=default_mode)
    p.add_argument('--revise-stages',action='store_true',default=default_stage_revision,
        help='Unconditional A10/B20/C10 stages, sparse validation and PPG beta floor .25')
    p.add_argument('--iterations',type=int,default=1 if default_mode=='on' else 4)
    p.add_argument('--share-target',type=float,choices=(.25,.4,.5),default=.25)
    p.add_argument('--device',default='cuda');p.add_argument('--mono',default='mono')
    p.add_argument('--inference',choices=('keep','cpu','gpu'),default='keep',
        help='cpu: in-worker ONNX Runtime for search, ordinary and rating lanes; gpu: original service; keep: last recorded choice')
    p.add_argument('--workers',type=int,default=64,help='CPU lanes; each owns one logical CPU')
    p.add_argument('--teacher-budget',choices=('count','timed'),default='count',help='Teacher search budget on the CPU path')
    p.add_argument('--teacher-full',type=int,default=256,help='Trajectories per full teacher search')
    p.add_argument('--teacher-fast',type=int,default=48,help='Trajectories per fast teacher search')
    p.add_argument('--no-pinning',action='store_true',help='Leave CPU lane placement to the operating system')
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
    out=ROOT/'artifacts/rust-backend'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+args.mode+'-training-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True);report=dict(passed=False,training_passed=False,teacher_mode=args.mode,branching=0.,
        tables=dict(train=28,held_out=4),share_target=args.share_target,qualification_iterations=4,expected_generation=args.expected_generation)
    try:
        report['execution_source']=execution()
        kwargs=dict(mode=args.mode,inference=inference_choice(args))
        if args.revise_stages:kwargs['revise_stages']=True
        state,config,remaining=prepare(ROOT,args.run,args.expected_generation,args.iterations,args.share_target,**kwargs)
        from brass_ai.recovery.collector_runtime import from_config
        options=read(config);report['collector_runtime']=from_config(options)
        revised='schedule' in options['teacher_phase']['search_value']
        report['stage_schedule']=options['teacher_phase']['search_value'].get('schedule')
        cadence=('tables=28 + 4 held-out every tenth iteration; qualification override='+
            str(report['stage_schedule']['validation_during_beta_qualification'])+'; ') if revised else 'tables=28+4; '
        print('Branching=0%; teacher='+args.mode+'; '+cadence+
            (('qualified fixed beta; residual stage restored from checkpoint.' if revised else 'qualified fixed beta; search head Stage A (w=0, frozen trunk).') if args.mode=='on' else
             'qualification=4 iterations; teacher gradients are diagnostic only.'),flush=True)
        if revised:print('Residual stages: A10 head-only; B20 blend warm-up; C10 trunk warm-up; KL cap .003; PPG beta floor .25.',flush=True)
        from brass_ai.learning.post_transfer_reward import is_revised
        if is_revised(options):print('Transferred reward: win=1, placement=.05, VP=.001 (cap 180); insolvency charge=0.',flush=True)
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
            teacher_mode=state['signature']['config']['teacher_phase']['mode'],search_value=state.get('search_value'))
        require(report['teacher_mode']==args.mode,'Continuation teacher mode changed unexpectedly')
        print('Teacher '+args.mode+': '+('qualified fixed beta='+str(qualification['beta']) if report['passed'] else
            'qualification Pending; inspect readiness, table coverage and finite gradient measurements.'),flush=True)
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
