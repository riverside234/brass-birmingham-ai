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


def share_target_arg(text):
    from brass_ai.recovery.teacher_gradient_qualification import valid_target
    try:value=float(text)
    except ValueError:raise argparse.ArgumentTypeError('share target must be a number')
    if not valid_target(value):raise argparse.ArgumentTypeError('share target must be above 0 and at most 1')
    return value


def inference_choice(args):
    """Parse --inference: keep the recorded choice, or select CPU/GPU explicitly."""
    if args.inference=='keep':return 'keep'
    if args.inference=='gpu':return None
    from brass_ai.recovery.collector_runtime import DEFAULT_C_SCALE,cpu_block,validate_cpu
    profile=getattr(args,'teacher_count_search',None) or {}
    scale=profile.get('c_scale',DEFAULT_C_SCALE) if args.c_scale is None else args.c_scale
    budget=(dict(mode='timed') if args.teacher_budget=='timed' else
            dict(profile,mode='count',full=args.teacher_full,full_k=profile.get('full_k',16),fast=args.teacher_fast,
                 fast_k=args.teacher_fast_k,c_scale=scale))
    block=cpu_block(workers=args.workers,teacher_budget=budget,pinning='none' if args.no_pinning else 'one-lane-per-cpu',
                    overlap=not args.no_overlap,readers=args.readers)
    validate_cpu(block)
    return block


def prepare(root,run_path,expected,iterations,share_target=.25,*,mode='shadow',revise_stages=False,inference='keep',revision=None,
            skip_utility_qualification=False,adaptive_teacher_beta=False,teacher_search_head_weight=None,
            update_opening_sampling=False,update_cotton_sampling=False):
    from brass_ai.bank import read,sha
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.recovery.collector_runtime import configured,cpu_block,from_config,recipe,validate_teacher_budget
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
        if skip_utility_qualification and 'teacher_phase' in working:
            working=dict(working,teacher_phase=dict(working['teacher_phase'],skip_utility_qualification=True))
        from brass_ai.recovery import training_revision_g1491 as g1491
        require(revision in (None,'g1491'),'Unknown training revision')
        if mode=='shadow' and (revision=='g1491' or g1491.applied(before)):
            # g1491: the on recipe returns to a two-iteration shadow that re-measures beta.
            require(not g1491.applied(before) or is_shadow(before),'The g1491 calibration already switched teaching on')
            config=g1491.configure(working,state);config['teacher_phase']['term']['share_target']=share_target
        elif mode=='on':
            config=enable(working,share_target,revise_stages=revise_stages);authorize(state,config)
            from brass_ai.recovery.training_revision_g1294 import configure as revise_training,due
            from brass_ai.recovery import training_revision_g1307 as lr_floor
            if revise_stages and due(state):config=revise_training(config,state)
            if revise_stages and lr_floor.due(state):config=lr_floor.configure(config)
        elif is_shadow(before):
            config=dict(working,teacher_phase=dict(working['teacher_phase'],
                term=dict(working['teacher_phase']['term'],share_target=share_target,shadow_iterations=4)))
        else:config=configure(before,share_target)
        if skip_utility_qualification:config['teacher_phase']['skip_utility_qualification']=True
        from brass_ai.recovery import adaptive_teacher_beta as adaptive
        if adaptive_teacher_beta and mode=='on':
            config['teacher_phase'][adaptive.KEY] = adaptive.settings()
        if revise_stages and mode=='shadow':
            from brass_ai.recovery.search_value_schedule import revise
            config=revise(config)
        if teacher_search_head_weight is not None:
            config=g1491.configure_teacher_games(config,teacher_search_head_weight)
        from brass_ai.recovery import training_revision_g2090 as g2090
        from brass_ai.recovery import training_revision_g2240 as g2240
        if update_opening_sampling:
            from brass_ai.recovery.config import validate_config
            config=validate_config(g2090.configure(config))
        if update_cotton_sampling:
            from brass_ai.recovery.config import validate_config
            config=validate_config(g2240.configure(config))
        require(is_on(config) if mode=='on' else is_shadow(config),'Invalid teacher continuation recipe')
        diagnostic_only=qualification_upgrade(before,config)
        require(config==before or diagnostic_only or state['iteration']==expected,'Change shadow settings at the requested starting generation')
        tag=str(share_target).replace('.','')
        prefix='-rust-teacher-on-stage-a-share' if mode=='on' else '-rust-shadow32-no-branch-share'
        if 'schedule' in config['teacher_phase']['search_value']:
            prefix='-rust-teacher-on-stages-v2-share' if mode=='on' else '-rust-shadow-stages-v2-share'
        if is_revised(config):prefix=prefix.replace('-share','-no-insolvency-share')
        # Each revised recipe gets its own file; its runtime settings start from the newest
        # earlier one, so --inference keep still keeps CPU inference.
        stages=[prefix]
        from brass_ai.recovery.training_revision_g1294 import applied as training_revised,summary as revision_summary
        from brass_ai.recovery import training_revision_g1307 as lr_floor
        if training_revised(config):
            entropy=revision_summary(config)['entropy']
            prefix=prefix.replace('-share',('-entropy-raise-g'+str(entropy['start_generation']) if entropy else '-g1294-revision')+'-share')
            stages.append(prefix)
        if lr_floor.applied(config):
            prefix=prefix.replace('-share','-lr-floor-4e-6-share');stages.append(prefix)
        stage_a=config['teacher_phase']['search_value'].get('schedule',{}).get('stage_a_iterations',10)
        if stage_a!=10:
            prefix=prefix.replace('-share','-stage-a'+str(stage_a)+'-share');stages.append(prefix)
        if g1491.applied(config):
            prefix=prefix.replace('-share','-g1491-revision-share');stages.append(prefix)
            if 'cotton' in config['teacher_phase'][g1491.KEY]['exploration']:
                # A prepared, older g1491 config may already exist at this boundary.
                # Keep it as a recovery recipe; never overwrite it with new behavior.
                prefix=prefix.replace('-share','-cotton20-share');stages.append(prefix)
            if g1491.variant_of(config) in g1491.RESTARTS:
                prefix=prefix.replace('-share','-g1490-restart-share');stages.append(prefix)
            if g1491.variant_of(config) in (g1491.WIDE_RESTART, g1491.SEARCH_REFRESH):
                # Look up the prior wide-run CPU config before choosing the new filename below.
                prefix=prefix.replace('-share','-rows256-share');stages.append(prefix)
        from brass_ai.learning.post_transfer_reward import charged,restored
        if restored(config):
            prefix=prefix.replace('-share','-placement01-share');stages.append(prefix)
        if charged(config):
            prefix=prefix.replace('-share','-insolvency-kept-share');stages.append(prefix)
        if config['teacher_phase'].get('skip_utility_qualification',False):
            prefix=prefix.replace('-share','-skip-utility-share');stages.append(prefix)
        from brass_ai.recovery.search_value_schedule import dedicated_holdout
        if dedicated_holdout(config['teacher_phase']['search_value']):
            prefix='-rust-search32-share';stages.append(prefix)
        if adaptive.KEY in config['teacher_phase']:
            prefix=prefix.replace('-share','-adaptive-beta-share');stages.append(prefix)
        profile=g1491.count_search(config)
        if profile is not None:
            prefix=prefix.replace('-share','-teacher-k8-c015-share');stages.append(prefix)
        weight=g1491.teacher_search_head_weight(config)
        if weight is not None:
            previous_weight=g1491.teacher_search_head_weight(before)
            if previous_weight is not None and previous_weight != weight:
                stages.append(prefix.replace('-share',f'-cotton30-w{previous_weight}-share'))
            prefix=prefix.replace('-share',f'-cotton30-w{weight}-share');stages.append(prefix)
        if profile is not None and 'cotton_c_scale' in profile:
            prefix=prefix.replace('-teacher-k8-c015-cotton30-', '-teacher-k8-c020-cotton010-mix25-');stages.append(prefix)
        if g1491.seated_cotton(config):
            stages.append(prefix.replace('-teacher-k8-c015-cotton30-', '-teacher-k8-c020-cotton010-mix25-'))
            prefix=prefix.replace('-cotton30-', '-seated-cotton30-selfplay50-20-')
            stages.append(prefix)
            if config['value_phase']['max_epochs'] == 1:
                prefix=prefix.replace('-share', '-ppg-epoch1-share')
                stages.append(prefix)
        if g2090.applied(config):
            if weight is not None and previous_weight is not None and previous_weight != weight:
                prior_prefix=prefix.replace(f'-w{weight}-', f'-w{previous_weight}-')
                if g2090.applied(before):
                    prior_prefix=prior_prefix.replace('-share', '-opening-t2-cotton10-share')
                stages.append(prior_prefix)
                if g2240.applied(before):
                    stages.append(prior_prefix.replace('-opening-t2-cotton10-share', '-cotton-g2240-share'))
            prefix=prefix.replace('-share', '-opening-t2-cotton10-share')
            stages.append(prefix)
        if g2240.applied(config):
            prefix=prefix.replace('-opening-t2-cotton10-share', '-cotton-g2240-share')
            stages.append(prefix)
        output=root/'configs/step5'/(run.name+prefix+tag+'.json')
        adopting_cotton=g2240.applied(config) and not g2240.applied(before)
        existing=read(output) if output.exists() else state['signature']['config']
        require(not output.exists() or recipe(existing)==config or qualification_upgrade(recipe(existing),config),'Existing shadow config differs')
        earlier=next((p for p in (root/'configs/step5'/(run.name+name+tag+'.json') for name in reversed(stages[:-1])) if p.exists()),None)
        settings=from_config(existing if output.exists() or earlier is None else read(earlier))
        if inference is None:settings.pop('native_inference',None)
        elif inference!='keep':settings['native_inference']=inference
        elif profile is not None:
            block=settings.get('native_inference') or cpu_block()
            budget=dict(block['teacher_budget'],**profile)
            if 'cotton_c_scale' not in profile:budget.pop('cotton_c_scale',None)
            settings['native_inference']=dict(block,teacher_budget=budget)
        validate_teacher_budget(config,settings.get('native_inference'))
        if settings.get('native_inference') is not None:
            # Fail before any config write if the CPU workers are not prepared.
            from brass_ai.rust_backend.lab_cpu_runtime import resolve as resolve_cpu
            resolve_cpu(root,teacher_core=teacher_build(state)['core_source'])
        config=configured(config,settings)
        if recipe(config)!=before and not diagnostic_only:
            target=dict(state['signature'],config=recipe(config))
            if adopting_cotton:
                # Journal the prepared boundary so a retry cannot archive work
                # that already belongs to the revised recipe.
                from brass_ai.recovery.retention import safe
                directory=safe(run, run/'configuration-history'/f"cotton-sampling-{state['iteration']:04d}")
                marker=safe(run, directory/'recipe.json')
                from brass_ai.learning.contracts import digest
                adoption=dict(checkpoint=entry, config_sha256=digest(recipe(config)))
                if marker.exists():
                    require(read(marker)==adoption,'Interrupted cotton recipe differs from the prepared continuation')
                else:
                    from brass_ai.recovery.utility_width_upgrade import archive_pending as archive_iteration
                    archived=archive_iteration(run,directory,state)
                    write_json(marker,adoption)
                    if archived:
                        print(f'Preserved {len(archived)} interrupted paths at {directory}.', flush=True)
            elif preserves_pending(state,target):
                require(not list(run.glob(f"generation-{state['iteration']+1:04d}.pt*")),
                    'Uncommitted checkpoint exists; recover it before enabling teacher training')
            else:archive_pending(root,run,state)
        write_json(output,config)
        return state,output,expected+iterations-state['iteration']


def parser(*,default_mode='shadow',default_stage_revision=False,workers=64,teacher_fast=96,inference='keep',
           run='artifacts/step5/joint7-v11-2-g1290-trained-restart',iterations=None,description=__doc__):
    from brass_ai.recovery.collector_runtime import C_SCALES

    p=argparse.ArgumentParser(description=description)
    p.add_argument('--run',default=run)
    p.add_argument('--expected-generation',type=int,required=True)
    p.add_argument('--mode',choices=('shadow','on'),default=default_mode)
    p.add_argument('--revise-stages',action='store_true',default=default_stage_revision,
        help='Unconditional A10/B20/C10 stages, sparse validation and PPG beta floor .25')
    p.add_argument('--iterations',type=int,default=iterations or (1 if default_mode=='on' else 4))
    p.add_argument('--share-target',type=share_target_arg,default=.25,help='Teacher gradient share, any value above 0 and at most 1')
    p.add_argument('--skip-utility-qualification',action='store_true',
        help='Persistently allow teacher collection and beta qualification without waiting for utility readiness')
    p.add_argument('--adaptive-teacher-beta',action='store_true',
        help='Persistently adapt teacher-on beta toward the gradient share target from the accepted beta')
    p.add_argument('--device',default='cuda');p.add_argument('--mono',default='mono')
    p.add_argument('--inference',choices=('keep','cpu','gpu'),default=inference,
        help='cpu: in-worker ONNX Runtime for search, ordinary and rating lanes; gpu: original service; keep: last recorded choice')
    p.add_argument('--workers',type=int,default=workers,help='CPU lanes; each owns one logical CPU')
    p.add_argument('--teacher-budget',choices=('count','timed'),default='count',help='Teacher search budget on the CPU path')
    p.add_argument('--teacher-full',type=int,default=640,help='Trajectories per full teacher search')
    p.add_argument('--teacher-fast',type=int,default=teacher_fast,help='Trajectories per quick teacher search')
    p.add_argument('--teacher-fast-k',type=int,choices=(2,4,8,16),default=8,help='Considered actions per quick teacher search')
    p.add_argument('--no-pinning',action='store_true',help='Leave CPU lane placement to the operating system')
    p.add_argument('--c-scale',type=float,choices=C_SCALES,
        help='Gumbel c_scale for count-budget teacher searches (committed value or 0.1 by default)')
    p.add_argument('--no-overlap',action='store_true',help='Run teacher tables before ordinary games instead of beside them')
    p.add_argument('--readers',type=int,default=24,help='Replay reader processes on the CPU path')
    return p


def main(*,default_mode='shadow',default_stage_revision=False):
    args=parser(default_mode=default_mode,default_stage_revision=default_stage_revision).parse_args()
    continuation(args,mode=args.mode,expected=args.expected_generation,iterations=args.iterations,
        revise_stages=args.revise_stages)


def continuation(args,*,mode,expected,iterations,revise_stages=False,revision=None):
    """One reviewed continuation: prepare the recipe, train, and export its evidence."""
    import os
    os.environ['CUDA_VISIBLE_DEVICES']=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',',1)[0]
    from datetime import datetime,timezone
    import uuid
    from brass_ai.bank import read
    from brass_ai.learning.contracts import project_path,require
    from brass_ai.step5.league import write_json
    from rust_lab_results import export
    from brass_ai.rust_backend.lab_replay_compat import execution
    from brass_ai.recovery import training_revision_g1491 as g1491
    from brass_ai.recovery import training_revision_g2090 as g2090
    out=ROOT/'artifacts/rust-backend'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+mode+'-training-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True);report=dict(passed=False,training_passed=False,teacher_mode=mode,branching=0.,
        tables=dict(train=28,held_out=4),share_target=args.share_target,qualification_iterations=4,expected_generation=expected)
    try:
        report['execution_source']=execution()
        kwargs=dict(mode=mode)
        choice=inference_choice(args)
        if choice!='keep':kwargs['inference']=choice
        if revise_stages:kwargs['revise_stages']=True
        if revision is not None:kwargs['revision']=revision
        if getattr(args,'skip_utility_qualification',False):kwargs['skip_utility_qualification']=True
        if getattr(args,'adaptive_teacher_beta',False):kwargs['adaptive_teacher_beta']=True
        teacher_weight=getattr(args,'teacher_search_head_weight',None)
        if teacher_weight is not None:kwargs['teacher_search_head_weight']=teacher_weight
        if getattr(args,'update_opening_sampling',False):kwargs['update_opening_sampling']=True
        if getattr(args,'update_cotton_sampling',False):kwargs['update_cotton_sampling']=True
        state,config,remaining=prepare(ROOT,args.run,expected,iterations,args.share_target,**kwargs)
        from brass_ai.recovery.collector_runtime import from_config
        options=read(config);report['collector_runtime']=from_config(options)
        if options['teacher_phase'].get('skip_utility_qualification',False):
            report['utility_qualification']='skipped'
            print('Utility qualification skipped; teacher collection does not wait for utility readiness.',flush=True)
        tables=g1491.tables(options);calibration=options['teacher_phase']['term']['shadow_iterations']
        report.update(tables=tables,qualification_iterations=g1491.shadow_iterations(options))
        revised='schedule' in options['teacher_phase']['search_value']
        report['stage_schedule']=options['teacher_phase']['search_value'].get('schedule')
        cadence=(f"tables={tables['train']} + {tables['held_out']} held-out every {report['stage_schedule']['validation_interval']} iterations; qualification override="+
            str(report['stage_schedule']['validation_during_beta_qualification'])+'; ') if revised else f"tables={tables['train']}+{tables['held_out']}; "
        from brass_ai.recovery import adaptive_teacher_beta as adaptive
        beta_mode = 'adaptive beta' if adaptive.KEY in options['teacher_phase'] else 'qualified fixed beta'
        print('Branching=0%; teacher='+mode+'; '+cadence+
            ((beta_mode+'; residual stage restored from checkpoint.' if revised else beta_mode+'; search head Stage A (w=0, frozen trunk).') if mode=='on' else
             f'qualification={calibration} iterations; teacher gradients are diagnostic only.'),flush=True)
        from brass_ai.recovery.search_value_schedule import validation_plan
        if g2090.applied(options) and validation_plan(options,state).get('forced_by_beta_remeasurement'):
            print('Teacher beta will be re-measured with teaching on and four held-out tables.',flush=True)
        if revised:
            schedule=report['stage_schedule'];a=schedule['stage_a_iterations'];b=schedule['stage_b_iterations'];c=schedule['stage_c_iterations']
            floor_stage = 'B' if 'ppg_beta_floor_after_stage_b' in schedule else 'A'
            base_floor = options['value_phase']['beta_clone_min']
            floor = schedule.get(f'ppg_beta_floor_after_stage_{floor_stage.lower()}', base_floor)
            floor_change = f', {floor:g} after Stage {floor_stage}' if floor != base_floor else ''
            print(f'Residual stages: A{a} head-only; B{b} blend warm-up; C{c} trunk warm-up; KL cap .003; '
                  f'PPG beta floor {base_floor:g}{floor_change}. Ordinary PPG epoch cap {options["value_phase"]["max_epochs"]}.', flush=True)
        if g2090.applied(options):
            from brass_ai.recovery.ordinary_cotton import KEY
            opening=options['opening']
            cotton=options['table_plan'][KEY]
            opening_scope='PPO and teacher cotton sampling' if g1491.cotton_only(options) else 'PPO and teacher'
            print(f'{opening_scope} opening temperature {opening["temperature"]:g} for the first '
                  f'{opening["action_decisions_per_seat"]} actions per seat. PPO cotton targeting {cotton["probability"]:.0%}.', flush=True)
            if g1491.cotton_only(options):
                fraction=g1491.cotton_settings(options)['table_fraction']
                print(f'Teacher cotton tables {fraction:.0%}; normal teacher sampling off. '
                      f'PPO cotton self-play {cotton["cotton_selfplay"]}% of all PPO tables.', flush=True)
        from brass_ai.learning.post_transfer_reward import is_revised
        if is_revised(options):
            r=options['reward'];cost=options['insolvency']
            print(f"Transferred reward: win={r['win']:g}, placement={r['placement']:g}, VP={r['vp']:g} (cap {r['vp_cap']}); "
                  f"insolvency event cost={cost['event_cost']:g}, max cost={cost['max_cost']:g}.",flush=True)
        from brass_ai.recovery.training_revision_g1294 import applied as training_revised,summary as revision_summary
        if training_revised(options):
            report['training_revision']=revision_summary(options)
            print('Training revision: '+str(report['training_revision']),flush=True)
        from brass_ai.recovery import training_revision_g1307 as lr_floor
        if lr_floor.applied(options):
            report['ppo_lr_floor']=lr_floor.summary(options)
            print('PPO LR floor: '+str(report['ppo_lr_floor']),flush=True)
        if g1491.applied(options):
            report['g1491_revision']=g1491.summary(options)
            print('g1491 revision: '+str(report['g1491_revision']),flush=True)
        print('Collector runtime: '+str(report['collector_runtime'])+' (training config)',flush=True)
        if remaining:
            argv=[sys.executable,'-u',str(ROOT/'scripts/step5.py'),'--root',str(ROOT),'--config',str(config),
                '--device',args.device,'--mono',args.mono,'train','--run',args.run,'--expected-generation',str(state['iteration']),
                '--iterations',str(remaining),'--imitation-run',state['signature']['imitation_source']['bc_run'],'--resume-reviewed-source']
            report['training_command']=argv;write_json(out/'qualification.json',report)
            run_training(argv,out/'training.log')
        state=read(project_path(ROOT,args.run)/'state.json')
        require(state['iteration']==expected+iterations,'Requested training boundary did not commit')
        weight=state.get('teacher_phase',{}).get('weight',{})
        qualification=weight.get('gradient_qualification')
        measuring=adaptive.pending(weight)
        report.update(training_passed=True,generation=state['iteration'],checkpoint=state['history'][-1],
            gradient_qualification=qualification,passed=bool(qualification and qualification['passed'] and not measuring),
            teacher_mode=state['signature']['config']['teacher_phase']['mode'],search_value=state.get('search_value'))
        for key in (adaptive.KEY, adaptive.REMEASUREMENT):
            if key in weight:report[key]=weight[key]
        require(report['teacher_mode']==mode,'Continuation teacher mode changed unexpectedly')
        if measuring:
            status='beta re-measurement pending; inspect table coverage and gradient measurements.'
        elif report['passed']:
            status=f'{beta_mode}={weight.get("beta", qualification["beta"])}'
        else:
            status='qualification pending; inspect readiness, table coverage and gradient measurements.'
        print('Teacher '+mode+': '+status,flush=True)
        return state
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
