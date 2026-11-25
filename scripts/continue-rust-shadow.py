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


NAME_LIMIT=255  # bytes per path component (Linux); names that fit keep their old form


def recipe_path(root,run_name,prefix,tag):
    """configs/step5 file for a recipe stage. Each adopted revision lengthens the descriptive
    name; past NAME_LIMIT bytes it becomes the run name plus a hash of the full name, so the
    same recipe always maps to the same file and lookups of earlier stages never fail."""
    import hashlib
    name=run_name+prefix+tag+'.json'
    if len(name.encode())>NAME_LIMIT:
        name=run_name+'-rust-recipe-'+hashlib.sha256(name.encode()).hexdigest()[:24]+'-share'+tag+'.json'
    return root/'configs/step5'/name


def share_target_arg(text):
    from brass_ai.recovery.teacher_gradient_qualification import valid_target
    try:value=float(text)
    except ValueError:raise argparse.ArgumentTypeError('share target must be a number')
    if not valid_target(value):raise argparse.ArgumentTypeError('share target must be above 0 and at most 1')
    return value


def workers_arg(text):
    from brass_ai.recovery.collector_runtime import MAX_WORKERS
    try:value=int(text)
    except ValueError:raise argparse.ArgumentTypeError('workers must be an integer')
    if not 1<=value<=MAX_WORKERS:raise argparse.ArgumentTypeError(f'workers must be between 1 and {MAX_WORKERS}')
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
            update_opening_sampling=False,update_cotton_sampling=False,update_cotton_network=False,update_rail_cotton=False,update_cotton_timing=False,update_search_lambda=False,value_revision=None,rating_every=None,
            teacher_revision=False,soft_policy=False,soft_policy_weight=None,ppg_beta_floor=None,teacher_search_revision=False,
            teacher_training_revision=False,table_revision=False,era_head_revision=False,rail_end_weight=None,
            optimizer_revision=False,auxiliary_revision=False,clip_revision=False):
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
        from brass_ai.recovery import training_revision_g2400 as g2400
        if update_cotton_network:
            from brass_ai.recovery.config import validate_config
            config=validate_config(g2400.configure(config))
        from brass_ai.recovery import training_revision_g2654 as g2654
        if update_rail_cotton:
            from brass_ai.recovery.config import validate_config
            config=validate_config(g2654.configure(config))
        from brass_ai.recovery import training_revision_g3471 as g3471
        from brass_ai.recovery import training_revision_g3528 as g3528
        if update_cotton_timing:
            # The newest reviewed cotton timing within each era: g3528 (from the g3471 or the g2654 recipe).
            from brass_ai.recovery.config import validate_config
            config=validate_config(g3528.configure(config))
        from brass_ai.recovery import training_revision_g2880 as g2880
        if update_search_lambda:
            from brass_ai.recovery.config import validate_config
            config=validate_config(g2880.configure(config))
        from brass_ai.recovery import training_revision_g3140 as g3140
        if value_revision:
            # Adopted at this committed generation; earlier items persist with the recipe.
            from brass_ai.recovery.config import validate_config
            config=validate_config(g3140.configure(config,g3140.parse(value_revision),state['iteration']))
        from brass_ai.recovery import rating_cadence
        if rating_every is not None:
            # Ratings run every `rating_every` committed iterations from this generation on.
            from brass_ai.recovery.config import validate_config
            config=validate_config(rating_cadence.configure(config,rating_every,state['iteration']))
        from brass_ai.recovery import training_revision_g3270 as g3270
        if teacher_revision:
            # Build steps 2-3 (40 tables, rebuilt targets, 8-iteration labels, search-value rotation); then kept.
            require(mode=='on','The g3270 teacher revision is adopted with teaching on')
            config=g3270.configure(config,state['iteration'])
        from brass_ai.recovery import training_revision_g3423 as g3423
        if soft_policy:
            # Soft policy S1: a trainer-owned soft head on the teacher term's long-label rows; then kept.
            require(mode=='on','The g3423 soft policy head is adopted with teaching on')
            config=g3423.configure(config,state['iteration'])
        if soft_policy_weight is not None:
            # The soft head's weight x beta (8 -> 6 at g3471); head, Adam and warm-up carry over.
            require(mode=='on','The soft policy weight changes with teaching on')
            config=g3423.set_weight(config,soft_policy_weight,state['iteration'])
        from brass_ai.recovery import training_revision_g3658 as g3658
        if ppg_beta_floor is not None:
            # PPG beta floor .25 -> .125 (value phase and search-head Stage C); the first resume sets beta to .125.
            require(ppg_beta_floor==g3658.FLOOR,'Only the reviewed .125 PPG beta floor')
            from brass_ai.recovery.config import validate_config
            config=validate_config(g3658.configure(config))
        from brass_ai.recovery import training_revision_g4138 as g4138
        if teacher_search_revision:
            # 44+4 tables, short search 128 at C_scale .6; search-value games rebuilt from here; then kept.
            require(mode=='on','The g4138 teacher search revision is adopted with teaching on')
            config=g4138.configure(config,state['iteration'])
        from brass_ai.recovery import training_revision_g4475 as g4475
        if teacher_training_revision:
            # 160 teacher rows per PPO step, search-value head paused (era heads keep the search tables); then kept.
            require(mode=='on','The g4475 teacher training revision is adopted with teaching on')
            config=g4475.configure(config,state['iteration'])
        from brass_ai.recovery import training_revision_g4551 as g4551
        if table_revision:
            # Standard PPO self-play tables (teacher cotton kept), 50+4 teacher tables, 176 teacher rows per PPO step.
            require(mode=='on','The g4551 table revision is adopted with teaching on')
            config=g4551.configure(config,state['iteration'])
        from brass_ai.recovery import training_revision_g4770 as g4770
        if era_head_revision:
            # Rail-end heads back at .05 (PPG only), PPO-game era-head rows on actual maps, search pool 4 of 8 iterations.
            config=g4770.configure(config,state['iteration'])
        if rail_end_weight is not None:
            # The restored Rail-end head's weight, .05 -> .25 only.
            config=g4770.set_rail_end_weight(config,rail_end_weight)
        from brass_ai.recovery import training_revision_g5402 as g5402
        if optimizer_revision:
            # PPO clip 5.83 (PPG keeps 1.0), PPO rate -20% as the new base, soft head and era heads on the main rates.
            adopted=g5402.applied(config)
            config=g5402.configure(config,state['iteration'],
                multiplier=None if adopted else g5402.committed_multiplier(state),
                gradient_norms=None if adopted else g5402.committed_gradient_norms(state))
        from brass_ai.recovery import training_revision_g5466 as g5466
        if auxiliary_revision:
            # Rail-end preserved in PPO like Canal/Rail, PPG clip 2.2 with a one-time PPG Adam rescale (from the
            # last 30 committed PPG phases' logged norms), the era heads on the fixed PPO rate in PPG.
            adopted=g5466.applied(config)
            config=g5466.configure(config,state['iteration'],
                gradient_norms=None if adopted else g5466.committed_ppg_norms(run,state))
        from brass_ai.recovery import training_revision_g5685 as g5685
        if clip_revision:
            # PPO clip 6.5 and PPG clip 2.4, each with a one-time Adam rescale from the last 30 committed
            # iterations' logged norms.
            adopted=g5685.applied(config)
            config=g5685.configure(config,state['iteration'],
                gradient_norms=None if adopted else g5685.committed_norms(run,state))
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
            if config['value_phase']['max_epochs'] == 1 or g2400.applied(config):
                # Retain the old recipe's full filename in the runtime lookup
                # chain before choosing the new two-epoch filename below.
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
            if profile in (g1491.UPDATED_SEARCH, g1491.G2600_SEARCH):
                prefix=prefix.replace('-teacher-k8-c015-', '-teacher-k8-c010-c070-')
                stages.append(prefix)
        if g2400.applied(config):
            prefix=prefix.replace('-ppg-epoch1-', '-ppg-epoch2-').replace('-share', '-cotton-network-share');stages.append(prefix)
        from brass_ai.recovery import training_revision_g2600 as g2600
        if g2600.applied(config):
            # The g2400 filename above (and its other blend) stays in the runtime lookup chain.
            if weight is not None and previous_weight is not None and previous_weight != weight:
                stages.append(prefix.replace(f'-w{weight}-', f'-w{previous_weight}-'))
            prefix=(prefix.replace('-teacher-k8-c010-c070-', '-teacher-k8-c020-c070-').replace('-ppg-epoch2-', '-ppg-epoch1-')
                    .replace('-share', '-g2600-opening13-floor025-share'))
            stages.append(prefix)
        if g2654.applied(config):
            if weight is not None and previous_weight is not None and previous_weight != weight:
                stages.append(prefix.replace(f'-w{weight}-', f'-w{previous_weight}-'))
            prefix=prefix.replace('-share', '-rail-cotton-l3-share');stages.append(prefix)
        if g2880.applied(config):
            if weight is not None and previous_weight is not None and previous_weight != weight:
                stages.append(prefix.replace(f'-w{weight}-', f'-w{previous_weight}-'))
            prefix=prefix.replace('-share', '-search-lambda095-share');stages.append(prefix)
        if g3140.applied(config):
            if weight is not None and previous_weight is not None and previous_weight != weight:
                stages.append(prefix.replace(f'-w{weight}-', f'-w{previous_weight}-'))
            if g3140.applied(before) and g3140.tag(before)!=g3140.tag(config):
                # The recipe file the run used last stays in the runtime lookup chain.
                stages.append(prefix.replace('-share', '-value-'+g3140.tag(before)+'-share'))
            prefix=prefix.replace('-share', '-value-'+g3140.tag(config)+'-share');stages.append(prefix)
        if rating_cadence.applied(config):
            prefix=prefix.replace('-share', '-pl'+str(config[rating_cadence.KEY]['every_iterations'])+'-share');stages.append(prefix)
        if g3270.applied(config):
            prefix=prefix.replace('-share', '-'+g3270.tag(config)+'-share');stages.append(prefix)
        if g3423.applied(config):
            if g3423.applied(before) and g3423.tag(before)!=g3423.tag(config):
                stages.append(prefix.replace('-share', '-'+g3423.tag(before)+'-share'))
            prefix=prefix.replace('-share', '-'+g3423.tag(config)+'-share');stages.append(prefix)
        if g3471.applied(config) or g3528.applied(config):
            prefix=prefix.replace('-share', '-cotton-timed-share');stages.append(prefix)
        if g3528.applied(config):
            prefix=prefix.replace('-share', '-retimed-share');stages.append(prefix)
        if g3658.applied(config):
            prefix=prefix.replace('-share', '-ppg-floor0125-share');stages.append(prefix)
        if g4138.applied(config):
            prefix=prefix.replace('-share', '-'+g4138.tag(config)+'-share');stages.append(prefix)
        if g4475.applied(config):
            prefix=prefix.replace('-share', '-'+g4475.tag(config)+'-share');stages.append(prefix)
        if g4551.applied(config):
            prefix=prefix.replace('-share', '-'+g4551.tag(config)+'-share');stages.append(prefix)
        if g4770.applied(config):
            prefix=prefix.replace('-share', '-'+g4770.tag(config)+'-share');stages.append(prefix)
        if g5402.applied(config):
            prefix=prefix.replace('-share', '-'+g5402.tag(config)+'-share');stages.append(prefix)
        if g5466.applied(config):
            prefix=prefix.replace('-share', '-'+g5466.tag(config)+'-share');stages.append(prefix)
        if g5685.applied(config):
            prefix=prefix.replace('-share', '-'+g5685.tag(config)+'-share');stages.append(prefix)
        output=recipe_path(root,run.name,prefix,tag)
        adopting_teacher=g3270.applied(config) and not g3270.applied(before)
        adopting_soft=g3423.applied(config) and config.get(g3423.KEY)!=before.get(g3423.KEY)
        adopting_floor=g3658.applied(config) and not g3658.applied(before)
        adopting_search=g4138.applied(config) and not g4138.applied(before)
        adopting_training=g4475.applied(config) and not g4475.applied(before)
        adopting_tables=g4551.applied(config) and not g4551.applied(before)
        adopting_era_heads=g4770.applied(config) and not g4770.applied(before)
        adopting_optimizer=g5402.applied(config) and not g5402.applied(before)
        adopting_auxiliary=g5466.applied(config) and not g5466.applied(before)
        adopting_clips=g5685.applied(config) and not g5685.applied(before)
        adopting_cotton=(g2240.applied(config) and not g2240.applied(before) or
                         g2400.applied(config) and not g2400.applied(before) or
                         g2654.applied(config) and not g2654.applied(before) or
                         g3471.applied(config) and not g3471.applied(before) or
                         g3528.applied(config) and not g3528.applied(before) or
                         g2880.applied(config) and not g2880.applied(before) or
                         g3140.changed(before,config) or adopting_teacher or adopting_soft or adopting_floor or adopting_search or
                         adopting_training or adopting_tables or adopting_era_heads or adopting_optimizer or
                         adopting_auxiliary or adopting_clips)
        existing=read(output) if output.exists() else state['signature']['config']
        require(not output.exists() or recipe(existing)==config or qualification_upgrade(recipe(existing),config),'Existing shadow config differs')
        earlier=next((p for p in (recipe_path(root,run.name,name,tag) for name in reversed(stages[:-1])) if p.exists()),None)
        settings=from_config(existing if output.exists() or earlier is None else read(earlier))
        if inference is None:settings.pop('native_inference',None)
        elif inference!='keep':settings['native_inference']=inference
        elif profile is not None:
            block=settings.get('native_inference') or cpu_block()
            # g4138: the committed profile with the revision's short search when adopted.
            budget=dict(block['teacher_budget'],**g4138.search_profile(config))
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
                journal=('teacher-revision' if adopting_teacher else 'teacher-search-revision' if adopting_search else
                         'teacher-training-revision' if adopting_training else
                         'table-revision' if adopting_tables else
                         'era-head-revision' if adopting_era_heads else
                         'optimizer-revision' if adopting_optimizer else
                         'auxiliary-revision' if adopting_auxiliary else
                         'soft-policy' if adopting_soft else
                         'value-revision' if g3140.changed(before,config) else 'ppg-beta-floor' if adopting_floor else
                         'search-lambda' if g2880.applied(config) and not g2880.applied(before) else 'cotton-sampling')
                directory=safe(run, run/'configuration-history'/f"{journal}-{state['iteration']:04d}")
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
    p.add_argument('--workers',type=workers_arg,default=workers,
        help='CPU lanes; each owns one logical CPU (at most 64, the CPUs the lab trainer sees)')
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
        if getattr(args,'update_cotton_network',False):kwargs['update_cotton_network']=True
        if getattr(args,'update_rail_cotton',False):kwargs['update_rail_cotton']=True
        if getattr(args,'update_cotton_timing',False):kwargs['update_cotton_timing']=True
        if getattr(args,'update_search_lambda',False):kwargs['update_search_lambda']=True
        if getattr(args,'value_revision',None):kwargs['value_revision']=args.value_revision
        if getattr(args,'rating_every',None) is not None:kwargs['rating_every']=args.rating_every
        if getattr(args,'teacher_revision',False):kwargs['teacher_revision']=True
        if getattr(args,'soft_policy',False):kwargs['soft_policy']=True
        if getattr(args,'soft_policy_weight',None) is not None:kwargs['soft_policy_weight']=args.soft_policy_weight
        if getattr(args,'ppg_beta_floor',None) is not None:kwargs['ppg_beta_floor']=args.ppg_beta_floor
        if getattr(args,'teacher_search_revision',False):kwargs['teacher_search_revision']=True
        if getattr(args,'teacher_training_revision',False):kwargs['teacher_training_revision']=True
        if getattr(args,'table_revision',False):kwargs['table_revision']=True
        if getattr(args,'era_head_revision',False):kwargs['era_head_revision']=True
        if getattr(args,'rail_end_weight',None) is not None:kwargs['rail_end_weight']=args.rail_end_weight
        if getattr(args,'optimizer_revision',False):kwargs['optimizer_revision']=True
        if getattr(args,'auxiliary_revision',False):kwargs['auxiliary_revision']=True
        if getattr(args,'clip_revision',False):kwargs['clip_revision']=True
        state,config,remaining=prepare(ROOT,args.run,expected,iterations,args.share_target,**kwargs)
        from brass_ai.recovery.collector_runtime import from_config
        options=read(config);report['collector_runtime']=from_config(options)
        if options['teacher_phase'].get('skip_utility_qualification',False):
            report['utility_qualification']='skipped'
            print('Utility qualification skipped; teacher collection does not wait for utility readiness.',flush=True)
        from brass_ai.recovery import training_revision_g3270 as g3270
        tables=g3270.tables(options);calibration=options['teacher_phase']['term']['shadow_iterations']
        report.update(tables=tables,qualification_iterations=g1491.shadow_iterations(options))
        revised='schedule' in options['teacher_phase']['search_value']
        report['stage_schedule']=options['teacher_phase']['search_value'].get('schedule')
        cadence=(f"tables={tables['train']} + {tables['held_out']} validation every iteration; " if g3270.every_iteration(options) else
            f"tables={tables['train']} + {tables['held_out']} held-out every {report['stage_schedule']['validation_interval']} iterations; qualification override="+
            str(report['stage_schedule']['validation_during_beta_qualification'])+'; ') if revised else f"tables={tables['train']}+{tables['held_out']}; "
        from brass_ai.recovery import adaptive_teacher_beta as adaptive
        beta_mode = 'adaptive beta' if adaptive.KEY in options['teacher_phase'] else 'qualified fixed beta'
        print('Branching=0%; teacher='+mode+'; '+cadence+
            ((beta_mode+'; residual stage restored from checkpoint.' if revised else beta_mode+'; search head Stage A (w=0, frozen trunk).') if mode=='on' else
             f'qualification={calibration} iterations; teacher gradients are diagnostic only.'),flush=True)
        from brass_ai.recovery.search_value_schedule import validation_plan
        if g2090.applied(options) and validation_plan(options,state).get('forced_by_beta_remeasurement'):
            print('Teacher beta will be re-measured with teaching on and four held-out tables. '
                  'Adaptive moving averages start fresh.',flush=True)
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
                from brass_ai.recovery.ordinary_cotton import selfplay_settings
                fraction=g1491.cotton_settings(options)['table_fraction']
                # g4551: every PPO self-play table is standard; the teacher cotton tables stay.
                selfplay=(f'PPO cotton self-play {cotton["cotton_selfplay"]}% of all PPO tables.'
                          if selfplay_settings(options) is not None else 'PPO cotton self-play off (all self-play tables standard).')
                print(f'Teacher cotton tables {fraction:.0%}; normal teacher sampling off. '+selfplay, flush=True)
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
        from brass_ai.recovery import training_revision_g2600 as g2600
        if g2600.applied(options):
            report['g2600_revision']=g2600.summary(options)
            print('g2600 revision: '+str(report['g2600_revision']),flush=True)
        from brass_ai.recovery import training_revision_g2654 as g2654
        if g2654.applied(options):
            report['g2654_revision']=g2654.summary(options)
            print('Cotton exploration skips Rail-era builds below level three: '+str(report['g2654_revision']),flush=True)
        from brass_ai.recovery import training_revision_g3471 as g3471
        if g3471.applied(options):
            report['g3471_cotton_timing']=g3471.summary(options)
            print('Cotton exploration timing (actions of a seat within its era): '+str(report['g3471_cotton_timing']),flush=True)
        from brass_ai.recovery import training_revision_g3528 as g3528
        if g3528.applied(options):
            report['g3528_cotton_timing']=g3528.summary(options)
            print('Cotton exploration timing (actions of a seat within its era): '+str(report['g3528_cotton_timing']),flush=True)
        from brass_ai.recovery import training_revision_g3140 as g3140
        if g3140.applied(options):
            report['g3140_value_revision']=g3140.summary(options)
            print('g3140 value revision: '+str(report['g3140_value_revision']),flush=True)
        from brass_ai.recovery import training_revision_g3658 as g3658
        if g3658.applied(options):
            report['g3658_ppg_beta_floor']=g3658.summary(options)
            print('PPG beta floor: '+str(report['g3658_ppg_beta_floor']),flush=True)
        from brass_ai.recovery import rating_cadence
        if rating_cadence.applied(options):
            report['rating_cadence']=rating_cadence.summary(options)
            print('Strength rating cadence: '+str(report['rating_cadence']),flush=True)
        if g3270.applied(options):
            report['g3270_teacher_revision']=g3270.summary(options)
            r=report['g3270_teacher_revision'];v=r['search_value']
            # The tables that run (g4138 raises them to 44+4); the rest is the g3270 block.
            print(f"g3270 teacher revision (from g{r['from_generation']}): {tables['train']}+{tables['held_out']} tables every "
                  f"iteration, labels {r['buffer']['iterations']} iterations (ages {r['buffer']['age_weights']}), "
                  f"{g3270.phase_options(options,options['teacher_phase'])['term']['rows_per_step']} rows per PPO step "
                  f"without repeats, targets rebuilt from the current policy at scale "
                  f"{r['targets']['label_scale']:g}, long-search labels only; search value buffer {v['buffer_iterations']} iterations, "
                  f"at most {v['steps_per_iteration']} steps, {v['heldout_games']} rotating held-out games.",flush=True)
        from brass_ai.recovery import training_revision_g3423 as g3423
        if g3423.applied(options):
            report['g3423_soft_policy']=g3423.summary(options)
            r=report['g3423_soft_policy'];w=r['warmup']
            print(f"g3423 soft policy {r['step']} (from g{r['from_generation']}): a trainer-owned soft head predicts the rebuilt "
                  f"label target at T={r['temperature']:g} on the teacher term's long-label rows, weight {r['weight']:g} x beta, "
                  f"PPO-row factor {r['ppo_row_factor']:g}; {w['head_only_iterations']} head-only iterations, then a "
                  f"{w['ramp_iterations']}-iteration trunk ramp.",flush=True)
        from brass_ai.recovery import training_revision_g4138 as g4138
        if g4138.applied(options):
            report['g4138_teacher_search']=g4138.summary(options)
            r=report['g4138_teacher_search'];s=r['short_search']
            print(f"g4138 teacher search revision (from g{r['from_generation']}): {r['tables']['train']}+{r['tables']['held_out']} "
                  f"tables every iteration; short search {s['simulations']} simulations, K={s['k']}, C_scale={s['c_scale']:g} "
                  f"(long search unchanged). Search value trains on games collected from g{r['from_generation']} on, "
                  f"after {r['search_value']['required_ages']} such ages; long labels, beta and held-out games kept.",flush=True)
        from brass_ai.recovery import training_revision_g4475 as g4475
        if g4475.applied(options):
            report['g4475_teacher_training']=g4475.summary(options)
            r=report['g4475_teacher_training']
            print(f"g4475 teacher training revision (from g{r['from_generation']}): {r['rows_per_step']} teacher label rows "
                  f"per PPO step; the search-value head's training is paused (head, Adam and stage kept); the search "
                  f"tables still publish their value shards and train the era heads.",flush=True)
        from brass_ai.recovery import training_revision_g4551 as g4551
        if g4551.applied(options):
            report['g4551_table_revision']=g4551.summary(options)
            r=report['g4551_table_revision']
            print(f"g4551 table revision (from g{r['from_generation']}): every PPO self-play table is a standard game "
                  f"(teacher cotton tables kept); {r['tables']['train']}+{r['tables']['held_out']} teacher tables every "
                  f"iteration; {r['rows_per_step']} teacher label rows per PPO step.",flush=True)
        from brass_ai.recovery import training_revision_g4770 as g4770
        if g4770.applied(options):
            report['g4770_era_head_revision']=g4770.summary(options)
            r=report['g4770_era_head_revision'];pool=r['search_pool']
            print(f"g4770 era-head revision (from g{r['from_generation']}): Rail-end heads restored at g{r['rail_end_restored']}, "
                  f"weight {r['local_weights']['rail_end']:g} (PPG only, same batches as Canal/Rail; no PPO snapshot or "
                  f"preservation); PPO-game rows on actual maps for every era head; search rows lambda "
                  f"{r['search_targets']['lam']:g}; search pool a {pool['sampled_iterations']}-iteration sample of the last "
                  f"{pool['window_iterations']}.",flush=True)
        from brass_ai.recovery import training_revision_g5402 as g5402
        if g5402.applied(options):
            report['g5402_optimizer_revision']=g5402.summary(options)
            r=report['g5402_optimizer_revision'];b,a,s=r['before'],r['after'],r['adam_rescale']
            print(f"g5402 optimizer revision (from g{r['from_generation']}): PPO grad clip {b['grad_clip']:g} -> "
                  f"{r['ppo_grad_clip']:g}, PPG keeps {r['ppg_grad_clip']:g}; PPO rate {b['learning_rate']*b['multiplier']:.3g} -> "
                  f"{a['learning_rate']:.3g} as the base (multiplier 1, floor 4e-06); PPO Adam moments rescaled once "
                  f"(exp_avg x{s['first_moment']:.3f}, exp_avg_sq x{s['second_moment']:.2f}, from {s['steps']} committed "
                  f"steps); the soft head steps at the PPO rate and the era heads at the PPG rate (the PPO rate in PPO).",flush=True)
        from brass_ai.recovery import training_revision_g5466 as g5466
        if g5466.applied(options):
            report['g5466_auxiliary_revision']=g5466.summary(options)
            r=report['g5466_auxiliary_revision'];s=r['adam_rescale']
            print(f"g5466 auxiliary revision (from g{r['from_generation']}): Rail-end heads frozen and preserved in PPO "
                  f"like Canal/Rail; PPG grad clip {s['old_clip']:g} -> {r['ppg_grad_clip']:g} (PPO keeps its clip, the "
                  f"heads keep 1.0); PPG Adam moments rescaled once (exp_avg x{s['first_moment']:.3f}, exp_avg_sq "
                  f"x{s['second_moment']:.3f}, from {s['steps']} logged steps of {len(s['norms'])} iterations; fit clip "
                  f"rate {s['predicted_clipped_percent']:.1f}% vs logged {s['logged_clipped_percent']:.1f}%); the era "
                  f"heads step at the PPO rate in PPG, fixed for the phase.",flush=True)
        from brass_ai.recovery import training_revision_g5685 as g5685
        if g5685.applied(options):
            report['g5685_clip_revision']=g5685.summary(options)
            r=report['g5685_clip_revision']
            for name in ('ppo','ppg'):
                s=r[name+'_adam_rescale']
                print(f"g5685 clip revision (from g{r['from_generation']}): {name.upper()} grad clip {s['old_clip']:g} -> "
                      f"{s['new_clip']:g}; {name.upper()} Adam moments rescaled once (exp_avg x{s['first_moment']:.4f}, "
                      f"exp_avg_sq x{s['second_moment']:.4f}, from {s['steps']} logged steps of {len(s['norms'])} "
                      f"iterations; fit clip rate {s['predicted_clipped_percent']:.1f}% vs logged "
                      f"{s['logged_clipped_percent']:.1f}%).",flush=True)
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
