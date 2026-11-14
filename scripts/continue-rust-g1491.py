#!/usr/bin/env python3
"""Continue the g1290 restart run with the g1491 recipe revision, in one command.

1. If the run was stopped after the current iteration had sealed its teacher
   tables, that iteration is finished first under the recipe it was collected with.
2. Two shadow iterations re-measure the fixed teacher beta for the new label mix.
   No teacher gradient is applied, the search head waits and the PPO LR holds.
3. Teaching switches on through the reviewed transition and runs the rest.

--iterations counts every iteration from --expected-generation, calibration
included. After an interruption, rerun with --expected-generation set to the
last committed generation: the command resumes whichever step is unfinished.
"""
import copy
import importlib.util
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
RUN = 'artifacts/step5/joint7-v11-2-g1290-restart-g1290'
EXTRA_CALIBRATION = 2
# The short-search budget the run used up to g1491 (the continue scripts' default).
PREVIOUS_TEACHER_FAST = 96


def shared():
    path = Path(__file__).with_name('continue-rust-shadow.py')
    spec = importlib.util.spec_from_file_location('rust_shadow_continuation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_of(run):
    from brass_ai.bank import read
    return read(run/'state.json')


def ensure_cpu(run, state):
    """Rebuild a stale worker once, before adopting the new recipe. No benchmark."""
    from brass_ai.rust_backend.lab_cpu_runtime import resolve
    from brass_ai.rust_backend.lab_shadow import teacher_build
    core=teacher_build(state)['core_source']
    try:
        resolve(ROOT,teacher_core=core)
    except (ValueError, FileNotFoundError) as error:
        print(f'Preparing the updated CPU teacher: {error}',flush=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/prepare-rust-cpu-inference.py'),
            '--run',str(run),'--expected-generation',str(state['iteration']),'--focused-regressions'],check=True,cwd=ROOT)
        resolve(ROOT,teacher_core=core)


def interrupted(run, generation):
    """The current iteration sealed teacher work or a checkpoint before the stop."""
    return bool(list((run/'teacher-labels').glob(f'generation-{generation:06d}-*')) or
                list((run/'search-values').glob(f'{generation:06d}-*')) or
                list(run.glob(f'generation-{generation+1:04d}.pt*')))


def calibrated(state, *, skip_utility_qualification=False):
    from brass_ai.rust_backend.lab_shadow_config import is_shadow
    from brass_ai.recovery import training_revision_g1491 as g1491
    config = state['signature']['config']
    if not g1491.applied(config):
        return False, g1491.SHADOW_ITERATIONS
    if not is_shadow(config):
        return True, 0
    q = state.get('teacher_phase', {}).get('weight', {}).get('gradient_qualification')
    if q is not None and q['passed']:
        return True, 0
    tail = [o for o in (q or {}).get('observations', []) if o['generation'] == state['iteration']-1]
    # A changed reward first needs one value phase to re-fit the utility (teacher readiness);
    # that iteration plans no teacher tables.
    return False, g1491.shadow_iterations(config)-len(tail)+(0 if ready(state,skip_utility_qualification=skip_utility_qualification) else 1)


def ready(state, *, skip_utility_qualification=False):
    from brass_ai.learning.contracts import digest
    from brass_ai.recovery.utility import reward_spec
    config = state['signature']['config']
    if skip_utility_qualification or config.get('teacher_phase',{}).get('skip_utility_qualification',False):return True
    receipt = state.get('teacher_phase', {}).get('readiness') or {}
    return (receipt.get('ready') is True and receipt.get('utility_qualification') != 'skipped'
            and receipt.get('reward_id') == digest(reward_spec(config)))


def main(*, run=RUN, teacher_fast=128, description=__doc__, share_target_required=False):
    module = shared()
    parser = module.parser(default_mode='on', default_stage_revision=True, workers=64, teacher_fast=teacher_fast, inference='cpu',
                           run=run, iterations=10, description=description)
    migrations = parser.add_mutually_exclusive_group()
    migrations.add_argument('--upgrade-utility-head',action='store_true',
        help='Widen the g1190 run utility 32 to 256 once, extend Stage A from 50 to 100 and re-measure beta')
    migrations.add_argument('--refresh-search-head', action='store_true',
        help='Replace the g1190 search head with width 32 and re-measure beta with the teacher on')
    migrations.add_argument('--reset-search-head', action='store_true',
        help='Restart Stage A with a fresh 32-wide head, the 50/20/30 ordinary mix and seated teacher cotton exploration')
    migrations.add_argument('--update-opening-sampling', action='store_true',
        help='Use temperature 2 for three opening actions, 10%% ordinary cotton targeting and a .5 PPG beta floor. '
             'Re-measure beta once with teaching on.')
    migrations.add_argument('--update-cotton-sampling', action='store_true',
        help='Adopt the single-seat cotton mix and disable normal teacher sampling at the committed boundary.')
    migrations.add_argument('--retune-teacher-search', action='store_true',
        help='Use long K=8 with C_scale=0.15 and short C_scale=0.5, re-measure beta with teaching on and refill the search buffers')
    migrations.add_argument('--split-teacher-c-scale', action='store_true',
        help='Use long C_scale=0.20 on standard tables and 0.10 on cotton tables with 25%% cotton exploration. '
             'Refill the search buffers and re-measure beta with teaching on.')
    migrations.add_argument('--update-teacher-scales', action='store_true',
        help='Use long C_scale=0.10 and short C_scale=0.70 for all teacher tables and refill their training buffers.')
    parser.add_argument('--teacher-search-head-weight', type=int, choices=(0, 1),
        help='Override the teacher search-head weight in the retuned recipe. '
             'After --reset-search-head the override applies from Stage C. The setting persists.')
    if share_target_required:
        # No silent default: the target is the run's own, passed the same way every time.
        action = next(a for a in parser._actions if a.dest == 'share_target')
        action.required, action.default = True, None
    args = parser.parse_args()
    from brass_ai.learning.contracts import project_path, require
    run = project_path(ROOT, args.run)
    state = state_of(run)
    generation = state['iteration']
    require(generation == args.expected_generation and args.iterations > 0,
            f"The run is at g{generation}; pass --expected-generation {generation}")
    from brass_ai.recovery import training_revision_g1491 as g1491
    profile = g1491.count_search(state['signature']['config'])
    if args.reset_search_head:
        profile = g1491.RETUNED_SEARCH
    elif args.split_teacher_c_scale:
        profile = g1491.COTTON_SPLIT_SEARCH
    elif args.update_teacher_scales:
        profile = g1491.UPDATED_SEARCH
    elif args.retune_teacher_search and profile is None:
        profile = g1491.RETUNED_SEARCH
    args.teacher_count_search = profile
    if args.teacher_search_head_weight is not None:
        require(profile is not None, 'Teacher game settings require the retuned g1190 recipe')
    if profile is not None:
        require(args.inference != 'gpu' and args.teacher_budget == 'count' and
                args.c_scale in (None, profile['c_scale']), 'Use the committed CPU teacher count-search profile')
        if args.inference == 'cpu':
            module.inference_choice(args)
    if getattr(args,'upgrade_utility_head',False):
        from brass_ai.recovery.utility_width_upgrade import prepare
        state=prepare(ROOT,run,expected_generation=generation)
    if args.refresh_search_head:
        from brass_ai.recovery.search_head_refresh import prepare
        state=prepare(ROOT,run,expected_generation=generation,adaptive_beta=args.adaptive_teacher_beta)
    ensure_cpu(run,state)
    if args.reset_search_head:
        from brass_ai.recovery.search_head_refresh import prepare
        state=prepare(ROOT,run,expected_generation=generation,adaptive_beta=args.adaptive_teacher_beta,reset=True)
    if args.retune_teacher_search or args.split_teacher_c_scale or args.update_teacher_scales:
        from brass_ai.recovery.teacher_search_retune import prepare
        options = dict(update_scales=True) if args.update_teacher_scales else dict(split_c_scale=args.split_teacher_c_scale)
        state=prepare(ROOT,run,expected_generation=generation,**options)
    end = generation+args.iterations
    if not g1491.applied(state['signature']['config']) and interrupted(run, generation):
        # Its tables were collected under the current recipe and the earlier short-search
        # budget (n=96): finish it with exactly those.
        print(f'Finishing interrupted iteration g{generation}->g{generation+1} under its own recipe first', flush=True)
        recovery = copy.copy(args); recovery.inference = 'cpu'
        recovery.teacher_fast = PREVIOUS_TEACHER_FAST
        state = module.continuation(recovery, mode='on', expected=generation, iterations=1, revise_stages=True)
        generation = state['iteration']
    done, needed = calibrated(state,skip_utility_qualification=args.skip_utility_qualification)
    attempts = 0
    while not done and generation < end:
        t = g1491.tables(state['signature']['config']) if g1491.applied(state['signature']['config']) else g1491.TABLES
        require(attempts <= EXTRA_CALIBRATION, 'Beta calibration did not qualify; inspect teacher_phase.term in the '
                f"latest report (full {t['train']}+{t['held_out']} tables, finite gradients) before continuing")
        count = min(needed if attempts == 0 else 1, end-generation)
        if not ready(state,skip_utility_qualification=args.skip_utility_qualification):
            print(f'Teacher readiness: the first shadow iteration re-fits the utility under the current reward; no teacher tables', flush=True)
        print(f'g1491 beta calibration: {count} shadow iteration(s) from g{generation}', flush=True)
        state = module.continuation(args, mode='shadow', expected=generation, iterations=count, revision='g1491')
        generation = state['iteration']; attempts += 1
        done, needed = calibrated(state,skip_utility_qualification=args.skip_utility_qualification)
    if generation < end:
        require(done, 'Beta calibration is unfinished')
        from brass_ai.recovery import adaptive_teacher_beta as adaptive
        if adaptive.pending(state.get('teacher_phase', {}).get('weight', {})):
            detail = 're-measuring beta with the teacher on'
        elif args.adaptive_teacher_beta or adaptive.KEY in state['signature']['config']['teacher_phase']:
            detail = 'using adaptive beta'
        else:
            detail = 'using the calibrated fixed beta'
        print(f'Teacher on from g{generation}, {detail}', flush=True)
        module.continuation(args, mode='on', expected=generation, iterations=end-generation, revise_stages=True)


if __name__ == '__main__':
    main()
