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
    migrations.add_argument('--update-cotton-network', action='store_true',
        help='Add one cotton network per era, use 15%% cotton teacher tables and cap ordinary PPG at two epochs.')
    migrations.add_argument('--retune-teacher-search', action='store_true',
        help='Use long K=8 with C_scale=0.15 and short C_scale=0.5, re-measure beta with teaching on and refill the search buffers')
    migrations.add_argument('--split-teacher-c-scale', action='store_true',
        help='Use long C_scale=0.20 on standard tables and 0.10 on cotton tables with 25%% cotton exploration. '
             'Refill the search buffers and re-measure beta with teaching on.')
    migrations.add_argument('--update-teacher-scales', action='store_true',
        help='Use long C_scale=0.10 and short C_scale=0.70 for all teacher tables and refill their training buffers.')
    migrations.add_argument('--update-rail-cotton', action='store_true',
        help='Exclude Rail-era level 1 and 2 cotton builds from PPO and teacher cotton exploration '
             '(level 1 stays excluded in both eras). Keeps labels, buffers and the search-head stage.')
    migrations.add_argument('--update-cotton-timing', action='store_true',
        help='Time cotton exploration within each era on PPO and teacher cotton tables by the newest reviewed rule '
             '(g3528, from the g3471 or g2654 recipe): none at a seat\'s last Canal action or its first two and last Rail '
             'actions; no cotton-city network at Canal 14 or Rail 15; cotton builds only at Rail 15. Keeps labels, '
             'buffers and the search-head stage; the setting persists (training_revision_g3528).')
    migrations.add_argument('--update-search-lambda', action='store_true',
        help='Use lambda .95 (was .8) for search-value returns. Buffered games are retargeted; '
             'keeps labels, buffers, the search head and its stage.')
    migrations.add_argument('--update-g2600-recipe', action='store_true',
        help='Use opening temperature 1.3 for three actions in PPO and teacher games, long C_scale=0.20 on all '
             'teacher tables, a .25 PPG beta floor and one-epoch ordinary PPG. Clears both teacher search buffers '
             'and re-measures beta with teaching on.')
    parser.add_argument('--value-revision', metavar='ITEMS',
        help='Adopt g3140 value-phase revision items at this committed generation, comma-separated: '
             'time, validation, heads, targets, weight=0.15, weight=0.5, vp, rail-end, ownership-stop, search-aux, '
             'search-rows, drop-rail-end. '
             'Items persist and only move forward (training_revision_g3140).')
    parser.add_argument('--rating-every', type=int, choices=(20,),
        help='From this committed generation on, run the PL strength rating every 20 iterations '
             '(was every 10). Committed slots are kept; set once (rating_cadence).')
    parser.add_argument('--teacher-revision', action='store_true',
        help='Adopt the g3270 teacher revision at this committed generation (build steps 2-3): 36+4 tables every '
             'iteration, 8-iteration labels, 128 rows per PPO step without repeats, targets rebuilt from the current '
             'policy; search value: 16-iteration buffer, 32 steps, 32 rotating held-out games. Set once '
             '(training_revision_g3270); beta is re-measured once with teaching on.')
    parser.add_argument('--soft-policy', action='store_true',
        help='Adopt the g3423 soft policy head at this committed generation (soft policy step S1): a trainer-owned '
             'Linear(192->1) on the hidden layer of the policy scorer predicts the rebuilt label target at T=4 on the '
             '128 long-label rows per PPO step of the teacher term, weight 8 x beta; 2 head-only iterations, then a '
             '5-iteration trunk ramp. Needs the g3270 teacher revision. Set once (training_revision_g3423).')
    parser.add_argument('--soft-policy-weight', type=float, choices=(2., 4., 6., 8., 10., 12., 16.),
        help='Set the weight of the adopted soft policy head to this many x beta at this committed generation (8 at '
             'adoption, 6 from g3471); the head, its Adam and warm-up carry over. Each change is kept until the next '
             '(training_revision_g3423).')
    parser.add_argument('--ppg-beta-floor', type=float, choices=(.125,),
        help='Lower the PPG beta floor .25 -> .125 at this committed generation, in the value phase and in the '
             'search-head Stage C schedule, and set the PPG beta to .125 once (hotfix); trips, Adams, RNG and replay '
             'carry over. The floor persists (training_revision_g3658).')
    parser.add_argument('--teacher-search-revision', action='store_true',
        help='Adopt the g4138 teacher search revision at this committed generation: 44+4 teacher tables every '
             'iteration and short searches of 128 simulations at C_scale 0.6 (long search unchanged). Long-search '
             'labels, beta and the search head carry over; the search-value head trains only on games collected '
             'from here, after eight such ages. Set once (training_revision_g4138); the CPU teacher is rebuilt.')
    parser.add_argument('--teacher-training-revision', action='store_true',
        help='Adopt the g4475 teacher training revision at this committed generation: 160 teacher label rows per '
             'PPO step (was 128), and the search-value head\'s training paused (no post-PPG pass, no stage '
             'advance; head, Adam and stage kept). Search tables still publish their value shards and train the '
             'era heads. Trainer-only; set once (training_revision_g4475).')
    parser.add_argument('--table-revision', action='store_true',
        help='Adopt the g4551 table revision at this committed generation: every PPO self-play table is a standard '
             'game (the teacher cotton tables stay), 50+4 teacher tables every iteration (was 44+4) and 176 teacher '
             'label rows per PPO step (was 160). Set once (training_revision_g4551).')
    parser.add_argument('--era-head-revision', action='store_true',
        help='Adopt the g4770 era-head revision at this committed generation: the Rail-end heads come back fresh at '
             'weight 0.05, trained only in PPG in the same batches as the Canal/Rail heads (no PPO snapshot or '
             'preservation; PPO-game rows on the actual game-end map, search rows on lambda 0.95 returns after a '
             'four-iteration warm-up); every era head trains PPO-game rows on actual maps (lambda 1); the search '
             'pool is a 4-iteration sample of the last 8 iterations\' tables. Set once (training_revision_g4770).')
    parser.add_argument('--rail-end-weight', type=float, choices=(0.25,),
        help='Raise the restored Rail-end heads\' weight from 0.05 to 0.25 at this committed generation '
             '(after --era-head-revision). The setting persists.')
    parser.add_argument('--optimizer-revision', action='store_true',
        help='Adopt the g5402 optimizer revision at this committed generation: PPO grad clip 5.83 (was 1.0) while '
             'PPG keeps its own 1.0; the PPO rate in force is cut by 20%% and becomes the base at multiplier 1 (the '
             'controller restarts its KL window and runs its cooldown; the 4e-6 floor stays) and the PPO Adam moments '
             'are rescaled once to the new clip; the soft policy head '
             'steps at the PPO rate and the era heads at the PPG rate (the PPO rate in PPO). Trainer-only; set once '
             '(training_revision_g5402).')
    parser.add_argument('--auxiliary-revision', action='store_true',
        help='Adopt the g5466 auxiliary revision at this committed generation (after --optimizer-revision): the '
             'Rail-end heads are frozen and preserved in PPO like the Canal/Rail heads; PPG clips at 2.2 (was 1.0) '
             'and the PPG Adam moments are rescaled once to the new clip from the logged norms of the last 30 '
             'committed PPG phases; the era heads step at the PPO rate in PPG too, fixed for the phase (no warm-up '
             'or decay). The era heads and the soft head keep their own clips at 1.0. Trainer-only; set once '
             '(training_revision_g5466).')
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
    elif args.update_g2600_recipe:
        profile = g1491.G2600_SEARCH
    elif args.retune_teacher_search and profile is None:
        profile = g1491.RETUNED_SEARCH
    from brass_ai.recovery import training_revision_g4138 as g4138
    if profile is not None and (args.teacher_search_revision or g4138.applied(state['signature']['config'])):
        # g4138: the recipe owns the short search (128 simulations, K 8, C_scale .6), not --teacher-fast.
        profile = g4138.short_search(profile)
        args.teacher_fast, args.teacher_fast_k = profile['fast'], profile['fast_k']
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
    if args.retune_teacher_search or args.split_teacher_c_scale or args.update_teacher_scales or args.update_g2600_recipe:
        from brass_ai.recovery.teacher_search_retune import prepare
        options = (dict(g2600=True) if args.update_g2600_recipe else dict(update_scales=True) if args.update_teacher_scales
                   else dict(split_c_scale=args.split_teacher_c_scale))
        state=prepare(ROOT,run,expected_generation=generation,**options)
    end = generation+args.iterations
    if not g1491.applied(state['signature']['config']) and interrupted(run, generation):
        # Its tables were collected under the current recipe and the earlier short-search
        # budget (n=96): finish it with exactly those.
        print(f'Finishing interrupted iteration g{generation}->g{generation+1} under its own recipe first', flush=True)
        recovery = copy.copy(args); recovery.inference = 'cpu'
        recovery.teacher_fast = PREVIOUS_TEACHER_FAST
        recovery.value_revision = None  # an interrupted iteration finishes under its own recipe
        recovery.rating_every = None
        recovery.teacher_revision = False
        recovery.soft_policy = False
        recovery.soft_policy_weight = None
        recovery.update_cotton_timing = False
        recovery.ppg_beta_floor = None
        recovery.teacher_search_revision = False
        recovery.teacher_training_revision = False
        recovery.table_revision = False
        recovery.era_head_revision = False
        recovery.rail_end_weight = None
        recovery.optimizer_revision = False
        recovery.auxiliary_revision = False
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
