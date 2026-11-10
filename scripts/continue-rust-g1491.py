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


def interrupted(run, generation):
    """The current iteration sealed teacher work or a checkpoint before the stop."""
    return bool(list((run/'teacher-labels').glob(f'generation-{generation:06d}-*')) or
                list((run/'search-values').glob(f'{generation:06d}-*')) or
                list(run.glob(f'generation-{generation+1:04d}.pt*')))


def calibrated(state):
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
    return False, g1491.SHADOW_ITERATIONS-len(tail)


def main():
    module = shared()
    args = module.parser(default_mode='on', default_stage_revision=True, workers=64, teacher_fast=128, inference='cpu',
                         run=RUN, iterations=10, description=__doc__).parse_args()
    from brass_ai.learning.contracts import project_path, require
    run = project_path(ROOT, args.run)
    state = state_of(run)
    generation = state['iteration']
    require(generation == args.expected_generation and args.iterations > 0,
            f"The run is at g{generation}; pass --expected-generation {generation}")
    end = generation+args.iterations
    from brass_ai.recovery import training_revision_g1491 as g1491
    if not g1491.applied(state['signature']['config']) and interrupted(run, generation):
        # Its tables were collected under the current recipe and the earlier short-search
        # budget (n=96): finish it with exactly those.
        print(f'Finishing interrupted iteration g{generation}->g{generation+1} under its own recipe first', flush=True)
        recovery = copy.copy(args); recovery.inference = 'cpu'
        recovery.teacher_fast = PREVIOUS_TEACHER_FAST
        state = module.continuation(recovery, mode='on', expected=generation, iterations=1, revise_stages=True)
        generation = state['iteration']
    done, needed = calibrated(state)
    attempts = 0
    while not done and generation < end:
        require(attempts <= EXTRA_CALIBRATION, 'Beta calibration did not qualify; inspect teacher_phase.term in the '
                'latest report (full 48+4 tables, finite gradients) before continuing')
        count = min(needed if attempts == 0 else 1, end-generation)
        print(f'g1491 beta calibration: {count} shadow iteration(s) from g{generation}', flush=True)
        state = module.continuation(args, mode='shadow', expected=generation, iterations=count, revision='g1491')
        generation = state['iteration']; attempts += 1
        done, needed = calibrated(state)
    if generation < end:
        require(done, 'Beta calibration is unfinished')
        print(f'Teacher on from g{generation} with the calibrated fixed beta', flush=True)
        module.continuation(args, mode='on', expected=generation, iterations=end-generation, revise_stages=True)


if __name__ == '__main__':
    main()
