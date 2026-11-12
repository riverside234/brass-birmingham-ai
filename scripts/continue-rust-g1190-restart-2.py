#!/usr/bin/env python3
"""Continue the second g1190 restart run (prepare-rust-g1190-restart-2.py), in one command.

Like continue-rust-g1190-restart.py, with 256 teacher rows per PPO step and the
insolvency charge kept (.0025, cap .025) in the reward:

1. With --skip-utility-qualification, start the one-iteration beta qualification
   immediately, without waiting for a utility fit. This choice persists across
   iterations and resumes. No teacher gradient is applied and the PPO LR holds.
   Without the flag, utility readiness is required before teacher collection.
2. Teaching switches on through the reviewed transition; the fresh search head
   starts Stage A (50 iterations), then Stage B (50) and Stage C (10).

The defaults are the CPU path with 64 workers and n=96 short searches (K=8);
long searches use n=640, K=16 until retuned. --iterations counts every iteration from
--expected-generation, calibration included. After an interruption, rerun with
--expected-generation set to the last committed generation: the command resumes
whichever step is unfinished.

--upgrade-utility-head applies a one-time same-run conversion at that committed
generation: utility hidden width 32 -> 256, Stage A budget 50 -> 100 (completed
iterations retained), then one fresh beta measurement without waiting for utility
qualification. Interrupted uncommitted work is archived before replaying it.
The learned utility/search weights, PPO/PPG moments, replay and RNG are retained.

--refresh-search-head replaces the 256-wide search head once with a fresh 32-wide
head at the committed generation. It restarts Stage A with w=0 and lambda=.8,
resets PPG beta to .25 and re-measures teacher beta with 96 teacher rows per PPO step while teaching stays on.
--adaptive-teacher-beta enables ongoing adjustment from the accepted beta.
Set --share-target at a committed iteration boundary. Adaptive beta carries
its current weight and running gradient norms into the new target.

--retune-teacher-search adopts long K=8 and C_scale=0.15 with short C_scale=0.5
at a committed Stage A boundary. It re-measures beta with the teacher on, retires
the old teacher labels and search values, and holds Stage A during buffer refill.
Each search-value fit runs for at most one epoch with the existing early stop.
Subsequent continuations use the committed search settings automatically.
"""
import importlib.util
from pathlib import Path

RUN = 'artifacts/step5/joint7-v11-2-g1190-restart-2'


def main():
    path = Path(__file__).with_name('continue-rust-g1491.py')
    spec = importlib.util.spec_from_file_location('rust_g1491_continuation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(run=RUN, teacher_fast=96, description=__doc__, share_target_required=True)


if __name__ == '__main__':
    main()
