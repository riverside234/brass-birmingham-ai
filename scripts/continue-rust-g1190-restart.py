#!/usr/bin/env python3
"""Continue the g1190 restart run (prepare-rust-g1190-restart.py), in one command.

1. Shadow iterations: the first re-fits the utility under the restart reward (no
   teacher tables until it is ready), the next re-measures the fixed teacher beta on
   long-search labels. No teacher gradient is applied and the PPO LR holds.
2. Teaching switches on through the reviewed transition; the fresh search head
   starts Stage A (50 iterations), then Stage B (50) and Stage C (10).

The defaults are the CPU path with 64 workers and n=96 short searches (K=8);
long searches stay n=640, K=16. --iterations counts every iteration from
--expected-generation, calibration included. After an interruption, rerun with
--expected-generation set to the last committed generation: the command resumes
whichever step is unfinished.
"""
import importlib.util
from pathlib import Path

RUN = 'artifacts/step5/joint7-v11-2-g1190-restart'


def main():
    path = Path(__file__).with_name('continue-rust-g1491.py')
    spec = importlib.util.spec_from_file_location('rust_g1491_continuation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(run=RUN, teacher_fast=96, description=__doc__, share_target_required=True)


if __name__ == '__main__':
    main()
