#!/usr/bin/env python3
"""Continue the g1490 restart run (prepare-rust-g1490-restart.py), in one command.

1. One shadow iteration re-measures the fixed teacher beta on long-search labels.
   No teacher gradient is applied, the search head waits and the PPO LR holds.
2. Teaching switches on through the reviewed transition and runs the rest.

The defaults are the CPU path with 64 workers and n=96 short searches (K=8);
long searches stay n=640, K=16. --iterations counts every iteration from
--expected-generation, calibration included. After an interruption, rerun with
--expected-generation set to the last committed generation: the command resumes
whichever step is unfinished.
"""
import importlib.util
from pathlib import Path

RUN = 'artifacts/step5/joint7-v11-2-g1490-restart'


def main():
    path = Path(__file__).with_name('continue-rust-g1491.py')
    spec = importlib.util.spec_from_file_location('rust_g1491_continuation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(run=RUN, teacher_fast=96, description=__doc__, share_target_required=True)


if __name__ == '__main__':
    main()
