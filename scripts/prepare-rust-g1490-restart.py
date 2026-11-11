#!/usr/bin/env python3
"""Prepare a new run that restarts the g1290 restart run from its g1490 weights.

The source run is read only; stop its trainer first. The full g1490 and g1491
checkpoints are gone, so the new run starts from the exact committed g1491 state
(the last one before the g1491 revision, verified against its export) with the
retained g1490 snapshot as the learner, counted as generation 1491. Optimizers are
new (the PPO warmup restarts; its LR multiplier is the one g1491 committed),
teacher labels and beta start empty, and the search head keeps Stage C while its
value buffer refills.

The recipe is the g1490 restart variant of the g1491 revision: 32 teacher tables
(+4 held out), long-search labels only, 72 teacher rows per PPO step, the
four-iteration 4:3:2:1 buffer, two PPO epochs, one shadow iteration that
re-measures beta with the PPO LR held, and the same visit-count and cotton
exploration. Then run continue-rust-g1490-restart.py.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

SOURCE='artifacts/step5/joint7-v11-2-g1290-restart-g1290'
SNAPSHOT='results/joint7-v11-2-g1290-restart-g1290-g1491-20261110T191929753967Z'
LATER='results/joint7-v11-2-g1290-restart-g1290-g1500-20261110T232721030490Z'
RUN='artifacts/step5/joint7-v11-2-g1490-restart'
WEIGHTS=SOURCE+'/population/weights-4a1f58e5f59319eb4389216ca0e2f06c47adf07f00f27d755bf2479b7ed68c45-aux-602cd59664e409fd.pt'


def main():
    from brass_ai.recovery.branch_restart import prepare
    from brass_ai.recovery.teacher_gradient_qualification import valid_target

    def share_target(text):
        try:value=float(text)
        except ValueError:raise argparse.ArgumentTypeError('share target must be a number')
        if not valid_target(value):raise argparse.ArgumentTypeError('share target must be above 0 and at most 1')
        return value
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--source-run',default=SOURCE)
    p.add_argument('--snapshot',default=SNAPSHOT,help='Results export of g1491, the committed state the new run starts from')
    p.add_argument('--later-export',default=LATER,
        help='A later export of the source run; its rating archive replaces rating slots the source has since compacted')
    p.add_argument('--run',default=RUN,help='New, empty run folder')
    p.add_argument('--weights',default=WEIGHTS,help='The retained g1490 snapshot (checked against the g1491 state)')
    p.add_argument('--share-target',type=share_target,required=True,
        help='Teacher gradient share above 0 and at most 1; pass the same value to continue-rust-g1490-restart.py')
    p.add_argument('--device',default='cpu',help='Device for rebuilding any missing opponent probe caches')
    a=p.parse_args()
    print(json.dumps(prepare(a.root.resolve(),a.source_run,a.snapshot,a.run,share_target=a.share_target,
        weights=a.weights,device=a.device,rollback=dict(later=a.later_export)),indent=2),flush=True)


if __name__=='__main__':main()
