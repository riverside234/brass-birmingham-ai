#!/usr/bin/env python3
"""Prepare a new run that restarts the g1290 restart run from its g1190 weights.

The source run is read only; stop its trainer first. The new run starts from the
exact committed g1491 state (the last one before the g1491 revision, verified
against its export) with the retained g1190 snapshot as the learner, counted as
generation 1491. g1190 predates the teacher phase and the post-transfer reward:
its tensors are kept, its utility moves to the restart reward (placement .1, zero
insolvency charge) and is re-fit by the first value phase, and a fresh search head
is adopted at Stage A when teaching starts.

The recipe is the g1490 restart recipe (32 teacher tables +4 held out, long-search
labels only, 72 rows per PPO step, the 4:3:2:1 buffer, two PPO epochs, one shadow
iteration re-measuring beta with the PPO LR held, visit-count and cotton
exploration) with placement .1 and the restart stage schedule: Stage A 50
iterations fitting with the Stage C held-out early stop, Stage B 50, and a PPG beta
floor of .5 once the head has left Stage A. Then run continue-rust-g1190-restart.py.
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
RUN='artifacts/step5/joint7-v11-2-g1190-restart'
LEARNER=1190
WEIGHTS=SOURCE+'/population/weights-d872ea925d6643ee3a3eeec99a52d16c6a78c2de1f757617f1aaaa47d175f864-aux-dd8dcbfaf5350001.pt'


def main():
    from brass_ai.recovery.branch_restart import prepare,SEED
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
    p.add_argument('--weights',default=WEIGHTS,help='The retained g1190 snapshot (checked against the g1491 state)')
    p.add_argument('--share-target',type=share_target,required=True,
        help='Teacher gradient share above 0 and at most 1; pass the same value to continue-rust-g1190-restart.py')
    p.add_argument('--device',default='cpu',help='Device for rebuilding any missing opponent probe caches')
    a=p.parse_args()
    print(json.dumps(prepare(a.root.resolve(),a.source_run,a.snapshot,a.run,share_target=a.share_target,
        weights=a.weights,device=a.device,rollback=dict(later=a.later_export,revision=SEED,learner_generation=LEARNER)),
        indent=2,default=str),flush=True)


if __name__=='__main__':main()
