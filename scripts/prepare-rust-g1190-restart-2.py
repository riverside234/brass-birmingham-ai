#!/usr/bin/env python3
"""Prepare the second g1190 restart: the first one again from g1491, with two changes.

The first attempt (prepare-rust-g1190-restart.py) ran g1491 to g1500. This restart
starts from the same exact committed g1491 state with the same retained g1190 snapshot
as the learner, in a new run folder, and changes only:

1. Teacher rows per PPO step: 256 (was 72).
2. The insolvency charge stays in the reward (event cost .0025, cap .025) instead of
   being removed. The reward is placement .1, win 1, VP .001 (cap 180) with the
   canal/final VP timing, plus that charge. The seed's utility moves to it with all
   tensors kept and its calibration cleared, so the first value phase re-fits it.

Everything else is as in the first attempt: 32 teacher tables (+4 held out), long-search
labels only, the 4:3:2:1 buffer, two PPO epochs, one shadow iteration re-measuring beta
with the PPO LR held, visit-count and cotton exploration, and a fresh 256-wide search
head (Stage A 50 iterations with the held-out early stop, Stage B 50, Stage C 10, PPG
beta floor .5 after Stage A). The source run is read only; stop any trainer first.
Then run continue-rust-g1190-restart-2.py.
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
RUN='artifacts/step5/joint7-v11-2-g1190-restart-2'
LEARNER=1190
WEIGHTS=SOURCE+'/population/weights-d872ea925d6643ee3a3eeec99a52d16c6a78c2de1f757617f1aaaa47d175f864-aux-dd8dcbfaf5350001.pt'


def main():
    from brass_ai.recovery.branch_restart import prepare,SEED_CHARGED
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
    p.add_argument('--run',default=RUN,help='New, empty run folder (not the first g1190 restart\'s)')
    p.add_argument('--weights',default=WEIGHTS,help='The retained g1190 snapshot (checked against the g1491 state)')
    p.add_argument('--share-target',type=share_target,required=True,
        help='Teacher gradient share above 0 and at most 1; pass the same value to continue-rust-g1190-restart-2.py')
    p.add_argument('--device',default='cpu',help='Device for rebuilding any missing opponent probe caches')
    a=p.parse_args()
    print(json.dumps(prepare(a.root.resolve(),a.source_run,a.snapshot,a.run,share_target=a.share_target,
        weights=a.weights,device=a.device,rollback=dict(later=a.later_export,revision=SEED_CHARGED,learner_generation=LEARNER)),
        indent=2,default=str),flush=True)


if __name__=='__main__':main()
