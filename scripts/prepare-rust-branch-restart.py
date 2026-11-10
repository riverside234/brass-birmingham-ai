#!/usr/bin/env python3
"""Prepare a new run folder that restarts the Rust teacher run from a committed generation.

The source run is read only. Stop its trainer first. The new run starts from the
exact committed state in the results export of that generation and its retained
weights, with new optimizers and empty buffers. Its first four iterations run the
teacher in shadow mode to measure a new beta; continue-rust-teacher.py then turns
teaching on from the fifth.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from brass_ai.recovery.branch_restart import prepare,STAGE_A
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--source-run',default='artifacts/step5/joint7-v11-2-g1290-trained-restart')
    p.add_argument('--snapshot',default='results/joint7-v11-2-g1290-trained-restart-g1290-20261109T041246972332Z',
        help='Results export of the restart generation (its state.json is verified against the export receipt)')
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-g1290-restart-g1290',help='New, empty run folder')
    p.add_argument('--weights',help='Optional: the retained weights file; it must be that generation\'s snapshot')
    p.add_argument('--share-target',type=float,default=.1,help='Teacher gradient share used to measure the new beta')
    p.add_argument('--stage-a-iterations',type=int,choices=(10,30),default=STAGE_A)
    p.add_argument('--device',default='cpu',help='Device for rebuilding any missing opponent probe caches')
    a=p.parse_args()
    print(json.dumps(prepare(a.root.resolve(),a.source_run,a.snapshot,a.run,share_target=a.share_target,
        stage_a_iterations=a.stage_a_iterations,weights=a.weights,device=a.device),indent=2),flush=True)


if __name__=='__main__':main()
