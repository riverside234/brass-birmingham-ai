"""Prepare a NEW neutral run from g1290 weights and the source pool through g1330."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    from brass_ai.recovery.fresh_selfplay import prepare
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--source-run',default='artifacts/step5/joint7-v11-2-grouped-lr-critic')
    p.add_argument('--expected-source-generation',type=int,default=1360)
    p.add_argument('--seed-generation',type=int,default=1290)
    p.add_argument('--history-through',type=int,default=1330)
    p.add_argument('--run',default='artifacts/step5/joint7-v11-2-neutral-from-g1290')
    p.add_argument('--output',default='configs/step5/joint7-v11-2-neutral-from-g1290.json')
    a=p.parse_args()
    print(json.dumps(prepare(a.root.resolve(),a.source_run,a.run,a.output,
        expected_generation=a.expected_source_generation,seed_generation=a.seed_generation,history_through=a.history_through),indent=2))


if __name__=='__main__':main()
