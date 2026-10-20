"""Preview/apply the reviewed g110 limit adjustment and write its continuation config."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from brass_ai.recovery.g110_hotfix import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True,help='Continuation config under configs/step5; no overwrite')
    parser.add_argument('--apply',action='store_true',help='Apply once; otherwise validate and preview')
    parser.add_argument('--refine-ppg',action='store_true',help='Also halve value LR and stratify whole-game critic holdouts')
    args=parser.parse_args()
    print(json.dumps(run(ROOT,args.run,args.output,apply=args.apply,refine_ppg=args.refine_ppg),indent=2))


if __name__=='__main__':main()
