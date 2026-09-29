"""Quick additional experiment: one legal decision per variant, no tournament."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from brass_ai.learning.contracts import LAB_ROOT,project_path,read_json,require
from brass_ai.searchlab.experiment import quick_smoke,validate_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=LAB_ROOT)
    parser.add_argument('--checkpoint',default='training/models/releases/step5-selfplay-medium-smoke.pt')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--seconds',type=float,default=1.)
    parser.add_argument('--future-turns',type=int,choices=[1,2],default=1)
    parser.add_argument('--config',default='configs/searchlab/small.json')
    parser.add_argument('--workers',type=int,default=1,help='Up to three independent processes, still only three decisions total')
    parser.add_argument('--mono')
    args=parser.parse_args();require(args.root.is_absolute(),'Use an absolute project root');root=args.root.resolve()
    require(args.device!='cuda' or torch.cuda.is_available(),'CUDA requested but unavailable')
    config=validate_config(read_json(project_path(root,args.config)))
    config.update(seconds=args.seconds,future_turns=args.future_turns)
    torch.set_num_threads(config['threads'])
    print(json.dumps(quick_smoke(root,args.checkpoint,config,workers=args.workers,device=args.device,mono=args.mono),indent=2))


if __name__=='__main__':main()
