"""Quick additional experiment: one legal decision per variant, no tournament."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from brass_ai.host import RulesHost
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.contracts import LAB_ROOT,project_path,read_json,require
from brass_ai.searchlab.experiment import make_policy,validate_config
from brass_ai.step5.policy import NeuralPolicy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=LAB_ROOT)
    parser.add_argument('--checkpoint',default='training/models/releases/step5-selfplay-medium-smoke.pt')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--seconds',type=float,default=1.)
    parser.add_argument('--future-turns',type=int,choices=[1,2],default=1)
    parser.add_argument('--mono')
    args=parser.parse_args();root=args.root.resolve()
    require(args.device!='cuda' or torch.cuda.is_available(),'CUDA requested but unavailable')
    torch.set_num_threads(2)
    config=read_json(root/'configs/searchlab/experiment.json')
    config.update(seconds=args.seconds,future_turns=args.future_turns)
    validate_config(config)
    model,raw=load_checkpoint(project_path(root,args.checkpoint),device=args.device)
    rows=[]
    with RulesHost(root/'artifacts/r1-host/guarded',mono=args.mono) as live:
        for variant in ('s2','r4','s3'):
            context=live.reset(61302)
            rollouts=[NeuralPolicy(model,args.device,exploration=.05,focus='neutral')]
            policy=make_policy(variant,model,root,args.device,config,rollouts,args.mono)
            try:
                began=time.monotonic();view=live.observe(context['actor']);choices=live.choices()
                index,_,teacher=policy.select_batch([(view,choices,random.Random(72050))])[0]
                require(type(index) is int and 0<=index<len(choices),'Illegal search output')
                live.apply(context,choices[index])
                rows.append(dict(variant=variant,legal=True,seconds=time.monotonic()-began,
                    teacher=teacher is not None,telemetry=policy.telemetry))
            finally:policy.close()
    print(json.dumps(dict(format='searchlab-quick-smoke/1.0.0',passed=True,model_id=raw['model_id'],
        decisions=rows,promoted=False,note='One decision per variant; no complete-game or strength claim.'),indent=2))


if __name__=='__main__':main()
