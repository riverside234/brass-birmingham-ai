"""Collect complete four-seat games with frozen bank and/or model opponents."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import LAB_ROOT
from brass_ai.learning.legacy_policy import LegacyPolicy
from brass_ai.learning.runner import collect_games,match_report,ModelPolicy,RandomPolicy
from brass_ai.learning.checkpoint import load_checkpoint
from brass_ai.learning.replay import write_shard
from brass_ai.learning.contracts import project_path
import json
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT);p.add_argument('--policies',nargs=4,required=True)
    p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--output',required=True);p.add_argument('--purpose',choices=['training','evaluation','diagnostic'],default='training');p.add_argument('--mono')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--threads',type=int,default=2)
    args=p.parse_args();root=args.root.resolve()
    if args.threads<1:p.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    with ExitStack() as stack:
        policies=[]
        for identity in args.policies:
            if identity.startswith('model:'):
                model,_=load_checkpoint(project_path(root,identity[6:]),device=args.device)
                policy=ModelPolicy(model,device=args.device)
            elif identity=='random':policy=RandomPolicy()
            else:
                policy=LegacyPolicy(root,identity,purpose=args.purpose,mono=args.mono);stack.callback(policy.close)
            policies.append(policy)
        games=collect_games(project_path(root,'artifacts/r1-host/guarded'),args.seeds,policies,workers=args.workers,mono=args.mono,sampling_seed=71519)
        write_shard(project_path(root,args.output),games,dict(producer='R3-bank/1.0.0',purpose=args.purpose,
            policy_ids=[p.policy_id for p in policies],requested_policies=args.policies))
        print(json.dumps(match_report(games)))

if __name__=='__main__':main()
