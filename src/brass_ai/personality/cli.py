"""Separate P0/P1 entry point. No Step 5 signatures or installed AI are changed."""
import argparse
import json
from pathlib import Path
import torch

from brass_ai.learning.contracts import LAB_ROOT,require
from .handoff import create,inspect
from .training import execute


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=LAB_ROOT)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--threads',type=int,default=4)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('inspect');p.add_argument('--from-run',required=True)
    p=sub.add_parser('handoff');p.add_argument('--from-run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--trace-shards',type=int,default=4);p.add_argument('--references',type=int,default=3)
    for name in ('warmup','joint-pilot','validate-goals'):
        p=sub.add_parser(name);p.add_argument('--handoff',required=True);p.add_argument('--run',required=True)
        p.add_argument('--epochs',type=int,default=3 if name=='warmup' else 2)
        p.add_argument('--games',type=int,default=0 if name=='warmup' else 20)
        p.add_argument('--workers',type=int,default=4);p.add_argument('--mono')
    a=parser.parse_args();require(1<=a.threads<=16,'Invalid P1 tensor thread count');torch.set_num_threads(a.threads)
    require(a.device!='cuda' or torch.cuda.is_available(),'CUDA unavailable in this Python environment')
    if a.command=='inspect':
        data=inspect(a.root,a.from_run)
        result=dict(eligible=True,parent=data['entry'],reward=data['reward'],calibration=data['model'].utility_calibration)
    elif a.command=='handoff':
        result=create(a.root,a.from_run,a.output,trace_shards=a.trace_shards,reference_count=a.references)
        result={k:result[k] for k in ('format','parent','reward','trace_status')}
    else:
        result=execute(a.root,a.handoff,a.run,mode={'warmup':'warmup','joint-pilot':'joint','validate-goals':'validate'}[a.command],
            epochs=a.epochs,games=a.games,workers=a.workers,device=a.device,mono=a.mono)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
