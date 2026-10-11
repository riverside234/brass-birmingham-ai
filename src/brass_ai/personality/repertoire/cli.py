"""P3 frozen-model discovery, train-only encoder fitting and immutable archives."""
import argparse
import json
from pathlib import Path
import torch
from brass_ai.learning.contracts import LAB_ROOT,require


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=LAB_ROOT)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--threads',type=int,default=4)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('propose');a.add_argument('--ready',required=True);a.add_argument('--output',required=True);a.add_argument('--count',type=int,default=64)
    a.add_argument('--seed',type=int,default=17701);a.add_argument('--seconds',type=float,default=12.);a.add_argument('--workers',type=int,default=4)
    a.add_argument('--decision-mode',choices=('direct','searched'),default='direct');a.add_argument('--from-archive')
    a=sub.add_parser('collect');a.add_argument('--proposal',required=True);a.add_argument('--output',required=True);a.add_argument('--setups',type=int,default=2)
    a.add_argument('--partition',choices=('train','validation'),default='train');a.add_argument('--seed',type=int,default=83000000)
    a.add_argument('--tables',type=int,default=24);a.add_argument('--mono')
    a=sub.add_parser('fit');a.add_argument('--collections',nargs='+',required=True);a.add_argument('--output',required=True)
    a.add_argument('--epochs',type=int,default=30);a.add_argument('--seed',type=int,default=17051)
    a=sub.add_parser('archive');a.add_argument('--proposal',required=True);a.add_argument('--encoder',required=True);a.add_argument('--output',required=True)
    a.add_argument('--discovery',nargs='*',default=[]);a.add_argument('--qualification',nargs='*',default=[]);a.add_argument('--previous')
    a.add_argument('--method',choices=('fps','kmeans'),default='fps');a.add_argument('--count',type=int,default=16)
    a=p.parse_args();require(1<=a.threads<=16,'Invalid tensor thread count');torch.set_num_threads(a.threads)
    require(a.device!='cuda' or torch.cuda.is_available(),'CUDA unavailable')
    if a.command=='propose':
        from .discovery import create
        result=create(a.root,a.ready,a.output,count=a.count,seed=a.seed,seconds=a.seconds,workers=a.workers,decision_mode=a.decision_mode,from_archive=a.from_archive)
    elif a.command=='collect':
        from .discovery import execute
        result=execute(a.root,a.proposal,a.output,setups=a.setups,partition=a.partition,seed=a.seed,tables=a.tables,device=a.device,mono=a.mono)
    elif a.command=='fit':
        from .descriptor import fit
        result=fit(a.root,a.collections,a.output,epochs=a.epochs,seed=a.seed,device=a.device)
    else:
        from .archive import build
        result=build(a.root,a.proposal,a.encoder,a.discovery,a.qualification,a.output,previous=a.previous,method=a.method,count=a.count,device=a.device)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
