"""P4 shared-network roster selection, matched calibration and blind review."""
import argparse
import json
from pathlib import Path
import torch
from brass_ai.learning.contracts import LAB_ROOT, require


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=LAB_ROOT)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--threads',type=int,default=4)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('select');a.add_argument('--archive',required=True);a.add_argument('--output',required=True)
    a.add_argument('--groups',type=int,default=16);a.add_argument('--members',type=int,default=8)
    a.add_argument('--method',choices=('fps','kmeans'),default='fps');a.add_argument('--seed',type=int,default=19019)
    a=sub.add_parser('cards');a.add_argument('--roster',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('plan');a.add_argument('--roster',required=True);a.add_argument('--output',required=True)
    a.add_argument('--setups',type=int,default=16);a.add_argument('--repetitions',type=int,default=2)
    a.add_argument('--seed',type=int,default=121000000);a.add_argument('--labels')
    a=sub.add_parser('collect');a.add_argument('--panel',required=True);a.add_argument('--output',required=True)
    a.add_argument('--start',type=int,default=0);a.add_argument('--games',type=int,default=128)
    a.add_argument('--tables',type=int,default=24);a.add_argument('--mono')
    a=sub.add_parser('check');a.add_argument('--panel',required=True);a.add_argument('--chunks',nargs='+',required=True)
    a.add_argument('--output',required=True);a.add_argument('--review')
    a=p.parse_args();require(1<=a.threads<=16,'Invalid tensor thread count');torch.set_num_threads(a.threads)
    require(a.device!='cuda' or torch.cuda.is_available(),'CUDA unavailable')
    if a.command=='select':
        from .roster import select
        result=select(a.root,a.archive,a.output,groups=a.groups,members=a.members,method=a.method,seed=a.seed)
    elif a.command=='cards':
        from .roster import cards
        result=cards(a.root,a.roster,a.output)
    elif a.command=='plan':
        from .panels import plan
        result=plan(a.root,a.roster,a.output,setups=a.setups,repetitions=a.repetitions,seed=a.seed,labels=a.labels)
    elif a.command=='collect':
        from .panels import execute
        result=execute(a.root,a.panel,a.output,start=a.start,games=a.games,tables=a.tables,device=a.device,mono=a.mono)
    else:
        from .evaluation import check
        result=check(a.root,a.panel,a.chunks,a.output,review=a.review,device=a.device)
    summary={k:result[k] for k in ('format','id','counts','full_size','p4_qualified','release_qualified','checks','scope') if k in result}
    if 'jobs' in result:summary['jobs']=len(result['jobs'])
    summary['output']=str(a.output)
    print(json.dumps(summary,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
