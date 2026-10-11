"""P2 lab-only search preparation, reserved teachers and policy distillation."""
import argparse
import json
from pathlib import Path
import torch
from brass_ai.learning.contracts import LAB_ROOT,require
from brass_ai.step5.league import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=LAB_ROOT)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--threads',type=int,default=4)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--handoff',required=True);a.add_argument('--pilot',required=True);a.add_argument('--output',required=True)
    a.add_argument('--games',type=int,default=20);a.add_argument('--calibration-games',type=int,default=8)
    a.add_argument('--workers',type=int,default=4);a.add_argument('--mono');a.add_argument('--distilled')
    a=sub.add_parser('teach');a.add_argument('--ready',required=True);a.add_argument('--pilot',required=True);a.add_argument('--run',required=True)
    a.add_argument('--tables',type=int,default=4);a.add_argument('--mono')
    a=sub.add_parser('distill');a.add_argument('--ready',required=True);a.add_argument('--collection',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('continue-pilot');a.add_argument('--pilot',required=True);a.add_argument('--distilled',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('probe');a.add_argument('--ready',required=True);a.add_argument('--seconds',type=float,default=12.)
    a.add_argument('--workers',type=int,default=4);a.add_argument('--repeats',type=int,default=2);a.add_argument('--unpruned',action='store_true')
    a.add_argument('--mono');a.add_argument('--report',type=Path)
    a=p.parse_args();require(1<=a.threads<=16,'Invalid tensor thread count');torch.set_num_threads(a.threads)
    require(a.device!='cuda' or torch.cuda.is_available(),'CUDA unavailable')
    if a.command=='prepare':
        from .readiness import prepare
        result=prepare(a.root,a.handoff,a.pilot,a.output,games=a.games,calibration_games=a.calibration_games,
            workers=a.workers,device=a.device,mono=a.mono,distilled=a.distilled)
    elif a.command=='teach':
        from .teachers import collect_due
        result=collect_due(a.root,a.ready,a.pilot,a.run,tables=a.tables,device=a.device,mono=a.mono)
    elif a.command=='distill':
        from .teachers import distill
        result=distill(a.root,a.ready,a.collection,a.output,device=a.device)
    elif a.command=='continue-pilot':
        from .continuation import continue_pilot
        result=continue_pilot(a.root,a.pilot,a.distilled,a.output)
    else:
        from .probe import probe
        result=probe(a.root,a.ready,seconds=a.seconds,workers=a.workers,repeats=a.repeats,device=a.device,mono=a.mono,unpruned=a.unpruned)
        if a.report:write_json(a.report,result)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
