"""Preregister, run/resume, summarize and admit immutable four-player opponents."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import LAB_ROOT, read, below
from brass_ai.qualification import make_spec, run_spec, summarize, publish_pool, write_new

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('preregister');q.add_argument('--bank',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('run');q.add_argument('--spec',required=True);q.add_argument('--output',required=True);q.add_argument('--mono')
    q=sub.add_parser('report');q.add_argument('--spec',required=True);q.add_argument('--matches',required=True);q.add_argument('--checks',required=True);q.add_argument('--output',required=True);q.add_argument('--admit',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    if a.command=='preregister':
        spec=make_spec(root,a.bank);write_new(below(root,a.output),spec);print(spec['id'])
    elif a.command=='run':run_spec(root,read(below(root,a.spec)),below(root,a.output),mono=a.mono)
    else:
        report=summarize(root,read(below(root,a.spec)),below(root,a.matches),read(below(root,a.checks)))
        path=below(root,a.output);write_new(path,report)
        if a.admit:publish_pool(root,report,path)
        print({'games':report['total_games'],'failed':report['failed_games'],'reply_comparison':report['reply_comparison']})

if __name__=='__main__':main()
