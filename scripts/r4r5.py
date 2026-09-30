"""Additional search experiments. Current Step 5 runs are never resumed or edited."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.learning.contracts import LAB_ROOT,read_json,require
from brass_ai.searchlab.runtime import install

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=LAB_ROOT);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--mono')
    p.add_argument('--config',default='configs/searchlab/experiment.json',help='Larger lab preset; use configs/searchlab/small.json for previous small defaults')
    sub=p.add_subparsers(dest='command',required=True);sub.add_parser('install-runtime')
    c=sub.add_parser('compare');c.add_argument('--checkpoint',required=True);c.add_argument('--references',nargs='+',required=True)
    c.add_argument('--output',required=True);c.add_argument('--setups',type=int);c.add_argument('--seed',type=int,default=3910000)
    c.add_argument('--partition',choices=['validation','test']);c.add_argument('--variants',nargs='+',choices=['s2','r4','s3'])
    d=sub.add_parser('distill-s3');d.add_argument('--checkpoint',required=True);d.add_argument('--output',required=True)
    d.add_argument('--games',type=int);d.add_argument('--seed',type=int,default=4920000)
    d.add_argument('--minibatch-size',type=int);d.add_argument('--distill-steps',type=int)
    for q in (c,d):
        q.add_argument('--seconds',type=float);q.add_argument('--future-turns',type=int,choices=[1,2]);q.add_argument('--simulations',type=int)
        q.add_argument('--workers',type=int,help='Independent processes; use 1 for a promotable equal-budget comparison')
        q.add_argument('--batch-size',type=int,help='Inference batch limit; current schema shares batches across workers')
    m=sub.add_parser('promote');m.add_argument('--report',required=True);m.add_argument('--variant',choices=['r4','s3'],required=True)
    m.add_argument('--output',default='artifacts/searchlab/promoted.json')
    return p


def main(argv=None):
    a=parser().parse_args(argv);require(a.root.is_absolute(),'Use an absolute project root');root=a.root.resolve()
    if a.command=='install-runtime':result=install(root)
    else:
        import torch
        from brass_ai.learning.contracts import project_path
        from brass_ai.searchlab.experiment import compare,promote,distill_s3,validate_config
        require(a.device!='cuda' or torch.cuda.is_available(),'CUDA requested but unavailable')
        config=validate_config(read_json(project_path(root,a.config)))
        for field in ('seconds','future_turns','simulations','batch_size','minibatch_size','distill_steps'):
            if getattr(a,field,None) is not None:config[field]=getattr(a,field)
        if 'execution' in config:
            config['execution']=dict(config['execution'],batch_size=config['batch_size'])
            config['update']=dict(config['update'],microbatch_size=config['minibatch_size'])
        if getattr(a,'workers',None) is not None:
            config['distill_workers' if a.command=='distill-s3' else 'workers']=a.workers
        validate_config(config)
        if a.command=='compare':
            result=compare(root,a.checkpoint,a.references,a.output,config,setups=a.setups,seed=a.seed,
                partition=a.partition,variants=a.variants,device=a.device,mono=a.mono)['summary']
        elif a.command=='distill-s3':result=distill_s3(root,a.checkpoint,a.output,config,games=a.games,seed=a.seed,device=a.device,mono=a.mono)
        else:result=promote(root,a.report,a.variant,a.output,device=a.device,mono=a.mono)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
