"""Freeze original source and profile settings; binaries stay private and ignored."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import freeze,LAB_ROOT

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT)
p.add_argument('--profiles',nargs='+',required=True);p.add_argument('--fixed-work',action='store_true')
args=p.parse_args()
print(freeze(args.root,args.profiles,fixed_work=args.fixed_work))
