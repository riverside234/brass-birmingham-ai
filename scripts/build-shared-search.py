"""Build shared official v4 search for IL/self-play without replacing frozen banks."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import read
from brass_ai.search.legacy_runtime import build

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--bank',action='append');p.add_argument('--managed');p.add_argument('--compiler');p.add_argument('--mono')
    a=p.parse_args();root=a.root.resolve()
    pool=read(root/'training/opponents/pool.json')
    banks=sorted(set(a.bank or [p.split('/')[0] for kind in ('training','evaluation') for p in pool[kind]]))
    for bank in banks:
        ref=build(root,bank,managed=a.managed,compiler=a.compiler,mono=a.mono)
        print(json.dumps(dict(bank=bank,search_runtime=ref['id'],strength_qualified=False)),flush=True)

if __name__=='__main__':main()
