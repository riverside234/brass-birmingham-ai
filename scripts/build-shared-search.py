"""Build shared v5 search (official v4 ancestry) without replacing frozen banks."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import read
from brass_ai.search.legacy_runtime import build,publish

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--bank',action='append');p.add_argument('--managed');p.add_argument('--compiler');p.add_argument('--mono')
    p.add_argument('--compile',action='store_true',help='Use the local pinned compiler instead of the portable prebuilt')
    p.add_argument('--publish',action='store_true',help='Compile locally and publish only the authored executable and source receipt for lab use')
    a=p.parse_args();root=a.root.resolve()
    pool=read(root/'training/opponents/pool.json')
    banks=sorted(set(a.bank or [p.split('/')[0] for kind in ('training','evaluation') for p in pool[kind]]))
    for bank in banks:
        try:
            ref=build(root,bank,managed=a.managed,compiler=a.compiler,mono=a.mono,use_prebuilt=not(a.compile or a.publish))
            bundle=publish(root,ref).relative_to(root).as_posix() if a.publish else None
        except (ValueError,OSError) as error:p.exit(1,'Shared search setup failed: '+str(error)+'\n')
        print(json.dumps(dict(bank=bank,search_runtime=ref['id'],strength_qualified=False,bundle=bundle)),flush=True)

if __name__=='__main__':main()
