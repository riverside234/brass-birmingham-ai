"""Choose a reproducible bootstrap table from qualified immutable pool identities."""
import argparse
from pathlib import Path
import random
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import LAB_ROOT,read,admission,require

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT);p.add_argument('--seed',type=int,required=True)
p.add_argument('--purpose',choices=['training','evaluation'],default='training');a=p.parse_args()
pool=read(a.root/'training/opponents/pool.json');ids=sorted(pool[a.purpose]);require(ids,'No qualified opponents in selected pool')
for identity in ids:admission(a.root,identity,a.purpose)
rng=random.Random(a.seed)
table=rng.sample(ids,4) if len(ids)>=4 else rng.choices(ids,k=4)
print(json.dumps(dict(purpose=a.purpose,policies=table,sampling_seed=a.seed,distribution='uniform available qualified versions; four distinct profiles when available')))
