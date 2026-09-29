"""Compare all six frozen S1 styles on the same real permitted positions.

Deterministic fixed-work diagnostic; not a wall-clock strength tournament.
Position selection is chronological and independent of each profile's decisions.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from brass_ai.bank import LAB_ROOT,FrozenOpponent,digest,read,sha
from brass_ai.learning.replay import read_shards
from brass_ai.qualification import STYLES,write_new

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT);p.add_argument('--bank',required=True)
p.add_argument('--shard',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=a.root.resolve()
games=read_shards([root/a.shard]);positions=[];seen=set()
for game in games:
 for row in game['records']:
  c=row['view']['context'];key=(game['game_id'],c['era'],c['round'])
  if c['phase']=='action' and key not in seen:
   seen.add(key);positions.append(row)
opponent=FrozenOpponent(root,a.bank+'/balanced-v2',purpose='diagnostic');rows=[]
try:
 profiles=read(root/'training/opponents/bank'/a.bank/'manifest.json')['payload']['profiles']
 for row in positions:
  results={}
  for style in STYLES:
   d=opponent.worker.call('plan',observation=row['view'],choices=row['choices'],profile=profiles[style+'-v2'],fixed_work=True,sampling_seed=1)
   results[style]=dict(selected=d['selected'],nodes=d['nodes'])
  rows.append(dict(step=row['step'],view_hash=digest(row['view']),context=row['view']['context'],profiles=results))
  print(json.dumps(dict(step=row['step'],distinct=len({r['selected']['signature'] for r in results.values()}))),flush=True)
finally:opponent.close()
differences={s:sum(r['profiles'][s]['selected']['signature']!=r['profiles']['balanced']['selected']['signature'] for r in rows) for s in STYLES if s!='balanced'}
report=dict(format='legacy-style-diagnostic/1.0.0',bank=a.bank,source_shard_sha256=sha(root/a.shard),positions=len(rows),
 fixed_work=True,selection='first ordinary action position in each era/round from one complete game',differences_from_balanced=differences,
 each_specialist_differed=all(n>0 for n in differences.values()),rows=rows)
write_new(root/a.output,report);print(json.dumps(differences))
