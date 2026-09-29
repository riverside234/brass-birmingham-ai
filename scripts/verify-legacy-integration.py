"""Validate a real complete-game bank shard, learn four-seat values, reject misuse."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from brass_ai.bank import LAB_ROOT,load_bank,require,sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.legacy_policy import LegacyPolicy
from brass_ai.learning.model import FourSeatNet,ModelConfig
from brass_ai.learning.replay import read_shards
from brass_ai.learning.training import prepared,supervised_updates,on_policy_update
from brass_ai.qualification import write_new

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=LAB_ROOT);p.add_argument('--bank',required=True)
    p.add_argument('--shard',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=a.root.resolve()
    load_bank(root,a.bank);games=read_shards([root/a.shard]);rows=[r for g in games for r in g['records']]
    require(all(r['behavior']['kind']=='external' and r['behavior']['policy_id'].startswith(a.bank+'/') and r['teacher'] is None for r in rows),'Wrong shard provenance')
    require(all(sum(r['style'])==1 for r in rows),'Missing per-profile style context')
    require(any(any(r['intent']) for r in rows),'Missing permitted intent features')
    require(all({r['seat'] for r in g['records']}=={0,1,2,3} for g in games),'Missing seat trajectories')
    torch.manual_seed(710);torch.set_num_threads(2)
    model=FourSeatNet(ModelConfig.preset('small'));before=model_id(model);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    metrics=supervised_updates(model,optimizer,prepared(games),2,8,mode='value')
    require(before!=model_id(model),'Value training failed to change weights')
    rejected=False
    try:on_policy_update(model,optimizer,games)
    except ValueError:rejected=True
    require(rejected,'Legacy behavior incorrectly accepted as current neural policy')
    first=rows[0];identity=first['behavior']['policy_id'];policy=LegacyPolicy(root,identity,purpose='diagnostic')
    try:
        one,two=random.Random(2),random.Random(2);view,choices=first['view'],first['choices']
        chosen=policy.select_batch([(view,choices,one)])[0][0]
        require(0<=chosen<len(choices),'Illegal bank choice')
        policy.select_batch([(view,choices,two)])
        require(len(policy.contexts)==2,'Separate game/seat contexts were merged')
        # A stale queued ID is never applied blindly. Restoring the same revision
        # discards branch state and reconstructs a legal incumbent.
        policy.contexts[one]['queue'].clear();policy.contexts[one]['queue'].append('invalid-cached-id')
        repeated=policy.select_batch([(view,choices,one)])[0][0]
        require(0<=repeated<len(choices),'Restore used a stale action')
        del two
        require(len(policy.contexts)==1,'Completed game contexts leaked')
    finally:policy.close()
    report=dict(passed=True,bank=a.bank,shard_sha256=sha(root/a.shard),games=len(games),decisions=len(rows),
        four_seat_values_trained=True,legacy_on_policy_rejected=True,style_and_intent_preserved=True,
        separate_contexts=True,restore_invalidates_cached_action=True,last_metrics=metrics[-1])
    write_new(root/a.output,report);print(json.dumps(report))

if __name__=='__main__':main()
