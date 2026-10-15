"""Reuse existing Features-4 data for tiny v9 IL, self-play handoff and ONNX.

One IL update; no conversion, new games, PPO iterations or timed teacher search.
Requires a tiny existing manifest (one training and one validation game).
"""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import torch

from brass_ai.bank import sha
from brass_ai.learning.contracts import SOURCE_ROOT,compatibility,project_path,read_json,require
from brass_ai.learning.checkpoint import save_checkpoint,load_checkpoint
from brass_ai.recovery.demonstrations import train_bc,completed_imitation,load_dataset
from brass_ai.recovery.config import validate_config
from brass_ai.recovery.utility import migrate,reward_spec
from brass_ai.personality.model import import_parent
from brass_ai.personality.laptop import export as portable
from brass_ai.step5.league import implementation_id,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();root=SOURCE_ROOT
    manifest=project_path(root,args.manifest);out=project_path(root,args.output)
    meta=read_json(manifest)
    require(len(meta['entries'])==2 and {e['split'] for e in meta['entries']}=={'train','validation'},
            'Smoke accepts only one complete game per train/validation partition')
    require(not out.exists(),'Use a new isolated smoke output')
    files={manifest}
    for entry in meta['entries']:
        p=project_path(root,entry['path']);files.add(p)
        files.update(p.parent.glob(p.name+'.features-v4.*'))
    before={str(p):sha(p) for p in files}
    out.mkdir(parents=True);began=time.monotonic();source_id=implementation_id();torch.set_num_threads(2)
    config=read_json(root/'configs/step5/joint7-latest.json')
    update=dict(config['update'],effective_batch_size=16,microbatch_size=8,max_candidate_cells=8192,epochs=1)
    selected={}
    def tiny_loader(*a,**kw):
        data,header=load_dataset(*a,**kw)
        selected[a[2]]=dict(available=len(data),used=min(8,len(data)))
        return data[:8],header
    # Production validation/loader and BC optimizer; only the sample count is a fixture.
    with patch('brass_ai.recovery.demonstrations.load_dataset',side_effect=tiny_loader):
        bc=train_bc(root,out/'imitation',manifest,None,update,epochs=1,min_epochs=1,patience=1,workers=1)
    model,origin=completed_imitation(root,out/'imitation')
    require(bc['history'][0]['train']['optimizer_steps']==1,'Expected one tiny IL update')
    # Use the actual documented configuration CLI, binding both IL and rehearsal.
    command=[sys.executable,str(root/'scripts/feature-engineering.py'),'configure',
             '--imitation-run',str(out/'imitation'),'--output',str(out/'selfplay.json')]
    subprocess.run(command,cwd=root,check=True,capture_output=True,text=True)
    resolved=validate_config(read_json(out/'selfplay.json'))
    require(resolved['imitation_checkpoint']==origin and resolved['rehearsal_manifest']==origin['manifest'],
            'Configuration lost IL/rehearsal lineage')
    for field in config.keys()-{'imitation_checkpoint','rehearsal_manifest'}:
        require(resolved[field]==config[field],'Configuration changed '+field)
    weights={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'])
    receipt=migrate(model,optimizer,reward_spec(config))
    for name,value in weights.items():torch.testing.assert_close(value,model.state_dict()[name],rtol=0,atol=0)
    require(not optimizer.state,'Fresh PPO optimizer must be empty at IL handoff')
    save_checkpoint(out/'utility.pt',model,optimizer,step=0,training=dict(diagnostic=True,origin=origin))
    model,_=load_checkpoint(out/'utility.pt')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(661);actor,actor_receipt=import_parent(model)
    samples=portable.samples
    with patch.object(portable,'samples',side_effect=lambda *a,**kw:samples(*a,**kw)[:2]):
        exported=portable.export(actor,out/'actor.onnx')
    require(before=={str(p):sha(p) for p in files},'Existing IL dataset/cache files changed')
    require(source_id==implementation_id(),'Source changed during smoke')
    write_json(out/'report.json',dict(format='compact-widths-smoke/1.0.0',passed=True,
        implementation=source_id,elapsed_seconds=time.monotonic()-began,
        reused_manifest=manifest.relative_to(root).as_posix(),source_files_unchanged=True,
        compatibility_changes=[k for k,v in compatibility().items() if meta['compatibility'][k]!=v],
        IL=dict(epochs=1,optimizer_steps=1,selection=selected,origin=origin),
        handoff=dict(IL_weights_exact=True,PPO_optimizer_empty=True,utility=receipt,personality=actor_receipt),
        teacher_work_budget=resolved['teacher_work_budget'],
        teacher_workers_per_game=resolved['execution']['teacher_search_workers_per_turn'],
        export=exported,scope='One tiny IL update and export parity; no new games, PPO update, timed search or strength measurement.'))
    print(out/'report.json')


if __name__=='__main__':main()
