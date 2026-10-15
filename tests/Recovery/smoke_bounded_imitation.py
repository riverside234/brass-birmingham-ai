"""One small real IL epoch through disk-backed ingress; no gameplay/evaluation.

Accepts only one train and one validation shard (at most 1,024 rows per split).
Outputs must be new. Source replays, manifests and prior checkpoints stay intact.
"""
import argparse
from pathlib import Path
import time
from unittest.mock import patch

import torch

from brass_ai.bank import read,sha
from brass_ai.learning.contracts import SOURCE_ROOT,project_path,require
from brass_ai.recovery import demonstrations as demos
from brass_ai.recovery.config import UPDATE_V21
from brass_ai.step5.league import implementation_id,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=SOURCE_ROOT;manifest=project_path(root,args.manifest);meta=read(manifest)
    require(len(meta['entries'])==2 and {e['split'] for e in meta['entries']}=={'train','validation'},
            'Smoke requires exactly one train and one validation game')
    out=project_path(root,args.output);require(not out.exists(),'Use a new isolated smoke directory')
    sources={manifest}
    for e in meta['entries']:
        path=project_path(root,e['path']);sources.add(path)
        sources.update(path.parent.glob(path.name+'.features-v4.*'))
    before={str(p):sha(p) for p in sources};implementation=implementation_id()
    began=time.monotonic();torch.set_num_threads(2)
    config=dict(UPDATE_V21,effective_batch_size=1024,microbatch_size=8,max_candidate_cells=8192)
    production=demos.load_dataset
    def bounded(*a,**kw):
        rows,header=production(*a,**kw)
        if len(rows)>1024:
            rows.close();raise ValueError('Smoke refuses more than 1,024 rows per split')
        return rows,header
    with patch.object(demos,'load_dataset',bounded):
        first=demos.train_bc(root,out/'imitation',manifest,None,config,epochs=1,min_epochs=1,patience=1,workers=1)
        unchanged={str(p):sha(p) for p in (out/'imitation').iterdir() if p.suffix=='.pt' or p.name=='state.json'}
        repeated=demos.train_bc(root,out/'imitation',manifest,None,config,epochs=1,min_epochs=1,patience=1,workers=1)
    require(first['history'][0]['train']['optimizer_steps']==1,'Smoke must make exactly one optimizer update')
    require(first['history']==repeated['history'],'Repeated completed run changed history')
    require(unchanged=={p:sha(Path(p)) for p in unchanged},'Repeat changed checkpoints/state')
    require(all(r['cache_hits']==r['shards'] for r in repeated['preparation'].values()),'Cache was not reused')
    _,origin=demos.completed_imitation(root,out/'imitation')
    require(before=={str(p):sha(p) for p in sources},'Source dataset/cache changed')
    require(implementation==implementation_id(),'Source changed during smoke')
    report=dict(passed=True,implementation=implementation,elapsed_seconds=time.monotonic()-began,
        preparation=first['preparation'],repeated_preparation=repeated['preparation'],
        training=first['history'][0]['train'],validation=first['history'][0]['validation'],
        source_files_unchanged=True,repeat_checkpoints_unchanged=True,origin=origin,
        scope='One bounded CPU IL epoch, completed-run reuse and handoff; no long training, search or playing-strength evaluation.')
    write_json(out/'report.json',report);print(out/'report.json')


if __name__=='__main__':main()
