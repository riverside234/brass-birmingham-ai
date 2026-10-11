"""One real four-player P4 collection/check, using explicitly unqualified fixtures."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT/'src',ROOT/'tests/R2',ROOT/'tests/Personality'):sys.path.insert(0,str(path))

import torch
from smoke_search import fixture
from test_deployment import candidate
from test_repertoire import records
from brass_ai.bank import sha
from brass_ai.learning.batching import model_id
from brass_ai.learning.contracts import digest,require
from brass_ai.personality.repertoire import descriptor
from brass_ai.personality.repertoire.contracts import identity as repertoire_identity
from brass_ai.personality.repertoire.traces import schema
from brass_ai.personality.search.contracts import controller,Profile
from brass_ai.personality.search.readiness import load
from brass_ai.personality.deployment import panels,evaluation,roster
from brass_ai.personality.deployment.contracts import sealed
from brass_ai.search.common import SearchConfig
from brass_ai.step5.league import implementation_id,write_json
from brass_ai.personality.contracts import source_id


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--mono')
    a=p.parse_args();out=a.output.resolve();require(out.is_relative_to(ROOT/'artifacts') and not out.exists(),'New smoke directory required')
    torch.set_num_threads(2);torch.manual_seed(11791);start=time.monotonic()
    before=implementation_id();p1=source_id();ready=fixture(out)
    _,parents,r,_,pool=load(ROOT,ready)
    enc=out/'encoder';enc.mkdir();payload,report=descriptor.fit_arrays(records(),epochs=1)
    torch.save(payload,enc/'encoder.pt')
    manifest=dict(format=descriptor.VERSION,contract=repertoire_identity(),schema=schema(),weights=dict(path='encoder.pt',sha256=sha(enc/'encoder.pt')),
                  training_groups=['authored-0','authored-1'],report=report,scope='Authored fixture encoder; not learned style evidence')
    manifest['id']=digest(manifest);write_json(enc/'manifest.json',manifest)
    v={k:x for k,x in candidate().items() if k!='id'}
    v['shared_model']=dict(ready=ready.relative_to(ROOT).as_posix(),ready_id=r['id'],model_id=r['model']['model_id'])
    v['pool_id']=pool;v['decision_mode']='direct';cfg=SearchConfig(**v['search']).validate()
    for m in v['members']:
        c=controller(r['model']['model_id'],model_id(parents[0]),Profile.parse(m['profile']),cfg,pool,decision_mode='direct')
        m.update(model_id=r['model']['model_id'],candidate_id=c['candidate_id'])
    v['controls']=[controller(r['model']['model_id'],model_id(parents[0]),Profile.parse(c['profile']),cfg,pool,decision_mode='direct') for c in v['controls']]
    v['encoder']=dict(path=enc.relative_to(ROOT).as_posix(),id=manifest['id'],sha256=sha(enc/'manifest.json'))
    v['dependencies']=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in (ready/'ready.json',ready/'model.pt',enc/'manifest.json',enc/'encoder.pt')]
    directory=out/'candidate';directory.mkdir();write_json(directory/'manifest.json',sealed(v))
    planned=panels.plan(ROOT,directory,out/'panel',setups=2,repetitions=1,seed=133000000)
    chunk=panels.execute(ROOT,out/'panel',out/'chunk-0000',games=1,tables=1,mono=a.mono)
    checked=evaluation.check(ROOT,out/'panel',[out/'chunk-0000'],out/'checked')
    require(not checked['p4_qualified'] and not checked['checks']['complete_panel'] and checked['games']==1,'Incomplete fixture panel received qualification')
    require(not roster.load(out/'checked')['release_qualified'],'Smoke cannot qualify deployment')
    require(before==implementation_id() and p1==source_id(),'Step 5/P0/P1 source changed')
    result=dict(scope='One native four-player direct preview with untrained fixture model; no skill/roster qualification',
                seconds=time.monotonic()-start,planned_games=len(planned['jobs']),collected=1,checks=checked['checks'],
                step5_implementation=before,p1_source=p1,chunk_id=chunk['id'])
    write_json(out/'smoke-report.json',result);print(result,flush=True)


if __name__=='__main__':main()
