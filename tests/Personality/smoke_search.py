"""Authored model/readiness fixtures, real rules-host P2 search, no qualification."""
import argparse
import copy
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT/'src',ROOT/'tests/R2',ROOT/'tests/Personality'):sys.path.insert(0,str(path))

import torch
from personality_fixtures import qualified_run
from brass_ai.bank import sha
from brass_ai.learning.contracts import digest,require,read_json
from brass_ai.step5.league import write_json,implementation_id,seeds_for
from brass_ai.personality import handoff,checkpoint
from brass_ai.personality.contracts import source_id,FACTORS
from brass_ai.personality.model import import_parent
from brass_ai.personality.search.contracts import READY,identity,Profile,search_config,controller
from brass_ai.personality.search.readiness import load
from brass_ai.personality.search.collection import collect,read
from brass_ai.personality.search.probe import probe
from brass_ai.personality.repertoire.traces import from_collection
from brass_ai.learning.batching import model_id


def fixture(out):
    source=qualified_run(out/'fixture-source',traces=False)
    handoff.create(ROOT,source,out/'handoff',trace_shards=0,reference_count=1)
    manifest,parent=handoff.load(out/'handoff');model,_=import_parent(parent)
    with torch.no_grad():
        for name,head in model.goal_heads.items():
            head.weight.zero_();head.bias.fill_(-3.)
            value=dict(family='industry_sequence',industry='CottonMill',region='unspecified',horizon=4)[name]
            head.bias[FACTORS[name].index(value)]=3.
        model.confidence_head.weight.zero_();model.confidence_head.bias.fill_(3.)
    directory=out/'ready';ident=checkpoint.save(directory/'model.pt',model,training=dict(fixture=True))
    receipt=dict(format=READY,contract=identity(),model=dict(path='model.pt',sha256=sha(directory/'model.pt'),model_id=ident),
        handoff=(out/'handoff').relative_to(ROOT).as_posix(),handoff_sha256=sha(out/'handoff/manifest.json'),
        reward=manifest['reward'],utility_ready=True,goal_calibration=dict(temperature=1.,bias_ready=True,scope='Authored smoke calibration fixture only'),
        data=[],scope='Untrained model/authored readiness fixture, never production qualification')
    receipt['id']=digest(receipt);write_json(directory/'ready.json',receipt)
    return directory


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--mono')
    a=p.parse_args();out=a.output.resolve();require(out.is_relative_to((ROOT/'artifacts').resolve()) and not out.exists(),'New artifacts smoke directory required')
    torch.set_num_threads(2);torch.manual_seed(7183);began=time.monotonic();before=implementation_id();p1=source_id()
    ready=fixture(out)
    timing=probe(ROOT,ready,seconds=12.,workers=4,repeats=2,mono=a.mono)
    write_json(out/'paired-root-timing.json',timing)
    model,parents,r,_,pool=load(ROOT,ready);cfg=search_config(r['reward'],seconds=.35,workers=4)
    profiles=[Profile(style='industrialist')]*4
    controllers=[controller(model_id_value,model_id(parents[0]),p,cfg,pool)['candidate_id']
        for model_id_value,p in [(r['model']['model_id'],p) for p in profiles]]
    seed=seeds_for(93200000,1,'train')[0][0]
    jobs=[dict(index=0,seed=seed,sampling_seed=172,partition='train',neural_seats=list(range(4)),profiles=[p.to_dict() for p in profiles],
        controllers=controllers,parent_indices=[0]*4)]
    game=collect(ROOT,ready,out/'teacher',jobs,cfg,tables=1,mono=a.mono,teacher_game=True)
    manifest,pairs=read(out/'teacher',teacher_game=True);g,side=pairs[0]
    from brass_ai.personality.repertoire.traces import trajectory
    shapes=[trajectory(g,s)[0].shape for s in range(4)]
    require(all(len(x)==2 and x[0]==32 for x in shapes),'Public per-seat trajectory failed')
    require(before==implementation_id() and p1==source_id(),'Step 5/P0/P1 fingerprint changed')
    require(all(sum(x['scheduled_turns'] for x in seat.values())==4 for seat in side['phases']),'Teacher reservation lost')
    # One inexpensive P3 direct-policy table against frozen parent references.
    direct_jobs=copy.deepcopy(jobs)
    direct_jobs[0]['neural_seats']=[0]
    direct_jobs[0]['controllers']=[controller(r['model']['model_id'],model_id(parents[0]),p,cfg,pool,decision_mode='direct')['candidate_id'] for p in profiles]
    direct=collect(ROOT,ready,out/'direct',direct_jobs,cfg,tables=1,mono=a.mono,decision_mode='direct')
    _,features=from_collection(out/'direct',include_parents=True)
    require(len(features)==4,'Discovery must extract all four permitted public trajectories')
    result=dict(scope='Fixture model and readiness only; two real full games and paired roots, not strength qualification',
        seconds=time.monotonic()-began,step5_implementation=before,p1_source=p1,timing=timing,
        game=dict(records=len(g['records']),result=g['result'],phases=side['phases'],search=side['telemetry'],inference=manifest['inference']),
        trajectory_shapes=shapes,direct=dict(games=len(direct['records']),seconds=direct['seconds'],trajectories=len(features),inference=direct['inference']))
    write_json(out/'report.json',result);print(f'P2 native smoke passed in {result["seconds"]:.1f}s: {out / "report.json"}',flush=True)


if __name__=='__main__':main()
